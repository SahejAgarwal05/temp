#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "graphics.cuh"
#include <thrust/execution_policy.h>
#include <thrust/sort.h>


constexpr int MAX_LEAF_SIZE = 2;
constexpr int BLOCK_SIZE = 256;
constexpr unsigned long IDX_MASK = 0xffffffffULL;
using namespace std;
struct Hit {
    HitRecord record;
    int object_index = -1;
    int triangle_index = -1;
    __device__ bool hit() const { return object_index != -1; }
};

struct ObjectMaterial {
    Vec3 emission;
    Vec3 albedo;
};

// Preorder layout: the left child is index + 1; rightNodeIndex == -1 marks a leaf.
// Every range begins at a multiple of MAX_LEAF_SIZE, including internal ranges.
struct LeafNode {
    Vec3 highCorner;
    Vec3 lowCorner;
    int rightNodeIndex;
    int n;
    int startIndex;
    int parent_id;
};

struct GPUTriangle {
    Triangle triangle;
    int object_index;
    int triangle_index; // Original flattened ID, preserved through sorting.
    int node_index;
    Vec3 highCorner = Vec3(fmaxf(triangle.p0.x, fmaxf(triangle.p1.x, triangle.p2.x)),
                           fmaxf(triangle.p0.y, fmaxf(triangle.p1.y, triangle.p2.y)),
                           fmaxf(triangle.p0.z, fmaxf(triangle.p1.z, triangle.p2.z)));
    Vec3 lowCorner = Vec3(fminf(triangle.p0.x, fminf(triangle.p1.x, triangle.p2.x)),
                          fminf(triangle.p0.y, fminf(triangle.p1.y, triangle.p2.y)),
                          fminf(triangle.p0.z, fminf(triangle.p1.z, triangle.p2.z)));
    __host__ __device__ GPUTriangle() = default;
    __host__ __device__ GPUTriangle(int object_index, int triangle_index, const Triangle& triangle):
        triangle(triangle), object_index(object_index), triangle_index(triangle_index), node_index(-1) {
            highCorner = Vec3(fmaxf(triangle.p0.x, fmaxf(triangle.p1.x, triangle.p2.x)),
                             fmaxf(triangle.p0.y, fmaxf(triangle.p1.y, triangle.p2.y)),
                             fmaxf(triangle.p0.z, fmaxf(triangle.p1.z, triangle.p2.z)));
            lowCorner = Vec3(fminf(triangle.p0.x, fminf(triangle.p1.x, triangle.p2.x)),
                            fminf(triangle.p0.y, fminf(triangle.p1.y, triangle.p2.y)),
                            fminf(triangle.p0.z, fminf(triangle.p1.z, triangle.p2.z)));
        }
};

struct NodeBox {
    Vec3 lowCorner = Vec3(INF, INF, INF);
    Vec3 highCorner = Vec3(-INF, -INF, -INF);

    __host__ __device__  inline void insert(const Vec3& point) {
        lowCorner = Vec3(fminf(lowCorner.x, point.x), fminf(lowCorner.y, point.y),
                         fminf(lowCorner.z, point.z));
        highCorner = Vec3(fmaxf(highCorner.x, point.x), fmaxf(highCorner.y, point.y),
                          fmaxf(highCorner.z, point.z));
    }
    __host__ __device__ inline void insert(const Triangle& triangle) {
        insert(triangle.p0); insert(triangle.p1); insert(triangle.p2);
    }
    __host__ __device__  inline void insert(const NodeBox& other) {
        insert(other.lowCorner); insert(other.highCorner);
    }
    __device__ void pad() {
        const float scale = 8.0f * 1.1920928955078125e-7f;
        const Vec3 padding(scale * fmaxf(1.f, fmaxf(fabsf(lowCorner.x), fabsf(highCorner.x))),
                           scale * fmaxf(1.f, fmaxf(fabsf(lowCorner.y), fabsf(highCorner.y))),
                           scale * fmaxf(1.f, fmaxf(fabsf(lowCorner.z), fabsf(highCorner.z))));
        lowCorner -= padding;
        highCorner += padding;
    }
};

__host__ __device__ inline int countNodes(int n) {
    if (n == 0) return 0;
    const int leaves = n / MAX_LEAF_SIZE + (n % MAX_LEAF_SIZE != 0);
    return leaves + (leaves - 1);
}

__device__ inline unsigned gridLocation(float value, float low, float high) {
    return high > low
        ? unsigned(fminf(1023.f, fmaxf(0.f, (value - low) / (high - low) * 1024.f))) : 0;
}

__device__ unsigned long mortonEncoding(int idx, const GPUTriangle triangle,
                               unsigned long* mortonCodes, NodeBox fullSpace) {
    const Vec3 center = triangle.lowCorner * 0.5f + triangle.highCorner * 0.5f;
    const unsigned x = gridLocation(center.x, fullSpace.lowCorner.x, fullSpace.highCorner.x);
    const unsigned y = gridLocation(center.y, fullSpace.lowCorner.y, fullSpace.highCorner.y);
    const unsigned z = gridLocation(center.z, fullSpace.lowCorner.z, fullSpace.highCorner.z);
    unsigned long morton = 0;
    for (int i = 0; i < 10; ++i) {
        morton |= static_cast<unsigned long>((x >> i) & 1) << (3 * i + 2);
        morton |= static_cast<unsigned long>((y >> i) & 1) << (3 * i + 1);
        morton |= static_cast<unsigned long>((z >> i) & 1) << (3 * i);
    }
    // Thirty spatial bits plus a unique 32-bit ID fit in a 64-bit key.
    return (morton << 32) | static_cast<unsigned long>(idx);
}

__global__ void calcDelta(int n, const unsigned long* mortonCodes, unsigned long* deltas) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n - 1) return;
    const unsigned long difference = mortonCodes[idx] ^ mortonCodes[idx + 1];
    const unsigned long rank = static_cast<unsigned long>(64 - __clzll(difference));
    deltas[idx] = (rank << 32) | (IDX_MASK - static_cast<unsigned long>(idx));
}

__global__ void initNodes(int n_nodes, int n_triangles, LeafNode* nodes) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_nodes) return;
    nodes[idx].rightNodeIndex = -1;
    nodes[idx].n = idx == 0 ? n_triangles : 0;
    nodes[idx].startIndex = 0;
    nodes[idx].parent_id = -1;
}

// One block finds a split between whole leaf groups. The low key bits already
// break ties, so the winning key must stay intact until its index is decoded.
__device__ void splitNode(int n_triangles, int n_nodes, const GPUTriangle* triangles,
                          const unsigned long* deltas, unsigned long* best,
                          int node_index, LeafNode* nodes) {
    const int start = nodes[node_index].startIndex;
    const int n = nodes[node_index].n;
    if (n <= MAX_LEAF_SIZE) return;
    __shared__ unsigned long keys[BLOCK_SIZE];
    unsigned long key = 0;
    for (int i = start + MAX_LEAF_SIZE - 1 + threadIdx.x * MAX_LEAF_SIZE;
         i < start + n - 1; i += blockDim.x * MAX_LEAF_SIZE) {
        if (deltas[i] > key) key = deltas[i];
    }
    keys[threadIdx.x] = key;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset; offset /= 2) {
        if (threadIdx.x < offset && keys[threadIdx.x + offset] > keys[threadIdx.x])
            keys[threadIdx.x] = keys[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *best = keys[0];
        const int split = static_cast<int>(IDX_MASK - (keys[0] & IDX_MASK));
        const int left_n = split + 1 - start;
        const int left = node_index + 1;
        const int right = left + countNodes(left_n);
        nodes[node_index].rightNodeIndex = right;
        nodes[left].n = left_n;
        nodes[left].startIndex = start;
        nodes[left].parent_id = node_index;
        nodes[right].n = n - left_n;
        nodes[right].startIndex = start + left_n;
        nodes[right].parent_id = node_index;
    }
    __syncthreads();
}

__global__ void buildBVHSubtree(
    int n_triangles, int n_nodes, GPUTriangle* triangles,
    const unsigned long* deltas, LeafNode* nodes,
    unsigned long* best, int first_node, int second_node)
{
    // Preorder guarantees parents are initialized before their children.
    // A single cooperating block avoids device-side recursive launches and
    // their extra compiler/linker requirements on the server.
    for (int index = first_node; index < n_nodes; ++index) {
        splitNode(n_triangles, n_nodes, triangles, deltas, best + index, index, nodes);
        __syncthreads();
    }
}

__device__ void calcNodeBox(int n_triangles, GPUTriangle* triangles, int start,
                            int end, Vec3* lowCorner, Vec3* highCorner,
                            int node_index, bool write_idx) {
    __shared__ float lows[BLOCK_SIZE][3], highs[BLOCK_SIZE][3];
    const int t = threadIdx.x;
    NodeBox box;
    for (int idx = start + t; idx < end; idx += blockDim.x) {
        box.insert(triangles[idx].lowCorner);
        box.insert(triangles[idx].highCorner);
        if (write_idx) triangles[idx].node_index = node_index;
    }
    lows[t][0] = box.lowCorner.x; lows[t][1] = box.lowCorner.y; lows[t][2] = box.lowCorner.z;
    highs[t][0] = box.highCorner.x; highs[t][1] = box.highCorner.y; highs[t][2] = box.highCorner.z;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset; offset /= 2) {
        if (t < offset) {
            for (int axis = 0; axis < 3; ++axis) {
                lows[t][axis] = fminf(lows[t][axis], lows[t + offset][axis]);
                highs[t][axis] = fmaxf(highs[t][axis], highs[t + offset][axis]);
            }
        }
        __syncthreads();
    }
    if (t == 0) {
        box.lowCorner = Vec3(lows[0][0], lows[0][1], lows[0][2]);
        box.highCorner = Vec3(highs[0][0], highs[0][1], highs[0][2]);
        box.pad();
        *lowCorner = box.lowCorner;
        *highCorner = box.highCorner;
    }
}

__global__ void buildBVHBounds(int n_nodes, LeafNode* nodes, GPUTriangle* triangles) {
    const int index = blockIdx.x;
    if (index >= n_nodes) return;
    const int start = nodes[index].startIndex;
    const int end = start + nodes[index].n;
    calcNodeBox(end, triangles, start, end, &nodes[index].lowCorner,
                &nodes[index].highCorner, index, nodes[index].rightNodeIndex == -1);
}

// Call after initNodes, Morton sorting, and calcDelta.
void buildBVH(int n_nodes, int n_triangles, GPUTriangle* gpu_triangles,
              LeafNode* gpu_nodes, const unsigned long* deltas) {
    if (n_nodes == 0) return;
    unsigned long* best = nullptr;
    cudaMalloc(&best, n_nodes * sizeof(unsigned long));
    buildBVHSubtree<<<1, BLOCK_SIZE>>>(n_triangles, n_nodes, gpu_triangles,
                                           deltas, gpu_nodes, best, 0, -1);
    buildBVHBounds<<<n_nodes, BLOCK_SIZE>>>(n_nodes, gpu_nodes, gpu_triangles);
}

__global__ void convertToGPUTriagles(int n_triangles, unsigned long* morton_codes, const Triangle* triangles, GPUTriangle* gpu_triangles, int* object_indices, NodeBox fullSpace) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_triangles) return;
    gpu_triangles[idx] = GPUTriangle(object_indices[idx], idx, triangles[idx]);
    morton_codes[idx] = mortonEncoding(idx, gpu_triangles[idx], morton_codes, fullSpace);
}
// Slab intersection, including rays parallel to a face of the box.
__device__ bool intersectNode(const Ray& ray, const LeafNode& node, float closest) {
    // Triangle intersection and slab division can round the same distance
    // differently, especially for coplanar triangles. Only pruning gets slack.
    float near = 0.0f;
    float far = closest + EPSILON * fmaxf(1.0f, fabsf(closest));
    for (int axis = 0; axis < 3; ++axis) {
        const float origin = axis == 0 ? ray.origin.x : axis == 1 ? ray.origin.y : ray.origin.z;
        const float direction = axis == 0 ? ray.direction.x : axis == 1 ? ray.direction.y : ray.direction.z;
        const float low = axis == 0 ? node.lowCorner.x : axis == 1 ? node.lowCorner.y : node.lowCorner.z;
        const float high = axis == 0 ? node.highCorner.x : axis == 1 ? node.highCorner.y : node.highCorner.z;
        if (direction == 0.0f) {
            if (origin < low || origin > high) return false;
        } else {
            const float a = (low - origin) / direction;
            const float b = (high - origin) / direction;
            near = fmaxf(near, fminf(a, b));
            far = fminf(far, fmaxf(a, b));
            if (near > far) return false;
        }
    }
    return true;
}

__device__ Hit closestHit(const Ray& ray, const GPUTriangle* triangles,
                          const LeafNode* nodes, int n_nodes, int previous_triangle) {
    Hit hit;
    // Preorder ranges allow a missed subtree to be skipped without a stack.
    for (int index = 0; index < n_nodes;) {
        const LeafNode& node = nodes[index];
        if (!intersectNode(ray, node, hit.record.t)) {
            index += countNodes(node.n);
            continue;
        }
        if (node.rightNodeIndex == -1) {
            for (int i = node.startIndex; i < node.startIndex + node.n; ++i) {
                const GPUTriangle& triangle = triangles[i];
                if (triangle.triangle_index == previous_triangle) continue;
                HitRecord record;
                if (intersect_triangle(ray, triangle.triangle, &record) &&
                    (record.t < hit.record.t ||
                     (record.t == hit.record.t && triangle.triangle_index < hit.triangle_index))) {
                    hit.record = record;
                    hit.object_index = triangle.object_index;
                    hit.triangle_index = triangle.triangle_index;
                }
            }
        }
        ++index;
    }
    return hit;
}

__device__ Vec3 tracePath(Ray ray, const RenderSettings& settings,
                          const GPUTriangle* triangles, const LeafNode* nodes,
                          int n_nodes, const ObjectMaterial* materials, RandomGPU& random) {
    Vec3 light, throughput(1.0f, 1.0f, 1.0f);
    int previous_triangle = -1;
    for (int bounce = 0; ; ++bounce) {
        const Hit hit = closestHit(ray, triangles, nodes, n_nodes, previous_triangle);
        if (!hit.hit()) {
            light += multiply(throughput, settings.background_light);
            break;
        }
        const ObjectMaterial& material = materials[hit.object_index];
        light += multiply(throughput, material.emission);
        if (bounce >= settings.max_bounces) break;
        throughput = multiply(throughput, material.albedo);
        ray = cosine_weighted_sampling(hit.record, random);
        previous_triangle = hit.triangle_index;
    }
    return light;
}

__global__ void renderPixels(RenderSettings settings, const GPUTriangle* triangles,
                             const LeafNode* nodes, int n_nodes,
                             const ObjectMaterial* materials, Vec3* subpixel_colors, size_t n_subpixels) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n_subpixels) return;
    const size_t width = size_t(settings.width) * settings.supersampling_factor;
    const int x = int(index % width), y = int(index / width);
    // Each subpixel independently traces its camera ray and samples.
    RandomGPU random(index);
    const Ray ray = make_camera_ray(settings, x, y);
    Vec3 color;
    for (int sample = 0; sample < settings.samples_per_pixel; ++sample)
        color += tracePath(ray, settings, triangles, nodes, n_nodes, materials, random);
    subpixel_colors[index] = color / float(settings.samples_per_pixel);
}

__global__ void averageSubpixels(RenderSettings settings, const Vec3* subpixel_colors,
                                 Pixel* pixels, size_t n_pixels) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n_pixels) return;
    const int ss = settings.supersampling_factor;
    const size_t width = size_t(settings.width) * ss;
    const size_t start = (index / settings.width * ss) * width + (index % settings.width) * ss;
    Vec3 color;
    for (int dy = 0; dy < ss; ++dy) {
        for (int dx = 0; dx < ss; ++dx)
            color += subpixel_colors[start + size_t(dy) * width + dx];
    }
    // Average linear colors before tone mapping and gamma correction.
    pixels[index] = to_pixel(color / (float(ss) * ss));
}

std::vector<Pixel> render_scene(const RenderSettings& settings, const std::vector<Object>& objects) {
    const size_t n_pixels = size_t(settings.width) * settings.height;
    const size_t n_subpixels = n_pixels * settings.supersampling_factor * settings.supersampling_factor;
    // 1. Collect triangles and materials on CPU
    std::vector<Triangle> triangles;
    std::vector<int> triangle_object_indices;
    std::vector<ObjectMaterial> materials(objects.size());
    NodeBox fullSpace;

    for (int object_index = 0; object_index < objects.size(); ++object_index) {
        const Object& object = objects[object_index];

        materials[object_index] = ObjectMaterial{
            object.emission,
            object.albedo
        };

        for (const Triangle& triangle : object.triangles) {
            fullSpace.insert(triangle);
            triangles.push_back(triangle);
            triangle_object_indices.push_back(object_index);
        }
    }

    int n_triangles = static_cast<int>(triangles.size());
    int n_nodes = countNodes(n_triangles);


    // 2. Declare GPU pointers
    Triangle* gpu_raw_triangles = nullptr;
    GPUTriangle* gpu_triangles = nullptr;
    LeafNode* gpu_nodes = nullptr;
    ObjectMaterial* gpu_materials = nullptr;
    int* gpu_triangle_object_indices = nullptr;
    unsigned long* mortonCodes = nullptr;
    unsigned long* deltas = nullptr;
    Pixel* gpu_pixels = nullptr;
    Vec3* gpu_subpixel_colors = nullptr;
    auto release = [&]() {
        cudaFree(gpu_raw_triangles);
        cudaFree(gpu_triangle_object_indices);
        cudaFree(gpu_triangles);
        cudaFree(gpu_materials);
        cudaFree(gpu_nodes);
        cudaFree(mortonCodes);
        cudaFree(deltas);
        cudaFree(gpu_pixels);
        cudaFree(gpu_subpixel_colors);
    };
    cudaMalloc(&gpu_raw_triangles, n_triangles * sizeof(Triangle));
    cudaMalloc(&gpu_triangle_object_indices, n_triangles * sizeof(int));
    cudaMalloc(&gpu_triangles, n_triangles * sizeof(GPUTriangle));
    cudaMalloc(&gpu_materials, objects.size() * sizeof(ObjectMaterial));
    cudaMalloc(&gpu_nodes, n_nodes * sizeof(LeafNode));
    cudaMalloc(&mortonCodes, n_triangles * sizeof(unsigned long));
    cudaMalloc(&deltas, n_triangles * sizeof(unsigned long));


        cudaMemcpy(
            gpu_raw_triangles,
            triangles.data(),
            n_triangles * sizeof(Triangle),
            cudaMemcpyHostToDevice
        );

        cudaMemcpy(
            gpu_triangle_object_indices,
            triangle_object_indices.data(),
            n_triangles * sizeof(int),
            cudaMemcpyHostToDevice
        );

        cudaMemcpy(
            gpu_materials,
            materials.data(),
            objects.size() * sizeof(ObjectMaterial),
            cudaMemcpyHostToDevice
        );


        // 5. Convert triangles into GPU triangle structures
        convertToGPUTriagles<<<
            (n_triangles + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            n_triangles,
            mortonCodes,
            gpu_raw_triangles,
            gpu_triangles,
            gpu_triangle_object_indices,
            fullSpace
        );


        // 6. Free temporary GPU arrays
   
        cudaFree(gpu_raw_triangles);
        gpu_raw_triangles = nullptr;
        cudaFree(gpu_triangle_object_indices);
        gpu_triangle_object_indices = nullptr;


        // 7. Initialize BVH nodes
        initNodes<<<
            (n_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            n_nodes,
            n_triangles,
            gpu_nodes
        );
        // 9. Sort triangles by Morton codes
        thrust::sort_by_key(
            thrust::device,
            mortonCodes,
            mortonCodes + n_triangles,
            gpu_triangles
        );


        // 10. Calculate Morton code deltas
        calcDelta<<<
            (n_triangles + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            n_triangles,
            mortonCodes,
            deltas
        );


        // 11. Build BVH
        buildBVH(
            n_nodes,
            n_triangles,
            gpu_triangles,
            gpu_nodes,
            deltas
        );

        // 12. Trace samples, average subpixels, and return the final image.
        cudaMalloc(&gpu_pixels, n_pixels * sizeof(Pixel));
        cudaMalloc(&gpu_subpixel_colors, n_subpixels * sizeof(Vec3));
        renderPixels<<<unsigned((n_subpixels + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(
            settings, gpu_triangles, gpu_nodes, n_nodes, gpu_materials, gpu_subpixel_colors, n_subpixels);
        averageSubpixels<<<unsigned((n_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(
            settings, gpu_subpixel_colors, gpu_pixels, n_pixels);
        std::vector<Pixel> pixels(n_pixels);
        cudaMemcpy(pixels.data(), gpu_pixels, n_pixels * sizeof(Pixel), cudaMemcpyDeviceToHost);
        release();
        return pixels;
}
