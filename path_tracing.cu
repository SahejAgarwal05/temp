#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "graphics.cuh"
#include <thrust/execution_policy.h>
#include <thrust/sort.h>


constexpr int MAX_LEAF_SIZE = 2;
constexpr int BLOCK_SIZE = 256;
constexpr unsigned long long IDX_MASK = 0xffffffffULL;
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

    __host__ __device__ LeafNode(int triangle_count = 0, int first_triangle = 0, int parent = -1)
        : rightNodeIndex(-1), n(triangle_count), startIndex(first_triangle), parent_id(parent) {}
};

struct GPUTriangle {
    int object_index;
    int triangle_index; // Original flattened ID, preserved through sorting.
    Vec3 edge0;
    Vec3 edge1;
    Vec3 normal;
    Vec3 highCorner;
    Vec3 lowCorner;
    Vec3 p0;
    __host__ __device__ GPUTriangle() = default;
    __host__ __device__ GPUTriangle(int object_index, int triangle_index, const Triangle& triangle2):
        object_index(object_index), triangle_index(triangle_index) {
            highCorner = Vec3(fmaxf(triangle2.p0.x, fmaxf(triangle2.p1.x, triangle2.p2.x)),
                             fmaxf(triangle2.p0.y, fmaxf(triangle2.p1.y, triangle2.p2.y)),
                             fmaxf(triangle2.p0.z, fmaxf(triangle2.p1.z, triangle2.p2.z)));
            lowCorner = Vec3(fminf(triangle2.p0.x, fminf(triangle2.p1.x, triangle2.p2.x)),
                            fminf(triangle2.p0.y, fminf(triangle2.p1.y, triangle2.p2.y)),
                            fminf(triangle2.p0.z, fminf(triangle2.p1.z, triangle2.p2.z)));
            edge0 = triangle2.p1 - triangle2.p0;
            edge1 = triangle2.p2 - triangle2.p0;
            p0 = triangle2.p0;
            normal = normalize(cross(edge0, edge1));
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

__device__ unsigned long long mortonEncoding(int idx, const GPUTriangle triangle,
                               unsigned long long* mortonCodes, NodeBox fullSpace) {
    const Vec3 center = triangle.lowCorner * 0.5f + triangle.highCorner * 0.5f;
    const unsigned x = gridLocation(center.x, fullSpace.lowCorner.x, fullSpace.highCorner.x);
    const unsigned y = gridLocation(center.y, fullSpace.lowCorner.y, fullSpace.highCorner.y);
    const unsigned z = gridLocation(center.z, fullSpace.lowCorner.z, fullSpace.highCorner.z);
    unsigned long long morton = 0;
    for (int i = 0; i < 10; ++i) {
        morton |= static_cast<unsigned long long>((x >> i) & 1) << (3 * i + 2);
        morton |= static_cast<unsigned long long>((y >> i) & 1) << (3 * i + 1);
        morton |= static_cast<unsigned long long>((z >> i) & 1) << (3 * i);
    }
    // Thirty spatial bits plus a unique 32-bit ID fit in a 64-bit key.
    return (morton << 32) | static_cast<unsigned long long>(idx);
}

__global__ void calcDelta(int n, const unsigned long long* mortonCodes, unsigned long long* deltas) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n - 1) return;
    const unsigned long long difference = mortonCodes[idx] ^ mortonCodes[idx + 1];
    const unsigned long long rank = static_cast<unsigned long long>(64 - __clzll(difference));
    deltas[idx] = (rank << 32) | (IDX_MASK - static_cast<unsigned long long>(idx));
}

// One block finds a split between whole leaf groups. The low key bits already
// break ties, so the winning key must stay intact until its index is decoded.
__device__ void splitNode(int n_triangles, int n_nodes, const GPUTriangle* triangles,
                          const unsigned long long* deltas, unsigned long long* best,
                          int node_index, LeafNode* nodes) {
    const int start = nodes[node_index].startIndex;
    const int n = nodes[node_index].n;
    if (n <= MAX_LEAF_SIZE) return;

    if (threadIdx.x == 0) *best = 0;
    __syncthreads();
    // Consider only boundaries between complete leaf groups.
    for (int i = start + MAX_LEAF_SIZE - 1 + threadIdx.x * MAX_LEAF_SIZE;
         i < start + n - 1; i += blockDim.x * MAX_LEAF_SIZE)
        atomicMax(best, deltas[i]);
    __syncthreads();

    if (threadIdx.x == 0) {
        // The key already contains a unique split index; keep it intact.
        const int split = static_cast<int>(IDX_MASK - (*best & IDX_MASK));
        const int left_n = split + 1 - start;
        const int left = node_index + 1;
        const int right = left + countNodes(left_n);
        nodes[node_index].rightNodeIndex = right;
        nodes[left] = LeafNode(left_n, start, node_index);
        nodes[right] = LeafNode(n - left_n, start + left_n, node_index);
    }
    __syncthreads();
}

__global__ void buildBVHSubtree(
    int n_triangles, int n_nodes, GPUTriangle* triangles,
    const unsigned long long* deltas, LeafNode* nodes,
    unsigned long long* best, int first_node, int second_node)
{
    if (threadIdx.x == 0) nodes[first_node] = LeafNode(n_triangles);
    __syncthreads();
    // Preorder guarantees parents are initialized before their children.
    // A single cooperating block avoids device-side recursive launches and
    // their extra compiler/linker requirements on the server.
    for (int index = first_node; index < n_nodes; ++index) {
        splitNode(n_triangles, n_nodes, triangles, deltas, best + index, index, nodes);
        __syncthreads();
    }
}

__device__ NodeBox calcNodeBox(GPUTriangle* triangles, const int startIndex, const int n) {
    NodeBox box;
    for (int i = startIndex; i < startIndex + n; ++i) {
        box.insert(triangles[i].lowCorner);
        box.insert(triangles[i].highCorner);
    }
    box.pad();
    return box;
}



__global__ void buildBVHBounds(int n_nodes, LeafNode* gpu_nodes, GPUTriangle* gpu_triangles) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    // Reverse preorder ensures both child boxes are ready before their parent.
    for(int i = n_nodes - 1; i >= 0; --i) {
        if(gpu_nodes[i].rightNodeIndex == -1) {
            // Leaf node
            NodeBox box = calcNodeBox(gpu_triangles, gpu_nodes[i].startIndex, gpu_nodes[i].n);
            gpu_nodes[i].lowCorner = box.lowCorner;
            gpu_nodes[i].highCorner = box.highCorner;
        } else {
            // Internal node
            NodeBox box;
            LeafNode leftChild = gpu_nodes[i + 1];
            LeafNode rightChild = gpu_nodes[gpu_nodes[i].rightNodeIndex];
            box.insert(leftChild.lowCorner);
            box.insert(leftChild.highCorner);
            box.insert(rightChild.lowCorner);
            box.insert(rightChild.highCorner);
            gpu_nodes[i].lowCorner = box.lowCorner;
            gpu_nodes[i].highCorner = box.highCorner;
        }
    }
}

// Call after Morton sorting and calcDelta. The builder initializes its own nodes.
void buildBVH(int n_nodes, int n_triangles, GPUTriangle* gpu_triangles,
              LeafNode* gpu_nodes, const unsigned long long* deltas) {
    if (n_nodes == 0) return;
    unsigned long long* best = nullptr;
    cudaMalloc(&best, n_nodes * sizeof(unsigned long long));
    buildBVHSubtree<<<1, BLOCK_SIZE>>>(n_triangles, n_nodes, gpu_triangles,
                                           deltas, gpu_nodes, best, 0, -1);
    buildBVHBounds<<<1, 1>>>(n_nodes, gpu_nodes, gpu_triangles);
    cudaFree(best);
}

__global__ void convertToGPUTriagles(int n_triangles, unsigned long long* morton_codes, const Triangle* triangles, GPUTriangle* gpu_triangles, int* object_indices, NodeBox fullSpace) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_triangles) return;
    gpu_triangles[idx] = GPUTriangle(object_indices[idx], idx, triangles[idx]);
    morton_codes[idx] = mortonEncoding(idx, gpu_triangles[idx], morton_codes, fullSpace);
}
// Slab intersection, including rays parallel to a face of the box.
__device__ bool intersectNode(const Ray& ray, const LeafNode& node, float closest, float& entry_distance) {
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
    entry_distance = near;
    return true;
}
__device__ bool intersect_triangle_gpu(const GPUTriangle& triangle, const Ray& ray, HitRecord* record) {
    // Implementation for GPU triangle intersection
    const Vec3 p = cross(ray.direction, triangle.edge1);
    const float determinant = dot(triangle.edge0, p);
    if (std::fabs(determinant) <= EPSILON) {
        return false;
    }

    const float inverse_determinant = 1.0 / determinant;
    const Vec3 offset = ray.origin - triangle.p0;
    const float u = dot(offset, p) * inverse_determinant;
    if (u < 0.0 || u > 1.0) {
        return false;
    }

    const Vec3 q = cross(offset, triangle.edge0);
    const float v = dot(ray.direction, q) * inverse_determinant;
    if (v < 0.0 || u + v > 1.0) {
        return false;
    }

    const float t = dot(triangle.edge1, q) * inverse_determinant;
    if (t <= EPSILON) {
        return false;
    }
    record->unit_normal = triangle.normal;
    if (dot(triangle.normal, ray.direction) > 0.0) {
        record->unit_normal *= -1;
    }
    record->t = t;
    record->position = ray.origin + t * ray.direction;
    return true;
}
__device__ void testTriangleHit(const Ray& ray, const GPUTriangle& triangle,
                                int previous_triangle, Hit& closest) {
    if (triangle.triangle_index == previous_triangle) return;
    HitRecord record;
    if (!intersect_triangle_gpu(triangle, ray, &record)) return;
    if (record.t < closest.record.t ||
        (record.t == closest.record.t && triangle.triangle_index < closest.triangle_index)) {
        closest.record = record;
        closest.object_index = triangle.object_index;
        closest.triangle_index = triangle.triangle_index;
    }
}



__device__ Hit closestHit(const Ray& ray, const GPUTriangle* triangles,
                          const LeafNode* nodes, int n_nodes, int previous_triangle) {
    Hit hit;
    if (n_nodes == 0) return hit;
    constexpr int STACK_CAPACITY = 64;
    int pending_nodes[STACK_CAPACITY];
    int pending_count = 0;
    int node_index = 0;

    while (node_index != -1) {
        const LeafNode& node = nodes[node_index];
        float entry_distance;
        // Recheck deferred nodes against any closer hit found in the meantime.
        if (intersectNode(ray, node, hit.record.t, entry_distance)) {
            if (node.rightNodeIndex == -1) {
                testTriangleHit(ray, triangles[node.startIndex], previous_triangle, hit);
                if (node.n > 1)
                    testTriangleHit(ray, triangles[node.startIndex + 1], previous_triangle, hit);
            } else {
                const int left_child = node_index + 1;
                const int right_child = node.rightNodeIndex;
                float left_entry, right_entry;
                const bool hit_left = intersectNode(ray, nodes[left_child], hit.record.t, left_entry);
                const bool hit_right = intersectNode(ray, nodes[right_child], hit.record.t, right_entry);
                if (hit_left && hit_right) {
                    if (pending_count == STACK_CAPACITY) {
                        // Keep correctness even for a tree deeper than the local stack.
                        for (int i = 0; i < nodes[0].n; ++i)
                            testTriangleHit(ray, triangles[i], previous_triangle, hit);
                        return hit;
                    }
                    const bool left_is_nearer = left_entry <= right_entry;
                    // Save the farther child and immediately descend into the nearer one.
                    pending_nodes[pending_count++] = left_is_nearer ? right_child : left_child;
                    node_index = left_is_nearer ? left_child : right_child;
                    continue;
                }
                else if (hit_left || hit_right) {
                    node_index = hit_left ? left_child : right_child;
                    continue;
                }
            }
        }
        node_index = pending_count > 0 ? pending_nodes[--pending_count] : -1;
    }
    return hit;
}

__device__ Vec3 tracePath(Ray ray, const RenderSettings& settings,
                          const GPUTriangle* triangles, const LeafNode* nodes,
                          int n_nodes, const ObjectMaterial* materials, RandomGPU& random,
                          const Hit* first_hit = nullptr) {
    Vec3 light, throughput(1.0f, 1.0f, 1.0f), zero(0,0,0);
    int previous_triangle = -1;
    for (int bounce = 0; ; ++bounce) {
        // All samples of a subpixel share the deterministic camera intersection.
        const Hit hit = bounce == 0 && first_hit != nullptr
            ? *first_hit : closestHit(ray, triangles, nodes, n_nodes, previous_triangle);
        if (!hit.hit() || length(throughput - zero) < EPSILON) {
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

__global__ void findSubpixelFirstHits(RenderSettings settings, const GPUTriangle* triangles,
                                 const LeafNode* nodes, int bvh_node_count,
                                 Hit* subpixel_first_hits, size_t subpixel_count, size_t subpixel_grid_width) {
    const size_t subpixel_index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (subpixel_index >= subpixel_count) return;
    const Ray camera_ray = make_camera_ray(settings, int(subpixel_index % subpixel_grid_width),
                                           int(subpixel_index / subpixel_grid_width));
    subpixel_first_hits[subpixel_index] = closestHit(camera_ray, triangles, nodes, bvh_node_count, -1);
}

// CUDA atomics operate on scalar components, so add an RGB color componentwise.
__device__ void atomicAddColor(Vec3* sum, const Vec3& color) {
    atomicAdd(&sum->x, color.x);
    atomicAdd(&sum->y, color.y);
    atomicAdd(&sum->z, color.z);
}

__global__ void traceSubpixelSamples(RenderSettings settings, const GPUTriangle* triangles,
                                     const LeafNode* nodes, int bvh_node_count,
                                     const ObjectMaterial* materials, Vec3* subpixel_radiance_sums,
                                     const Hit* subpixel_first_hits, size_t subpixel_count) {
    // Each thread traces one path sample from its subpixel's cached first hit.
    const size_t path_sample_index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t subpixel_index = path_sample_index / settings.samples_per_pixel;
    if (subpixel_index >= subpixel_count) return;

    RandomGPU random(path_sample_index);
    const Vec3 sample_color = tracePath(Ray{}, settings, triangles, nodes, bvh_node_count,materials, random, &subpixel_first_hits[subpixel_index]);

    atomicAddColor(subpixel_radiance_sums + subpixel_index, sample_color);
}

__global__ void averageSubpixelSamples(RenderSettings settings, const Vec3* subpixel_radiance_sums,
                                       Vec3* pixel_radiance_sums, size_t subpixel_count) {
    const size_t subpixel_index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (subpixel_index >= subpixel_count) return;
    const size_t subpixels_per_axis = settings.supersampling_factor;
    const size_t subpixel_grid_width = size_t(settings.width) * subpixels_per_axis;
    const size_t pixel_x = (subpixel_index % subpixel_grid_width) / subpixels_per_axis;
    const size_t pixel_y = (subpixel_index / subpixel_grid_width) / subpixels_per_axis;
    const size_t pixel_index = pixel_y * settings.width + pixel_x;
    const Vec3 subpixel_color = subpixel_radiance_sums[subpixel_index] / float(settings.samples_per_pixel);
    atomicAddColor(pixel_radiance_sums + pixel_index, subpixel_color);
}

__global__ void finalizeOutputPixels(RenderSettings settings, const Vec3* pixel_radiance_sums,
                                 Pixel* output_pixels, size_t output_pixel_count) {
    const size_t output_pixel_index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_pixel_index >= output_pixel_count) return;
    const int subpixels_per_axis = settings.supersampling_factor;
    // Average linear colors before tone mapping and gamma correction.
    output_pixels[output_pixel_index] = to_pixel(pixel_radiance_sums[output_pixel_index] /
        (float(subpixels_per_axis) * subpixels_per_axis));
}

std::vector<Pixel> render_scene(const RenderSettings& settings, const std::vector<Object>& objects) {
    // Each subpixel has one camera ray and one cached first hit.
    // samples_per_pixel is the existing setting for path samples per subpixel.
    const int subpixels_per_axis = settings.supersampling_factor;
    const int samples_per_subpixel = settings.samples_per_pixel;
    const size_t output_pixel_count = size_t(settings.width) * settings.height;
    const size_t subpixel_count = output_pixel_count * subpixels_per_axis * subpixels_per_axis;
    // 1. Collect triangles and materials on CPU
    std::vector<Triangle> triangles;
    std::vector<int> triangle_object_indices;
    std::vector<ObjectMaterial> materials(objects.size());
    NodeBox scene_bounds;

    for (int object_index = 0; object_index < objects.size(); ++object_index) {
        const Object& object = objects[object_index];

        materials[object_index] = ObjectMaterial{
            object.emission,
            object.albedo
        };

        for (const Triangle& triangle : object.triangles) {
            scene_bounds.insert(triangle);
            triangles.push_back(triangle);
            triangle_object_indices.push_back(object_index);
        }
    }

    int triangle_count = static_cast<int>(triangles.size());
    int bvh_node_count = countNodes(triangle_count);


    // 2. Declare GPU pointers
    Triangle* gpu_raw_triangles = nullptr;
    GPUTriangle* gpu_triangles = nullptr;
    LeafNode* gpu_nodes = nullptr;
    ObjectMaterial* gpu_materials = nullptr;
    int* gpu_triangle_object_indices = nullptr;
    unsigned long long* mortonCodes = nullptr;
    unsigned long long* deltas = nullptr;
    Pixel* gpu_output_pixels = nullptr;
    Vec3* gpu_pixel_radiance_sums = nullptr;
    Vec3* gpu_subpixel_radiance_sums = nullptr;
    Hit* gpu_subpixel_first_hits = nullptr;
    auto release = [&]() {
        cudaFree(gpu_raw_triangles);
        cudaFree(gpu_triangle_object_indices);
        cudaFree(gpu_triangles);
        cudaFree(gpu_materials);
        cudaFree(gpu_nodes);
        cudaFree(mortonCodes);
        cudaFree(deltas);
        cudaFree(gpu_output_pixels);
        cudaFree(gpu_pixel_radiance_sums);
        cudaFree(gpu_subpixel_radiance_sums);
        cudaFree(gpu_subpixel_first_hits);
    };
    cudaMalloc(&gpu_raw_triangles, triangle_count * sizeof(Triangle));
    cudaMalloc(&gpu_triangle_object_indices, triangle_count * sizeof(int));
    cudaMalloc(&gpu_triangles, triangle_count * sizeof(GPUTriangle));
    cudaMalloc(&gpu_materials, objects.size() * sizeof(ObjectMaterial));
    cudaMalloc(&gpu_nodes, bvh_node_count * sizeof(LeafNode));
    cudaMalloc(&mortonCodes, triangle_count * sizeof(unsigned long long));
    cudaMalloc(&deltas, triangle_count * sizeof(unsigned long long));


        cudaMemcpy(
            gpu_raw_triangles,
            triangles.data(),
            triangle_count * sizeof(Triangle),
            cudaMemcpyHostToDevice
        );

        cudaMemcpy(
            gpu_triangle_object_indices,
            triangle_object_indices.data(),
            triangle_count * sizeof(int),
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
            (triangle_count + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            triangle_count,
            mortonCodes,
            gpu_raw_triangles,
            gpu_triangles,
            gpu_triangle_object_indices,
            scene_bounds
        );


        // 6. Free temporary GPU arrays
   
        cudaFree(gpu_raw_triangles);
        gpu_raw_triangles = nullptr;
        cudaFree(gpu_triangle_object_indices);
        gpu_triangle_object_indices = nullptr;


        // 9. Sort tr/angles by Morton codes
        thrust::sort_by_key(
            thrust::device,
            mortonCodes,
            mortonCodes + triangle_count,
            gpu_triangles
        );


        // 10. Calculate Morton code deltas
        calcDelta<<<
            (triangle_count + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            triangle_count,
            mortonCodes,
            deltas
        );

        cudaFree(mortonCodes);
        mortonCodes = nullptr;
        // 11. Build BVH
        buildBVH(
            bvh_node_count,
            triangle_count,
            gpu_triangles,
            gpu_nodes,
            deltas
        );

        // 12. Trace samples, average subpixels, and return the final image.
        cudaMalloc(&gpu_output_pixels, output_pixel_count * sizeof(Pixel));
        cudaMalloc(&gpu_pixel_radiance_sums, output_pixel_count * sizeof(Vec3));
        cudaMemset(gpu_pixel_radiance_sums, 0, output_pixel_count * sizeof(Vec3));
        cudaMalloc(&gpu_subpixel_radiance_sums, subpixel_count * sizeof(Vec3));
        cudaMemset(gpu_subpixel_radiance_sums, 0, subpixel_count * sizeof(Vec3));
        cudaMalloc(&gpu_subpixel_first_hits, subpixel_count * sizeof(Hit));
        const size_t subpixel_grid_width = size_t(settings.width) * settings.supersampling_factor;
        findSubpixelFirstHits<<<unsigned((subpixel_count + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(
            settings, gpu_triangles, gpu_nodes, bvh_node_count, gpu_subpixel_first_hits, subpixel_count, subpixel_grid_width);
        const size_t path_sample_count = subpixel_count * samples_per_subpixel;
        traceSubpixelSamples<<<unsigned((path_sample_count + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(
            settings, gpu_triangles, gpu_nodes, bvh_node_count, gpu_materials, gpu_subpixel_radiance_sums,
            gpu_subpixel_first_hits, subpixel_count);
        averageSubpixelSamples<<<unsigned((subpixel_count + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(
            settings, gpu_subpixel_radiance_sums, gpu_pixel_radiance_sums, subpixel_count);
        finalizeOutputPixels<<<unsigned((output_pixel_count + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(
            settings, gpu_pixel_radiance_sums, gpu_output_pixels, output_pixel_count);
        std::vector<Pixel> output_pixels(output_pixel_count);
        cudaMemcpy(output_pixels.data(), gpu_output_pixels, output_pixel_count * sizeof(Pixel), cudaMemcpyDeviceToHost);
        release();
        return output_pixels;
}
