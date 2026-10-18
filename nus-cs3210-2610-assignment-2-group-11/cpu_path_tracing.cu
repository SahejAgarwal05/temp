#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "graphics.cuh"
#include <cuda/atomic>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace {
constexpr int MAX_LEAF_SIZE = 2;
constexpr int BLOCK_SIZE = 256;
constexpr unsigned long long IDX_MASK = 0xffffffffULL;

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
};

struct NodeBox {
    Vec3 lowCorner = Vec3(INF, INF, INF);
    Vec3 highCorner = Vec3(-INF, -INF, -INF);

    __host__ __device__ void insert(const Vec3& point) {
        lowCorner = Vec3(fminf(lowCorner.x, point.x), fminf(lowCorner.y, point.y),
                         fminf(lowCorner.z, point.z));
        highCorner = Vec3(fmaxf(highCorner.x, point.x), fmaxf(highCorner.y, point.y),
                          fmaxf(highCorner.z, point.z));
    }
    __host__ __device__ void insert(const Triangle& triangle) {
        insert(triangle.p0); insert(triangle.p1); insert(triangle.p2);
    }
    __host__ __device__ void insert(const NodeBox& other) {
        insert(other.lowCorner); insert(other.highCorner);
    }
    __device__ void pad() {
        const float scale = 8.0f * std::numeric_limits<float>::epsilon();
        const Vec3 padding(scale * fmaxf(1.f, fmaxf(fabsf(lowCorner.x), fabsf(highCorner.x))),
                           scale * fmaxf(1.f, fmaxf(fabsf(lowCorner.y), fabsf(highCorner.y))),
                           scale * fmaxf(1.f, fmaxf(fabsf(lowCorner.z), fabsf(highCorner.z))));
        lowCorner -= padding;
        highCorner += padding;
    }
};

void checkCuda(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(result));
}

template <typename T> struct GPUArray {
    T* data = nullptr;
    explicit GPUArray(size_t size) {
        if (size) checkCuda(cudaMalloc(reinterpret_cast<void**>(&data), size * sizeof(T)));
    }
    ~GPUArray() { if (data) cudaFree(data); }
    GPUArray(const GPUArray&) = delete;
    GPUArray& operator=(const GPUArray&) = delete;
};

__host__ __device__ int countNodes(int n) {
    if (n == 0) return 0;
    const int leaves = n / MAX_LEAF_SIZE + (n % MAX_LEAF_SIZE != 0);
    return leaves + (leaves - 1);
}

__device__ unsigned gridLocation(float value, float low, float high) {
    return high > low
        ? unsigned(fminf(1023.f, fmaxf(0.f, (value - low) / (high - low) * 1024.f))) : 0;
}

__global__ void mortonEncoding(int n, const GPUTriangle* triangles,
                               unsigned long long* mortonCodes, NodeBox fullSpace) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;
    NodeBox box;
    box.insert(triangles[idx].triangle);
    const Vec3 center = box.lowCorner * 0.5f + box.highCorner * 0.5f;
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
    mortonCodes[idx] = (morton << 32) | static_cast<unsigned>(idx);
}

__global__ void calcDelta(int n, const unsigned long long* mortonCodes,
                          unsigned long long* deltas) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n - 1) return;
    const unsigned long long difference = mortonCodes[idx] ^ mortonCodes[idx + 1];
    // Rank the most significant differing bit, retaining 32 bits for the split index.
    deltas[idx] = static_cast<unsigned long long>(64 - __clzll(difference)) << 32;
}

__global__ void initNodes(int n_nodes, int n_triangles, LeafNode* nodes) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_nodes) return;
    nodes[idx].rightNodeIndex = -1;
    nodes[idx].n = idx == 0 ? n_triangles : 0;
    nodes[idx].startIndex = 0;
    nodes[idx].parent_id = -1;
}

__global__ void splitNode(int n_triangles, const GPUTriangle* triangles,
                          const unsigned long long* deltas, unsigned long long* best) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_triangles - 1 || (i + 1) % MAX_LEAF_SIZE != 0) return;
    const int node = triangles[i].node_index;
    if (triangles[i + 1].node_index != node) return;
    atomicMax(&best[node], deltas[i] | static_cast<unsigned long long>(i + 1));
}

__global__ void split_nodes(int n_nodes, LeafNode* nodes,
                            const unsigned long long* best, int* changed) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_nodes) return;
    // The immutable candidate array identifies nodes active before this launch.
    // Do not read a child while its parent may be initializing its range.
    if (best[idx] == 0) return;
    const LeafNode node = nodes[idx];
    if (node.n <= MAX_LEAF_SIZE || node.rightNodeIndex != -1) return;
    const int split = static_cast<int>(best[idx] & IDX_MASK);
    const int left_count = split - node.startIndex;
    const int right = idx + 1 + countNodes(left_count);
    // Child slots were initialized by initNodes. Only their parent writes them.
    nodes[idx + 1].n = left_count;
    nodes[idx + 1].startIndex = node.startIndex;
    nodes[idx + 1].parent_id = idx;
    nodes[right].n = node.n - left_count;
    nodes[right].startIndex = split;
    nodes[right].parent_id = idx;
    nodes[idx].rightNodeIndex = right;
    atomicExch(changed, 1);
}

__global__ void reassign(int n_triangles, GPUTriangle* triangles, const LeafNode* nodes) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_triangles) return;
    const int node = triangles[i].node_index;
    const int right = nodes[node].rightNodeIndex;
    if (right != -1)
        triangles[i].node_index = i < nodes[right].startIndex ? node + 1 : right;
}

void constructNodes(int n_triangles, int n_nodes, GPUTriangle* triangles,
                    const unsigned long long* deltas, LeafNode* nodes) {
    GPUArray<unsigned long long> best(n_nodes);
    GPUArray<int> d_changed(1);
    const int blocksT = (n_triangles - 1) / BLOCK_SIZE + 1;
    const int blocksN = (n_nodes - 1) / BLOCK_SIZE + 1;
    initNodes<<<blocksN, BLOCK_SIZE>>>(n_nodes, n_triangles, nodes);
    checkCuda(cudaGetLastError());
    int changed;
    do {
        checkCuda(cudaMemset(best.data, 0, n_nodes * sizeof(unsigned long long)));
        checkCuda(cudaMemset(d_changed.data, 0, sizeof(int)));
        splitNode<<<blocksT, BLOCK_SIZE>>>(n_triangles, triangles, deltas, best.data);
        checkCuda(cudaGetLastError());
        split_nodes<<<blocksN, BLOCK_SIZE>>>(n_nodes, nodes, best.data, d_changed.data);
        checkCuda(cudaGetLastError());
        reassign<<<blocksT, BLOCK_SIZE>>>(n_triangles, triangles, nodes);
        checkCuda(cudaGetLastError());
        checkCuda(cudaMemcpy(&changed, d_changed.data, sizeof(int), cudaMemcpyDeviceToHost));
    } while (changed);
}

__global__ void computeBoxes(int n_nodes, const GPUTriangle* triangles,
                             LeafNode* nodes, int* visits) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_nodes || nodes[idx].rightNodeIndex != -1) return;
    NodeBox box;
    for (int j = nodes[idx].startIndex; j < nodes[idx].startIndex + nodes[idx].n; ++j)
        box.insert(triangles[j].triangle);
    box.pad();
    nodes[idx].lowCorner = box.lowCorner;
    nodes[idx].highCorner = box.highCorner;
    int cur = idx;
    while (nodes[cur].parent_id != -1) {
        const int p = nodes[cur].parent_id;
        // Publish child bounds and acquire the sibling's writes across blocks.
        cuda::atomic_ref<int, cuda::thread_scope_device> completed(visits[p]);
        if (completed.fetch_add(1, cuda::memory_order_acq_rel) == 0) return;
        NodeBox merged;
        merged.insert(nodes[p + 1].lowCorner); merged.insert(nodes[p + 1].highCorner);
        const int right = nodes[p].rightNodeIndex;
        merged.insert(nodes[right].lowCorner); merged.insert(nodes[right].highCorner);
        nodes[p].lowCorner = merged.lowCorner;
        nodes[p].highCorner = merged.highCorner;
        cur = p;
    }
}

__device__ bool intersectBox(const Ray& ray, const LeafNode& node, float limit) {
    float near = 0.f, far = limit;
    const float origins[] = {ray.origin.x, ray.origin.y, ray.origin.z};
    const float directions[] = {ray.direction.x, ray.direction.y, ray.direction.z};
    const float low[] = {node.lowCorner.x, node.lowCorner.y, node.lowCorner.z};
    const float high[] = {node.highCorner.x, node.highCorner.y, node.highCorner.z};
    for (int axis = 0; axis < 3; ++axis) {
        if (directions[axis] == 0.f) {
            if (origins[axis] < low[axis] || origins[axis] > high[axis]) return false;
        } else {
            const float a = (low[axis] - origins[axis]) / directions[axis];
            const float b = (high[axis] - origins[axis]) / directions[axis];
            near = fmaxf(near, fminf(a, b));
            far = fminf(far, fmaxf(a, b));
            if (near > far) return false;
        }
    }
    return true;
}

__device__ Hit find_closest_hit(const Ray& ray, const GPUTriangle* triangles,
                               const LeafNode* nodes, int n_nodes, int last_hit) {
    Hit closest;
    int idx = 0;
    while (idx < n_nodes) {
        const LeafNode& node = nodes[idx];
        if (!intersectBox(ray, node, closest.record.t)) {
            // Preorder makes every subtree contiguous, so traversal needs no stack.
            idx += countNodes(node.n);
            continue;
        }
        if (node.rightNodeIndex == -1) {
            for (int j = node.startIndex; j < node.startIndex + node.n; ++j) {
                const GPUTriangle& triangle = triangles[j];
                if (triangle.triangle_index == last_hit) continue;
                HitRecord record;
                if (intersect_triangle(ray, triangle.triangle, &record) &&
                    (record.t < closest.record.t ||
                     (record.t == closest.record.t && triangle.triangle_index < closest.triangle_index))) {
                    closest.record = record;
                    closest.object_index = triangle.object_index;
                    closest.triangle_index = triangle.triangle_index;
                }
            }
        }
        ++idx;
    }
    return closest;
}

__device__ Vec3 trace_ray(Ray ray, const RenderSettings& settings,
                          const GPUTriangle* triangles, const ObjectMaterial* materials,
                          const LeafNode* nodes, int n_nodes, RandomGPU& random) {
    Vec3 color, throughput(1.f, 1.f, 1.f);
    int last_hit = -1;
    for (int depth = 0; ; ++depth) {
        const Hit hit = find_closest_hit(ray, triangles, nodes, n_nodes, last_hit);
        if (!hit.hit()) {
            color += multiply(throughput, settings.background_light);
            break;
        }
        const ObjectMaterial& material = materials[hit.object_index];
        color += multiply(throughput, material.emission);
        if (depth >= settings.max_bounces) break;
        throughput = multiply(throughput, material.albedo);
        ray = cosine_weighted_sampling(hit.record, random);
        last_hit = hit.triangle_index;
    }
    return color;
}

__global__ void renderPixels(RenderSettings settings, const GPUTriangle* triangles,
                             const ObjectMaterial* materials, const LeafNode* nodes,
                             int n_nodes, Pixel* pixels, size_t pixel_count) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= pixel_count) return;
    const int x = idx % settings.width;
    const int y = idx / settings.width;
    RandomGPU random(settings.seed + static_cast<std::uint32_t>(idx));
    Vec3 color;
    const int factor = settings.supersampling_factor;
    for (int sy = 0; sy < factor; ++sy) {
        for (int sx = 0; sx < factor; ++sx) {
            const Ray ray = make_camera_ray(settings, x * factor + sx, y * factor + sy);
            for (int sample = 0; sample < settings.samples_per_pixel; ++sample)
                color += trace_ray(ray, settings, triangles, materials, nodes, n_nodes, random);
        }
    }
    pixels[idx] = to_pixel(color / (float(settings.samples_per_pixel) * factor * factor));
}
} // namespace

std::vector<Pixel> render_scene(const RenderSettings& settings, const std::vector<Object>& objects) {
    if (settings.width <= 0 || settings.height <= 0 || settings.samples_per_pixel <= 0 ||
        settings.supersampling_factor <= 0 || settings.max_bounces < 0 ||
        settings.width > std::numeric_limits<int>::max() / settings.supersampling_factor ||
        settings.height > std::numeric_limits<int>::max() / settings.supersampling_factor)
        throw std::invalid_argument("Invalid render dimensions or sampling settings");

    std::vector<GPUTriangle> triangles;
    std::vector<ObjectMaterial> materials;
    NodeBox fullSpace;
    for (size_t object_index = 0; object_index < objects.size(); ++object_index) {
        if (object_index > size_t(std::numeric_limits<int>::max()) ||
            objects[object_index].triangles.size() > size_t(std::numeric_limits<int>::max()) - triangles.size())
            throw std::length_error("Scene has too many objects or triangles");
        const Object& object = objects[object_index];
        materials.push_back(ObjectMaterial{object.emission, object.albedo});
        for (const Triangle& triangle : object.triangles) {
            triangles.push_back(GPUTriangle{triangle, static_cast<int>(object_index),
                                           static_cast<int>(triangles.size()), 0});
            fullSpace.insert(triangle);
        }
    }
    const size_t pixel_count = size_t(settings.width) * settings.height;
    std::vector<Pixel> pixels(pixel_count);
    const int n_triangles = static_cast<int>(triangles.size());
    if (n_triangles == 0) {
        for (Pixel& pixel : pixels) pixel = to_pixel(settings.background_light);
        return pixels;
    }
    const int n_nodes = countNodes(n_triangles);
    GPUArray<GPUTriangle> gpu_triangles(triangles.size());
    GPUArray<ObjectMaterial> gpu_materials(materials.size());
    GPUArray<Pixel> gpu_pixels(pixel_count);
    GPUArray<LeafNode> gpu_nodes(n_nodes);
    GPUArray<unsigned long long> mortonCodes(n_triangles), deltas(n_triangles - 1);
    GPUArray<int> visits(n_nodes);
    checkCuda(cudaMemcpy(gpu_triangles.data, triangles.data(), triangles.size() * sizeof(GPUTriangle), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(gpu_materials.data, materials.data(), materials.size() * sizeof(ObjectMaterial), cudaMemcpyHostToDevice));
    const int blocks = (n_triangles - 1) / BLOCK_SIZE + 1;
    mortonEncoding<<<blocks, BLOCK_SIZE>>>(n_triangles, gpu_triangles.data, mortonCodes.data, fullSpace);
    checkCuda(cudaGetLastError());
    thrust::sort_by_key(thrust::device, mortonCodes.data, mortonCodes.data + n_triangles, gpu_triangles.data);
    calcDelta<<<blocks, BLOCK_SIZE>>>(n_triangles, mortonCodes.data, deltas.data);
    checkCuda(cudaGetLastError());
    constructNodes(n_triangles, n_nodes, gpu_triangles.data, deltas.data, gpu_nodes.data);
    checkCuda(cudaMemset(visits.data, 0, n_nodes * sizeof(int)));
    computeBoxes<<<(n_nodes - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(n_nodes, gpu_triangles.data, gpu_nodes.data, visits.data);
    checkCuda(cudaGetLastError());
    renderPixels<<<(pixel_count - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(settings, gpu_triangles.data, gpu_materials.data, gpu_nodes.data, n_nodes, gpu_pixels.data, pixel_count);
    checkCuda(cudaGetLastError());
    checkCuda(cudaMemcpy(pixels.data(), gpu_pixels.data, pixel_count * sizeof(Pixel), cudaMemcpyDeviceToHost));
    return pixels;
}
