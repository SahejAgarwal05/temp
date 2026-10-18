// GPU pipeline: upload scene -> build Morton BVH -> trace paths -> average pixels.
#include "graphics.cuh"
#include "geometry.cuh"

#include <cuda/atomic>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace {
constexpr int kStack = 64;
constexpr int kBuildBlock = 256;
constexpr int kLeafSize = 2;

struct Material { float4 albedo, emission; };
__constant__ Material materials[128];

// Leaf: triangles[link..link+count). Internal: children at index+1 and link.
struct alignas(16) Node { Vec3 lo; int link; Vec3 hi; int count; };

// Keep the original ID through sorting for self-hit exclusion and distance ties.
struct alignas(16) Primitive { Triangle triangle; int object, id, padding; };
static_assert(sizeof(Node) == 32 && sizeof(Primitive) == 48, "Unexpected device layout");

struct Box {
    Vec3 lo = Vec3(INF, INF, INF), hi = Vec3(-INF, -INF, -INF);
    // Expand this box to include a point.
    __host__ __device__ void add(const Vec3& p) {
        lo = Vec3(fminf(lo.x, p.x), fminf(lo.y, p.y), fminf(lo.z, p.z));
        hi = Vec3(fmaxf(hi.x, p.x), fmaxf(hi.y, p.y), fmaxf(hi.z, p.z));
    }
    // Expand this box to include all three corners of a triangle.
    __device__ void add(const Triangle& triangle) {
        add(triangle.p0); add(triangle.p1); add(triangle.p2);
    }
    // Expand this box to include a tree node's entire bounding box.
    __device__ void add(const Node& node) {
        add(node.lo); add(node.hi);
    }
    // Slightly enlarge the box to include flat triangles and edges despite rounding.
    __device__ void pad() {
        const float scale = 8 * 1.1920929e-7f;
        const Vec3 p(scale * fmaxf(1.f, fmaxf(fabsf(lo.x), fabsf(hi.x))),
                     scale * fmaxf(1.f, fmaxf(fabsf(lo.y), fabsf(hi.y))),
                     scale * fmaxf(1.f, fmaxf(fabsf(lo.z), fabsf(hi.z))));
        lo -= p; hi += p;
    }
};

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    // Allocate GPU memory for n elements; an empty buffer keeps a null pointer.
    explicit DeviceBuffer(size_t n) { if (n) cudaMalloc(&data, n * sizeof(T)); }
    // Release GPU memory automatically when this owner leaves scope.
    ~DeviceBuffer() { if (data) cudaFree(data); }
    // Disable copying so two owners cannot free the same allocation.
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// Return the storage needed for all leaves and internal nodes, or zero if empty.
int bvh_node_count(int count) {
    const int leaves = (count + kLeafSize - 1) / kLeafSize;
    return leaves ? 2 * leaves - 1 : 0;
}

// One GPU block finds the box enclosing every triangle and stores it in nodes[0].
__global__ void scene_bounds(const Primitive* triangles, int count, Node* nodes) {
    __shared__ float low[kBuildBlock][3], high[kBuildBlock][3];
    const int t = threadIdx.x;
    Box box;
    for (int i = t; i < count; i += blockDim.x) box.add(triangles[i].triangle);
    low[t][0] = box.lo.x; low[t][1] = box.lo.y; low[t][2] = box.lo.z;
    high[t][0] = box.hi.x; high[t][1] = box.hi.y; high[t][2] = box.hi.z;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset; offset /= 2) {
        if (t < offset) for (int axis = 0; axis < 3; ++axis) {
            low[t][axis] = fminf(low[t][axis], low[t + offset][axis]);
            high[t][axis] = fmaxf(high[t][axis], high[t + offset][axis]);
        }
        __syncthreads();
    }
    if (t == 0) {
        nodes[0].lo = Vec3(low[0][0], low[0][1], low[0][2]);
        nodes[0].hi = Vec3(high[0][0], high[0][1], high[0][2]);
    }
}

// Map a coordinate into the integer grid 0..1023; a flat axis maps to zero.
__device__ unsigned grid_position(float value, float low, float high) {
    return high > low ? unsigned(fminf(1023.f, fmaxf(0.f, (value - low) / (high - low) * 1024.f))) : 0;
}

// Give each triangle a sortable key based on the center of its bounding box.
// Morton codes interleave 10 bits per axis; appended IDs make every key unique.
__global__ void spatial_keys(const Primitive* triangles, int count, const Node* scene,
                              unsigned long long* keys) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    Box box;
    box.add(triangles[i].triangle);
    const Vec3 center = box.lo * .5f + box.hi * .5f;
    const unsigned x = grid_position(center.x, scene[0].lo.x, scene[0].hi.x);
    const unsigned y = grid_position(center.y, scene[0].lo.y, scene[0].hi.y);
    const unsigned z = grid_position(center.z, scene[0].lo.z, scene[0].hi.z);
    unsigned morton = 0;
    for (int bit = 0; bit < 10; ++bit) {
        morton |= ((x >> bit) & 1u) << (3 * bit + 2);
        morton |= ((y >> bit) & 1u) << (3 * bit + 1);
        morton |= ((z >> bit) & 1u) << (3 * bit);
    }
    keys[i] = (static_cast<unsigned long long>(morton) << 32) | unsigned(triangles[i].id);
}

// Return the last leaf of the left child by splitting at the highest differing key bit.
__device__ int split_leaves(const unsigned long long* keys, int first, int last) {
    const auto start = keys[first * kLeafSize];
    const int prefix = __clzll(start ^ keys[last * kLeafSize]);
    int left = first, right = last;
    while (left + 1 < right) {
        const int middle = (left + right) / 2;
        if (__clzll(start ^ keys[middle * kLeafSize]) > prefix) left = middle;
        else right = middle;
    }
    return left;
}

// Build leaf boxes and triangle ranges, plus child and parent links for the tree.
// Each GPU thread handles one leaf and walks from the root to find its array index.
__global__ void build_leaves(const Primitive* triangles, int count,
                             const unsigned long long* keys, Node* nodes, int* parents) {
    const int leaf = blockIdx.x * blockDim.x + threadIdx.x;
    const int leaves = (count + kLeafSize - 1) / kLeafSize;
    if (leaf >= leaves) return;
    int current = 0, parent = -1, first = 0, last = leaves - 1;
    while (first < last) {
        const int split = split_leaves(keys, first, last);
        // A left subtree with L leaves occupies 2*L-1 slots.
        const int right = current + 2 * (split - first + 1);
        // Only the first leaf in a range writes its internal node.
        if (leaf == first) {
            nodes[current].link = right; nodes[current].count = 0;
            parents[current] = parent;
        }
        parent = current;
        if (leaf <= split) { ++current; last = split; }
        else { current = right; first = split + 1; }
    }
    const int begin = leaf * kLeafSize;
    const int size = count - begin < kLeafSize ? count - begin : kLeafSize;
    Box box;
    for (int i = begin; i < begin + size; ++i) box.add(triangles[i].triangle);
    box.pad();
    nodes[current].lo = box.lo; nodes[current].hi = box.hi;
    nodes[current].link = begin; nodes[current].count = size;
    parents[current] = parent;
}

// Walk upward from leaves, combining child boxes until every parent encloses its subtree.
__global__ void merge_bounds(Node* nodes, const int* parents, int* finished, int count) {
    int current = blockIdx.x * blockDim.x + threadIdx.x;
    if (current >= count || nodes[current].count == 0) return;
    while (parents[current] != -1) {
        const int parent = parents[current];
        cuda::atomic_ref<int, cuda::thread_scope_device> children(finished[parent]);
        // First child stops; second acquires its sibling's box and merges upward.
        if (children.fetch_add(1, cuda::memory_order_acq_rel) == 0) return;
        Box box;
        box.add(nodes[parent + 1]);
        box.add(nodes[nodes[parent].link]);
        nodes[parent].lo = box.lo; nodes[parent].hi = box.hi;
        current = parent;
    }
}

// CPU coordinator: launch GPU bounds, key generation, sorting, tree links, and box merging.
// Sorting reorders the triangle array so each leaf refers to a contiguous range.
void build_bvh(Primitive* triangles, int count, Node* nodes) {
    if (!count) return;
    const int node_count = bvh_node_count(count), leaves = (count + kLeafSize - 1) / kLeafSize;
    DeviceBuffer<unsigned long long> keys(count);
    DeviceBuffer<int> parents(node_count), finished(node_count);
    cudaMemset(finished.data, 0, node_count * sizeof(int));
    // Default-stream ordering keeps build stages sequential on the GPU.
    // nodes[0] temporarily stores scene bounds for key normalization.
    scene_bounds<<<1, kBuildBlock>>>(triangles, count, nodes);
    spatial_keys<<<(count + kBuildBlock - 1) / kBuildBlock, kBuildBlock>>>(triangles, count, nodes, keys.data);
    thrust::sort_by_key(thrust::device, thrust::device_pointer_cast(keys.data),
                        thrust::device_pointer_cast(keys.data + count), thrust::device_pointer_cast(triangles));
    build_leaves<<<(leaves + kBuildBlock - 1) / kBuildBlock, kBuildBlock>>>(triangles, count, keys.data, nodes, parents.data);
    merge_bounds<<<(node_count + kBuildBlock - 1) / kBuildBlock, kBuildBlock>>>(nodes, parents.data, finished.data, node_count);
}

// Return whether the ray enters this box before best, and write its entry distance.
// Intersect the allowed ray-distance intervals along x, y, and z.
__device__ bool box_hit(const Node& node, const Ray& ray, const Vec3& inverse, float best, float& entry) {
    float near = 0, far = best;
    for (int axis = 0; axis < 3; ++axis) {
        const float o = axis == 0 ? ray.origin.x : axis == 1 ? ray.origin.y : ray.origin.z;
        const float d = axis == 0 ? ray.direction.x : axis == 1 ? ray.direction.y : ray.direction.z;
        const float inv = axis == 0 ? inverse.x : axis == 1 ? inverse.y : inverse.z;
        const float lo = axis == 0 ? node.lo.x : axis == 1 ? node.lo.y : node.lo.z;
        const float hi = axis == 0 ? node.hi.x : axis == 1 ? node.hi.y : node.hi.z;
        // Avoid 0*infinity for parallel rays and reciprocal overflow for tiny d.
        if (d == 0) { if (o < lo || o > hi) return false; }
        else {
            const float a = isfinite(inv) ? (lo - o) * inv : (lo - o) / d;
            const float b = isfinite(inv) ? (hi - o) * inv : (hi - o) / d;
            near = fmaxf(near, fminf(a, b));
            far = fminf(far, fmaxf(a, b));
            if (near > far) return false;
        }
    }
    entry = near;
    return true;
}
struct Closest { float t = INF; int index = -1, id = 2147483647; };
// Test triangle i and update the nearest hit, skipping the previous bounce's triangle.
__device__ void candidate(const Ray& ray, const Primitive* triangles, int i, int last, Closest& hit) {
    const Primitive& p = triangles[i];
    if (p.id == last) return;
    HitRecord record;
    // Equal-distance hits follow original scene order, as in the CPU reference.
    if (intersect_triangle(ray, p.triangle, &record) &&
        (record.t < hit.t || (record.t == hit.t && p.id < hit.id))) {
        hit.t = record.t; hit.index = i; hit.id = p.id;
    }
}

// Search the BVH for the nearest triangle, visiting nearer boxes first to prune work.
// Return index = -1 when the ray misses all eligible triangles.
__device__ Closest closest_hit(const Ray& ray, const Primitive* triangles, int count,
                               const Node* nodes, int last) {
    Closest hit;
    if (!count) return hit;
    const Vec3 inverse(1.f / ray.direction.x, 1.f / ray.direction.y, 1.f / ray.direction.z);
    // Unique 62-bit keys limit tree depth; 64 slots also fit the pending leaf.
    int stack[kStack], size = 0;
    stack[size++] = 0;
    while (size) {
        const int current = stack[--size];
        const Node node = nodes[current];
        float entry;
        if (!box_hit(node, ray, inverse, hit.t, entry)) continue;
        if (node.count) {
            for (int i = node.link; i < node.link + node.count; ++i)
                candidate(ray, triangles, i, last, hit);
            continue;
        }

        const int left = current + 1, right = node.link;
        float left_entry, right_entry;
        const bool hit_left = box_hit(nodes[left], ray, inverse, hit.t, left_entry);
        const bool hit_right = box_hit(nodes[right], ray, inverse, hit.t, right_entry);
        // Push farther first so the nearer child is popped first. Each popped
        // node is checked again against any closer hit found in the meantime.
        if (hit_left && hit_right) {
            const bool left_first = left_entry <= right_entry;
            stack[size++] = left_first ? right : left;
            stack[size++] = left_first ? left : right;
        } else if (hit_left || hit_right) {
            stack[size++] = hit_left ? left : right;
        }
    }
    return hit;
}
// Scramble an integer to give neighboring GPU workers different random seeds.
__device__ uint32_t hash(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16);
}

// Follow one ray through its bounces, accumulating emitted light and background
// weighted by the surface colors encountered along the path.
__device__ Vec3 trace_path(Ray ray, const RenderSettings& settings,
                          const Primitive* triangles, int count, const Node* nodes,
                          RandomGPU& random) {
    Vec3 throughput(1, 1, 1), light;
    int last = -1;
    for (int depth = 0; ; ++depth) {
        const Closest hit = closest_hit(ray, triangles, count, nodes, last);
        if (hit.index < 0) {
            light += multiply(throughput, settings.background_light);
            break;
        }
        const Primitive p = triangles[hit.index];
        const Material material = materials[p.object];
        // The final surface emits even when the bounce limit prevents scattering.
        light += multiply(throughput, Vec3(material.emission.x, material.emission.y, material.emission.z));
        if (depth >= settings.max_bounces) break;
        throughput = multiply(throughput, Vec3(material.albedo.x, material.albedo.y, material.albedo.z));

        HitRecord record;
        record.t = hit.t;
        record.position = ray.origin + hit.t * ray.direction;
        record.unit_normal = normalize(cross(p.triangle.p1 - p.triangle.p0,
                                             p.triangle.p2 - p.triangle.p0));
        if (dot(record.unit_normal, ray.direction) > 0) record.unit_normal = -record.unit_normal;
        ray = cosine_weighted_sampling(record, random);
        last = p.id;
    }
    return light;
}

// Trace all samples on the GPU. Each group of split threads shares one subpixel,
// traces different samples, and adds their combined light to sums[subpixel].
__global__ void paths(RenderSettings settings, const Primitive* triangles, int count,
                      const Node* nodes, Vec3* sums,
                      size_t subpixels, int split) {
    const size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t subpixel = tid / split;
    const int part = tid & (split - 1);
    Vec3 sum;
    if (subpixel < subpixels) {
        RandomGPU random(hash(uint32_t(tid) ^ hash(settings.seed)));
        const int width = settings.width * settings.supersampling_factor;
        const Ray primary = make_camera_ray(settings, int(subpixel % width), int(subpixel / width));
        for (size_t sample = part; sample < size_t(settings.samples_per_pixel); sample += split)
            sum += trace_path(primary, settings, triangles, count, nodes, random);
    }
    // All lanes, including padded ones, must reach these full-mask shuffles.
    const int width = split < 32 ? split : 32;
    for (int offset = width / 2; offset; offset /= 2) {
        sum.x += __shfl_down_sync(0xffffffffu, sum.x, offset, width);
        sum.y += __shfl_down_sync(0xffffffffu, sum.y, offset, width);
        sum.z += __shfl_down_sync(0xffffffffu, sum.z, offset, width);
    }
    if (subpixel < subpixels && (threadIdx.x & (width - 1)) == 0) {
        // Atomics combine warp sums when a subpixel has more than 32 workers.
        atomicAdd(&sums[subpixel].x, sum.x);
        atomicAdd(&sums[subpixel].y, sum.y);
        atomicAdd(&sums[subpixel].z, sum.z);
    }
}

// One GPU thread per output pixel averages its samples and supersampled subpixels.
__global__ void average(RenderSettings settings, const Vec3* sums, Vec3* colors, size_t pixels) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= pixels) return;
    const int ss = settings.supersampling_factor, width = settings.width * ss;
    const size_t start = (i / settings.width * ss) * width + (i % settings.width) * ss;
    Vec3 color;
    for (int y = 0; y < ss; ++y)
        for (int x = 0; x < ss; ++x) color += sums[start + size_t(y) * width + x];
    colors[i] = color / (float(settings.samples_per_pixel) * ss * ss);
}
} // namespace

// CPU entry point: upload geometry/materials, build the GPU BVH, trace and average
// samples, then copy colors back and convert them into final image pixels.
std::vector<Pixel> render_scene(
    const RenderSettings& settings,
    const std::vector<Object>& objects
) {
    std::vector<Primitive> host_triangles;
    std::vector<Material> host_materials;
    host_triangles.reserve(objects.size() * 256);
    for (int o = 0; o < int(objects.size()); ++o) {
        const Vec3 a = objects[o].albedo, e = objects[o].emission;
        host_materials.push_back({make_float4(a.x, a.y, a.z, 0), make_float4(e.x, e.y, e.z, 0)});
        for (const Triangle& t : objects[o].triangles)
            host_triangles.push_back({t, o, int(host_triangles.size()), 0});
    }
    const int count = int(host_triangles.size());

    int device; cudaGetDevice(&device);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);
    const size_t pixels = size_t(settings.width) * settings.height;
    const size_t subpixels = pixels * settings.supersampling_factor * settings.supersampling_factor;
    int split = 1;
    // Share a subpixel's samples across power-of-two groups to occupy small renders.
    const size_t target = size_t(prop.multiProcessorCount) * 2048 * 4;
    while (subpixels * split < target && split < settings.samples_per_pixel && split < (1 << 24)) split *= 2;
    const int block = 128;
    DeviceBuffer<Primitive> triangles(count);
    DeviceBuffer<Node> nodes(bvh_node_count(count));
    DeviceBuffer<Vec3> sums(subpixels), colors(pixels);
    if (count) cudaMemcpy(triangles.data, host_triangles.data(), count * sizeof(Primitive), cudaMemcpyHostToDevice);
    build_bvh(triangles.data, count, nodes.data);
    if (!host_materials.empty()) {
        cudaMemcpyToSymbol(materials, host_materials.data(), host_materials.size() * sizeof(Material));
    }
    cudaMemset(sums.data, 0, subpixels * sizeof(Vec3));
    const size_t blocks = (subpixels * split + block - 1) / block;
    paths<<<unsigned(blocks), block>>>(settings, triangles.data, count, nodes.data, sums.data, subpixels, split);
    average<<<unsigned((pixels + 255) / 256), 256>>>(settings, sums.data, colors.data, pixels);
    std::vector<Vec3> host_colors(pixels);
    cudaMemcpy(host_colors.data(), colors.data, pixels * sizeof(Vec3), cudaMemcpyDeviceToHost);
    std::vector<Pixel> result(pixels);
    for (size_t i = 0; i < pixels; ++i) result[i] = to_pixel(host_colors[i]);
    return result;
}
