// CUDA renderer with selectable sampled uniform-grid and exact BVH searches.
// Host: pack/upload triangles and materials, allocate buffers using GPU counts.
// Sampled GPU build: subdivide triangle surfaces at spacing <= D, prefix-scan
// counts, generate samples, count/scan/scatter references into a uniform grid.
// GPU rendering: paths<true> uses grid DDA; paths<false> uses the original LBVH.
// Both share material shading, bounce sampling, sample distribution, averaging,
// tone mapping and gamma conversion. Only final RGB bytes return to the CPU.
#include "graphics.cuh"
#include "sampled_config.cuh"
#include "geometry.cuh"

#include <cuda/atomic>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
constexpr int kStack = 64;
constexpr int kBuildBlock = 256;
constexpr int kLeafSize = 2;

// Each object has one material shared by all its triangles. xyz holds RGB;
// float4's unused fourth component keeps the values aligned for GPU loads.
// albedo is the fraction of light reflected; emission is light produced here.
struct Material { float4 albedo, emission; };
__constant__ Material materials[128];

/*
Node is one box in a flat tree array; lo/hi are its minimum/maximum x,y,z.
  count > 0: leaf; link is the first triangle index, count is how many to test.
  count = 0: internal node; left child is this node's index + 1, right is link.
For example, link=20,count=3 means triangles[20..22]. An internal nodes[5]
with link=12 has children nodes[6] and nodes[12]. The GPU builder creates this
same compact layout, so traversal still needs no pointers or separate objects.
*/
struct alignas(16) Node { Vec3 lo; int link; Vec3 hi; int count; };

// object selects materials[object]. id is the original flattened triangle ID,
// which survives GPU sorting. Example: reordered array indices [0,1,2] might
// now contain IDs [2,0,1]. A hit at index 0 must exclude ID 2 next bounce.
// padding has no scene meaning; it makes the record a multiple of 16 bytes.
struct alignas(16) Primitive { Triangle triangle; int object, id, padding; };
static_assert(sizeof(Node) == 32 && sizeof(Primitive) == 48, "Unexpected device layout");

// Box only does geometry; unlike Node it has no tree links. Start with inverted
// bounds, then add points/boxes to expand lo/hi to enclose them. GPU construction
// uses it for triangle and parent boxes; the test driver also uses it on the CPU.
struct Box {
    Vec3 lo = Vec3(INF, INF, INF), hi = Vec3(-INF, -INF, -INF);
    __host__ __device__ void add(const Vec3& p) {
        lo = Vec3(fminf(lo.x, p.x), fminf(lo.y, p.y), fminf(lo.z, p.z));
        hi = Vec3(fmaxf(hi.x, p.x), fmaxf(hi.y, p.y), fmaxf(hi.z, p.z));
    }
    __device__ void pad() {
        // Expand a few floating-point rounding steps to avoid missing a flat
        // triangle or shared edge because a computed intersection is rounded.
        const float scale = 8 * 1.1920929e-7f;
        const Vec3 p(scale * fmaxf(1.f, fmaxf(fabsf(lo.x), fabsf(hi.x))),
                     scale * fmaxf(1.f, fmaxf(fabsf(lo.y), fabsf(hi.y))),
                     scale * fmaxf(1.f, fmaxf(fabsf(lo.z), fabsf(hi.z))));
        lo -= p; hi += p;
    }
};

/*
DeviceBuffer<T> owns a GPU array from the CPU. T is an element type, so
DeviceBuffer<Node>(100) allocates room for 100 Nodes, not 100 bytes.
  data: GPU pointer passed to kernels/cudaMemcpy; do not read data[i] on the CPU.
  Constructor: allocate. Destructor (~DeviceBuffer): free when it leaves scope.
  = delete: prevent two copied owners from accidentally freeing the same pointer.
The owner is on the CPU; its pointed-to array is on the GPU.
*/
void cuda_check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(error));
}

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t n) { allocate(n); }
    void allocate(size_t n) {
        if (data) throw std::runtime_error("Device buffer already allocated");
        if (n > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::runtime_error("Device allocation size overflow");
        if (n) cuda_check(cudaMalloc(&data, n * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// Up to two adjacent sorted triangles form a leaf. A binary tree with L leaves has
// L-1 internal nodes, so it needs 2*L-1 slots. An empty scene needs no slots.
int bvh_node_count(int count) {
    const int leaves = (count + kLeafSize - 1) / kLeafSize;
    return leaves ? 2 * leaves - 1 : 0;
}

// GPU build stage 1: find scene bounds using one cooperating block. Each thread
// scans a different strided subset, then the block reduces their minimum/maximum
// coordinates. nodes[0] temporarily holds this box for spatial-key normalization.
__global__ void scene_bounds(const Primitive* triangles, int count, Node* nodes) {
    __shared__ float low[kBuildBlock][3], high[kBuildBlock][3];
    const int t = threadIdx.x;
    Box box;
    for (int i = t; i < count; i += blockDim.x) {
        const Triangle p = triangles[i].triangle;
        box.add(p.p0); box.add(p.p1); box.add(p.p2);
    }
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

// Map one coordinate to a 10-bit grid position. A flat scene axis maps to zero.
__device__ unsigned grid_position(float value, float low, float high) {
    return high > low ? unsigned(fminf(1023.f, fmaxf(0.f, (value - low) / (high - low) * 1024.f))) : 0;
}

// GPU build stage 2: give every triangle a sortable spatial key. Interleaving
// x/y/z bits produces a Morton code: nearby codes tend to describe nearby space.
// Append the original ID so even coincident centers have distinct sortable keys.
__global__ void spatial_keys(const Primitive* triangles, int count, const Node* scene,
                              unsigned long long* keys) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const Triangle p = triangles[i].triangle;
    Box box; box.add(p.p0); box.add(p.p1); box.add(p.p2);
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

// Split a range of leaves at the highest differing spatial-key bit. Each leaf
// is represented by its first triangle's key. __clzll counts equal leading bits
// in two keys after XOR. Binary search finds the final leaf sharing the next bit
// with the first leaf. Distinct appended IDs make the endpoint XOR nonzero.
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

// GPU build stage 3: one thread per leaf writes its triangle range and box.
// Each thread walks its own key ranges to calculate its node's flat-array index.
// Only the first leaf in a range writes that range's internal-node links, so
// threads never race to initialize a node. No CPU recursion or node queue is used.
__global__ void build_leaves(const Primitive* triangles, int count,
                             const unsigned long long* keys, Node* nodes, int* parents) {
    const int leaf = blockIdx.x * blockDim.x + threadIdx.x;
    const int leaves = (count + kLeafSize - 1) / kLeafSize;
    if (leaf >= leaves) return;
    int current = 0, parent = -1, first = 0, last = leaves - 1;
    while (first < last) {
        const int split = split_leaves(keys, first, last);
        // A left subtree with L leaves occupies 2*L-1 nodes. Its root is next
        // after the parent, so the right root starts at parent + 2*L.
        const int right = current + 2 * (split - first + 1);
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
    for (int i = begin; i < begin + size; ++i) {
        const Triangle p = triangles[i].triangle;
        box.add(p.p0); box.add(p.p1); box.add(p.p2);
    }
    box.pad();
    nodes[current].lo = box.lo; nodes[current].hi = box.hi;
    nodes[current].link = begin; nodes[current].count = size;
    parents[current] = parent;
}

// GPU build stage 4: start at leaves and merge boxes upward. At a parent, the
// first finished child stops; the second merges both child boxes and continues.
// Acquire/release publishes a child's box before its sibling reads it, even if
// the threads run in different blocks. The root's parent is -1, ending the walk.
__global__ void merge_bounds(Node* nodes, const int* parents, int* finished, int count) {
    int current = blockIdx.x * blockDim.x + threadIdx.x;
    if (current >= count || nodes[current].count == 0) return;
    while (parents[current] != -1) {
        const int parent = parents[current];
        cuda::atomic_ref<int, cuda::thread_scope_device> children(finished[parent]);
        if (children.fetch_add(1, cuda::memory_order_acq_rel) == 0) return;
        Box box;
        box.add(nodes[parent + 1].lo); box.add(nodes[parent + 1].hi);
        box.add(nodes[nodes[parent].link].lo); box.add(nodes[nodes[parent].link].hi);
        nodes[parent].lo = box.lo; nodes[parent].hi = box.hi;
        current = parent;
    }
}

// Host coordinator only: launch GPU stages in default-stream order. Thrust is
// CUDA's sorting library; it sorts the keys and their attached triangle records
// together on the GPU. Temporary arrays live only for this build, not for rays.
void build_bvh(Primitive* triangles, int count, Node* nodes) {
    if (!count) return;
    const int node_count = bvh_node_count(count), leaves = (count + kLeafSize - 1) / kLeafSize;
    DeviceBuffer<unsigned long long> keys(count);
    DeviceBuffer<int> parents(node_count), finished(node_count);
    cudaMemset(finished.data, 0, node_count * sizeof(int));
    scene_bounds<<<1, kBuildBlock>>>(triangles, count, nodes);
    spatial_keys<<<(count + kBuildBlock - 1) / kBuildBlock, kBuildBlock>>>(triangles, count, nodes, keys.data);
    thrust::sort_by_key(thrust::device, thrust::device_pointer_cast(keys.data),
                        thrust::device_pointer_cast(keys.data + count), thrust::device_pointer_cast(triangles));
    build_leaves<<<(leaves + kBuildBlock - 1) / kBuildBlock, kBuildBlock>>>(triangles, count, keys.data, nodes, parents.data);
    merge_bounds<<<(node_count + kBuildBlock - 1) / kBuildBlock, kBuildBlock>>>(nodes, parents.data, finished.data, node_count);
}

// Slab test: intersect the ray's distance interval with each axis's box interval.
// Starting far at best also prunes boxes beyond the closest triangle found so far.
// inverse is reused for all nodes visited by this ray; entry orders children.
// A ray is origin + t*direction. Each axis gives an allowed interval of t values.
// Example: x allows [2,6], y allows [3,8], z allows [1,4]; their intersection is
// [3,4], so the ray enters at t=3. Disjoint intervals mean it misses the box.
__device__ bool box_hit(const Node& node, const Ray& ray, const Vec3& inverse, float best, float& entry) {
    float near = 0, far = best;
    // Handle parallel rays explicitly: 0 * infinity must not become a NaN.
    for (int axis = 0; axis < 3; ++axis) {
        const float o = axis == 0 ? ray.origin.x : axis == 1 ? ray.origin.y : ray.origin.z;
        const float d = axis == 0 ? ray.direction.x : axis == 1 ? ray.direction.y : ray.direction.z;
        const float inv = axis == 0 ? inverse.x : axis == 1 ? inverse.y : inverse.z;
        const float lo = axis == 0 ? node.lo.x : axis == 1 ? node.lo.y : node.lo.z;
        const float hi = axis == 0 ? node.hi.x : axis == 1 ? node.hi.y : node.hi.z;
        if (d == 0) { if (o < lo || o > hi) return false; }
        else {
            // Very small nonzero components can overflow the reciprocal.
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
// Closest is local to one ray search on the GPU, not a scene-wide hit list.
//   t: nearest distance found so far; INF initially means no distance limit.
//   index: position in the reordered triangle array; -1 means no hit was found.
//   id: original scene triangle ID, used for self-hit exclusion and equal-distance
//       tie-breaking. Its initial large value lets a real ID win the first tie.
// Position/normal are absent because only the final winner needs that shading data.
struct Closest { float t = INF; int index = -1, id = 2147483647; };
// Test one candidate triangle and update the caller's Closest record by reference.
// i is a reordered array index; last is an original ID from the previous bounce.
__device__ void candidate(const Ray& ray, const Primitive* triangles, int i, int last, Closest& hit) {
    const Primitive& p = triangles[i];
    // Ignore exactly the triangle the path just left, as in the CPU reference.
    if (p.id == last) return;
    HitRecord record;
    // Keep the supplied intersection arithmetic. Only t is consumed here, so
    // inlining eliminates candidate position/normal work; shade the winner below.
    // Equal-distance hits prefer the original scene order, matching the CPU's
    // strict less-than update even when the BVH visits triangles out of order.
    if (intersect_triangle(ray, p.triangle, &record) &&
        (record.t < hit.t || (record.t == hit.t && p.id < hit.id))) {
        hit.t = record.t; hit.index = i; hit.id = p.id;
    }
}

// Walk the BVH with a small stack, visiting the nearer child first.
__device__ Closest closest_hit(const Ray& ray, const Primitive* triangles, int count,
                               const Node* nodes, int last) {
    Closest hit;
    if (!count) return hit;
    const Vec3 inverse(1.f / ray.direction.x, 1.f / ray.direction.y, 1.f / ray.direction.z);
    int stack[kStack], size = 0, current = 0;
    // current is the node being processed; stack[0..size-1] holds nodes postponed
    // until later. This stack stores node indices, not triangles or full rays.
    // Each split consumes a differing bit of a unique 62-bit key, so a path
    // has at most 62 internal levels and fits the 64-entry traversal stack.
    while (true) {
        const Node node = nodes[current];
        float entry;
        if (box_hit(node, ray, inverse, hit.t, entry)) {
            if (node.count) {
                for (int i = node.link; i < node.link + node.count; ++i)
                    candidate(ray, triangles, i, last, hit);
            } else {
                // Ask which child boxes this ray can enter before its best hit.
                // If both qualify, visit the nearer one now and push the other.
                // If just one qualifies, visit it without growing the stack.
                // If neither qualifies, fall through to pop a deferred node.
                int first = current + 1, second = node.link;
                float a, b;
                bool left = box_hit(nodes[first], ray, inverse, hit.t, a);
                bool right = box_hit(nodes[second], ray, inverse, hit.t, b);
                if (left && right) {
                    // A nearby hit tightens hit.t and prunes later box tests.
                    if (b < a) { int tmp = first; first = second; second = tmp; }
                    stack[size++] = second;
                    current = first;
                    continue;
                }
                if (left || right) { current = left ? first : second; continue; }
            }
        }
        if (!size) break;
        // Recheck deferred nodes against the latest closest distance.
        current = stack[--size];
    }
    return hit;
}
#include "sampled_grid.cuh"

// Mix neighboring thread indices before seeding the supplied cuRAND wrapper.
// The shifts/xors/multiplications scramble integer bits. This is a seed mixer;
// RandomGPU itself is the random-number generator used for hemisphere sampling.
__device__ uint32_t hash(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16);
}

// Follow one ray through all its bounces. throughput is the accumulated albedo;
// light is the emission/background weighted by that accumulated color.
template<bool Sampled>
__device__ Vec3 trace_path(Ray ray, const RenderSettings& settings,
                          const Primitive* triangles, int count, const Node* nodes,
                          SampleGrid grid, RandomGPU& random) {
    Vec3 throughput(1, 1, 1), light;
    int last = -1;
    // A path starts with all color channels weighted by 1. After surfaces with
    // albedos A and B, later contributions are weighted by A*B component by
    // component. Emission is added at every hit; background is added on a miss.
    for (int depth = 0; ; ++depth) {
        HitRecord record;
        int object, triangle_id;
        if constexpr (Sampled) {
            const SampleClosest hit = sampled_hit(ray, grid, last);
            if (hit.index < 0) {
                light += multiply(throughput, settings.background_light);
                break;
            }
            const SurfaceSample p = grid.samples[hit.index];
            record.t = hit.t; record.position = p.position; record.unit_normal = p.normal;
            object = p.object; triangle_id = p.triangle;
        } else {
            const Closest hit = closest_hit(ray, triangles, count, nodes, last);
            if (hit.index < 0) {
                light += multiply(throughput, settings.background_light);
                break;
            }
            const Primitive p = triangles[hit.index];
            record.t = hit.t;
            record.position = ray.origin + hit.t * ray.direction;
            record.unit_normal = normalize(cross(p.triangle.p1 - p.triangle.p0,
                                                 p.triangle.p2 - p.triangle.p0));
            object = p.object; triangle_id = p.id;
        }
        const Material material = materials[object];
        // A hit still emits at the bounce limit; only further scattering stops.
        // depth=0 is the camera ray's first hit. With max_bounces=1, that hit
        // may scatter once; a second hit at depth=1 emits but does not scatter.
        light += multiply(throughput, Vec3(material.emission.x, material.emission.y, material.emission.z));
        if (depth >= settings.max_bounces) break;
        throughput = multiply(throughput, Vec3(material.albedo.x, material.albedo.y, material.albedo.z));

        if (dot(record.unit_normal, ray.direction) > 0) record.unit_normal = -record.unit_normal;
        ray = cosine_weighted_sampling(record, random);
        // The supplied sampler starts exactly at record.position. Remember the
        // original ID so the next search ignores only the triangle just left.
        last = triangle_id;
    }
    return light;
}

// One thread runs its share of a subpixel's complete paths, without recursion.
// Groups larger than a warp emit one atomic sum per warp. Padded lanes always
// reach every shuffle, even when there are fewer subpixels than launch threads.
template<bool Sampled>
__global__ void paths(RenderSettings settings, const Primitive* triangles, int count,
                      const Node* nodes, SampleGrid grid, Vec3* sums,
                      size_t subpixels, int split) {
    const size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    // Consecutive groups of split threads share a subpixel. The power-of-two
    // split lets a bit mask select each thread's position within its group.
    const size_t subpixel = tid / split;
    const int part = tid & (split - 1);
    // Example: split=4 assigns threads 0..3 to subpixel 0, threads 4..7 to
    // subpixel 1, etc. Within each group, part is 0,1,2,3.
    // With 10 samples, these four threads process:
    //   part 0: samples 0,4,8; part 1: 1,5,9; part 2: 2,6; part 3: 3,7.
    // Thus all 10 samples are evaluated exactly once without changing the count.
    Vec3 sum;
    if (subpixel < subpixels) {
        RandomGPU random(hash(uint32_t(tid) ^ hash(settings.seed)));
        const int width = settings.width * settings.supersampling_factor;
        // Camera rays have no jitter in the reference: reuse the same primary
        // ray for this subpixel, while scattering draws new random directions.
        const Ray primary = make_camera_ray(settings, int(subpixel % width), int(subpixel / width));
        // Strided ownership visits every sample exactly once, including when
        // samples_per_pixel is not divisible by split. Some threads may do none.
        for (size_t sample = part; sample < size_t(settings.samples_per_pixel); sample += split)
            sum += trace_path<Sampled>(primary, settings, triangles, count, nodes, grid, random);
    }
    // Power-of-two subgroups align with warp boundaries. Out-of-range lanes
    // contribute zero but still execute every shuffle under the full warp mask.
    const int width = split < 32 ? split : 32;
    // A CUDA warp contains 32 lanes. Reduce within groups of min(split,32):
    // offset halves each time, passing partial sums from higher lanes to lower
    // lanes until the group's first lane holds their total. A split=4 group
    // uses offsets 2 then 1. A split=64 subpixel uses two 32-lane reductions.
    // Do not return early for padded launch threads: the full-mask shuffles
    // require all 32 lanes to execute them, even when their contribution is zero.
    for (int offset = width / 2; offset; offset /= 2) {
        sum.x += __shfl_down_sync(0xffffffffu, sum.x, offset, width);
        sum.y += __shfl_down_sync(0xffffffffu, sum.y, offset, width);
        sum.z += __shfl_down_sync(0xffffffffu, sum.z, offset, width);
    }
    if (subpixel < subpixels && (threadIdx.x & (width - 1)) == 0) {
        // Each subgroup emits one partial sum. Atomics also combine the separate
        // warps/blocks belonging to a subpixel when split is greater than 32.
        atomicAdd(&sums[subpixel].x, sum.x);
        atomicAdd(&sums[subpixel].y, sum.y);
        atomicAdd(&sums[subpixel].z, sum.z);
    }
}

// Average linear radiance, then tone-map and convert to RGB on the GPU.
__global__ void average(RenderSettings settings, const Vec3* sums, Pixel* colors, size_t pixels) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= pixels) return;
    const int ss = settings.supersampling_factor, width = settings.width * ss;
    // Map the output pixel to the top-left of its ss-by-ss subpixel grid.
    const size_t start = (i / settings.width * ss) * width + (i % settings.width) * ss;
    // Subpixels are stored row by row across the whole enlarged image. For
    // ss=2, each output pixel owns a 2x2 block, spanning two subpixel rows.
    // sums already contains totals over samples; divide once by samples * ss^2.
    Vec3 color;
    for (int y = 0; y < ss; ++y)
        for (int x = 0; x < ss; ++x) color += sums[start + size_t(y) * width + x];
    colors[i] = to_pixel(color / (float(settings.samples_per_pixel) * ss * ss));
}
} // namespace

std::vector<Pixel> render_scene(
    const RenderSettings& settings,
    const std::vector<Object>& objects
) {
    // Scene input is host-owned. Upload once; construction and searches stay
    // on device for either backend. Materials use the existing constant table.
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

    if (objects.size() > 128) throw std::runtime_error("At most 128 materials supported");
    int device; cuda_check(cudaGetDevice(&device));
    cudaDeviceProp prop; cuda_check(cudaGetDeviceProperties(&prop, device));
    // pixels is the final width*height count. Supersampling factor ss creates
    // ss^2 subpixels per pixel, each receiving samples_per_pixel complete paths.
    const size_t pixels = size_t(settings.width) * settings.height;
    const size_t subpixels = pixels * settings.supersampling_factor * settings.supersampling_factor;
    int split = 1;
    // Small images need multiple sample workers per subpixel to occupy the GPU.
    // Aim for several waves of threads per SM; large images already have enough
    // subpixels and naturally use split = 1. This only redistributes samples.
    const size_t target = size_t(prop.multiProcessorCount) * 2048 * 4;
    // multiProcessorCount is the number of GPU streaming multiprocessors (SMs).
    // 2048*4 is a heuristic thread budget per SM, not a change to sample count.
    // Double split until there is enough parallel work or samples run out. A
    // power of two keeps thread groups aligned with the warp reduction above.
    while (subpixels * split < target && split < settings.samples_per_pixel && split < (1 << 24)) split *= 2;
    const int block = 128;
    DeviceBuffer<Primitive> triangles(count);
    DeviceBuffer<Node> nodes(settings.sampled_intersections ? 0 : bvh_node_count(count));
    OwnedSampleGrid grid;
    DeviceBuffer<Vec3> sums(subpixels);
    DeviceBuffer<Pixel> colors(pixels);
    // The sampled backend does not allocate or build BVH nodes. Cubes share
    // samples through integer references, avoiding repeated attribute storage.
    if (count) cuda_check(cudaMemcpy(triangles.data, host_triangles.data(), count * sizeof(Primitive), cudaMemcpyHostToDevice));
    if (settings.sampled_intersections) build_sample_grid(triangles.data, count, settings, grid);
    else build_bvh(triangles.data, count, nodes.data);
    if (!host_materials.empty()) {
        cuda_check(cudaMemcpyToSymbol(materials, host_materials.data(), host_materials.size() * sizeof(Material)));
    }
    // Atomic accumulation requires a zeroed sum for every subpixel.
    cuda_check(cudaMemset(sums.data, 0, subpixels * sizeof(Vec3)));
    const size_t blocks = (subpixels * split + block - 1) / block;
    // Round up because the last block may be partially used. paths() guards its
    // scene work for those extra threads while still including them in shuffles.
    if (settings.sampled_intersections)
        paths<true><<<unsigned(blocks), block>>>(settings, triangles.data, count, nodes.data, grid.view, sums.data, subpixels, split);
    else
        paths<false><<<unsigned(blocks), block>>>(settings, triangles.data, count, nodes.data, grid.view, sums.data, subpixels, split);
    // Both kernels use the default stream, so averaging waits for all path sums.
    average<<<unsigned((pixels + 255) / 256), 256>>>(settings, sums.data, colors.data, pixels);
    cuda_check(cudaGetLastError());
    std::vector<Pixel> result(pixels);
    // Tone mapping and gamma conversion also run on the GPU. Only final RGB
    // bytes cross back to the host; this copy synchronizes and reports failures.
    cuda_check(cudaMemcpy(result.data(), colors.data, pixels * sizeof(Pixel), cudaMemcpyDeviceToHost));
    return result;
}
