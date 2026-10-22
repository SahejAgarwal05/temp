#include "graphics.cuh"

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

constexpr int MAX_LEAF_SIZE = 2;
constexpr int BLOCK_SIZE = 256;

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

// Complete binary tree in heap order: children of i are 2*i+1 and 2*i+2.
// Leaves occupy the final level. Ranges are inclusive Morton-sorted positions;
// endIndex < startIndex marks an empty padding node.
struct LeafNode {
    Vec3 highCorner;
    Vec3 lowCorner;
    int startIndex;
    int endIndex;

    __host__ __device__ LeafNode(int first_triangle = 0, int last_triangle = -1)
        : startIndex(first_triangle), endIndex(last_triangle) {}
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

    __host__ __device__ GPUTriangle(int object_index, int triangle_index, const Triangle& triangle)
        : object_index(object_index), triangle_index(triangle_index) {
        highCorner = Vec3(fmaxf(triangle.p0.x, fmaxf(triangle.p1.x, triangle.p2.x)),
                          fmaxf(triangle.p0.y, fmaxf(triangle.p1.y, triangle.p2.y)),
                          fmaxf(triangle.p0.z, fmaxf(triangle.p1.z, triangle.p2.z)));
        lowCorner = Vec3(fminf(triangle.p0.x, fminf(triangle.p1.x, triangle.p2.x)),
                         fminf(triangle.p0.y, fminf(triangle.p1.y, triangle.p2.y)),
                         fminf(triangle.p0.z, fminf(triangle.p1.z, triangle.p2.z)));
        edge0 = triangle.p1 - triangle.p0;
        edge1 = triangle.p2 - triangle.p0;
        p0 = triangle.p0;
        normal = normalize(cross(edge0, edge1));
    }
};

struct NodeBox {
    Vec3 lowCorner = Vec3(INF, INF, INF);
    Vec3 highCorner = Vec3(-INF, -INF, -INF);

    __host__ __device__ NodeBox() = default;

    __host__ __device__ void insert(const Vec3& point) {
        lowCorner = Vec3(fminf(lowCorner.x, point.x), fminf(lowCorner.y, point.y),
                         fminf(lowCorner.z, point.z));
        highCorner = Vec3(fmaxf(highCorner.x, point.x), fmaxf(highCorner.y, point.y),
                          fmaxf(highCorner.z, point.z));
    }

    __host__ __device__ void insert(const Triangle& triangle) {
        insert(triangle.p0);
        insert(triangle.p1);
        insert(triangle.p2);
    }

    __host__ __device__ void insert(const GPUTriangle& triangle) {
        lowCorner = Vec3(fminf(lowCorner.x, triangle.lowCorner.x),
                         fminf(lowCorner.y, triangle.lowCorner.y),
                         fminf(lowCorner.z, triangle.lowCorner.z));
        highCorner = Vec3(fmaxf(highCorner.x, triangle.highCorner.x),
                          fmaxf(highCorner.y, triangle.highCorner.y),
                          fmaxf(highCorner.z, triangle.highCorner.z));
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
    int capacity = 1;
    while (capacity < leaves) capacity *= 2;
    return capacity + (capacity - 1);
}

__device__ inline unsigned gridLocation(float value, float low, float high) {
    return high > low
        ? unsigned(fminf(1023.f, fmaxf(0.f, (value - low) / (high - low) * 1024.f))) : 0;
}

__device__ unsigned long long mortonEncoding(int idx, const GPUTriangle& triangle,
                                            const NodeBox& scene_bounds) {
    const Vec3 center = triangle.lowCorner * 0.5f + triangle.highCorner * 0.5f;
    const unsigned x = gridLocation(center.x, scene_bounds.lowCorner.x, scene_bounds.highCorner.x);
    const unsigned y = gridLocation(center.y, scene_bounds.lowCorner.y, scene_bounds.highCorner.y);
    const unsigned z = gridLocation(center.z, scene_bounds.lowCorner.z, scene_bounds.highCorner.z);
    unsigned long long morton = 0;
    for (int i = 0; i < 10; ++i) {
        morton |= static_cast<unsigned long long>((x >> i) & 1) << (3 * i + 2);
        morton |= static_cast<unsigned long long>((y >> i) & 1) << (3 * i + 1);
        morton |= static_cast<unsigned long long>((z >> i) & 1) << (3 * i);
    }
    // Thirty spatial bits plus a unique 32-bit ID fit in a 64-bit key.
    return (morton << 32) | static_cast<unsigned long long>(idx);
}

struct TriangleBounds {
    __host__ __device__ NodeBox operator()(const GPUTriangle& triangle) const {
        NodeBox box;
        box.insert(triangle);
        return box;
    }
};

struct MergeBounds {
    __host__ __device__ NodeBox operator()(const NodeBox& a, const NodeBox& b) const {
        NodeBox box;
        box.lowCorner = Vec3(fminf(a.lowCorner.x, b.lowCorner.x),
                             fminf(a.lowCorner.y, b.lowCorner.y),
                             fminf(a.lowCorner.z, b.lowCorner.z));
        box.highCorner = Vec3(fmaxf(a.highCorner.x, b.highCorner.x),
                              fmaxf(a.highCorner.y, b.highCorner.y),
                              fmaxf(a.highCorner.z, b.highCorner.z));
        return box;
    }
};

__device__ NodeBox calcNodeBox(const GPUTriangle* triangles, int start_index, int end_index) {
    if (start_index > end_index) return NodeBox{};
    // Each GPU thread reduces its node's range. Explicit seq keeps the Thrust
    // call on that thread; no nested kernel launches or -rdc=true are required.
    NodeBox box = thrust::transform_reduce(
        thrust::seq, triangles + start_index, triangles + end_index + 1,
        TriangleBounds{}, NodeBox{}, MergeBounds{});
    box.pad();
    return box;
}

// Calculate each node's inclusive sorted-triangle range directly from its index.
// At depth d, node i spans capacity/2^d leaves starting at i-(2^d-1).
// Each node independently reduces its own range to compute its bounds.
__global__ void buildBVHNodes(int n_nodes, int n_triangles, const GPUTriangle* triangles,
                              LeafNode* nodes) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_nodes) return;
    const int leaf_capacity = n_nodes / 2 + 1;
    const int depth = 31 - __clz(static_cast<unsigned>(idx) + 1u);
    const unsigned level_width = 1u << depth;
    const unsigned position = static_cast<unsigned>(idx) - (level_width - 1u);
    const long long triangles_per_node =
        static_cast<long long>(leaf_capacity >> depth) * MAX_LEAF_SIZE;
    const long long start = position * triangles_per_node;
    const long long stop = start + triangles_per_node;
    nodes[idx].startIndex = start < n_triangles ? static_cast<int>(start) : n_triangles;
    nodes[idx].endIndex = (stop < n_triangles ? static_cast<int>(stop) : n_triangles) - 1;
    const NodeBox box = calcNodeBox(triangles, nodes[idx].startIndex, nodes[idx].endIndex);
    nodes[idx].lowCorner = box.lowCorner;
    nodes[idx].highCorner = box.highCorner;
}

// Call after Morton sorting. All nodes compute their ranges and bounds in one
// launch, without reading child nodes or waiting for level-by-level merges.
void buildBVH(int n_nodes, int n_triangles, const GPUTriangle* gpu_triangles,
              LeafNode* gpu_nodes) {
    if (n_nodes == 0) return;
    buildBVHNodes<<<(n_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        n_nodes, n_triangles, gpu_triangles, gpu_nodes);
}

__global__ void convertToGPUTriangles(int n_triangles, unsigned long long* morton_codes,
                                     const Triangle* triangles, GPUTriangle* gpu_triangles,
                                     const int* object_indices, NodeBox scene_bounds) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_triangles) return;
    gpu_triangles[idx] = GPUTriangle(object_indices[idx], idx, triangles[idx]);
    morton_codes[idx] = mortonEncoding(idx, gpu_triangles[idx], scene_bounds);
}

// Slab intersection, including rays parallel to a face of the box.
__device__ bool intersectNode(const Ray& ray, const LeafNode& node, float closest, float& entry_distance) {
    if (node.endIndex < node.startIndex) return false;
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
            if (node_index >= n_nodes / 2) {
                testTriangleHit(ray, triangles[node.startIndex], previous_triangle, hit);
                if (node.endIndex > node.startIndex)
                    testTriangleHit(ray, triangles[node.startIndex + 1], previous_triangle, hit);
            } else {
                const int left_child = 2 * node_index + 1;
                const int right_child = left_child + 1;
                float left_entry, right_entry;
                const bool hit_left = intersectNode(ray, nodes[left_child], hit.record.t, left_entry);
                const bool hit_right = intersectNode(ray, nodes[right_child], hit.record.t, right_entry);
                if (hit_left && hit_right) {
                    if (pending_count == STACK_CAPACITY) {
                        // Keep correctness even for a tree deeper than the local stack.
                        for (int i = 0; i <= nodes[0].endIndex; ++i)
                            testTriangleHit(ray, triangles[i], previous_triangle, hit);
                        return hit;
                    }
                    const bool left_is_nearer = left_entry <= right_entry;
                    // Save the farther child and immediately descend into the nearer one.
                    pending_nodes[pending_count++] = left_is_nearer ? right_child : left_child;
                    node_index = left_is_nearer ? left_child : right_child;
                    continue;
                } else if (hit_left || hit_right) {
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

struct ColorGroupIndex {
    size_t colors_per_group;
    __host__ __device__ size_t operator()(size_t color_index) const {
        return color_index / colors_per_group;
    }
};

// One thread traces one branch, with samples stored together by output pixel.
__global__ void tracePixelSamples(RenderSettings settings, const GPUTriangle* triangles,
                                  const LeafNode* nodes, int node_count,
                                  const ObjectMaterial* materials, const Hit* first_hits,
                                  Vec3* sample_colors, size_t sample_count) {
    const size_t sample_index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (sample_index >= sample_count) return;
    const size_t ss = settings.supersampling_factor;
    const size_t grouped_subpixel = sample_index / settings.samples_per_pixel;
    const size_t pixel_index = grouped_subpixel / (ss * ss);
    const size_t local_subpixel = grouped_subpixel % (ss * ss);
    const size_t subpixel_x = (pixel_index % settings.width) * ss + local_subpixel % ss;
    const size_t subpixel_y = (pixel_index / settings.width) * ss + local_subpixel / ss;
    const size_t subpixel_index = subpixel_y * size_t(settings.width) * ss + subpixel_x;
    const size_t branch_index = sample_index % settings.samples_per_pixel;
    RandomGPU random(subpixel_index * settings.samples_per_pixel + branch_index);
    sample_colors[sample_index] = tracePath(Ray{}, settings, triangles, nodes, node_count,
                                           materials, random, &first_hits[subpixel_index]);
}

__global__ void finalizeOutputPixels(RenderSettings settings, const Vec3* pixel_radiance_sums,
                                 Pixel* output_pixels, size_t output_pixel_count) {
    const size_t output_pixel_index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_pixel_index >= output_pixel_count) return;
    const int subpixels_per_axis = settings.supersampling_factor;
    // Average linear colors before tone mapping and gamma correction.
    output_pixels[output_pixel_index] = to_pixel(pixel_radiance_sums[output_pixel_index] /
        (float(subpixels_per_axis) * subpixels_per_axis * settings.samples_per_pixel));
}

std::vector<Pixel> render_scene(const RenderSettings& settings, const std::vector<Object>& objects) {
    // Each subpixel has one camera ray and one cached first hit.
    // samples_per_pixel is the existing setting for path samples per subpixel.
    const int subpixels_per_axis = settings.supersampling_factor;
    const size_t output_pixel_count = size_t(settings.width) * settings.height;
    const size_t subpixel_count = output_pixel_count * subpixels_per_axis * subpixels_per_axis;
    const size_t samples_per_output_pixel = size_t(subpixels_per_axis) * subpixels_per_axis
                                           * settings.samples_per_pixel;
    const size_t sample_count = output_pixel_count * samples_per_output_pixel;
    // Collect triangles and materials on the CPU.
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

    const int triangle_count = static_cast<int>(triangles.size());
    const int bvh_node_count = countNodes(triangle_count);
    // Allocate GPU scene data.
    Triangle* gpu_raw_triangles = nullptr;
    GPUTriangle* gpu_triangles = nullptr;
    LeafNode* gpu_nodes = nullptr;
    ObjectMaterial* gpu_materials = nullptr;
    int* gpu_triangle_object_indices = nullptr;
    unsigned long long* morton_codes = nullptr;
    Pixel* gpu_output_pixels = nullptr;
    Vec3* gpu_pixel_radiance_sums = nullptr;
    Vec3* gpu_sample_colors = nullptr;
    Hit* gpu_subpixel_first_hits = nullptr;
    auto release = [&]() {
        cudaFree(gpu_raw_triangles);
        cudaFree(gpu_triangle_object_indices);
        cudaFree(gpu_triangles);
        cudaFree(gpu_materials);
        cudaFree(gpu_nodes);
        cudaFree(morton_codes);
        cudaFree(gpu_output_pixels);
        cudaFree(gpu_pixel_radiance_sums);
        cudaFree(gpu_sample_colors);
        cudaFree(gpu_subpixel_first_hits);
    };
    cudaMalloc(&gpu_raw_triangles, triangle_count * sizeof(Triangle));
    cudaMalloc(&gpu_triangle_object_indices, triangle_count * sizeof(int));
    cudaMalloc(&gpu_triangles, triangle_count * sizeof(GPUTriangle));
    cudaMalloc(&gpu_materials, objects.size() * sizeof(ObjectMaterial));
    cudaMalloc(&gpu_nodes, bvh_node_count * sizeof(LeafNode));
    cudaMalloc(&morton_codes, triangle_count * sizeof(unsigned long long));
    cudaMemcpy(gpu_raw_triangles, triangles.data(), triangle_count * sizeof(Triangle),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_triangle_object_indices, triangle_object_indices.data(),
               triangle_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_materials, materials.data(), objects.size() * sizeof(ObjectMaterial),
               cudaMemcpyHostToDevice);

    // Convert triangles and compute Morton keys.
    if (triangle_count > 0) {
        convertToGPUTriangles<<<(triangle_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            triangle_count, morton_codes, gpu_raw_triangles, gpu_triangles,
            gpu_triangle_object_indices, scene_bounds);
    }

    // Free temporary input arrays.
    cudaFree(gpu_raw_triangles);
    gpu_raw_triangles = nullptr;
    cudaFree(gpu_triangle_object_indices);
    gpu_triangle_object_indices = nullptr;

    // Sort triangles directly by Morton keys.
    if (triangle_count > 0) {
        thrust::sort_by_key(thrust::device, morton_codes, morton_codes + triangle_count,
                            gpu_triangles);
    }
    cudaFree(morton_codes);
    morton_codes = nullptr;

    // Compute every node's range and Thrust-reduced bounds independently.
    buildBVH(bvh_node_count, triangle_count, gpu_triangles, gpu_nodes);

    // Trace every branch, sum by output pixel, and average once.
    cudaMalloc(&gpu_output_pixels, output_pixel_count * sizeof(Pixel));
    cudaMalloc(&gpu_pixel_radiance_sums, output_pixel_count * sizeof(Vec3));
    cudaMalloc(&gpu_sample_colors, sample_count * sizeof(Vec3));
    cudaMalloc(&gpu_subpixel_first_hits, subpixel_count * sizeof(Hit));
    const size_t subpixel_grid_width = size_t(settings.width) * settings.supersampling_factor;
    findSubpixelFirstHits<<<unsigned((subpixel_count + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(
        settings, gpu_triangles, gpu_nodes, bvh_node_count, gpu_subpixel_first_hits, subpixel_count, subpixel_grid_width);
    tracePixelSamples<<<unsigned((sample_count + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(
        settings, gpu_triangles, gpu_nodes, bvh_node_count, gpu_materials,
        gpu_subpixel_first_hits, gpu_sample_colors, sample_count);
    const auto sample_indices = thrust::make_counting_iterator<size_t>(0);
    const auto pixel_keys = thrust::make_transform_iterator(
        sample_indices, ColorGroupIndex{samples_per_output_pixel});
    thrust::reduce_by_key(thrust::device, pixel_keys, pixel_keys + sample_count,
                          gpu_sample_colors, thrust::make_discard_iterator(), gpu_pixel_radiance_sums,
                          thrust::equal_to<size_t>(), thrust::plus<Vec3>());
    cudaFree(gpu_sample_colors);
    gpu_sample_colors = nullptr;
    finalizeOutputPixels<<<unsigned((output_pixel_count + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(
        settings, gpu_pixel_radiance_sums, gpu_output_pixels, output_pixel_count);
    std::vector<Pixel> output_pixels(output_pixel_count);
    cudaMemcpy(output_pixels.data(), gpu_output_pixels, output_pixel_count * sizeof(Pixel), cudaMemcpyDeviceToHost);
    release();
    return output_pixels;
}
