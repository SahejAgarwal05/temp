#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "graphics.cuh"
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>


constexpr int MAX_LEAF_SIZE = 2;
constexpr int BLOCK_SIZE = 256;
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
    int triangle_index;
    Vec3 edge0;
    Vec3 edge1;
    Vec3 normal;
    Vec3 highCorner;
    Vec3 lowCorner;
    Vec3 p0;
    __host__ __device__ GPUTriangle() = default;
    __host__ __device__ GPUTriangle(int object_index, int triangle_index, const Triangle& triangle2):
        object_index(object_index), triangle_index(triangle_index) {
            highCorner = Vec3(fmaxf(triangle2.p0.x, fmaxf(triangle2.p1.x, triangle2.p2.x)), fmaxf(triangle2.p0.y, fmaxf(triangle2.p1.y, triangle2.p2.y)), fmaxf(triangle2.p0.z, fmaxf(triangle2.p1.z, triangle2.p2.z)));
            lowCorner = Vec3(fminf(triangle2.p0.x, fminf(triangle2.p1.x, triangle2.p2.x)), fminf(triangle2.p0.y, fminf(triangle2.p1.y, triangle2.p2.y)), fminf(triangle2.p0.z, fminf(triangle2.p1.z, triangle2.p2.z)));
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
        lowCorner = Vec3(fminf(lowCorner.x, point.x), fminf(lowCorner.y, point.y), fminf(lowCorner.z, point.z));
        highCorner = Vec3(fmaxf(highCorner.x, point.x), fmaxf(highCorner.y, point.y), fmaxf(highCorner.z, point.z));
    }
    __host__ __device__ inline void insert(const Triangle& triangle) {
        insert(triangle.p0); insert(triangle.p1); insert(triangle.p2);
    }
    __host__ __device__  inline void insert(const NodeBox& other) {
        insert(other.lowCorner); insert(other.highCorner);
    }
    __device__ void pad() {
        const float scale = 8.0f * 1.1920928955078125e-7f;
        const Vec3 padding(scale * fmaxf(1.f, fmaxf(fabsf(lowCorner.x), fabsf(highCorner.x))), scale * fmaxf(1.f, fmaxf(fabsf(lowCorner.y), fabsf(highCorner.y))), scale * fmaxf(1.f, fmaxf(fabsf(lowCorner.z), fabsf(highCorner.z))));
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

__device__ unsigned long long mortonEncoding(int idx, const GPUTriangle triangle, unsigned long long* mortonCodes, NodeBox fullSpace) {
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

    return (morton << 32) | static_cast<unsigned long long>(idx);
}

__device__ NodeBox calcNodeBox(const GPUTriangle* triangles, const int startIndex, const int endIndex) {
    NodeBox box;
    for (int i = startIndex; i <= endIndex; ++i) {
        box.insert(triangles[i].lowCorner);
        box.insert(triangles[i].highCorner);
    }
    box.pad();
    return box;
}


__global__ void buildBVHNodes(int n_nodes, int n_triangles, LeafNode* nodes) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_nodes) return;
    const int leaf_capacity = n_nodes / 2 + 1;
    const int depth = 31 - __clz(static_cast<unsigned>(idx) + 1u);
    const unsigned level_width = 1u << depth;
    const unsigned position = static_cast<unsigned>(idx) - (level_width - 1u);
    const long long triangles_per_node = static_cast<long long>(leaf_capacity >> depth) * MAX_LEAF_SIZE;
    const long long start = position * triangles_per_node;
    const long long stop = start + triangles_per_node;
    nodes[idx].startIndex = start < n_triangles ? static_cast<int>(start) : n_triangles;
    nodes[idx].endIndex = (stop < n_triangles ? static_cast<int>(stop) : n_triangles) - 1;
}

__global__ void buildBVHLeafBounds(int leaf_capacity, const GPUTriangle* triangles, LeafNode* nodes) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= leaf_capacity) return;
    LeafNode& leaf = nodes[leaf_capacity - 1 + idx];
    NodeBox box;
    if (leaf.startIndex <= leaf.endIndex)
        box = calcNodeBox(triangles, leaf.startIndex, leaf.endIndex);
    leaf.lowCorner = box.lowCorner;
    leaf.highCorner = box.highCorner;
}

__global__ void buildBVHInternalBounds(int first_parent, int parent_count, LeafNode* nodes) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= parent_count) return;
    const int parent_index = first_parent + idx;
    const LeafNode& left = nodes[2 * parent_index + 1];
    const LeafNode& right = nodes[2 * parent_index + 2];
    LeafNode& parent = nodes[parent_index];
    NodeBox box;
    if (left.startIndex <= left.endIndex) {
        box.insert(left.lowCorner);
        box.insert(left.highCorner);
    }
    if (right.startIndex <= right.endIndex) {
        box.insert(right.lowCorner);
        box.insert(right.highCorner);
    }
    parent.lowCorner = box.lowCorner;
    parent.highCorner = box.highCorner;
}



void buildBVH(int n_nodes, int n_triangles, GPUTriangle* gpu_triangles, LeafNode* gpu_nodes) {
    if (n_nodes == 0) return;
    const int leaf_capacity = n_nodes / 2 + 1;
    buildBVHNodes<<<(n_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(n_nodes, n_triangles, gpu_nodes);
    buildBVHLeafBounds<<<(leaf_capacity + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(leaf_capacity, gpu_triangles, gpu_nodes);
    for (int parent_count = leaf_capacity / 2; parent_count > 0; parent_count /= 2) {
        buildBVHInternalBounds<<<(parent_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(parent_count - 1, parent_count, gpu_nodes);
    }
}

__global__ void convertToGPUTriagles(int n_triangles, unsigned long long* morton_codes, const Triangle* triangles, GPUTriangle* gpu_triangles, int* object_indices, NodeBox fullSpace) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_triangles) return;
    gpu_triangles[idx] = GPUTriangle(object_indices[idx], idx, triangles[idx]);
    morton_codes[idx] = mortonEncoding(idx, gpu_triangles[idx], morton_codes, fullSpace);
}

__device__ bool intersectNode(const Ray& ray, const LeafNode& node, float closest, float& entry_distance) {
    if (node.endIndex < node.startIndex) return false;


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
__device__ void testTriangleHit(const Ray& ray, const GPUTriangle& triangle, int previous_triangle, Hit& closest) {
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



__device__ Hit closestHit(const Ray& ray, const GPUTriangle* triangles, const LeafNode* nodes, int n_nodes, int previous_triangle) {
    Hit hit;
    if (n_nodes == 0) return hit;
    constexpr int STACK_CAPACITY = 64;
    int pending_nodes[STACK_CAPACITY];
    int pending_count = 0;
    int node_index = 0;

    while (node_index != -1) {
        const LeafNode& node = nodes[node_index];
        float entry_distance;
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
                        for (int i = 0; i <= nodes[0].endIndex; ++i)
                            testTriangleHit(ray, triangles[i], previous_triangle, hit);
                        return hit;
                    }
                    const bool left_is_nearer = left_entry <= right_entry;
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

__device__ Vec3 tracePath(Ray ray, const RenderSettings& settings,const GPUTriangle* triangles, const LeafNode* nodes, int n_nodes, const ObjectMaterial* materials, RandomGPU& random, const Hit* first_hit = nullptr) {
    Vec3 light, throughput(1.0f, 1.0f, 1.0f), zero(0,0,0);
    int previous_triangle = -1;
    for (int bounce = 0; ; ++bounce) {

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

__global__ void findSubpixelFirstHits(RenderSettings settings, const GPUTriangle* triangles, const LeafNode* nodes, int bvh_node_count, Hit* subpixel_first_hits, size_t subpixel_count, size_t subpixel_grid_width) {
    const size_t subpixel_index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (subpixel_index >= subpixel_count) return;
    const Ray camera_ray = make_camera_ray(settings, int(subpixel_index % subpixel_grid_width), int(subpixel_index / subpixel_grid_width));
    subpixel_first_hits[subpixel_index] = closestHit(camera_ray, triangles, nodes, bvh_node_count, -1);
}

struct ColorGroupIndex {
    size_t colors_per_group;
    __host__ __device__ size_t operator()(size_t color_index) const {
        return color_index / colors_per_group;
    }
};


__global__ void tracePixelSamples(RenderSettings settings, const GPUTriangle* triangles, const LeafNode* nodes, int node_count, const ObjectMaterial* materials, const Hit* first_hits, Vec3* sample_colors, size_t sample_count) {
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
    sample_colors[sample_index] = tracePath(Ray{}, settings, triangles, nodes, node_count, materials, random, &first_hits[subpixel_index]);
}

__global__ void finalizeOutputPixels(RenderSettings settings, const Vec3* pixel_radiance_sums, Pixel* output_pixels, size_t output_pixel_count) {
    const size_t output_pixel_index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_pixel_index >= output_pixel_count) return;
    const int subpixels_per_axis = settings.supersampling_factor;

    output_pixels[output_pixel_index] = to_pixel(pixel_radiance_sums[output_pixel_index] / (float(subpixels_per_axis) * subpixels_per_axis * settings.samples_per_pixel));
}

std::vector<Pixel> render_scene(const RenderSettings& settings, const std::vector<Object>& objects) {


    const int subpixels_per_axis = settings.supersampling_factor;
    const size_t output_pixel_count = size_t(settings.width) * settings.height;
    const size_t subpixel_count = output_pixel_count * subpixels_per_axis * subpixels_per_axis;
    const size_t samples_per_output_pixel = size_t(subpixels_per_axis) * subpixels_per_axis
                                           * settings.samples_per_pixel;
    const size_t sample_count = output_pixel_count * samples_per_output_pixel;

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



    Triangle* gpu_raw_triangles = nullptr;
    GPUTriangle* gpu_triangles = nullptr;
    LeafNode* gpu_nodes = nullptr;
    ObjectMaterial* gpu_materials = nullptr;
    int* gpu_triangle_object_indices = nullptr;
    unsigned long long* mortonCodes = nullptr;
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
        cudaFree(mortonCodes);
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
    cudaMalloc(&mortonCodes, triangle_count * sizeof(unsigned long long));


        cudaMemcpy(gpu_raw_triangles, triangles.data(), triangle_count * sizeof(Triangle), cudaMemcpyHostToDevice);

        cudaMemcpy(gpu_triangle_object_indices, triangle_object_indices.data(), triangle_count * sizeof(int), cudaMemcpyHostToDevice);

        cudaMemcpy(gpu_materials, materials.data(), objects.size() * sizeof(ObjectMaterial), cudaMemcpyHostToDevice);



        convertToGPUTriagles<<<(triangle_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(triangle_count, mortonCodes, gpu_raw_triangles, gpu_triangles, gpu_triangle_object_indices, scene_bounds);




        cudaFree(gpu_raw_triangles);
        gpu_raw_triangles = nullptr;
        cudaFree(gpu_triangle_object_indices);
        gpu_triangle_object_indices = nullptr;



        thrust::sort_by_key(thrust::device, mortonCodes, mortonCodes + triangle_count, gpu_triangles);


        cudaFree(mortonCodes);
        mortonCodes = nullptr;

        buildBVH(bvh_node_count, triangle_count,gpu_triangles, gpu_nodes);

        cudaMalloc(&gpu_output_pixels, output_pixel_count * sizeof(Pixel));
        cudaMalloc(&gpu_pixel_radiance_sums, output_pixel_count * sizeof(Vec3));
        cudaMalloc(&gpu_sample_colors, sample_count * sizeof(Vec3));
        cudaMalloc(&gpu_subpixel_first_hits, subpixel_count * sizeof(Hit));
        const size_t subpixel_grid_width = size_t(settings.width) * settings.supersampling_factor;
        findSubpixelFirstHits<<<unsigned((subpixel_count + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(settings, gpu_triangles, gpu_nodes, bvh_node_count, gpu_subpixel_first_hits, subpixel_count, subpixel_grid_width);
        tracePixelSamples<<<unsigned((sample_count + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(settings, gpu_triangles, gpu_nodes, bvh_node_count, gpu_materials, gpu_subpixel_first_hits, gpu_sample_colors, sample_count);
        const auto sample_indices = thrust::make_counting_iterator<size_t>(0);
        const auto pixel_keys = thrust::make_transform_iterator(sample_indices, ColorGroupIndex{samples_per_output_pixel});
        thrust::reduce_by_key(thrust::device, pixel_keys, pixel_keys + sample_count, gpu_sample_colors, thrust::make_discard_iterator(), gpu_pixel_radiance_sums, thrust::equal_to<size_t>(), thrust::plus<Vec3>());
        cudaFree(gpu_sample_colors);
        gpu_sample_colors = nullptr;
        finalizeOutputPixels<<<unsigned((output_pixel_count + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(settings, gpu_pixel_radiance_sums, gpu_output_pixels, output_pixel_count);
        std::vector<Pixel> output_pixels(output_pixel_count);
        cudaMemcpy(output_pixels.data(), gpu_output_pixels, output_pixel_count * sizeof(Pixel), cudaMemcpyDeviceToHost);
        release();
        return output_pixels;
}
