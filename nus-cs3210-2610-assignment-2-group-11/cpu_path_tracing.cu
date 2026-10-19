#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "graphics.cuh"
#include <cuda/atomic>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <cstdint.h>


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
    GPUTriangle(int object_index, int triangle_index, const Triangle& triangle):
        triangle(triangle), object_index(object_index), triangle_index(triangle_index) {
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


__host__ __device__ inline int countNodes(int n) {
    if (n == 0) return 0;
    const int leaves = n / MAX_LEAF_SIZE + (n % MAX_LEAF_SIZE != 0);
    return leaves + (leaves - 1);
}

__device__ inline unsigned gridLocation(float value, float low, float high) {
    return high > low
        ? unsigned(fminf(1023.f, fmaxf(0.f, (value - low) / (high - low) * 1024.f))) : 0;
}

__global__ void mortonEncoding(int n, const GPUTriangle* triangles,
                               unsigned long* mortonCodes, NodeBox fullSpace) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;
    NodeBox box;
    box.insert(triangles[idx].triangle);
    const Vec3 center = box.lowCorner * 0.5f + box.highCorner * 0.5f;
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
    mortonCodes[idx] = (morton << 32) | static_cast<unsigned long>(idx);
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

//funciton to find splt for a single node, given the deltas and the triangles
__global__ void splitNode(int n_triangles, int n_nodes, const GPUTriangle* triangles, const unsigned long* deltas, unsigned long* best, int node_index, LeafNode* nodes) {
    const int i = (threadIdx.x + blockIdx.x * blockDim.x) + nodes[node_index].startIndex + 1;
    if (nodes[node_index].n <= MAX_LEAF_SIZE || i > nodes[node_index].startIndex + nodes[node_index].n - 2){
        
    } else {
        atomicMax(best, deltas[i]);
    }
    __syncthreads();
    if (nodes[node_index].n <= MAX_LEAF_SIZE || i > nodes[node_index].startIndex + nodes[node_index].n - 2){
    } else {
        if (deltas[i] == *best) {
            atomicMin(best, static_cast<unsigned long>(i));
        }
    }
    __syncthreads();
    if(threadIdx.x == 0 && blockIdx.x == 0){
        const unsigned long key = *best;
        if (key == 0) return; 
        const int start = nodes[node_index].startIndex; 
        const int n = nodes[node_index].n;                                    // no valid split
        const int split = static_cast<int>(IDX_MASK - (key & IDX_MASK));
        const int left_n = split + 1 - start;
        const int left = node_index + 1;
        const int right = left + countNodes(left_n);
        nodes[node_index].rightNodeIndex = right;
        nodes[left].rightNodeIndex = -1;
        nodes[left].n = left_n;
        nodes[left].startIndex = start;
        nodes[left].parent_id = node_index;
        nodes[right].rightNodeIndex = -1;
        nodes[right].n = n - left_n;
        nodes[right].startIndex = start + left_n;
        nodes[right].parent_id = node_index;
    }
}


__global__ void buildBVHSubtree(
    int n_triangles,
    int n_nodes,
    GPUTriangle* triangles,
    const unsigned long* deltas,
    LeafNode* nodes,
    unsigned long* best,
    int first_node,
    int second_node)
{
    const int index = blockIdx.x == 0 ? first_node : second_node;
    if (index < 0 || index >= n_nodes) return;
    const int n = nodes[index].n;

    if (n <= MAX_LEAF_SIZE) return;

    const int candidates = (n - 1) / MAX_LEAF_SIZE;
    const int blocks = (candidates + BLOCK_SIZE - 1) / BLOCK_SIZE;

    splitNode<<<blocks, BLOCK_SIZE>>>(
        n_triangles,
        n_nodes,
        triangles,
        deltas,
        best + index,
        index,
        nodes);

    buildBVHSubtree<<<2, 1>>>(
        n_triangles,
        n_nodes,
        triangles,
        deltas,
        nodes,
        best,
        index + 1,
        nodes[index].rightNodeIndex);
}

__global__ void calcNodeBox(int n_triangles, GPUTriangle* triangles, int start, int end, Vec3* lowCorner, Vec3* highCorner, int node_index, bool write_idx){ 
    int idx = threadIdx.x + blockIdx.x * blockDim.x + start;
    if (idx < end){
        atomicMin(&lowCorner->x, triangles[idx].lowCorner.x);
        atomicMin(&lowCorner->y, triangles[idx].lowCorner.y);
        atomicMin(&lowCorner->z, triangles[idx].lowCorner.z);
        atomicMax(&highCorner->x, triangles[idx].highCorner.x);
        atomicMax(&highCorner->y, triangles[idx].highCorner.y);
        atomicMax(&highCorner->z, triangles[idx].highCorner.z);
        if (write_idx) {
            triangles[idx].node_index = node_index;
        }
    }
}


__global__ void buildBVHBounds(int n_nodes, LeafNode* nodes,GPUTriangle* triangles,){
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n_nodes) return;
    const int start = nodes[index].startIndex;
    const int end = start + nodes[index].n;
    const bool is_leaf = nodes[index].rightNodeIndex == -1;
    calcNodeBox<<<(end - start + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(n_triangles, triangles, start, end, &nodes[index].lowCorner, &nodes[index].highCorner, index, is_leaf);
}


// Call after your existing initNodes, Morton sorting, and calcDelta.
void buildBVH(int n_nodes,int n_triangles,GPUTriangle* gpu_triangles,LeafNode* gpu_nodes,const unsigned long* deltas){
    unsigned long* best;
    cudaMalloc(&best, n_nodes * sizeof(unsigned long));
    cudaMemset(best, 0, n_nodes * sizeof(unsigned long));
    buildBVHSubtree<<<1, 1>>>(n_triangles,n_nodes,gpu_triangles,deltas,gpu_nodes,best,0,-1);
    checkCuda(cudaDeviceSynchronize());
    buildBVHBounds<<<(n_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE,BLOCK_SIZE>>>(n_nodes,gpu_nodes,gpu_triangles);
    checkCuda(cudaDeviceSynchronize());
    cudaFree(best);
}

__device__ void convertToGPUTriagles(int n_triangles, const Triangle* triangles, GPUTriangle* gpu_triangles, int* object_indices) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_triangles) return;
    gpu_triangles[idx] = GPUTriangle(object_indices[idx], idx, triangles[idx]);
}
std::vector<Pixel> render_scene(const RenderSettings& settings, const std::vector<Object>& objects) {
    // 1. Collect triangles and materials on CPU
std::vector<Triangle> triangles;
std::vector<int> triangle_object_indices;
std::vector<ObjectMaterial> materials(objects.size());

for (int object_index = 0; object_index < objects.size(); ++object_index) {
    const Object& object = objects[object_index];

    materials[object_index] = ObjectMaterial{
        object.emission,
        object.albedo
    };

    for (const Triangle& triangle : object.triangles) {
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


// 3. Allocate GPU memory
cudaMalloc(&gpu_raw_triangles, n_triangles * sizeof(Triangle));
cudaMalloc(&gpu_triangle_object_indices, n_triangles * sizeof(int));
cudaMalloc(&gpu_triangles, n_triangles * sizeof(GPUTriangle));
cudaMalloc(&gpu_materials, objects.size() * sizeof(ObjectMaterial));
cudaMalloc(&gpu_nodes, n_nodes * sizeof(LeafNode));
cudaMalloc(&mortonCodes, n_triangles * sizeof(unsigned long));
cudaMalloc(&deltas, n_triangles * sizeof(unsigned long));


// 4. Copy CPU data to GPU
checkCuda(cudaMemcpy(
    gpu_raw_triangles,
    triangles.data(),
    n_triangles * sizeof(Triangle),
    cudaMemcpyHostToDevice
));

checkCuda(cudaMemcpy(
    gpu_triangle_object_indices,
    triangle_object_indices.data(),
    n_triangles * sizeof(int),
    cudaMemcpyHostToDevice
));

checkCuda(cudaMemcpy(
    gpu_materials,
    materials.data(),
    objects.size() * sizeof(ObjectMaterial),
    cudaMemcpyHostToDevice
));


// 5. Convert triangles into GPU triangle structures
convertToGPUTriagles<<<
    (n_triangles + BLOCK_SIZE - 1) / BLOCK_SIZE,
    BLOCK_SIZE
>>>(
    n_triangles,
    gpu_raw_triangles,
    gpu_triangles,
    gpu_triangle_object_indices
);


// 6. Free temporary GPU arrays
cudaFree(gpu_raw_triangles);
cudaFree(gpu_triangle_object_indices);


// 7. Initialize BVH nodes
initNodes<<<
    (n_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE,
    BLOCK_SIZE
>>>(
    n_nodes,
    n_triangles,
    gpu_nodes
);


// 8. Generate Morton codes
// Call your Morton-code generation kernel here.
// It must populate mortonCodes before sorting.


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
}
