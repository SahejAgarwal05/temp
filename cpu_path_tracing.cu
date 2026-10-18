#define MAX_LEAF_SIZE = 2;

#include <cstdint>
#include <iostream>


#include "graphics.cuh"
#include "thrust/sort.h"

MAX_LEAF_SIZE = 2;
struct Hit {
    HitRecord record;
    int object_index;
    int triangle_index;

    Hit(): object_index(-1), triangle_index(-1) {}
    bool hit() const { return object_index != -1; }
};



struct ObjectMaterial{
    Vec3 emission;
    Vec3 albedo;
}


struct LeafNode{
    Vec3 highCorner;
    Vec3 lowCorner;
    int rightNodeIndex;
    int n;
    int startIndex = 0;
    int parent_id;
}

struct GPUTriangle{
    Triangle triangle;
    int object_index;
    int triangle_index;
    int padding;
    int node_index;
}

struct NodeBox{

    Vec3 lowCorner = Vec3(INF, INF, INF);
    Vec3 highCorner = Vec3(-INF, -INF, -INF);
    __device__ void insert(const Vec3& point){
        lowCorner.x = fminf(lowCorner.x, point.x);
        lowCorner.y = fminf(lowCorner.y, point.y);
        lowCorner.z = fminf(lowCorner.z, point.z);

        highCorner.x = fmaxf(highCorner.x, point.x);
        highCorner.y = fmaxf(highCorner.y, point.y);
        highCorner.z = fmaxf(highCorner.z, point.z);
    }

    __device__ void insert(const Triangle& triangle){
        insert(triangle.p0);
        insert(triangle.p1);
        insert(triangle.p2);
    }

    __device__ void insert(const NodeBox& other){
        insert(other.lowCorner);
        insert(other.highCorner);
    }
}

template <typename T> GPUArray{
    T* data = nullptr;
    GPUArray(size_t size){
        cudaMalloc(&data, size * sizeof(T));
    }
}


int countNodes(int n){
    int leaves = (n + MAX_LEAF_SIZE - 1) / MAX_LEAF_SIZE;
    return fmax(0, leaves * 2 - 1);
}

morton


__device__ unsigned gridLocation (float value, float low, float high) {
    return high > low ? unsigned(fminf(1023.f, fmaxf(0.f, (value - low) / (high - low) * 1024.f))) : 0;
}

__device__ mortonEncoding(int n; GPUTriangle* triangles, unsigned long long* mortonCodes, NodeBox *fullSpace){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;
    NodeBox temp;
    temp.insert(triangles[idx].triangle);
    NodeBox full = *fullSpace;
    Vec3 center = (temp.lowCorner + temp.highCorner) / 2.0
    unsigned long long tempMorton = 0;
    unsigned long long x = gridLocation(center.x, full.lowCorner.x, full.highCorner.x);
    unsigned long long y = gridLocation(center.y, full.lowCorner.y, full.highCorner.y);
    unsigned long long z = gridLocation(center.z, full.lowCorner.z, full.highCorner.z);
    for (int i = 0; i < 10; ++i) {
        tempMorton |= ((x >> i) & 1) << (3 * i + 2);
        tempMorton |= ((y >> i) & 1) << (3 * i + 1);
        tempMorton |= ((z >> i) & 1) << (3 * i);
    }
    mortonCodes[idx] = (tempMorton << 44) | idx;
}

__device__ void calcDelta(int n_triangles, GPUArray<unsigned long long > morotnCodes, GPUArray<unsingned long long> deltas){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_triangles - 1) return;
    unsigned long long leftCode = morotnCodes[idx];
    unsigned long long rightCode = morotnCodes[idx + 1];
    unsigned long long delta = leftCode ^ rightCode;
    deltas[idx] = delta;
}

// splitNode: read node_index from the triangles
__global__ void splitNode(int n_triangles, const GPUTriangle* triangles,
                          const unsigned long long* deltas,
                          unsigned long long* best) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_triangles - 1) return;
    int node = triangles[i].node_index;
    if (triangles[i + 1].node_index != node) return;
    unsigned long long key = (deltas[i] & ~IDX_MASK) | (unsigned long long)(i + 1);
    atomicMax(&best[node], key);
}

// split_nodes: add the "not active this round" check from last time
//   if (node.n <= 1 || node.rightNodeIndex != -1) return;
//   if (best[idx] == 0) return;
//   int s = (int)(best[idx] & IDX_MASK) - 1;

// reassign: take a raw pointer, not GPUArray
__global__ void reassign(int n_triangles, GPUTriangle* triangles, const LeafNode* nodes) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_triangles) return;
    int node = triangles[i].node_index;
    int right = nodes[node].rightNodeIndex;
    if (right == -1) return;
    int s = nodes[right].startIndex - 1;
    triangles[i].node_index = (i <= s) ? node + 1 : right;
}

// constructNodes: no nodeOf, takes the triangles pointer
void constructNodes(int n_triangles, int n_nodes, NodeBox fullSpace,
                    GPUTriangle* triangles, const unsigned long long* deltas, LeafNode* nodes) {
    GPUArray<unsigned long long> best(n_nodes);
    GPUArray<int> d_changed(1);

    initNodes<<<(n_nodes + 255) / 256, 256>>>(n_nodes, n_triangles, nodes, fullSpace);

    int blocksT = (n_triangles + 255) / 256;
    int blocksN = (n_nodes + 255) / 256;
    int changed = 1;
    while (changed) {
        cudaMemset(best.data, 0, n_nodes * sizeof(unsigned long long));
        cudaMemset(d_changed.data, 0, sizeof(int));
        splitNode<<<blocksT, 256>>>(n_triangles, triangles, deltas, best.data);
        split_nodes<<<blocksN, 256>>>(n_nodes, nodes, best.data, d_changed.data);
        reassign<<<blocksT, 256>>>(n_triangles, triangles, nodes);
        cudaMemcpy(&changed, d_changed.data, sizeof(int), cudaMemcpyDeviceToHost);
    }
}

__global__ void computeBoxes(int n_nodes, const GPUTriangle* tris, LeafNode* nodes, int* visits) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_nodes || nodes[idx].rightNodeIndex != -1) return;   // leaves only

    NodeBox box;
    for (int j = nodes[idx].startIndex; j < nodes[idx].startIndex + nodes[idx].n; ++j)
        box.insert(tris[j].triangle);
    nodes[idx].lowCorner = box.lowCorner;
    nodes[idx].highCorner = box.highCorner;

    int cur = idx;
    while (nodes[cur].parent_id != -1) {
        int p = nodes[cur].parent_id;
        __threadfence();                                  // publish our box
        if (atomicAdd(&visits[p], 1) == 0) return;        // first child: sibling finishes it
        __threadfence();
        NodeBox b;
        b.insert(nodes[p + 1].lowCorner);  b.insert(nodes[p + 1].highCorner);
        int r = nodes[p].rightNodeIndex;
        b.insert(nodes[r].lowCorner);      b.insert(nodes[r].highCorner);
        nodes[p].lowCorner = b.lowCorner;
        nodes[p].highCorner = b.highCorner;
        cur = p;
    }
}

std::vector<Pixel> render_scene(const RenderSettings& settings,const std::vector<Object>& objects) {
    std::vector<GPUTriangle> triangles;
    std::vector<Material> materials;
    int n_triangles = 0;
    NodeBox fullSpace;
    for(int object_index = 0; object_index < objects.size(); ++object_index) {
        const Object& object = objects[object_index];
        materials.push_back(object.material);
        n_triangles += object.triangles.size();
        for (int triangle_index = 0; triangle_index < object.triangles.size(); ++triangle_index) {
            const Triangle& triangle = object.triangles[triangle_index];
            GPUTriangle gpu_triangle;
            gpu_triangle.triangle = triangle;
            gpu_triangle.object_index = object_index;
            gpu_triangle.triangle_index = triangle_index;
            gpu_triangle.id = 0;
            triangles.push_back(gpu_triangle);
            fullSpace.insert(triangle);
        }
    }
    GPUArray<GPUTriangle> gpu_triangles(n_triangles);
    cudaMemcpy(gpu_triangles.data, triangles.data(), n_triangles * sizeof(GPUTriangle), cudaMemcpyHostToDevice);
    GPUArray<Material> gpu_materials(materials.size());
    cudaMemcpy(gpu_materials.data, materials.data(), materials.size() * sizeof(Material), cudaMemcpyHostToDevice);
    GPUArray<Pixel> gpu_pixels(settings.width * settings.height);
    int n_nodes = countNodes(n_triangles);
    GPUArray<LeafNode> gpu_nodes(n_nodes);
    GPUArray<unsigned long long> deltas(n_triangles - 1);
    gpu_nodes[0] = fullSpace;
    GPUArray<unsigned long long> gpu_mortonCodes(n_triangles);
    mortonEncoding<<<(n_triangles + 255) / 256, 256>>>(n_triangles, gpu_triangles.data, gpu_mortonCodes.data, &fullSpace);
    thrust::sort_by_key(thrust::device, gpu_mortonCodes.data, gpu_mortonCodes.data + n_triangles, gpu_triangles.data);
    clacDelta<<<(n_triangles + 255) / 256, 256>>>(n_triangles, gpu_mortonCodes.data, deltas.data);
    constructBVD<<<(n_triangles + 255) / 256, 256>>>(n_triangles, n_nodes, gpu_triangles.data, gpu_mortonCodes.data, &fullSpace, gpu_nodes.data);
    
}
