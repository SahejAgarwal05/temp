# Understanding the GPU BVH renderer

This guide explains the implementation in [path_tracing.cu](path_tracing.cu), including its CUDA syntax, data structures, tree construction, ray traversal, and sampling. The geometry and camera helpers come from [geometry.cuh](geometry.cuh) and [graphics.cuh](graphics.cuh).

## 1. What the renderer does

A ray starts at the camera and travels into the scene. If it hits a triangle, the renderer collects that surface's emitted light and may bounce the ray in a random direction. Surface colors weight the light collected on later bounces. Repeating this with many samples produces an image.

Testing every triangle at every bounce is expensive. A **bounding volume hierarchy**, or **BVH**, groups triangles inside a tree of bounding boxes. If a ray misses a box, it can skip every triangle under that box.

```text
CPU: flatten objects into triangle and material arrays
  |
GPU: find scene bounds -> calculate spatial keys -> sort triangles
  |
GPU: build tree links and leaf boxes -> merge boxes up to the root
  |
GPU: trace samples, searching the BVH at each bounce
  |
GPU: average samples and subpixels into output colors
  |
CPU: copy colors back -> tone map -> gamma correct -> return pixels
```

The tree is constructed once per `render_scene()` call and reused for all rays in that render.

## 2. CUDA terms and syntax

| Code | Meaning here |
| --- | --- |
| `__host__` | Function executes on the CPU. |
| `__device__` | Function executes on the GPU. |
| `__host__ __device__` | Function can be called on either processor. |
| `__global__` | GPU kernel launched by the CPU. |
| `kernel<<<blocks, threads>>>(...)` | Launch that many thread blocks, with that many threads per block. |
| `blockIdx.x` | Current block's index. |
| `threadIdx.x` | Current thread's index within its block. |
| `blockDim.x` | Number of threads per block. |
| `__shared__` | GPU memory shared by threads within one block. |
| `__syncthreads()` | Wait for all threads in this block and make their shared-memory writes visible to each other. |
| `__constant__` | GPU constant memory populated from the CPU and read by GPU code. |
| `atomicAdd(address, value)` | Add without losing contributions when threads update the same address. |

A typical global thread index is:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

With 256 threads per block, block 0 covers indices 0–255, block 1 covers 256–511, and so on. Blocks are scheduled onto the GPU's streaming multiprocessors, or SMs. Launching many blocks does not mean all blocks execute simultaneously.

The expression `(count + block - 1) / block` rounds division upward. For 1,000 items and 256 threads per block, it launches four blocks. The final 24 threads have no item, so a bounds check makes them stop or contribute zero.

## 3. Constants

| Constant | Purpose |
| --- | --- |
| `kStack = 64` | Maximum number of pending node indices stored by one ray's BVH search. This is a per-thread array, not a shared block-wide stack. |
| `kBuildBlock = 256` | Threads per block for the tree-building kernels. The scene-bounds reduction also uses shared arrays of this size. |
| `kLeafSize = 2` | Maximum triangles grouped into one leaf. The last leaf may contain only one. |

The `k` prefix is just a naming convention for constants. `constexpr` makes the values compile-time constants.

The traversal stack size follows from this tree's key construction: each split consumes a differing bit of a unique key containing 30 spatial bits and 32 ID bits. The depth is bounded by those 62 bits, so 64 stack slots accommodate the traversal. This bound depends on preserving that construction.

For `N` triangles:

```text
leaves = ceil(N / 2)
nodes  = 2 * leaves - 1, when leaves > 0
```

A binary tree with `L` leaves and two children per internal node has `L - 1` internal nodes. Empty scenes need zero nodes.

## 4. Data structures and memory

### `Material`

```cpp
struct Material { float4 albedo, emission; };
```

`albedo` is the RGB fraction of light reflected by a surface. `emission` is its own emitted RGB light. Each `float4` has four floats; only `x`, `y`, and `z` are used. The fourth component is unused and helps give a predictable aligned layout.

`materials[128]` is a constant-memory table. Each triangle stores an object index used to select its material. This implementation assumes no more than 128 objects.

### `Primitive`

```cpp
struct alignas(16) Primitive {
    Triangle triangle;
    int object, id, padding;
};
```

- `triangle`: three 3D vertices, occupying 36 bytes.
- `object`: material-table index.
- `id`: original triangle index before spatial sorting.
- `padding`: unused integer completing the explicit 48-byte record.

The original ID survives sorting. It lets the next bounce skip the triangle just hit and makes equal-distance hits follow original scene order.

### `Node`

```cpp
struct alignas(16) Node {
    Vec3 lo;
    int link;
    Vec3 hi;
    int count;
};
```

`lo` and `hi` are the minimum and maximum corners of an axis-aligned bounding box: a box whose faces follow the x, y, and z axes.

| Node kind | `count` | Meaning of `link` |
| --- | --- | --- |
| Leaf | Positive | First triangle index; test `count` consecutive triangles. |
| Internal | Zero | Right child's node index; left child is at this node's index + 1. |

For example, a leaf with `link = 8, count = 2` contains triangles 8 and 9 in the **sorted** triangle array.

The tree is stored in preorder: parent, entire left subtree, entire right subtree. This allows the left-child index to be implicit.

### What `alignas(16)` means

It requires a struct's starting address to be divisible by 16. It does **not** mean the struct is 16 bytes long or that every field is 16-byte aligned.

`Node` is 32 bytes: two 12-byte vectors and two 4-byte integers. `Primitive` is 48 bytes. Both sizes are multiples of 16, so successive elements in an array preserve the required alignment.

Predictable alignment can help the compiler generate suitable memory accesses. It does not itself guarantee a speedup. The `static_assert` checks the expected sizes at compile time.

### `Box`

`Box` is a temporary geometry helper with `lo` and `hi`, but no tree links. It starts with inverted bounds: minimum = positive infinity and maximum = negative infinity. Adding the first real point establishes actual bounds.

### `Closest`

```cpp
struct Closest {
    float t = INF;
    int index = -1, id = 2147483647;
};
```

This is one ray's best hit so far. `t` is distance along the ray, `index` selects the sorted triangle, and `id` is its original ID. `index = -1` means no hit. The large initial ID lets a real ID win a distance tie.

### `DeviceBuffer<T>`

This CPU-side owner allocates an array of `T` in GPU memory. Its `data` pointer is passed to kernels; the CPU must not directly dereference it.

- Constructor: `cudaMalloc` allocates `n * sizeof(T)` bytes.
- Destructor: `cudaFree` releases the allocation when the owner leaves scope.
- Deleted copy constructor and assignment: prevent two owners from freeing the same allocation.
- `explicit`: prevents accidental implicit conversion from a number into a buffer.

This pattern is called RAII: resource lifetime follows the owning object's lifetime. The current wrapper does not check CUDA error return values.

## 5. Building the BVH

### `Box::add(point)`, `Box::add(triangle)`, `Box::add(node)`

These overloads expand the box to include a point, all three triangle vertices, or another node's box. Every coordinate updates its minimum with `fminf` and maximum with `fmaxf`.

### `Box::pad()`

Slightly enlarges a leaf box using a margin scaled to coordinate magnitude. This helps avoid missing flat triangles or shared edges because floating-point calculations round slightly differently. Parent boxes inherit this margin when leaf boxes are merged.

### `bvh_node_count(count)`

Calculates the required number of nodes from the number of triangle groups. Returns zero for an empty scene.

### `scene_bounds(triangles, count, nodes)`

One GPU block computes the box enclosing the entire scene:

1. Each thread scans a strided subset of triangles: thread `t` handles `t`, `t + 256`, `t + 512`, etc.
2. Each thread writes its local bounds into shared memory.
3. A reduction combines pairs of bounds, halving the active thread count each round.
4. Thread 0 writes the final bounds into `nodes[0]` temporarily.

The reduction uses offsets 128, 64, 32, ..., 1. Every thread reaches each barrier. A thread that scanned no triangles contributes inverted bounds, which do not enlarge a valid box.

### `grid_position(value, low, high)`

Maps one coordinate into a 10-bit integer grid, from 0 to 1023. It normalizes relative to the scene bounds, multiplies by 1024, clamps, and converts to an unsigned integer. If the axis has no extent, it returns zero to avoid division by zero.

### `spatial_keys(...)`

One GPU thread creates a key for each triangle:

1. Find the center of the triangle's bounding box. This is not necessarily the triangle's centroid.
2. Map x, y, and z into 10-bit grid coordinates.
3. Interleave their bits into a 30-bit **Morton code**.
4. Append the original 32-bit triangle ID.

For a three-bit illustration:

```text
x = x2 x1 x0
y = y2 y1 y0
z = z2 z1 z0

Morton code = x2 y2 z2 x1 y1 z1 x0 y0 z0
```

Sorting these codes tends to put spatially nearby triangles near each other in the array. It is an approximation, not a guarantee that consecutive triangles are the closest pair.

```cpp
keys[i] = (static_cast<unsigned long long>(morton) << 32)
          | unsigned(triangles[i].id);
```

The cast makes the value 64-bit before shifting. `<< 32` moves the spatial code into the upper half, and `|` puts the ID into the lower half. IDs make keys unique even when triangle centers coincide.

### `__clzll`: what it actually measures

`__clzll(x)` counts the **leading zero bits** in a 64-bit integer: start at the most significant bit on the left and count zeros until the first `1`. `clz` means "count leading zeros"; `ll` refers to `long long`.

It does not count all zero bits or trailing zeros.

The tree builder uses it with XOR:

```cpp
__clzll(key_a ^ key_b)
```

XOR, written `^`, produces zero when two corresponding bits match and one when they differ. Therefore, the leading zeros of the XOR are exactly the **matching prefix bits of the original keys**.

An eight-bit illustration:

```text
key A:   11001000
key B:   11011100
XOR:     00010100
         ^^^
         3 leading zeros -> the keys share the first 3 bits, "110"
```

The real function works on 64 bits. If these eight-bit values were stored as positive 64-bit integers with all higher bits zero, it would report 59: the 56 additional leading zeros plus these 3. The examples omit that common padding to make the split easier to see.

Why is a matching prefix useful? Morton bits encode position from coarse to fine. Sharing more leading spatial bits means the keys occupy the same progressively smaller grid region. If spatial codes are identical, the appended ID bits still allow the builder to split the keys.

### `split_leaves(keys, first, last)`: using `__clzll` to find the split

This function returns the **last leaf assigned to the left child** of a subtree covering leaves `first..last`, inclusive.

Each leaf contains up to two sorted triangles. The leaf's representative key is its first triangle's key, accessed as `keys[leaf * kLeafSize]`.

First, compare the two endpoint keys:

```cpp
const auto start = keys[first * kLeafSize];
const int prefix = __clzll(start ^ keys[last * kLeafSize]);
```

`prefix` is the number of bits shared by the range's first and last keys. The next bit is where they differ. That bit determines this subtree's two children.

Consider four sorted representative keys, shown with eight bits:

```text
leaf 0: 00100000
leaf 1: 00110000
leaf 2: 01000000
leaf 3: 01100000
```

The first and last share only the first bit, `0`:

```text
00100000 XOR 01100000 = 01000000
                       ^
                       1 leading zero
```

The next bit separates them into:

```text
left child:  00...... -> leaves 0 and 1
right child: 01...... -> leaves 2 and 3
```

The binary search asks whether a middle key shares **more** prefix bits with the first key than the two endpoints share:

```cpp
if (__clzll(start ^ keys[middle * kLeafSize]) > prefix)
    left = middle;
else
    right = middle;
```

- More shared bits: the middle key agrees with the first key at the splitting bit, so it belongs in the left child.
- Same prefix length: the middle key differs at that bit, so it belongs in the right child.

For the example, the first check uses leaf 1. Its XOR with leaf 0 is `00010000`, giving 3 leading zeros. Since `3 > 1`, leaf 1 is on the left. The next check uses leaf 2; its XOR gives only 1 leading zero, so leaf 2 is on the right. The search stops with `left = 1` and returns 1.

Because the keys are sorted, the left-child group is contiguous. Binary search finds its boundary without scanning every leaf. Appended IDs keep distinct representative keys unique, so the endpoint XOR is nonzero. The function is called only when `first < last`.

### `build_leaves(...)`

One thread handles each leaf. It walks from the root through key-based splits to calculate where that leaf belongs in the flat node array.

At each internal node, only the first leaf in that range writes its child link and parent index. This gives each internal node one writer. Once a thread reaches its leaf, it writes the triangle range, padded bounding box, and parent index.

The right-child index is:

```cpp
right = current + 2 * (split - first + 1);
```

If the left child has `L = split - first + 1` leaves, its subtree has `2*L - 1` nodes. Those start at `current + 1`, so the next free position, the right-child root, is `current + 2*L`.

This calculation supports an uneven tree: the split follows keys, not necessarily the midpoint of the leaf count.

### `merge_bounds(...)`

Starts work at leaves and combines child boxes upward. Threads launched at internal-node indices immediately return.

At each parent, a counter records arriving children:

```cpp
cuda::atomic_ref<int, cuda::thread_scope_device> children(finished[parent]);
if (children.fetch_add(1, cuda::memory_order_acq_rel) == 0) return;
```

`fetch_add` atomically increments the counter and returns its **previous** value:

- First child sees 0 and stops.
- Second child sees 1, merges the two completed child boxes, and continues upward.

The acquire/release ordering publishes a child's box writes before the sibling's thread reads them. This works between blocks; a block-local `__syncthreads()` would not be enough. `atomic_ref` performs atomic operations on the existing counter in `finished`.

The root has parent `-1`, which ends the upward walk.

### `build_bvh(...)`

This CPU function allocates temporary GPU arrays for keys, parents, and completion counters, then coordinates the stages:

```text
scene_bounds -> spatial_keys -> thrust::sort_by_key -> build_leaves -> merge_bounds
```

`thrust::sort_by_key` sorts keys and moves their triangle records together. `device_pointer_cast` wraps raw GPU pointers for Thrust; it does not copy the arrays. `thrust::device` selects GPU execution.

The stages use the default stream, which preserves their ordering. This does not mean every individual kernel launch waits for completion on the CPU.

Temporary build arrays are freed when their owners leave scope. Triangles and nodes remain allocated by `render_scene()` for ray tracing. Empty scenes skip the build entirely.

## 6. Searching the tree

### `box_hit(node, ray, inverse, best, entry)`

A ray has the form `origin + t * direction`. For each axis, the box permits an interval of `t` values. A hit requires those three intervals to overlap before the best known triangle distance.

```text
x permits t = [2, 6]
y permits t = [3, 8]
z permits t = [1, 4]
overlap     = [3, 4] -> the ray enters at t = 3
```

`near` starts at zero and `far` starts at `best`. An empty overlap means a miss. The output reference `entry` receives the entry distance when the function returns true; it is used to order children.

`inverse` holds reciprocal ray directions and is reused across box tests. A zero direction component is handled explicitly to avoid `0 * infinity`. If a tiny nonzero component overflows its reciprocal, the code uses division for that axis.

Equal distances remain eligible: a box is rejected when `near > far`, allowing equal-distance triangle tie-breaking later.

### `candidate(ray, triangles, i, last, hit)`

Tests one triangle using `intersect_triangle()` from `geometry.cuh`. It skips the original triangle ID stored in `last`.

A successful intersection replaces `hit` if its distance is smaller. At equal distance, the smaller original ID wins. This matches scanning triangles in their original order and updating only for a strictly closer hit.

`Closest& hit` is a reference, so this function updates the caller's record directly.

### `closest_hit(...)`

Searches the BVH with a per-ray stack:

1. Push root index 0.
2. Pop a node and test its box against the current best distance.
3. For a leaf, test its triangles.
4. For an internal node, test both child boxes.
5. Push the farther child first and the nearer child second, so the nearer one is popped next.
6. Continue until no nodes remain.

Finding a nearby triangle early tightens the distance limit and can prune deferred nodes. Popped nodes are tested again because a closer hit may have been found since they were pushed.

An empty scene returns immediately without accessing a root node.

## 7. Following and combining light paths

### `hash(x)`

Scrambles integer bits with shifts, XORs, and multiplication to derive seeds. It helps distinguish neighboring worker seeds. The actual random-number generator is `RandomGPU`, which wraps cuRAND; `hash()` itself does not generate the sequence of bounce samples.

### `trace_path(...)`

Traces one sample through all allowed bounces. `throughput` starts at RGB `(1,1,1)` and accumulates component-wise surface albedos. `light` collects weighted emission and background.

At each depth:

1. Find the closest eligible triangle.
2. On a miss, add `throughput * background_light` and stop.
3. On a hit, add `throughput * material.emission`.
4. If the bounce limit is reached, stop.
5. Multiply throughput by the material's albedo.
6. Calculate the winning triangle's hit position and normal, facing the normal toward the incoming ray.
7. Sample a new outgoing ray and remember the hit triangle's original ID.

`multiply(a, b)` multiplies RGB channels component by component. For example, passing through two surfaces with red-channel albedos 0.5 and 0.2 gives later red light a weight of 0.1.

Depth 0 is the camera ray's first hit. With `max_bounces = 1`, that hit can scatter once; the next surface still contributes emission but cannot scatter again. Paths may stop earlier on a miss. There is no Russian-roulette termination in this implementation.

### `paths(...)`

The path-tracing kernel distributes samples across GPU workers. Consecutive groups of `split` threads share a subpixel:

```cpp
subpixel = tid / split;
part = tid & (split - 1);
```

Since `split` is a power of two, the bit mask computes the worker's position within its group, equivalent to `tid % split`.

For ten samples and `split = 4`:

```text
worker 0: samples 0, 4, 8
worker 1: samples 1, 5, 9
worker 2: samples 2, 6
worker 3: samples 3, 7
```

Every sample is evaluated exactly once, including when the sample count is not divisible by `split`. Each worker reuses its subpixel's primary camera ray and draws random directions for bounces. There is no random camera jitter here.

#### Warp reduction and `__shfl_down_sync`

A CUDA warp contains 32 lanes. The kernel combines sums within groups of `min(split, 32)` lanes using:

```cpp
sum.x += __shfl_down_sync(0xffffffffu, sum.x, offset, width);
```

This reads a value from a higher lane in the same subgroup. Repeating with successively halved offsets makes the subgroup's first lane hold its total. The code does this separately for red, green, and blue.

The mask `0xffffffffu` names all 32 warp lanes. Padded lanes contribute zero and still execute the shuffles. They must not return early before these operations. Launch blocks contain 128 threads, a multiple of the warp size.

The first lane of each subgroup atomically adds the result to `sums[subpixel]`. When `split > 32`, several warps contribute to the same subpixel, and atomics combine them safely. `sums` is zeroed before the kernel starts.

### `average(...)`

One thread handles each output pixel. With supersampling factor `ss`, each pixel owns an `ss` by `ss` grid of subpixels in the enlarged image.

The kernel adds their accumulated light and divides by:

```text
samples_per_pixel * ss * ss
```

For `ss = 2` and ten samples per subpixel, each output pixel averages 40 paths. Averaging occurs in linear color space before display conversion.

### `render_scene(settings, objects)`

The public CPU entry point coordinates the render:

1. Flatten objects into `host_triangles`, assigning original IDs and material indices.
2. Pack each object's albedo and emission into `host_materials`.
3. Query the GPU's SM count and choose a sample-worker group size, `split`.
4. Allocate GPU triangle, node, sum, and output-color buffers.
5. Upload triangles, build the BVH, and upload the material table.
6. Zero sample sums, launch `paths()`, and then launch `average()`.
7. Copy linear colors back to CPU memory.
8. Convert each color using `to_pixel()` and return the pixels.

The worker target is `SM_count * 2048 * 4`. This is a heuristic target for launched work, not a claim that all those threads are simultaneously resident. `split` doubles until there is enough work, the sample threshold is reached, or its cap is reached. It can exceed the sample count on its final doubling, leaving some workers with no sample.

Changing `split` preserves the number of samples, but can change the rendered estimate: it changes worker seeds, random-number sequences, and floating-point summation order. Atomic accumulation may also vary the addition order.

`cudaMemcpy` uploads/downloads ordinary arrays. `cudaMemcpyToSymbol` fills the constant-memory material table. The final blocking device-to-host copy waits for the GPU output before the CPU converts it.

## 8. Supporting functions in the headers

| Helper | Role in this renderer |
| --- | --- |
| `Vec3` operations | Add, subtract, scale, and divide positions, directions, and colors. Its default constructor produces `(0,0,0)`, so local sums start at zero. |
| `dot`, `cross`, `normalize` | Geometry operations used for intersection, normals, and direction vectors. |
| `intersect_triangle` | Finds ray/triangle intersections using edge vectors, determinant, and barycentric coordinates; rejects near-parallel rays and hits too close to the origin. Writes a `HitRecord`. |
| `make_camera_ray` | Makes a normalized ray from `(0,0,camera_position)` through a subpixel center on the screen at z = 0. |
| `RandomGPU` | Initializes and advances a cuRAND XORWOW random state for each worker. |
| `cosine_weighted_sampling` | Samples a random unit direction, adds the surface normal, and normalizes to choose an outgoing diffuse ray. Starts the new ray at the hit position. |
| `to_pixel` | Applies Reinhard tone mapping `c / (c + 1)`, then gamma correction with exponent `1 / 2.2`. |
| `Pixel` constructor | Converts display RGB values into 8-bit channels. |

`HitRecord` stores distance `t`, hit position, and unit surface normal. During candidate testing, only distance is retained in `Closest`; the final winning position and normal are calculated for scattering.

The GPU BVH path does not use the object bounding-sphere helpers. Its acceleration structure consists of triangle bounding boxes.

## 9. Can we launch more threads or blocks?

Most stages already scale their block count with the number of items:

| Stage | Work assignment |
| --- | --- |
| Scene bounds | One block, strided over all triangles. |
| Spatial keys | One thread per triangle. |
| Sorting | Thrust chooses its own parallel implementation. |
| Leaf construction | One thread per leaf. |
| Bounds merging | One launched thread per node, with internal-node threads returning; leaf threads merge upward. |
| Tracing | `split` workers per subpixel, using 128-thread blocks. |
| Averaging | One thread per output pixel, using 256-thread blocks. |

For 10,000 triangles, spatial-key generation launches 40 blocks of 256 threads. Changing the block size to 512 launches 20 blocks, leaving roughly the same number of total workers. It changes grouping rather than doubling the available work.

The single-block scene-bounds stage could be changed to a multi-block reduction: compute one box per block, then combine those boxes in another stage. That requires an implementation change, not just a larger launch grid, because the current code only coordinates threads inside one block.

Bounds merging has dependencies: a parent needs both child boxes. Less independent work remains near the root. Extra idle threads cannot remove that dependency.

Block size, per-thread registers, stack storage, and shared memory affect how much work can reside on the GPU. Measure the build stages and tracing separately before tuning them; their contributions depend on scene size and render settings.

## 10. Reading order

Start with `render_scene()` for the overall pipeline. Then read `build_bvh()` and its four kernels to understand tree construction. For rendering, read `paths()`, `trace_path()`, `closest_hit()`, and finally `average()`.

For the key-splitting logic, revisit the worked `__clzll` example before reading `split_leaves()`: the central question is whether a middle key shares more prefix bits with the first key than the range's endpoints share.
