# BVH tree creation and splitting, step by step

This walkthrough follows the actual builder in [path_tracing.cu](path_tracing.cu). It focuses on how sorted triangle keys become a tree, how `split_leaves()` chooses children, how GPU threads assign node indices, and how the bounding boxes are completed.

For the whole rendering pipeline, see [BVH_GUIDE.md](BVH_GUIDE.md).

## 1. What are we trying to create?

A bounding volume hierarchy, or BVH, is a tree of boxes. Every box contains all triangles beneath it. Internal nodes have two children; leaves contain actual triangle ranges.

```text
                    box containing all triangles
                         /             \
                box for one group    box for another group
                   /       \             /       \
                 ...       ...         ...       ...
```

A ray that misses a node's box can skip that entire subtree. This is why it helps to group triangles spatially: smaller, tighter boxes tend to eliminate more unnecessary triangle tests.

The builder has two separate tasks:

1. **Topology:** decide the parent/child relationships and triangle membership.
2. **Bounds:** compute the box belonging to each node.

In this implementation, the topology comes from sorted spatial keys. Leaf boxes come directly from triangle vertices. Internal boxes are computed afterward by combining child boxes.

## 2. The complete build pipeline

`build_bvh()` runs on the CPU and launches the following work on the GPU:

```text
triangle array
    |
    v
scene_bounds()          Find a box enclosing the whole scene
    |
    v
spatial_keys()          Give each triangle a key describing its position
    |
    v
thrust::sort_by_key()   Sort keys and triangle records together
    |
    v
build_leaves()          Construct tree links, parent indices, and leaf boxes
    |
    v
merge_bounds()         Combine leaf boxes upward to complete internal boxes
```

The CPU allocates the arrays and coordinates these stages. It does not recursively construct the tree. The stages execute in default-stream order, so later stages see the results of earlier ones.

## 3. Step 1: find the scene bounds

`scene_bounds()` calculates minimum and maximum coordinates across every triangle vertex.

For example:

```text
minimum corner = (-10, -2, 0)
maximum corner = ( 10,  6, 8)
```

These bounds define the coordinate range used to normalize triangle positions. They are temporarily written into `nodes[0].lo` and `nodes[0].hi`. Later, the real root box replaces them.

One block of 256 threads does the calculation. Thread `t` scans triangles `t`, `t + 256`, `t + 512`, and so on. The block then reduces the local boxes to one scene box using shared memory and barriers.

## 4. Step 2: give each triangle a spatial key

### 4.1 Use the center of its bounding box

Each triangle gets its own temporary box. Its center is:

```cpp
center = box.lo * 0.5f + box.hi * 0.5f;
```

This is the bounding-box center, which can differ from the average of the triangle's three vertices.

The center is used to choose a spatial ordering. The triangle's full extent is still included when calculating its actual leaf box.

### 4.2 Convert coordinates into integers

`grid_position()` maps each center coordinate into an integer from 0 through 1023, using the scene bounds:

```text
normalized coordinate = (value - scene_min) / (scene_max - scene_min)
grid coordinate       = clamp(normalized coordinate * 1024, 0, 1023)
```

Converting to an unsigned integer drops the fractional part. If an axis has zero extent, its grid coordinate is zero.

Each coordinate fits into 10 bits because 2^10 = 1024.

### 4.3 Interleave the coordinate bits

A Morton code interleaves x, y, and z bits. Using three bits per coordinate for illustration:

```text
x: x2 x1 x0
y: y2 y1 y0
z: z2 z1 z0

Morton: x2 y2 z2 x1 y1 z1 x0 y0 z0
```

Our code uses 10 bits per axis, producing 30 spatial bits. More significant bits describe coarser position; later bits describe finer position.

This gives us an integer that can be sorted. Nearby Morton codes often represent nearby regions, although the ordering does not guarantee that consecutive triangles are each other's nearest neighbors.

### 4.4 Append the original triangle ID

The final 64-bit key is:

```text
2 unused high bits | 30 Morton bits | 32 original triangle ID bits
```

```cpp
key = (static_cast<unsigned long long>(morton) << 32) | unsigned(id);
```

The original IDs are unique. Therefore, two triangles still have different complete keys even if their Morton codes match.

The ID also remains inside the triangle record after sorting. The renderer later uses it for self-hit exclusion and equal-distance tie-breaking.

## 5. Step 3: sort triangles and form leaf groups

Thrust sorts the keys and moves their associated `Primitive` records together:

```cpp
thrust::sort_by_key(... keys ..., ... triangles ...);
```

After sorting, triangle index 0 means the first triangle in spatial order. It does not necessarily mean original triangle ID 0.

With `kLeafSize = 2`, consecutive sorted triangles form leaves:

| Leaf index | Sorted triangle indices | Representative key |
| --- | --- | --- |
| 0 | 0, 1 | `keys[0]` |
| 1 | 2, 3 | `keys[2]` |
| 2 | 4, 5 | `keys[4]` |
| 3 | 6, 7 | `keys[6]` |
| 4 | 8, 9 | `keys[8]` |

Each leaf's first triangle supplies the key used to split the tree. This explains the repeated expression:

```cpp
keys[leaf_index * kLeafSize]
```

It translates a **leaf index** into its first **triangle index**.

Ten triangles give five leaves and nine total nodes:

```text
leaves         = ceil(10 / 2) = 5
internal nodes = leaves - 1  = 4
total nodes    = 2*leaves - 1 = 9
```

For an odd triangle count, the final leaf contains one triangle. Leaves are grouped before the hierarchy is determined; splitting changes how those leaf groups are arranged under internal nodes.

## 6. What exactly does `split_leaves()` return?

```cpp
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
```

`first` and `last` are inclusive leaf indices. The function returns the final leaf index of the left child:

```text
current range: first ......................... last
left child:    first ........ split
right child:                     split+1 ..... last
```

It does not return a triangle index, a node-array index, a bit position, or necessarily the numerical midpoint of the range.

## 7. Why XOR and `__clzll` reveal the splitting bit

`__clzll(value)` counts leading zero bits in a 64-bit value. Start at the most significant bit on the left and stop at the first `1`.

XOR, `^`, gives zero for matching bits and one for differing bits. Therefore:

```cpp
__clzll(a ^ b)
```

is the number of matching bits at the beginning of `a` and `b`.

```text
a:    11001000
b:    11011100
XOR:  00010100
      ^^^
      3 matching prefix bits
```

The examples below show eight-bit toy keys. Real keys have 64 bits. If an eight-bit value is stored with 56 zero bits above it, `__clzll` counts those too. All compared keys share that padding, so the relative comparisons still work; the toy examples omit it for readability.

For sorted keys, the first and last keys determine the prefix shared by the entire range. At the first differing bit, the first key has `0` and the last has `1`. All keys with `0` at that position come first, followed by those with `1`.

That bit gives a contiguous left group and right group. We use binary search to find the boundary.

## 8. A complete splitting example with five leaves

Use these sorted representative keys throughout the remaining examples:

| Leaf | Toy representative key |
| --- | --- |
| 0 | `00100000` |
| 1 | `00110000` |
| 2 | `01000000` |
| 3 | `01010000` |
| 4 | `01100000` |

These illustrate complete ordered keys; the real implementation calculates full 64-bit keys from Morton codes and IDs.

### 8.1 Root range: leaves 0 through 4

Compare endpoints:

```text
first: 00100000
last:  01100000
XOR:   01000000
```

They share one bit, `0`. The next bit separates `00......` from `01......`.

The function starts with `left = 0`, `right = 4`:

| Iteration | `left` | `right` | `middle` | First XOR middle | Shared bits | Update |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 4 | 2 | `01100000` | 1 | `right = 2`, since 1 is not greater than 1 |
| 2 | 0 | 2 | 1 | `00010000` | 3 | `left = 1`, since 3 is greater than 1 |

Now `left = 1`, `right = 2`: they are adjacent, so the loop stops and returns 1.

```text
root: leaves [0..4]
    left:  [0..1]  -> 00......
    right: [2..4]  -> 01......
```

### 8.2 Left range: leaves 0 through 1

There are only two leaves. `left + 1 < right` is false immediately, so the function returns `first`, which is 0.

```text
range [0..1]
    left:  [0..0]
    right: [1..1]
```

A range containing one leaf is already a leaf node; `build_leaves()` does not call `split_leaves()` for it.

### 8.3 Right range: leaves 2 through 4

```text
first: 01000000
last:  01100000
XOR:   00100000
```

The shared prefix has length 2: `01`. The next bit separates `010.....` and `011.....`.

The midpoint is leaf 3:

```text
01000000 XOR 01010000 = 00010000
```

It shares 3 bits with the first key. Since `3 > 2`, set `left = 3`. Now `right = 4` is adjacent, so return 3.

```text
range [2..4]
    left:  [2..3]
    right: [4..4]
```

The range `[2..3]` has two leaves and splits into leaf 2 and leaf 3.

### 8.4 The resulting hierarchy

```text
                         [0..4]
                       /        \
                  [0..1]        [2..4]
                  /    \        /    \
               leaf 0 leaf 1 [2..3]  leaf 4
                             /    \
                          leaf 2 leaf 3
```

Labels like `[2..4]` are inclusive leaf ranges, not coordinates or node-array indices.

## 9. Why the binary search is correct

At the start of the loop:

- `left` identifies a key on the first key's side of the splitting bit.
- `right` identifies a key on the last key's side.

For a midpoint key:

```cpp
__clzll(start ^ middle_key) > prefix
```

means it matches `start` beyond the endpoints' shared prefix. It therefore matches `start` at the splitting bit and belongs on the left.

Otherwise, it differs at the splitting bit and belongs on the right. Sorted order makes this a single boundary, so either update safely discards part of the search interval.

When `left` and `right` are adjacent, the boundary lies between them. Return `left`.

Every iteration shrinks the interval, so a split search takes logarithmic time in the number of leaves in that range. This is the cost of one split search; the entire GPU builder also repeats these searches along each leaf's root-to-leaf path.

The appended IDs keep representative keys distinct, making the endpoint XOR nonzero whenever `first < last`. If Morton codes coincide, the split can occur in the ID bits instead of the spatial bits.

## 10. How the tree fits into one node array

`Node` stores:

```cpp
struct alignas(16) Node {
    Vec3 lo;
    int link;
    Vec3 hi;
    int count;
};
```

For an internal node, `count = 0`, its left child is at `current + 1`, and `link` is its right child's index. For a leaf, `count > 0` and `link` is the first sorted triangle index.

Nodes use preorder layout:

```text
parent -> entire left subtree -> entire right subtree
```

The right-child index is computed from the left subtree's size:

```cpp
const int left_leaves = split - first + 1;
const int right = current + 2 * left_leaves;
```

Why? A subtree with `L` leaves contains `2*L - 1` nodes. It starts immediately after the parent, so the right subtree starts at:

```text
current + 1 + (2*L - 1) = current + 2*L
```

For our example, the root's left subtree has two leaves, so its right child begins at `0 + 2*2 = 4`.

The complete array is:

| Node index | Meaning | Left child | `link` | `count` | Parent |
| --- | --- | --- | --- | --- | --- |
| 0 | Internal, leaves `[0..4]` | 1 | 4 (right child) | 0 | -1 |
| 1 | Internal, leaves `[0..1]` | 2 | 3 (right child) | 0 | 0 |
| 2 | Leaf 0, triangles 0 and 1 | — | 0 (first triangle) | 2 | 1 |
| 3 | Leaf 1, triangles 2 and 3 | — | 2 (first triangle) | 2 | 1 |
| 4 | Internal, leaves `[2..4]` | 5 | 8 (right child) | 0 | 0 |
| 5 | Internal, leaves `[2..3]` | 6 | 7 (right child) | 0 | 4 |
| 6 | Leaf 2, triangles 4 and 5 | — | 4 (first triangle) | 2 | 5 |
| 7 | Leaf 3, triangles 6 and 7 | — | 6 (first triangle) | 2 | 5 |
| 8 | Leaf 4, triangles 8 and 9 | — | 8 (first triangle) | 2 | 4 |

```text
                         node 0
                       /        \
                   node 1       node 4
                   /    \       /    \
                node 2 node 3 node 5 node 8
                              /    \
                           node 6 node 7
```

Leaf index 2, sorted triangle index 2, original triangle ID 2, and node index 2 are different concepts. Always check which kind of index a variable holds.

## 11. How `build_leaves()` creates this in parallel

One GPU thread is assigned to each leaf. There is no single thread recursively visiting and constructing every node. Instead, each leaf thread independently calculates its path through the same deterministic splits.

The initial state is:

```cpp
current = 0;       // Root's node-array index
parent = -1;       // Root has no parent
first = 0;        // First leaf in the current range
last = leaves-1;  // Last leaf in the current range
```

At each step, the thread computes the split and right-child index, then follows the side containing its assigned leaf:

```cpp
parent = current;
if (leaf <= split) {
    ++current;       // Left child is immediately after the parent
    last = split;
} else {
    current = right;
    first = split + 1;
}
```

### Follow the thread assigned to leaf 3

| Step | Current node | Leaf range | Split | Right-child node | Decision |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | `[0..4]` | 1 | 4 | Leaf 3 > 1: go right to node 4 |
| 2 | 4 | `[2..4]` | 3 | 8 | Leaf 3 <= 3: go left to node 5 |
| 3 | 5 | `[2..3]` | 2 | 7 | Leaf 3 > 2: go right to node 7 |
| 4 | 7 | `[3..3]` | — | — | One leaf remains: stop |

It then writes:

```text
nodes[7].link  = 3 * 2 = 6
nodes[7].count = 2
parents[7]    = 5
nodes[7].box  = padded box enclosing sorted triangles 6 and 7
```

### Who writes the internal nodes?

Many leaf threads visit the same internal node. To avoid multiple writers, this condition chooses one:

```cpp
if (leaf == first) {
    nodes[current].link = right;
    nodes[current].count = 0;
    parents[current] = parent;
}
```

Only the first leaf in that node's range writes its metadata:

| Internal node | Leaf range | Writer thread |
| --- | --- | --- |
| 0 | `[0..4]` | Leaf 0 |
| 1 | `[0..1]` | Leaf 0 |
| 4 | `[2..4]` | Leaf 2 |
| 5 | `[2..3]` | Leaf 2 |

Every thread derives the same indices from the keys, so threads do not need to wait for an internal node to be written before calculating their next step. During this kernel, they navigate using calculated ranges and indices rather than reading links another thread is still creating.

The tradeoff is repeated work: several threads calculate the same ancestor splits. This keeps construction independent per leaf without a task queue or CPU recursion.

## 12. Leaf boxes: actual geometry enters the tree

After a thread reaches its leaf:

```cpp
begin = leaf * kLeafSize;
size = min(count - begin, kLeafSize); // Equivalent to the code's conditional
```

It expands a `Box` over every vertex of the leaf's one or two triangles, pads the box slightly for rounding tolerance, and writes it to the node.

The representative key controls grouping. It is not used as a substitute for the actual triangle bounds. Even if a triangle extends far away from its center, its full vertices are included in the leaf box.

At the end of `build_leaves()`, leaf boxes and all links are valid. Internal-node boxes are not complete yet.

## 13. Completing internal boxes with `merge_bounds()`

This kernel launches threads over node indices, but threads whose nodes are internal return immediately. The active threads begin at leaves.

For an internal node, its box is the component-wise union of its child boxes:

```text
parent.lo = min(left.lo, right.lo)
parent.hi = max(left.hi, right.hi)
```

Taking minima and maxima independently for x, y, and z gives the smallest axis-aligned box enclosing both child boxes.

### Why each parent needs a counter

A parent can be merged only after both children have finished. The array `finished` starts at zero. Each arriving child increments its parent's counter atomically:

```cpp
cuda::atomic_ref<int, cuda::thread_scope_device> children(finished[parent]);
if (children.fetch_add(1, cuda::memory_order_acq_rel) == 0) return;
```

`fetch_add(1)` returns the value **before** the increment:

| Arrival | Previous counter | New counter | Action |
| --- | --- | --- | --- |
| First completed child | 0 | 1 | Stop; sibling may still be unfinished |
| Second completed child | 1 | 2 | Merge both child boxes and continue upward |

The first child does not spin or wait. The second child's thread does the remaining work for their parent.

### A possible execution for our nine-node tree

GPU scheduling can vary. This is one valid arrival order:

1. Leaf node 2 arrives at parent 1. Counter becomes 1; its thread stops.
2. Leaf node 3 arrives at parent 1. Counter becomes 2; merge nodes 2 and 3 into node 1.
3. That thread continues to parent 0. Root counter becomes 1; it stops.
4. Leaf node 6 arrives at parent 5. Counter becomes 1; it stops.
5. Leaf node 7 arrives at parent 5. Counter becomes 2; merge nodes 6 and 7 into node 5.
6. That thread arrives at parent 4. Counter becomes 1; it stops.
7. Leaf node 8 arrives at parent 4. Counter becomes 2; merge nodes 5 and 8 into node 4.
8. That thread arrives at parent 0. Counter becomes 2; merge nodes 1 and 4 into the root.
9. Root parent is `-1`, so the upward walk ends.

Each completed subtree contributes exactly one arrival to its parent. A different arrival order still produces the same hierarchy and unions of boxes.

### Why acquire/release ordering matters

An atomic counter alone is not the complete explanation. A child must publish its box writes before the thread merging the parent reads them.

The release part makes prior box writes available; the acquire part lets the second arrival observe the sibling's published box. Device scope allows this communication across GPU blocks. A block-local barrier cannot coordinate arbitrary blocks.

Internal metadata was completed by the earlier `build_leaves()` kernel. Default-stream ordering ensures `merge_bounds()` starts after that stage finishes.

## 14. Important edge cases and misconceptions

### Empty scene

Zero triangles means zero leaves and nodes. `build_bvh()` returns without launching the build kernels.

### One leaf

One or two triangles produce one node. That node is both root and leaf, with parent `-1`. There are no splits and no parent boxes to merge.

### Odd triangle counts

Nine triangles still produce five leaves. The last leaf starts at triangle 8 and has `count = 1`; the earlier four have `count = 2`.

### Identical spatial codes

Triangles can quantize to the same Morton code. Their appended original IDs keep the complete keys distinct, allowing the builder to split using ID bits. Such a split separates records, but does not imply spatial separation.

### The tree need not be balanced

For toy keys `00000000`, `00000001`, `00000010`, `00000011`, `10000000`, the root divides four leaves from one. Splitting follows the first differing key bit, not the median leaf count.

### A key split does not cut triangles

It partitions a sorted list of leaf groups. No triangle is geometrically sliced or duplicated. Child boxes can overlap because triangles have extent and Morton ordering is approximate.

### More threads cannot remove parent dependencies

Leaf calculations are independent and run in parallel. An internal box still needs both children. Near the root, fewer independent merges remain. Increasing the launch size cannot make an unfinished child box ready sooner by itself.

## 15. How to check your understanding

Using the five-leaf example above, answer these before checking the explanations:

1. Why does the root return split 1 rather than midpoint 2? Because the highest differing endpoint bit separates `00......` from `01......`.
2. Why is root's right child node 4? Its left subtree has two leaves and occupies three nodes, immediately after root node 0.
3. Why does leaf 3 become node 7 rather than node 3? Leaf indices and preorder node-array indices count different things.
4. Why does the leaf-2 thread write two internal nodes? Leaf 2 is first in both ranges `[2..4]` and `[2..3]`.
5. Why does the first child arriving at a parent stop? Its sibling may not have a completed box; the second arrival performs the merge.
6. Why append IDs to keys? To keep complete keys unique when spatial codes coincide.

To follow the source, read `build_bvh()`, then `spatial_keys()`, `split_leaves()`, `build_leaves()`, and `merge_bounds()`. Keep the nine-node array table beside the code while tracing a leaf thread.
