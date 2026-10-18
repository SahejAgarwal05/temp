# How the BVH is stored: understanding every index

The tree in [path_tracing.cu](path_tracing.cu) is stored in an ordinary array. The confusing part is that the code uses several kinds of indices, and `link` has different meanings for internal nodes and leaves.

This guide follows one example all the way from the tree diagram to the actual arrays. For how the split positions are chosen, see [BVH_TREE_CREATION.md](BVH_TREE_CREATION.md).

## 1. Start with the two rules for reading a node

```cpp
struct alignas(16) Node {
    Vec3 lo;
    int link;
    Vec3 hi;
    int count;
};
```

`lo` and `hi` describe the node's box. To interpret the other fields, check `count` first:

| Condition | What this node is | How to interpret it |
| --- | --- | --- |
| `count == 0` | Internal node | Left child is `nodes[i + 1]`; right child is `nodes[node.link]`. |
| `count > 0` | Leaf | Its triangles start at `triangles[node.link]`, with `count` consecutive entries. |

Here, `i` is this node's position in the `nodes` array.

For example:

```cpp
// Internal node at nodes[4]:
nodes[4].count = 0;
nodes[4].link = 8;
// Children: nodes[5] and nodes[8].

// Leaf at nodes[7]:
nodes[7].count = 2;
nodes[7].link = 6;
// Triangles: triangles[6] and triangles[7].
```

The second leaf does not point to node 6. Its `link` refers to the triangle array because it is a leaf.

## 2. There are four different index systems

| Name | What it identifies | Where it appears |
| --- | --- | --- |
| Node index | An entry in the tree's node array | `nodes[current]`, traversal stack, `parents[]` values |
| Leaf index | A group of up to two sorted triangles | `leaf`, `first`, `last`, `split` in `build_leaves()` |
| Sorted triangle index | An entry in the reordered triangle array | `triangles[i]`, a leaf's `link`, `Closest.index` |
| Original triangle ID | A triangle's identity before sorting | `Primitive.id`, `Closest.id`, `last` in ray-intersection functions |

These indices can have the same numerical value by coincidence. They are not interchangeable.

Variable names also depend on the function: `last` in `build_leaves()` is a leaf-range endpoint; `last` in `closest_hit()` is the original triangle ID to skip after a bounce.

## 3. Our example: nine triangles, five leaves

With `kLeafSize = 2`, nine sorted triangles form five leaf groups:

| Leaf index | Sorted triangle indices | Triangle count |
| --- | --- | --- |
| 0 | 0, 1 | 2 |
| 1 | 2, 3 | 2 |
| 2 | 4, 5 | 2 |
| 3 | 6, 7 | 2 |
| 4 | 8 | 1 |

The first triangle of leaf `leaf` is:

```cpp
begin = leaf * kLeafSize;
```

Thus leaf 3 starts at sorted triangle index `3 * 2 = 6`.

Each leaf needs one node. Five leaves also need four internal nodes, so the node array has nine entries.

Assume the spatial-key splits give this shape:

```text
                         A: leaves [0..4]
                         /              \
                B: leaves [0..1]       C: leaves [2..4]
                   /      \               /       \
                leaf 0   leaf 1     D: leaves [2..3] leaf 4
                                       /      \
                                    leaf 2   leaf 3
```

The letters are temporary labels for internal nodes. They are not stored in the program. The bracketed ranges are inclusive leaf indices.

## 4. Assign node indices in preorder

The storage order is **parent, entire left subtree, entire right subtree**. This is called preorder.

Walk the diagram in that order:

```text
A -> B -> leaf 0 -> leaf 1 -> C -> D -> leaf 2 -> leaf 3 -> leaf 4
```

Assign consecutive node indices:

```text
                         node 0: A
                         /       \
                  node 1: B      node 4: C
                   /    \        /       \
           node 2: L0 node 3: L1 node 5: D node 8: L4
                                 /     \
                          node 6: L2 node 7: L3
```

The array is physically ordered like this:

```text
index:      0    1     2      3     4    5     6      7      8
nodes[]:   [A]  [B]  [L0]   [L1]  [C]  [D]  [L2]   [L3]   [L4]
```

Every subtree occupies one contiguous segment of the array. For example, C's subtree is `nodes[4..8]`, and B's subtree is `nodes[1..3]`.

The diagram explains the layout; the GPU builder computes these positions independently for each leaf rather than constructing the tree through a sequential preorder walk.

## 5. Why is the left child always `current + 1`?

The first item stored after a parent is its left-child root. So:

```text
node 0 -> left child 1
node 1 -> left child 2
node 4 -> left child 5
node 5 -> left child 6
```

This rule applies only to internal nodes. Leaves have no children.

The right child cannot generally be `current + 2`: the entire left subtree must fit before it. Root node 0's right child is node 4 because nodes 1, 2, and 3 belong to the left subtree.

## 6. Deriving the right-child formula

The builder knows the current inclusive leaf range `[first..last]` and its split:

```text
left child gets:  [first..split]
right child gets: [split+1..last]
```

The number of leaves on the left is:

```cpp
L = split - first + 1;
```

A full binary subtree with `L` leaves contains `L - 1` internal nodes, hence `2*L - 1` total nodes. That subtree begins at `current + 1`, so the right child begins after it:

```text
right = current + 1 + (2*L - 1)
      = current + 2*L
```

That is the expression in the code:

```cpp
right = current + 2 * (split - first + 1);
```

Examples from our tree:

| Current node | Leaf range | Split | Leaves on left | Right-child calculation |
| --- | --- | --- | --- | --- |
| 0 | `[0..4]` | 1 | 2 | `0 + 2*2 = 4` |
| 1 | `[0..1]` | 0 | 1 | `1 + 2*1 = 3` |
| 4 | `[2..4]` | 3 | 2 | `4 + 2*2 = 8` |
| 5 | `[2..3]` | 2 | 1 | `5 + 2*1 = 7` |

Notice that `split` is a leaf index. The calculation converts the size of a leaf range into the space needed in the node array.

## 7. The actual node records

Leaving out coordinates, these are the records for our nine-triangle example:

| Node index | Kind | `count` | `link` | Meaning of `link` | Parent |
| --- | --- | --- | --- | --- | --- |
| 0 | Internal A | 0 | 4 | Right child is node 4 | -1 |
| 1 | Internal B | 0 | 3 | Right child is node 3 | 0 |
| 2 | Leaf 0 | 2 | 0 | First triangle is 0 | 1 |
| 3 | Leaf 1 | 2 | 2 | First triangle is 2 | 1 |
| 4 | Internal C | 0 | 8 | Right child is node 8 | 0 |
| 5 | Internal D | 0 | 7 | Right child is node 7 | 4 |
| 6 | Leaf 2 | 2 | 4 | First triangle is 4 | 5 |
| 7 | Leaf 3 | 2 | 6 | First triangle is 6 | 5 |
| 8 | Leaf 4 | 1 | 8 | First triangle is 8 | 4 |

`count` tells us how to interpret `link`. A `link` of zero can be valid: leaf 0 points to the first triangle. It is not a null-pointer marker.

Each array index counts records, not bytes. `nodes[4]` selects the fifth `Node`; C++ automatically accounts for the 32-byte record size. `alignas(16)` changes alignment requirements, not these indexing rules.

## 8. Where the parent indices live

Parent indices are stored in a separate temporary array, not inside `Node`:

```text
node index:   0   1   2   3   4   5   6   7   8
parents[]:   -1   0   1   1   0   4   5   5   4
```

For example, `parents[7] = 5`: the parent of node 7 is node 5.

Following parents from leaf 3's node gives:

```text
node 7 -> node 5 -> node 4 -> node 0 -> -1
```

`merge_bounds()` uses this path to combine boxes upward. The `-1` sentinel means the root has no parent.

The parent array is freed after building. Ray traversal only needs the persistent `nodes` and `triangles` arrays.

## 9. Follow one leaf thread during construction

The `build_leaves()` thread assigned to **leaf index 3** must discover that its final **node index is 7**.

It starts at root node 0 with leaf range `[0..4]`:

| Step | `current` (node index) | `[first..last]` (leaf range) | Split | Action |
| --- | --- | --- | --- | --- |
| 1 | 0 | `[0..4]` | 1 | Leaf 3 is on the right: go to node 4. |
| 2 | 4 | `[2..4]` | 3 | Leaf 3 is on the left: go to node 5. |
| 3 | 5 | `[2..3]` | 2 | Leaf 3 is on the right: go to node 7. |
| 4 | 7 | `[3..3]` | — | Range has one leaf: write the leaf record. |

The final writes include:

```cpp
nodes[7].link = 6;   // leaf 3 starts at sorted triangle 3*2
nodes[7].count = 2;  // sorted triangles 6 and 7
parents[7] = 5;
```

Within this kernel, the global thread index identifies a leaf. In `merge_bounds()`, the global thread index identifies a node, and threads assigned to internal nodes return. The two launches assign different meanings to their indices.

## 10. Follow the same branch during ray traversal

Assume a ray visits the branch ending at node 7:

1. Read `nodes[0]`. It is internal. Its children are node `0 + 1 = 1` and node `link = 4`.
2. Visit node 4. It is internal. Its children are node `4 + 1 = 5` and node `link = 8`.
3. Visit node 5. It is internal. Its children are node `5 + 1 = 6` and node `link = 7`.
4. Visit node 7. Its `count = 2`, so it is a leaf. Test sorted triangles `link = 6` through `6 + 2 - 1 = 7`.

Actual traversal chooses child order using box entry distances and may skip boxes it misses. The example above isolates the indexing on one branch.

Entries in the traversal stack are **node indices**, such as 1 or 8. They do not identify triangles.

## 11. Sorted positions versus original triangle IDs

Suppose sorting produces this triangle array:

```text
sorted triangle index:  0  1  2  3  4  5  6  7  8
Primitive.id:           6  1  8  3  0  7  4  2  5
```

Node 7 contains sorted entries 6 and 7, which have original IDs 4 and 2.

If the winning triangle is `triangles[6]`, the hit record stores:

```text
Closest.index = 6  -> where to find its geometry in the sorted array
Closest.id    = 4  -> which original triangle it is
```

The next bounce skips original ID 4. It does not skip original ID 6 simply because the sorted position was 6.

The position-to-ID mapping depends on the spatial sorting; the numbers above are illustrative.

## 12. Rules to keep beside the code

| Task | Rule |
| --- | --- |
| Decide whether a node is internal | `nodes[i].count == 0` |
| Find an internal node's left child | `i + 1` |
| Find an internal node's right child | `nodes[i].link` |
| Find a leaf's triangle range | Start at `nodes[i].link`, read `nodes[i].count` entries |
| Convert leaf index to first triangle index | `leaf * kLeafSize` |
| Convert leaf index to node index | Follow the key-based splits and subtree-size calculations; no fixed multiplication works for every tree shape |
| Find a node's parent during building | `parents[i]` |
| Find a sorted triangle's original identity | `triangles[i].id` |

This is preorder storage, so the heap formulas `2*i + 1`, `2*i + 2`, and `(i - 1)/2` do not apply. Even the parent cannot generally be recovered by halving the node index.

When reading a line such as `nodes[nodes[parent].link]`, unpack it in two steps: read the parent's right-child index, then use that index to read the right child's node record. That expression is used on an internal parent, so `link` has its child-index meaning.
