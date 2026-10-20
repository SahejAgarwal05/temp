"""Exercise the production BVH kernels on the host without a CUDA toolkit.

Run with: python3 -m unittest discover -s tests -v
"""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BVHTopologyTest(unittest.TestCase):
    def test_heap_topology_and_bounds(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("A C++ compiler is required")

        source = (ROOT / "path_tracing.cu").read_text()
        geometry = (ROOT / "geometry.cuh").read_text()
        vec = geometry[geometry.index("struct Vec3 {"):geometry.index(
            "__host__ __device__ inline Vec3 operator*")]
        node = source[source.index("struct LeafNode {"):source.index("struct GPUTriangle")]
        box_and_count = source[source.index("struct NodeBox {"):source.index(
            "__device__ inline unsigned gridLocation")]
        kernels = source[source.index("__device__ NodeBox calcNodeBox"):source.index(
            "// Call after Morton sorting.")]
        program = r"""
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#define __host__
#define __device__
#define __global__
int __clz(unsigned value) { return __builtin_clz(value); }
constexpr int MAX_LEAF_SIZE = 2;
constexpr float INF = std::numeric_limits<float>::infinity();
struct LaunchIndex { int x = 0; } blockIdx, threadIdx, blockDim{32};
""" + vec + r"""
struct Triangle { Vec3 p0, p1, p2; };
struct GPUTriangle { Vec3 lowCorner, highCorner; int triangle_index, object_index; };
""" + node + box_and_count + kernels + r"""
bool equal(const Vec3& a, const Vec3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}
void contains(const LeafNode& node, const Vec3& point) {
    assert(node.lowCorner.x <= point.x && node.highCorner.x >= point.x);
    assert(node.lowCorner.y <= point.y && node.highCorner.y >= point.y);
    assert(node.lowCorner.z <= point.z && node.highCorner.z >= point.z);
}
int main() {
    std::mt19937 random(3210);
    assert(countNodes(0) == 0);
    for (int n = 1; n <= 1025; ++n) {
        const int leaves = (n + 1) / 2;
        const int node_count = countNodes(n);
        int capacity = 1;
        while (capacity < leaves) capacity *= 2;
        assert(node_count == 2 * capacity - 1);
        std::vector<GPUTriangle> triangles(n);
        for (int i = 0; i < n; ++i) {
            float x = float(int(random() % 10000) - 5000);
            float y = float(int(random() % 10000) - 5000);
            triangles[i] = {Vec3(x, y, -3), Vec3(x + 2, y + 4, -3), n - 1 - i, i % 7};
        }
        const auto original = triangles;
        std::vector<LeafNode> nodes(node_count, LeafNode(-999, -999));
        // Randomize thread order, including excess threads in the last block.
        auto launch = [&](int count, auto kernel) {
            std::vector<int> slots(((count + 31) / 32) * 32);
            for (int i = 0; i < int(slots.size()); ++i) slots[i] = i;
            std::shuffle(slots.begin(), slots.end(), random);
            for (int slot : slots) {
                blockIdx.x = slot / 32;
                threadIdx.x = slot % 32;
                kernel();
            }
        };
        // Topology must be complete before bounds, and must leave boxes alone.
        for (auto& node : nodes) {
            node.lowCorner = Vec3(123, 456, 789);
            node.highCorner = Vec3(-123, -456, -789);
        }
        launch(node_count, [&] { buildBVHNodes(node_count, n, nodes.data()); });
        std::function<void(int, int, int)> check_ranges = [&](int index, int start, int slots) {
            const auto& node = nodes[index];
            assert(node.startIndex == std::min(start * 2, n));
            assert(node.endIndex == std::min((start + slots) * 2, n) - 1);
            assert(equal(node.lowCorner, Vec3(123, 456, 789)));
            assert(equal(node.highCorner, Vec3(-123, -456, -789)));
            if (slots > 1) {
                check_ranges(2 * index + 1, start, slots / 2);
                check_ranges(2 * index + 2, start + slots / 2, slots / 2);
            }
        };
        check_ranges(0, 0, capacity);
        const auto topology = nodes;
        launch(capacity, [&] {
            buildBVHLeafBounds(capacity, triangles.data(), nodes.data());
        });
        int rounds = 0;
        for (int width = capacity / 2; width > 0; width /= 2) {
            launch(width, [&] {
                buildBVHInternalBounds(width - 1, width, nodes.data());
            });
            ++rounds;
        }
        int expected_rounds = 0;
        for (int width = 1; width < leaves; width *= 2) ++expected_rounds;
        assert(rounds == expected_rounds);
        std::vector<int> visited(node_count, 0), covered(n, 0);
        int leaf_visits = 0;
        std::function<int(int, int)> check = [&](int index, int depth) {
            assert(index >= 0 && index < node_count);
            assert(++visited[index] == 1);
            const auto& node = nodes[index];
            const int count = node.endIndex - node.startIndex + 1;
            assert(count >= 0);
            assert(node.startIndex == topology[index].startIndex);
            assert(node.endIndex == topology[index].endIndex);
            if (index >= capacity - 1) {
                const int sorted_start = 2 * (index - capacity + 1);
                assert(node.startIndex == std::min(sorted_start, n));
                assert(count == std::max(0, std::min(2, n - sorted_start)));
                if (count > 0) ++leaf_visits;
                NodeBox expected;
                if (count > 0) expected = calcNodeBox(triangles.data(), node.startIndex, node.endIndex);
                assert(equal(node.lowCorner, expected.lowCorner));
                assert(equal(node.highCorner, expected.highCorner));
                for (int i = node.startIndex; i <= node.endIndex; ++i) {
                    ++covered[i];
                    contains(node, triangles[i].lowCorner);
                    contains(node, triangles[i].highCorner);
                }
                return depth;
            }
            const int left_index = 2 * index + 1;
            const int right_index = left_index + 1;
            const auto& left = nodes[left_index];
            const auto& right = nodes[right_index];
            assert(node.startIndex == left.startIndex);
            assert(right.startIndex == left.endIndex + 1);
            assert(node.endIndex == right.endIndex);
            assert(count == left.endIndex - left.startIndex + 1
                            + right.endIndex - right.startIndex + 1);
            NodeBox expected;
            if (left.startIndex <= left.endIndex) {
                expected.insert(left.lowCorner); expected.insert(left.highCorner);
            }
            if (right.startIndex <= right.endIndex) {
                expected.insert(right.lowCorner); expected.insert(right.highCorner);
            }
            assert(equal(node.lowCorner, expected.lowCorner));
            assert(equal(node.highCorner, expected.highCorner));
            return std::max(check(left_index, depth + 1),
                            check(right_index, depth + 1));
        };
        assert(nodes[0].startIndex == 0 && nodes[0].endIndex == n - 1);
        assert(check(0, 0) == expected_rounds);
        assert(leaf_visits == leaves);
        for (int hits : visited) assert(hits == 1);
        for (int hits : covered) assert(hits == 1);
        for (int i = 0; i < n; ++i) {
            assert(triangles[i].triangle_index == original[i].triangle_index);
            assert(triangles[i].object_index == original[i].object_index);
            assert(equal(triangles[i].lowCorner, original[i].lowCorner));
            assert(equal(triangles[i].highCorner, original[i].highCorner));
        }
    }
    std::cout << "1025 heap BVH topology and bounds cases passed\n";
}
"""
        with tempfile.TemporaryDirectory() as directory:
            cpp = Path(directory) / "bvh_topology.cpp"
            binary = Path(directory) / "bvh_topology"
            cpp.write_text(program)
            subprocess.run(
                [compiler, "-std=c++17", "-O2", str(cpp), "-o", str(binary)],
                check=True, capture_output=True, text=True,
            )
            result = subprocess.run(
                [str(binary)], check=True, capture_output=True, text=True
            )
            self.assertIn("1025 heap BVH topology and bounds cases passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
