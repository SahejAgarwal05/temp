// Included inside path_tracing.cu's anonymous namespace, after BVH helpers.
// All geometry work and CSR construction execute on the GPU. The host only
// reads allocation sizes/bounds and launches kernels. No per-ray host traffic.
using GridOffset = unsigned long long;
struct alignas(16) SurfaceSample {
    Vec3 position; int triangle;
    Vec3 normal; int object; // object indexes the existing material table
};
static_assert(sizeof(SurfaceSample) == 32, "Unexpected sample layout");

struct SampleGrid {
    Vec3 lo, hi;
    int dims[3] = {0, 0, 0};
    float cell = 0, radius = 0;
    const SurfaceSample* samples = nullptr;
    const GridOffset* offsets = nullptr;
    const int* references = nullptr;
    int sample_count = 0;
};

__device__ float component(const Vec3& v, int a) {
    return a == 0 ? v.x : a == 1 ? v.y : v.z;
}
__device__ int cell_coordinate(float value, float low, float cell, int size) {
    return int(fminf(float(size - 1), fmaxf(0.f, floorf((value - low) / cell))));
}
__device__ size_t cell_index(const SampleGrid& g, int x, int y, int z) {
    return (size_t(z) * g.dims[1] + y) * g.dims[0] + x;
}

__global__ void sampling_counts(const Primitive* triangles, int count, float spacing,
                                int* divisions, GridOffset* counts) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > count) return;
    if (i == count) { counts[i] = 0; return; }
    const Triangle t = triangles[i].triangle;
    const Vec3 a = t.p1 - t.p0, b = t.p2 - t.p0;
    if (dot(cross(a, b), cross(a, b)) == 0.f) {
        divisions[i] = 0; counts[i] = 0; return;
    }
    const float longest = fmaxf(length(a), fmaxf(length(b), length(t.p2 - t.p1)));
    const double n = ceil(double(longest) / spacing);
    // Never cast an unrepresentable float to int or overflow a triangular count.
    if (!isfinite(n) || n > 65534) {
        divisions[i] = 0; counts[i] = GridOffset(1) << 40; return;
    }
    const int steps = n < 1 ? 1 : int(n);
    divisions[i] = steps;
    counts[i] = GridOffset(steps + 1) * (steps + 2) / 2;
}

__global__ void sample_surfaces(const Primitive* triangles, int count,
                               const int* divisions, const GridOffset* offsets,
                               SurfaceSample* samples, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    // Upper bound handles empty/degenerate triangles with repeated offsets.
    int lo = 0, hi = count;
    while (lo < hi) {
        const int m = lo + (hi - lo) / 2;
        if (offsets[m + 1] <= GridOffset(i)) lo = m + 1;
        else hi = m;
    }
    const int local = int(GridOffset(i) - offsets[lo]);
    int row = int((sqrt(8.0 * local + 1.0) - 1.0) * 0.5);
    while (GridOffset(row + 1) * (row + 2) / 2 <= GridOffset(local)) ++row;
    while (GridOffset(row) * (row + 1) / 2 > GridOffset(local)) --row;
    const int column = local - int(GridOffset(row) * (row + 1) / 2);
    const Primitive p = triangles[lo];
    const Vec3 a = p.triangle.p1 - p.triangle.p0, b = p.triangle.p2 - p.triangle.p0;
    const float inv = 1.f / divisions[lo];
    samples[i] = {p.triangle.p0 + a * ((row - column) * inv) + b * (column * inv),
                  p.id, normalize(cross(a, b)), p.object};
}

__device__ void overlapped_cells(const SampleGrid& g, const Vec3& p, int* low, int* high) {
    for (int a = 0; a < 3; ++a) {
        // Outward rounding includes cubes touching either closed AABB boundary.
        const float v = component(p, a), origin = component(g.lo, a);
        const float lower = nextafterf((v - g.radius - origin) / g.cell, -INF);
        const float upper = nextafterf((v + g.radius - origin) / g.cell, INF);
        low[a] = int(fminf(float(g.dims[a] - 1), fmaxf(0.f, floorf(lower))));
        high[a] = int(fminf(float(g.dims[a] - 1), fmaxf(0.f, floorf(upper))));
    }
}

// Two passes: count references per cell, scan counts, then scatter sample IDs.
// CSR gives contiguous candidate ranges without allocating one vector per cube.
// Each reference is just a 32-bit ID; sample attributes are stored only once.
template<bool Fill>
__global__ void grid_references(SampleGrid g, GridOffset* cursors, int* references) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.sample_count) return;
    int low[3], high[3];
    overlapped_cells(g, g.samples[i].position, low, high);
    for (int z = low[2]; z <= high[2]; ++z)
        for (int y = low[1]; y <= high[1]; ++y)
            for (int x = low[0]; x <= high[0]; ++x) {
                const size_t cell = cell_index(g, x, y, z);
                const GridOffset slot = atomicAdd(cursors + cell, GridOffset(1));
                if constexpr (Fill) references[g.offsets[cell] + slot] = i;
            }
}

struct SampleClosest {
    float distance_squared = INF, t = INF;
    int index = -1;
};
__device__ void sample_candidate(const Ray& ray, SampleGrid g, int i, int last,
                                 SampleClosest& best) {
    const SurfaceSample p = g.samples[i];
    // Preserve the renderer's exclusion of the triangle just left at a bounce.
    if (p.triangle == last) return;
    const Vec3 offset = p.position - ray.origin;
    const float t = dot(offset, ray.direction);
    if (t <= EPSILON) return;
    const Vec3 perpendicular = offset - ray.direction * t;
    if (dot(perpendicular, perpendicular) > g.radius * g.radius) return;
    const float distance_squared = dot(offset, offset);
    // Stable sample-ID tie break, independent of atomic scatter order.
    if (distance_squared < best.distance_squared ||
        (distance_squared == best.distance_squared && (best.index < 0 || i < best.index))) {
        best.distance_squared = distance_squared; best.t = t; best.index = i;
    }
}

__device__ SampleClosest sampled_hit(const Ray& ray, SampleGrid g, int last) {
    SampleClosest best;
    if (!g.sample_count) return best;
    float entry = 0.f, exit = INF;
    for (int a = 0; a < 3; ++a) {
        const float o = component(ray.origin, a), d = component(ray.direction, a);
        const float low = component(g.lo, a), high = component(g.hi, a);
        if (d == 0.f) { if (o < low || o > high) return best; }
        else {
            const float t0 = (low - o) / d, t1 = (high - o) / d;
            entry = fmaxf(entry, fminf(t0, t1));
            exit = fminf(exit, fmaxf(t0, t1));
        }
    }
    if (entry > exit) return best;
    const Vec3 start = ray.origin + ray.direction * entry;
    int cell[3], step[3];
    float next[3], delta[3];
    for (int a = 0; a < 3; ++a) {
        const float d = component(ray.direction, a), low = component(g.lo, a);
        cell[a] = cell_coordinate(component(start, a), low, g.cell, g.dims[a]);
        step[a] = d > 0.f ? 1 : -1;
        delta[a] = d == 0.f ? INF : g.cell / fabsf(d);
        const float boundary = low + (cell[a] + (d > 0.f ? 1 : 0)) * g.cell;
        next[a] = d == 0.f ? INF : fmaxf(entry, (boundary - component(ray.origin, a)) / d);
    }
    while (true) {
        const size_t c = cell_index(g, cell[0], cell[1], cell[2]);
        for (GridOffset j = g.offsets[c]; j < g.offsets[c + 1]; ++j)
            sample_candidate(ray, g, g.references[j], last, best);
        const float boundary = fminf(next[0], fminf(next[1], next[2]));
        // Do not stop at the first occupied cube or first accepted point.
        // Any better sample has projection <= its Euclidean distance and must
        // be referenced in a visited cube at that projection. Strict comparison
        // preserves equal-distance tie breaks at the boundary.
        if (boundary > exit || !isfinite(boundary) ||
            (best.index >= 0 && boundary > sqrtf(best.distance_squared))) break;
        bool outside = false;
        // Advance all tied axes at edges/corners. Closed overlap registration
        // ensures candidates on the skipped side cells also occur here.
        for (int a = 0; a < 3; ++a) if (next[a] == boundary) {
            cell[a] += step[a]; next[a] += delta[a];
            outside |= cell[a] < 0 || cell[a] >= g.dims[a];
        }
        if (outside) break;
    }
    return best;
}

struct OwnedSampleGrid {
    SampleGrid view;
    DeviceBuffer<SurfaceSample> samples;
    DeviceBuffer<GridOffset> offsets;
    DeviceBuffer<int> references;
    // Allocation is separate from construction so sizes can be read from scans.
    OwnedSampleGrid(): samples(0), offsets(0), references(0) {}
};

void build_sample_grid(const Primitive* triangles, int count, const RenderSettings& settings,
                       OwnedSampleGrid& owned) {
    if (!(std::isfinite(settings.sample_spacing) && settings.sample_spacing > 0 &&
          std::isfinite(settings.acceptance_radius) && settings.acceptance_radius > 0 &&
          std::isfinite(settings.cube_size) && settings.cube_size > 0))
        throw std::runtime_error("Sample spacing, acceptance radius and cube size must be finite and positive");
    if (!count) return;
    DeviceBuffer<int> divisions(count);
    DeviceBuffer<GridOffset> counts(count + 1), sample_offsets(count + 1);
    sampling_counts<<<(count + 256) / 256, 256>>>(triangles, count, settings.sample_spacing,
                                               divisions.data, counts.data);
    thrust::exclusive_scan(thrust::device, thrust::device_pointer_cast(counts.data),
        thrust::device_pointer_cast(counts.data + count + 1),
        thrust::device_pointer_cast(sample_offsets.data));
    GridOffset total;
    cuda_check(cudaMemcpy(&total, sample_offsets.data + count, sizeof(total), cudaMemcpyDeviceToHost));
    if (total > 2147483647ULL)
        throw std::runtime_error("Too many surface samples; increase --sample-spacing");
    if (!total) return;
    owned.samples.allocate(size_t(total));
    SampleGrid& g = owned.view;
    g.sample_count = int(total); g.samples = owned.samples.data;
    g.radius = settings.acceptance_radius; g.cell = settings.cube_size;
    sample_surfaces<<<unsigned((total + 255) / 256), 256>>>(triangles, count, divisions.data,
        sample_offsets.data, owned.samples.data, int(total));
    DeviceBuffer<Node> bounds(1);
    scene_bounds<<<1, kBuildBlock>>>(triangles, count, bounds.data);
    Node box;
    cuda_check(cudaMemcpy(&box, bounds.data, sizeof(box), cudaMemcpyDeviceToHost));
    size_t cells = 1;
    float lows[3] = {box.lo.x, box.lo.y, box.lo.z};
    float highs[3] = {box.hi.x, box.hi.y, box.hi.z};
    for (int a = 0; a < 3; ++a) {
        lows[a] = std::nextafter(lows[a] - g.radius, -INF);
        const double span = double(highs[a]) + g.radius - lows[a];
        const double n = std::floor(span / g.cell) + 1;
        if (!std::isfinite(n) || n < 1 || n > 16777216 || cells > 134217728 / size_t(n))
            throw std::runtime_error("Sample grid too large; increase --cube-size");
        g.dims[a] = int(n); cells *= size_t(n);
        highs[a] = lows[a] + g.dims[a] * g.cell;
        if (!std::isfinite(lows[a]) || !std::isfinite(highs[a]) ||
            !(lows[a] + g.cell > lows[a]))
            throw std::runtime_error("Grid coordinates exceed floating point resolution");
    }
    g.lo = Vec3(lows[0], lows[1], lows[2]); g.hi = Vec3(highs[0], highs[1], highs[2]);
    DeviceBuffer<GridOffset> cursors(cells + 1);
    owned.offsets.allocate(cells + 1); g.offsets = owned.offsets.data;
    cuda_check(cudaMemset(cursors.data, 0, (cells + 1) * sizeof(GridOffset)));
    const unsigned blocks = unsigned((total + 255) / 256);
    grid_references<false><<<blocks, 256>>>(g, cursors.data, nullptr);
    thrust::exclusive_scan(thrust::device, thrust::device_pointer_cast(cursors.data),
        thrust::device_pointer_cast(cursors.data + cells + 1),
        thrust::device_pointer_cast(owned.offsets.data));
    GridOffset reference_count;
    cuda_check(cudaMemcpy(&reference_count, owned.offsets.data + cells,
                          sizeof(reference_count), cudaMemcpyDeviceToHost));
    if (reference_count > std::numeric_limits<size_t>::max() / sizeof(int))
        throw std::runtime_error("Sample reference allocation overflows size_t");
    owned.references.allocate(size_t(reference_count)); g.references = owned.references.data;
    cuda_check(cudaMemset(cursors.data, 0, cells * sizeof(GridOffset)));
    grid_references<true><<<blocks, 256>>>(g, cursors.data, owned.references.data);
    cuda_check(cudaGetLastError());
    std::cerr << "Sample grid: " << total << " samples, " << cells << " cubes, "
              << reference_count << " references\n";
}
