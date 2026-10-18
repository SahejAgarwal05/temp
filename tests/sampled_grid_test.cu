// Compile as one translation unit to exercise the renderer's actual internals.
#include "../path_tracing.cu"
#include <random>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

__global__ void compare_searches(SampleGrid grid, const Ray* rays, int count, int* failures) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int last = i % 5 == 0 ? 0 : -1;
    const SampleClosest actual = sampled_hit(rays[i], grid, last);
    SampleClosest expected;
    // Exhaustive search is independent of cell assignment and DDA/pruning.
    for (int s = 0; s < grid.sample_count; ++s)
        sample_candidate(rays[i], grid, s, last, expected);
    if (actual.index != expected.index) atomicAdd(failures, 1);
}

__global__ void check_candidate_rules(const SurfaceSample* points, int* result) {
    SampleGrid grid; grid.samples = points; grid.radius = 1;
    const Ray ray{Vec3(0, 0, 0), Vec3(0, 0, 1)};
    SampleClosest best;
    // IDs 0/1: first has smaller projection but greater Euclidean distance.
    sample_candidate(ray, grid, 0, -1, best);
    sample_candidate(ray, grid, 1, -1, best);
    result[0] = best.index;
    // Behind origin, t == EPSILON, outside R, same previous triangle.
    for (int i = 2; i < 6; ++i) sample_candidate(ray, grid, i, 42, best);
    result[1] = best.index;
    // Exactly R is accepted. Repeated/equal-distance hits have stable IDs.
    best = SampleClosest();
    sample_candidate(ray, grid, 6, -1, best);
    sample_candidate(ray, grid, 7, -1, best);
    sample_candidate(ray, grid, 6, -1, best);
    result[2] = best.index;
}

void candidate_tests() {
    const SurfaceSample points[] = {
        {Vec3(.9f, 0, 1), 0, Vec3(0, 0, 1), 0},
        {Vec3(0, 0, 1.1f), 1, Vec3(0, 0, 1), 0},
        {Vec3(0, 0, -1), 2, Vec3(0, 0, 1), 0},
        {Vec3(0, 0, EPSILON), 3, Vec3(0, 0, 1), 0},
        {Vec3(1.01f, 0, .01f), 4, Vec3(0, 0, 1), 0},
        {Vec3(0, 0, .1f), 42, Vec3(0, 0, 1), 0},
        {Vec3(1, 0, 1), 6, Vec3(0, 0, 1), 0},
        {Vec3(1, 0, 1), 7, Vec3(0, 0, 1), 0}
    };
    DeviceBuffer<SurfaceSample> data(8);
    DeviceBuffer<int> output(3);
    cuda_check(cudaMemcpy(data.data, points, sizeof(points), cudaMemcpyHostToDevice));
    check_candidate_rules<<<1, 1>>>(data.data, output.data);
    int result[3];
    cuda_check(cudaMemcpy(result, output.data, sizeof(result), cudaMemcpyDeviceToHost));
    require(result[0] == 1, "Search must minimize Euclidean distance, not projection");
    require(result[1] == 1, "Candidate rejection failed");
    require(result[2] == 6, "Radius equality or deterministic tie break failed");
}

void grid_tests(float spacing, float radius, float cell_size) {
    const Primitive triangles[] = {
        {{Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)}, 0, 0, 0},
        // Degenerate triangle in the middle exercises repeated prefix offsets.
        {{Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0)}, 0, 1, 0},
        {{Vec3(-.2f, 0, .7f), Vec3(.8f, .4f, .7f), Vec3(0, .9f, 1.3f)}, 1, 2, 0}
    };
    DeviceBuffer<Primitive> data(3);
    cuda_check(cudaMemcpy(data.data, triangles, sizeof(triangles), cudaMemcpyHostToDevice));
    RenderSettings settings{};
    settings.sample_spacing = spacing; settings.acceptance_radius = radius; settings.cube_size = cell_size;
    OwnedSampleGrid owned;
    build_sample_grid(data.data, 3, settings, owned);
    const SampleGrid g = owned.view;
    std::vector<SurfaceSample> samples(g.sample_count);
    cuda_check(cudaMemcpy(samples.data(), g.samples, samples.size() * sizeof(SurfaceSample), cudaMemcpyDeviceToHost));
    const int n0 = int(std::ceil(std::sqrt(2.f) / spacing));
    require(samples[0].position.x == 0 && samples[0].position.y == 0, "Missing triangle vertex");
    int first_count = 0;
    for (const SurfaceSample& p : samples) {
        require(p.triangle != 1, "Degenerate triangle sampled");
        require(std::abs(length(p.normal) - 1) < 1e-5f, "Invalid sample normal");
        require(p.object == (p.triangle == 0 ? 0 : 1), "Incorrect sample material");
        if (p.triangle == 0) {
            ++first_count;
            require(p.position.x >= 0 && p.position.y >= 0 &&
                    p.position.x + p.position.y <= 1.000001f && p.position.z == 0,
                    "Sample outside triangle");
        }
    }
    require(first_count == (n0 + 1) * (n0 + 2) / 2, "Unexpected surface sample count");
    std::vector<Ray> rays;
    std::mt19937 random(12345);
    std::uniform_real_distribution<float> coord(-1.5f, 2.f);
    for (int i = 0; i < 12000; ++i) {
        const Vec3 origin(coord(random), coord(random), coord(random));
        Vec3 direction(coord(random), coord(random), coord(random));
        if (i % 4 == 0) direction = Vec3(0, 0, i % 8 ? 1 : -1);
        if (i % 4 == 1) direction = Vec3(0, i % 8 ? 1 : -1, 0);
        if (i % 4 == 2) direction = Vec3(i % 8 ? 1 : -1, 0, 0);
        rays.push_back({origin, normalize(direction)});
    }
    // Origins on grid faces, edges and corners, including rays staying on them.
    for (int axis = 0; axis < 3; ++axis)
        for (int sign : {-1, 1}) {
            Vec3 d; if (axis == 0) d.x = sign; if (axis == 1) d.y = sign; if (axis == 2) d.z = sign;
            for (int k = 0; k <= 6; ++k)
                rays.push_back({g.lo + Vec3(k * cell_size, k * cell_size, k * cell_size), d});
        }
    // Rays exactly R from vertices and just inside/outside the radius.
    for (const SurfaceSample& p : samples) {
        for (float factor : {.9999f, 1.f, 1.0001f})
            rays.push_back({p.position + Vec3(radius * factor, 0, -2), Vec3(0, 0, 1)});
    }
    DeviceBuffer<Ray> device_rays(rays.size());
    DeviceBuffer<int> failures(1);
    cuda_check(cudaMemcpy(device_rays.data, rays.data(), rays.size() * sizeof(Ray), cudaMemcpyHostToDevice));
    cuda_check(cudaMemset(failures.data, 0, sizeof(int)));
    compare_searches<<<unsigned((rays.size() + 255) / 256), 256>>>(g, device_rays.data, int(rays.size()), failures.data);
    int failed;
    cuda_check(cudaMemcpy(&failed, failures.data, sizeof(int), cudaMemcpyDeviceToHost));
    require(failed == 0, "Grid search differs from exhaustive sample search");
}

int main() try {
    candidate_tests();
    grid_tests(.13f, .07f, .2f);
    grid_tests(.13f, .23f, .1f); // R spans more than one cube in each direction
    grid_tests(.13f, .003f, .04f); // R much smaller than D and cube size
    OwnedSampleGrid empty;
    RenderSettings settings{};
    build_sample_grid(nullptr, 0, settings, empty);
    require(empty.view.sample_count == 0, "Empty grid has samples");
    std::cout << "Sample grid tests passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n'; return 1;
}
