#ifndef SAMPLED_CONFIG_CUH
#define SAMPLED_CONFIG_CUH

// Scene-unit distances. Edit these constants and rebuild with `make`.
// Optional compiler definitions let the benchmark sweep rebuild each variant.
#ifndef SAMPLED_SPACING
#define SAMPLED_SPACING 0.02f
#endif
#ifndef SAMPLED_CUBE_SIZE
#define SAMPLED_CUBE_SIZE 0.08f
#endif
struct SampleGridConfig {
    static constexpr float spacing = SAMPLED_SPACING;  // D
    static constexpr float cube_size = SAMPLED_CUBE_SIZE; // cube edge length
};
static_assert(SampleGridConfig::spacing > 0 && SampleGridConfig::spacing <= 3.402823466e38F,
              "Sample spacing must be finite and positive");
static_assert(SampleGridConfig::cube_size > 0 && SampleGridConfig::cube_size <= 3.402823466e38F,
              "Cube size must be finite and positive");
#endif
