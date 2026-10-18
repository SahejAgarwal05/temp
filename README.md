CUDA path tracer with sampled-point uniform-grid intersections (default) and the original exact LBVH backend (`--intersection bvh`).

Build and render on a CUDA machine:

```sh
make
./path_tracing scenes/cornell_box.rtscene --output sampled.ppm \
  --intersection sampled --sample-spacing 0.02 --acceptance-radius 0.02 --cube-size 0.08 \
  --correctness-check images/cornell_box.ppm
./path_tracing scenes/cornell_box.rtscene --output bvh.ppm \
  --intersection bvh --correctness-check images/cornell_box.ppm
make test
```

`--sample-spacing D`, `--acceptance-radius R`, and `--cube-size` are independent positive distances in scene units. The defaults above are starting values, **not measured or checker-certified settings**. The Makefile targets sm_80/sm_90; override `NVCCFLAGS` for other GPUs.

The GPU subdivides each nondegenerate triangle into a triangular lattice whose neighbor spacing is at most D, including vertices and edges. Each sample stores position, unit normal, original triangle ID, and a material-table index. GPU counts, prefix scans and atomic scatter build contiguous cube ranges of sample IDs. Every cube touching the closed `P ± R` box references that sample, including face/edge/corner contacts and radii larger than a cube.

Each ray uses DDA to visit crossed cubes, rejecting samples with `t <= EPSILON` or perpendicular squared distance greater than `R²`. It chooses the minimum **squared Euclidean distance from the origin**, with sample ID resolving ties. It keeps searching until no unseen sample can improve that distance. Hits use sample positions/normals/materials. Both backends share shading, normal orientation, hemisphere sampling, bounce limits, and exclusion of the original triangle just left. Averaging and RGB conversion also run on the GPU.

Only input packing, allocation coordination and output writing remain on the CPU. Sample attributes occupy 32 bytes each; cube references use 4 bytes each; CSR offsets use 8 bytes per cube plus a temporary 8-byte count/cursor per cube during construction. Smaller D increases sampling memory; larger R or smaller cubes increases reference duplication. Invalid sizes, excessive grids/samples and CUDA allocation failures report an error.

Compare and tune on the target GPU:

```sh
python3 scripts/tune_sample_grid.py \
  --case scenes/debug/debug_0.rtscene images/debug_0.ppm \
  --case scenes/cornell_box.rtscene images/cornell_box.ppm \
  --spacing .01,.02,.04 --radius .01,.02,.04 --cube-size .04,.08,.16 \
  --repeats 3 --output-dir grid_benchmark
```

Add `--case SCENE REFERENCE` for every scene that must pass. References need the checker's `Max error` metadata. The script runs the BVH baseline and every independent D/R/cube combination with the same seed and scene render settings. It uses the renderer's `Time taken` (including construction, rendering and readback), records checker errors and speedups in `results.csv`, and saves images/logs for each run. `best.json` selects the smallest sum of per-scene median times **only among configurations passing every case and repeat**. It reports if none pass. Copy its printed options to render with the winning configuration; a passing approximation may still be slower than the BVH.

Validation: `make test` compares actual GPU grid searches with exhaustive sample searches and checks sampling/materials, degenerates, rejection rules, ties, axis-parallel rays, cube boundaries, and R larger/smaller than cube size. `python3 tests/check_sample_grid_host.py` checks the same traversal math against exhaustive search without CUDA (requires clang++); it does not validate GPU kernels, scans, or atomics. CUDA compilation, image checks and timing require an NVIDIA/CUDA host; none were available on the development machine.
