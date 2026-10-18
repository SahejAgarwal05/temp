NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -gencode arch=compute_90,code=sm_90 -gencode arch=compute_80,code=sm_80 -Xcompiler -O3

HEADERS := geometry.cuh graphics.cuh utils.cuh sampled_grid.cuh sampled_config.cuh

.DEFAULT_GOAL := path_tracing
.PHONY: all clean test
all: cpu_path_tracing path_tracing

cpu_path_tracing: main.cu cpu_path_tracing.cu $(HEADERS)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -Xcompiler -fopenmp main.cu cpu_path_tracing.cu -o $@

path_tracing: main.cu path_tracing.cu $(HEADERS)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) main.cu path_tracing.cu -o $@

sampled_grid_test: tests/sampled_grid_test.cu path_tracing.cu $(HEADERS)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) tests/sampled_grid_test.cu -o $@

test: sampled_grid_test
	./sampled_grid_test

clean:
	rm -f cpu_path_tracing path_tracing sampled_grid_test
