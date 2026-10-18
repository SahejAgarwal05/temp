NVCC ?= nvcc
NVCCFLAGS ?= -g -std=c++17 -lineinfo -gencode arch=compute_90,code=sm_90 -gencode arch=compute_80,code=sm_80 -Xcompiler -O3

HEADERS := geometry.cuh graphics.cuh utils.cuh

.DEFAULT_GOAL := path_tracing
.PHONY: all clean
all: cpu_path_tracing path_tracing

cpu_path_tracing: main.cu cpu_path_tracing.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp main.cu cpu_path_tracing.cu -o $@

path_tracing: main.cu path_tracing.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) main.cu path_tracing.cu -o $@

clean:
	rm -f cpu_path_tracing path_tracing
