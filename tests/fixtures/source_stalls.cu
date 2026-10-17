// Small, deterministic SourceCounters target; not a throughput benchmark.
// Build with a toolkit supporting the target, e.g. -O2 -lineinfo -arch=sm_70.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); \
        std::exit(1); \
    } \
} while (0)

__global__ void source_stalls(const volatile uint32_t* next, uint32_t* output,
                              unsigned steps) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t index = tid;
    for (unsigned step = 0; step < steps; ++step) {
        index = next[index];  // SAMPLE_TARGET: dependent global-memory load
    }
    output[tid] = index;
}

int main() {
    constexpr unsigned count = 1u << 22;
    constexpr unsigned stride = 104729;
    constexpr unsigned threads = 128 * 256;
    constexpr unsigned steps = 1024;
    std::vector<uint32_t> links(count), result(threads);
    for (unsigned i = 0; i < count; ++i) links[i] = (i + stride) & (count - 1);
    uint32_t *device_links = nullptr, *device_result = nullptr;
    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    std::printf("GPU: %s; compute capability: %d.%d\n",
                properties.name, properties.major, properties.minor);
    CUDA_CHECK(cudaMalloc(&device_links, links.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&device_result, result.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(device_links, links.data(), links.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    source_stalls<<<128, 256>>>(device_links, device_result, steps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result.data(), device_result, result.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    for (unsigned tid = 0; tid < threads; ++tid) {
        uint32_t expected = static_cast<uint32_t>(
            (static_cast<uint64_t>(tid) + static_cast<uint64_t>(steps) * stride)
            & (count - 1));
        if (result[tid] != expected) {
            std::fprintf(stderr, "Mismatch at %u: %u != %u\n", tid, result[tid], expected);
            return 1;
        }
    }
    CUDA_CHECK(cudaFree(device_links));
    CUDA_CHECK(cudaFree(device_result));
    std::printf("PASS: %u chains, %u dependent loads each\n", threads, steps);
    return 0;
}
