#include "solve.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <nvtx3/nvToolsExt.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <string>
#include <random>
#include <iostream>
#include <cinttypes>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error: %s\n", cudaGetErrorString(e)); std::abort(); } } while(0)

#define CURAND_CHECK(x) do { curandStatus_t s=(x); if(s!=CURAND_STATUS_SUCCESS){ \
  std::fprintf(stderr,"cuRAND error: %d\n", int(s)); std::abort(); } } while(0)

void cpu_reduction_ref(const std::vector<float>& xs, float& y);

struct CaseResult {
    const char* name;
    bool pass = false;
    bool skipped = false;
    std::string note;
};

struct BenchCfg {
    const char* name;
    int N;
    float lo;
    float hi;
    float rtol;
};

__global__ void scale_to_range(float* data, size_t n, float lo, float range) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = lo + range * data[idx];
}

static void fill_random_device_uniform(float* dptr, size_t n, unsigned long long seed,
                                       float lo, float hi)
{
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CURAND_CHECK(curandGenerateUniform(gen, dptr, (size_t)n));
    CURAND_CHECK(curandDestroyGenerator(gen));

    const int threads = 256;
    const size_t blocks = (n + threads - 1) / threads;
    scale_to_range<<<(int)blocks, threads>>>(dptr, n, lo, hi - lo);
    CUDA_CHECK(cudaDeviceSynchronize());
}

static bool check_correctness(float* dX, float* dY, int N, float rtol) {

    // 1. Copy device inputs to host
    std::vector<float> hX(N);
    CUDA_CHECK(cudaMemcpy(hX.data(), dX, N * sizeof(float), cudaMemcpyDeviceToHost));

    // 2. Run CPU reference
    float cpu_ref;
    cpu_reduction_ref(hX, cpu_ref);

    // 3. Run GPU solve()
    solve(dX, dY, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4. Copy device output to host
    float gpu_out;
    CUDA_CHECK(cudaMemcpy(&gpu_out, dY, sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "CPU: " << cpu_ref << " GPU: " << gpu_out << std::endl;

    // 5. Compare element-wise
    if (!std::isfinite(gpu_out) || std::fabs(gpu_out - cpu_ref) >  rtol * max(std::fabs(gpu_out), std::fabs(cpu_ref))) return false;

    return true;
}

static float time_kernel(float *dX, float *dY, int N, int warmup, int iters, const char* cfg_name) {
    for (int i=0;i<warmup;++i) solve(dX, dY, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    const int profiled_iter = 0;
    for (int i=0;i<iters;++i) {
        if (i == profiled_iter) {
            nvtxRangePushA("steady_state");
            nvtxRangePushA(cfg_name);
            solve(dX, dY, N);
            nvtxRangePop();
            nvtxRangePop();
        } else {
            solve(dX, dY, N);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms=0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / (float)iters;
}


int main() {
    const int warmup = 5;
    const int iters  = 10;
    const unsigned long long seed_all = 42ULL;

    std::vector<BenchCfg> cfgs = {
        // {"all_zeros", 1024, 0.0f, 0.0f},
        // {"all_ones", 1024, 1.0f, 1.0f},
        // {"non_power_of_two", 5, 0.0f, 1.0f},
        {"large_random", 10000, -1000.0f, 1000.0f, 5e-4f},
        {"large_random_2", 15000000, 0.0f, 1000.0f, 1e-3f},
        {"perf", 4194304, 0.0f, 1000.0f, 5e-4f},
    };

    std::vector<CaseResult> results;

    for (const auto& cfg : cfgs) {
        size_t N = (size_t)cfg.N;

        // 1. Allocate device memory
        float *dX = nullptr, *dY = nullptr;
        CUDA_CHECK(cudaMalloc(&dX, N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dY, sizeof(float)));

        // 2. Fill with random data
        fill_random_device_uniform(dX, N, seed_all, cfg.lo, cfg.hi);
    
        // 3. Check correctness
        bool ok = check_correctness(dX, dY, N, cfg.rtol);
        results.push_back(CaseResult{
            cfg.name, ok, false, ok ? "" : "mismatch vs CPU reference"
        });

        // 4. Time kernel
        float avg_ms = time_kernel(dX, dY, N, warmup, iters, cfg.name);

        // 5. Free device memory
        cudaFree(dX); cudaFree(dY);
    }

    std::printf("\n==== Summary ====\n");
    int pass_cnt = 0;
    for (const auto& r : results) {
        if (r.skipped) {
            std::printf("[SKIP] %s  (%s)\n", r.name, r.note.c_str());
        } else if (r.pass) {
            std::printf("[PASS] %s\n", r.name);
            ++pass_cnt;
        } else {
            std::printf("[FAIL] %s  (%s)\n", r.name, r.note.c_str());
        }
    }
    std::printf("Passed: %d  Total: %zu\n", pass_cnt, results.size());
    return 0;
}
