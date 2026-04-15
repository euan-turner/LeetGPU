#!/usr/bin/env bash
# Scaffold a new GPU kernel problem directory with test harness.
# Usage: ./new_problem.sh <problem_name> [kernel_name]
#   problem_name: snake_case name for the directory and file prefix (e.g. "softmax", "mat_mul")
#   kernel_name:  optional GPU kernel function name (defaults to "gpu_<problem_name>")

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <problem_name> [kernel_name]"
    echo "  e.g. $0 softmax"
    echo "  e.g. $0 mat_mul gpu_matmul_tiled"
    exit 1
fi

PROBLEM="$1"
KERNEL_NAME="${2:-gpu_${PROBLEM}}"
DIR="$(cd "$(dirname "$0")" && pwd)/${PROBLEM}"

if [ -d "$DIR" ]; then
    echo "Error: directory '$DIR' already exists."
    exit 1
fi

mkdir -p "$DIR"

# --- solve.h ---
cat > "$DIR/solve.h" << 'SOLVEEOF'
#pragma once

// TODO: Define the solve() signature for this problem.
// void solve(...);
SOLVEEOF

# --- cpu_<problem>_ref.cu ---
cat > "$DIR/cpu_${PROBLEM}_ref.cu" << CPUEOF
#include <vector>
#include <cmath>

// TODO: Implement CPU reference.
// This function is called by the test harness to produce ground-truth outputs.
// Match the naming convention: cpu_${PROBLEM}_ref(...)
//
// void cpu_${PROBLEM}_ref(...) {
// }
CPUEOF

# --- gpu_<problem>.cu ---
cat > "$DIR/gpu_${PROBLEM}.cu" << GPUEOF
#include "solve.h"
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffffu
constexpr int THREADS_PER_WARP = 32;

// TODO: Implement GPU kernel.
// __global__ void ${KERNEL_NAME}(...) {
// }

// TODO: Implement solve() matching the signature in solve.h.
// void solve(...) {
// }
GPUEOF

# --- test_<problem>.cu ---
cat > "$DIR/test_${PROBLEM}.cu" << 'TESTEOF'
#include "solve.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <string>
#include <random>
#include <cinttypes>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error: %s\n", cudaGetErrorString(e)); std::abort(); } } while(0)

#define CURAND_CHECK(x) do { curandStatus_t s=(x); if(s!=CURAND_STATUS_SUCCESS){ \
  std::fprintf(stderr,"cuRAND error: %d\n", int(s)); std::abort(); } } while(0)

// TODO: Declare CPU reference function (defined in cpu_*_ref.cu).
// void cpu_${PROBLEM}_ref(...);

struct CaseResult {
    const char* name;
    bool pass = false;
    bool skipped = false;
    std::string note;
};

// TODO: Define a config struct for benchmark cases.
struct BenchCfg {
    const char* name;
    // TODO: problem-specific dimensions
};

static void fill_random_device_uniform(float* dptr, size_t n, unsigned long long seed)
{
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CURAND_CHECK(curandGenerateUniform(gen, dptr, (size_t)n));
    CURAND_CHECK(curandDestroyGenerator(gen));
}

// TODO: Adapt arguments to match solve() signature.
static bool check_correctness(/* device ptrs, dims */) {
    // 1. Copy device inputs to host
    // 2. Run CPU reference
    // 3. Run GPU solve()
    // CUDA_CHECK(cudaDeviceSynchronize());
    // 4. Copy device output to host
    // 5. Compare element-wise
    // const float atol = 1e-5f, rtol = 1e-5f;
    // for (size_t i=0;i<total;++i) {
    //     float a = out[i], b = ref[i];
    //     if (!std::isfinite(a) || std::fabs(a - b) > atol + rtol * std::fabs(b)) return false;
    // }
    return true;
}

// TODO: Adapt arguments to match solve() signature.
static float time_kernel(/* device ptrs, dims, */ int warmup, int iters) {
    // for (int i=0;i<warmup;++i) solve(...);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    // for (int i=0;i<iters;++i) solve(...);
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

    // TODO: Define benchmark configs.
    std::vector<BenchCfg> cfgs = {
        // {"case name", ...dims...},
    };

    std::vector<CaseResult> results;

    for (const auto& cfg : cfgs) {
        // TODO: Compute sizes from cfg dimensions.
        // size_t in_elems = ...;
        // size_t out_elems = ...;

        // 1. Allocate device memory
        // float *dX=nullptr, *dY=nullptr;
        // CUDA_CHECK(cudaMalloc(&dX, in_elems*sizeof(float)));
        // CUDA_CHECK(cudaMalloc(&dY, out_elems*sizeof(float)));

        // 2. Fill with random data
        // fill_random_device_uniform(dX, in_elems, seed_all);

        // 3. Check correctness
        // bool ok = check_correctness(...);
        // results.push_back({cfg.name, ok, false, ok ? "" : "mismatch vs CPU reference"});

        // 4. Time kernel
        // float avg_ms = time_kernel(..., warmup, iters);
        // std::printf("\n==== Benchmark: %s ====\n", cfg.name);
        // std::printf("Avg time per iter: %.3f ms\n", avg_ms);

        // 5. Free device memory
        // cudaFree(dX); cudaFree(dY);
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
TESTEOF

# --- test_script.sh ---
cat > "$DIR/test_script.sh" << SCRIPTEOF
nvcc -O2 -std=c++14 -lcurand -lineinfo -arch=sm_89 gpu_${PROBLEM}.cu cpu_${PROBLEM}_ref.cu test_${PROBLEM}.cu -o test_${PROBLEM}
./test_${PROBLEM}

nsys profile -w true -t cuda,nvtx,osrt --force-overwrite=true --stats=true --gpu-metrics-device=0 -x true -o test_${PROBLEM}_nsys_profile ./test_${PROBLEM}
ncu -f -o test_${PROBLEM}_ncu_profile --kernel-name ${KERNEL_NAME} --launch-count 1 --set full --cache-control none --import-source true --target-processes all ./test_${PROBLEM}
SCRIPTEOF

echo "Created problem scaffold in: $DIR/"
echo "  solve.h"
echo "  cpu_${PROBLEM}_ref.cu"
echo "  gpu_${PROBLEM}.cu"
echo "  test_${PROBLEM}.cu"
echo "  test_script.sh"
echo ""
echo "Next steps:"
echo "  1. Define solve() signature in solve.h"
echo "  2. Implement CPU reference in cpu_${PROBLEM}_ref.cu"
echo "  3. Fill in test configs, check_correctness(), and time_kernel() in test_${PROBLEM}.cu"
echo "  4. Implement GPU kernel in gpu_${PROBLEM}.cu"
echo "  5. Run: cd ${PROBLEM} && bash test_script.sh"
