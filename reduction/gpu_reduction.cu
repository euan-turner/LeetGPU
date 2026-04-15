#include "solve.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define FULL_MASK 0xffffffffu
#define CEIL_DIV(x, y) ((x) >= 0 ? (((x) + (y) - 1) / (y)) : ((x) / (y)))
#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error: %s\n", cudaGetErrorString(e)); std::abort(); } } while(0)

constexpr int THREADS_PER_WARP = 32;

__global__ void gpu_reduction(
  const float* __restrict__ input,
  float* __restrict__ output,
  int N) {
  __shared__ float warp_sums[32];

  const int tid = threadIdx.x;
  if (tid < 32) {
    warp_sums[tid] = 0.0f;
  }
  __syncthreads();

  const int wid = tid / THREADS_PER_WARP;
  const int lid = tid % THREADS_PER_WARP;

  // TODO: pipeline/unroll/vectorise based on register pressure
  // 1. Accumulate in threads
  float sum = 0.0f;
  for (int i = tid; i < N; i += blockDim.x) {
    sum += input[i];
  }

  // 2. Accumulate across each warp
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(FULL_MASK, sum, offset);
  }

  // 3. First warp accumulates shared results
  if (lid == 0) {
    warp_sums[wid] = sum;
  }

  __syncthreads();

  if (wid == 0) {
    float res = warp_sums[lid];
    for (int offset = 16; offset > 0; offset /= 2) {
      res += __shfl_down_sync(FULL_MASK, res, offset);
    }
    if (lid == 0) {
      *output = res;
    }
  }
}

// Sums input[chunk_start:chunk_start + chunk_len] to output[blockIdx.x]
// Chunk sizes determined by grid dim and N
// TODO: use co-operative groups to handle second kernel
__global__ void gpu_reduction_large(
  const float* __restrict__ input,
  float* __restrict__ block_outputs,
  float* __restrict__ output, // [gridDim.x]
  int N) {
  __shared__ float warp_sums[32];

  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int num_blocks = gridDim.x;
  const int wid = tid / THREADS_PER_WARP;
  const int lid = tid % THREADS_PER_WARP;

  const int base_blocks = N % num_blocks;

  cg::grid_group grid = cg::this_grid();

  int chunk_start;
  int chunk_len;
  if (bid < base_blocks) {
    // sums chunk_len + 1
    chunk_len = (N / num_blocks) + 1;
    chunk_start  = chunk_len * bid;
  } else {
    // sums chunk_len
    chunk_len = N / num_blocks;
    chunk_start = base_blocks * (chunk_len + 1) + (bid - base_blocks) * chunk_len;
  }

  // TODO: pipeline/unroll/vectorise based on register pressure
  // 1. Accumulate in threads
  float sum = 0.0f;
  // TODO: check the boundary case with N
  for (int i = chunk_start + tid; i < chunk_start + chunk_len; i += blockDim.x) {
    sum += input[i];
  }

  // 2. Accumulate across each warp
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(FULL_MASK, sum, offset);
  }

  // 3. First warp accumulates shared results
  if (lid == 0) {
    warp_sums[wid] = sum;
  }

  __syncthreads();

  if (wid == 0) {
    float res = warp_sums[lid];
    for (int offset = 16; offset > 0; offset /= 2) {
      res += __shfl_down_sync(FULL_MASK, res, offset);
    }
    if (lid == 0) {
      block_outputs[bid] = res;
    }
  }

  grid.sync();

  // 4. First block accumulates block_outputs
  cg::thread_block block = cg::this_thread_block();

  if (
    block.group_index().x == 0 && 
    block.group_index().y == 0 && 
    block.group_index().z == 0) {
      if (tid < 32) {
        warp_sums[tid] = 0.0f;
      }
      block.sync();

      // Accumulate in threads
      sum = 0.0f;
      for (int i = tid; i < grid.dim_blocks().x; i += block.num_threads()) {
        sum += block_outputs[i];
      }

      // Accumulate across each warp
      for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
      }

      // First warp accumulates shared results
      if (lid == 0) {
        warp_sums[wid] = sum;
      }

      block.sync();

      if (wid == 0) {
        float res = warp_sums[lid];
        for (int offset = 16; offset > 0; offset /= 2) {
          res += __shfl_down_sync(FULL_MASK, res, offset);
        }
        if (lid == 0) {
          *output = res;
        }
      }
  }
}


// Each thread in a block (32x32 = 1024) sums a subset of values (non-contiguous, offset by block size)
// Warp shuffle in each warp, then first warp sums results

// T4 has 40 SMs, 6 blocks/SM gives 240
// Launch 240 blocks to each sum chunks in parallel, then a final block to sum those
// 240 * 512 = 122880

// TODO: At what number of values to launch multiple blocks? 4194304

// if fewer inputs, launch fewer warps

void solve(const float* input, float* output, int N) {
  const int MAX_WARPS = 32;
  const int MAX_THREADS = MAX_WARPS * THREADS_PER_WARP;

  if (N < 122880) {
    // Single kernel
    int num_blocks = 1;
    dim3 grid(num_blocks);
    dim3 block(1024);

    gpu_reduction<<<grid, block>>>(input, output, N);
    cudaDeviceSynchronize();
  } else {
    // Hierarchy of kernels
    int num_blocks = 240;
    // First block computes final reduction

    float* temp;
    CUDA_CHECK(cudaMalloc(&temp, num_blocks * sizeof(float)));

    dim3 grid(num_blocks);
    dim3 block(512);


    void* args[] = {
      (void*)&input,
      (void*)&temp,
      (void*)&output,
      (void*)&N,
    };

    cudaError_t err = cudaLaunchCooperativeKernel(
      (void*)gpu_reduction_large,
      grid,
      block,
      args,
      0,
      0
    );
    CUDA_CHECK(err);
    cudaDeviceSynchronize();
    cudaFree(temp);
  }
}
