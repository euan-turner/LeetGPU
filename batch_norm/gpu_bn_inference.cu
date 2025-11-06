#include "solve.h"
#include <cuda_runtime.h>
#include <iostream>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define FULL_MASK 0xffffffffu
constexpr int THREADS_PER_WARP = 32;

__global__ void gpu_bn_inference_nchw(
    const float* __restrict__ x,      // [N,C,H,W]
    const float* __restrict__ gamma,  // [C]
    const float* __restrict__ beta,   // [C]
    float* __restrict__ y,            // [N,C,H,W]
    int N, int C, int H, int W,
    float eps,
    int tile_size)
{
    __shared__ float mean;        
    __shared__ float sum_x_sq;
    __shared__ float var;

    extern __shared__ float smem_raw[];
    float* smem_x = smem_raw;
    float* smem_x_sq = &smem_raw[tile_size];
    int total_blocks = gridDim.x;

    

    for (int c = blockIdx.x; c < C; c += total_blocks) 
    {
        int tid = threadIdx.x;

        if (tid == 0) {
            mean = 0.0f;
            sum_x_sq = 0.0f;
            var = 0.0f;
        }
        __syncthreads();

        int nhw = N * H * W;
        int hw = H * W;
        int chw = C * H * W;
        int num_tiles = (hw + tile_size - 1) / tile_size;

        // 1. Calculate channel mean and sum_x_sq in a single pass
        for (int n = 0; n < N; ++n) {
            int base = n * chw + c * hw;
            for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
                int tile_offset = tile_idx * tile_size;
                
                // Load tile into smem with zero padding
                for (int i = tid; i < tile_size; i += blockDim.x) {
                    int global_idx = tile_offset + i;
                    float val = (global_idx < hw) ? x[base + global_idx] : 0.0f;
                    smem_x[i] = val;
                    smem_x_sq[i] = val * val;
                }
                __syncthreads();

                // Tree reduction to 32 elements
                for (int stride = tile_size / 2; stride >= 32; stride >>= 1) {
                    for (int lane = tid; lane < stride; lane += blockDim.x) {
                        smem_x[lane] += smem_x[lane + stride];
                        smem_x_sq[lane] += smem_x_sq[lane + stride];
                    }
                    __syncthreads();
                }

                // Warp reduction
                if (tid < THREADS_PER_WARP) {
                    float reg_x = smem_x[tid];
                    float reg_x_sq = smem_x_sq[tid];

                    for (int stride = 16; stride > 0; stride >>= 1) {
                        reg_x += __shfl_xor_sync(FULL_MASK, reg_x, stride);
                        reg_x_sq += __shfl_xor_sync(FULL_MASK, reg_x_sq, stride);
                    }
                    if (tid == 0) {
                        mean += (reg_x / nhw);
                        sum_x_sq += (reg_x_sq / nhw);
                    }
                }
                __syncthreads();
            }
        }

        if (tid == 0) {
            // E[X^2] - (E[X])^2
            var = sum_x_sq - (mean * mean);
        }
        __syncthreads(); 

        float ch_gamma = gamma[c];
        float ch_beta = beta[c];
        float denom = sqrtf(var + eps);

        // 2. Apply normalization 
        for (int n = 0; n < N; ++n) {
            int base = n * chw + c * hw;
            for (int i = tid; i < hw; i += blockDim.x) {
                y[base + i] = ch_gamma * (x[base + i] - mean) / denom + ch_beta;
            }
        }
    } // End of grid-striding loop
}

// Launch with 256 threads/block
// Parallelising over (n,c)
__global__ void gpu_bn_inference_nchw_v2(
    const float* __restrict__ x,                // [N,C,H,W]
    const float* __restrict__ gamma,            // [C]
    const float* __restrict__ beta,             // [C]
    float* __restrict__ y,                      // [N,C,H,W]
    float* __restrict__ cn_sums,                // [C,N] - partial sums
    float* __restrict__ cn_sq_sums,             // [C,N] - partial sum of squares
    float* __restrict__ final_mean,             // [C] - computed mean per channel
    float* __restrict__ final_var,              // [C] - computed variance per channel
    int N, int C, int H, int W,
    float eps,
    int tile_size)
{
    extern __shared__ float smem_raw[]; // tile_size * 2
    float* smem_x = smem_raw;
    float* smem_x_sq = &smem_raw[tile_size];
    __shared__ float sum;
    __shared__ float sum_sq;

    // Get cooperative grid group for synchronization
    cg::grid_group grid = cg::this_grid();


    int nhw = N * H * W;
    int chw = C * H * W;
    int hw = H * W;

    int num_tiles = (hw + tile_size - 1) / tile_size;

    int tid = threadIdx.x;
    

    // Grid-stride loop over channels
    int n = blockIdx.y;
    int max_iters = (C + gridDim.x - 1) / gridDim.x;
    // All channels this thread block is responsible for
    for (int iter = 0; iter < max_iters; ++iter) {
        int c = blockIdx.x + iter * gridDim.x;
        int base = n * chw + c * hw;

        if (tid == 0) {
            sum = 0.0f;
            sum_sq = 0.0f;
        }
        __syncthreads();

        if (c < C) {
    
            // ============================================================
            // PHASE 1: Local reduction for this (n,c) block
            // ============================================================
            // - Reduce over H*W spatial dimensions for this (n,c) pair
            // - Compute sum and sum_sq for this block's slice
            // - Store results in global_partials[n*C + c] and global_partials_sq[n*C + c]
            
            for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
                int tile_offset = tile_idx * tile_size;

                // load tile into smem with zero padding
                for (int i = tid; i < tile_size; i += blockDim.x) {
                    int global_idx = tile_offset + i;
                    float val = (global_idx < hw) ? x[base + global_idx] : 0.0f;
                    smem_x[i] = val;
                    smem_x_sq[i] = val * val;
                }
                __syncthreads();

                // tree reduction into 32 elements
                for (int stride = tile_size / 2; stride >= 32; stride >>= 1) {
                    for (int lane = tid; lane < stride; lane += blockDim.x) {
                        smem_x[lane] += smem_x[lane + stride];
                        smem_x_sq[lane] += smem_x_sq[lane + stride];
                    }
                    __syncthreads();
                }

                // warp shuffle
                if (tid < THREADS_PER_WARP) {
                    float reg_x = smem_x[tid];
                    float reg_x_sq = smem_x_sq[tid];

                    for (int stride = 16; stride > 0; stride >>= 1) {
                        reg_x += __shfl_xor_sync(FULL_MASK, reg_x, stride);
                        reg_x_sq += __shfl_xor_sync(FULL_MASK, reg_x_sq, stride);
                    }
                    if (tid == 0) {
                        sum += reg_x;
                        sum_sq += reg_x_sq;
                    }
                }
                __syncthreads();
            }
            if (tid == 0) {
                cn_sums[c * N + n] = sum;
                cn_sq_sums[c * N + n] = sum_sq;
            }
        }

        grid.sync();  // Wait for all blocks to finish Phase 1
        
        // ============================================================
        // PHASE 2: Cross-batch reduction (one block per channel)
        // ============================================================
        if (c < C && blockIdx.y == 0) {
            // - Reduce across N dimension: sum all cn_sums[c * N + i] for i in [0,N)
            // - Compute final mean and variance for channel c
            // - Store in final_mean[c] and final_var[c]

            // Only the first warp (tid < 32) participates.
            if (tid < 32) { // THREADS_PER_WARP
                
                // --- 1. Load Data ---
                // Each of the first N threads loads one value.
                // Threads (tid >= N) load 0.0, which is correct for the sum.
                float reg_sum = 0.0f;
                float reg_sum_sq = 0.0f;
                int c_base = c * N;

                if (tid < N) {
                    reg_sum = cn_sums[c_base + tid];
                    reg_sum_sq = cn_sq_sums[c_base + tid];
                }

                // --- 2. Warp-wide Reduction ---
                // Sum the 32 registers (N of which have data)
                for (int stride = 16; stride > 0; stride >>= 1) { // 16, 8, 4, 2, 1
                    reg_sum += __shfl_xor_sync(FULL_MASK, reg_sum, stride);
                    reg_sum_sq += __shfl_xor_sync(FULL_MASK, reg_sum_sq, stride);
                }

                // --- 3. Final Calculation ---
                // Thread 0 has the final sum and writes the result.
                if (tid == 0) {
                    float mean = reg_sum / nhw;
                    float mean_sq = reg_sum_sq / nhw;
                    float var = mean_sq - mean * mean;
                    final_mean[c] = mean;
                    final_var[c] = var;
                }
            }
        }
        
        grid.sync();  // Wait for statistics computation
        
        // ============================================================
        // PHASE 3: Apply normalization
        // ============================================================
        // - All blocks read final_mean[c] and final_var[c]
        // - Apply normalization: y = gamma * (x - mean) / sqrt(var + eps) + beta
        // - Each block processes its (n,c) spatial slice
        if (c < C) {
            float ch_gamma = gamma[c];
            float ch_beta = beta[c];
            float denom = sqrtf(final_var[c] + eps);
            float mean = final_mean[c];
            for (int i = tid; i < hw; i += blockDim.x) {
                y[base + i] = ch_gamma * (x[base + i] - mean) / denom + ch_beta;
            }
        }
    }
}

void solve(const float* input, const float* gamma, const float* beta,
           float* output, int N, int C, int H, int W, float eps)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int num_blocks = prop.multiProcessorCount * 4;
    dim3 grid(num_blocks);
    dim3 block(1024);

    int max_smem_floats = prop.sharedMemPerBlock / sizeof(float);
    
    // 2 floats (x, x^2) per element in the tile
    // So max tile size is half the total floats that fit
    int max_tile_size_floats = max_smem_floats / 2;

    int tile_size = 1;
    // Find nearest power of 2 that fits
    while (tile_size * 2 <= max_tile_size_floats) {
        tile_size <<= 1;
    }
    // Handle cases where max_smem_floats is small (e.g. < 4)
    if (tile_size == 0) tile_size = 1; 

    size_t shBytes = tile_size * 2 * sizeof(float);

    gpu_bn_inference_nchw<<<grid, block, shBytes>>>(
        input, gamma, beta, output, N, C, H, W, eps, tile_size);
    cudaDeviceSynchronize();
}

// void solve(const float* input, const float* gamma, const float* beta,
//            float* output, int N, int C, int H, int W, float eps)
// {
//     cudaDeviceProp prop;
//     cudaGetDeviceProperties(&prop, 0);

//     // Thread block configuration
//     int block_size = 256;
//     dim3 block(block_size);

//     // Calculate tile size for shared memory
//     int max_smem_floats = prop.sharedMemPerBlock / sizeof(float);
//     int max_tile_size_floats = max_smem_floats / 2;
//     int tile_size = 1;
//     while (tile_size * 2 <= max_tile_size_floats) {
//         tile_size <<= 1;
//     }
//     if (tile_size == 0) tile_size = 1;
//     size_t shBytes = tile_size * 2 * sizeof(float);

//     // Determine max blocks that can run concurrently
//     int numBlocksPerSm;
//     cudaOccupancyMaxActiveBlocksPerMultiprocessor(
//         &numBlocksPerSm,
//         gpu_bn_inference_nchw_v2,
//         block_size,
//         shBytes
//     );
//     int maxActiveBlocks = numBlocksPerSm * prop.multiProcessorCount;

//     // Grid configuration: balance between C and N
//     // Reserve enough blocks for N dimension, distribute rest over C
//     int num_blocks_c = std::min(C, maxActiveBlocks / N);
//     if (num_blocks_c < 1) num_blocks_c = 1;
    
//     dim3 grid(num_blocks_c, N);

//     // Allocate temporary storage
//     float *d_cn_sums, *d_cn_sq_sums, *d_final_mean, *d_final_var;
//     cudaMalloc(&d_cn_sums, C * N * sizeof(float));
//     cudaMalloc(&d_cn_sq_sums, C * N * sizeof(float));
//     cudaMalloc(&d_final_mean, C * sizeof(float));
//     cudaMalloc(&d_final_var, C * sizeof(float));

//     // Prepare kernel arguments
//     void* args[] = {
//         (void*)&input,
//         (void*)&gamma,
//         (void*)&beta,
//         (void*)&output,
//         (void*)&d_cn_sums,
//         (void*)&d_cn_sq_sums,
//         (void*)&d_final_mean,
//         (void*)&d_final_var,
//         (void*)&N,
//         (void*)&C,
//         (void*)&H,
//         (void*)&W,
//         (void*)&eps,
//         (void*)&tile_size
//     };

//     // Launch cooperative kernel
//     cudaError_t err = cudaLaunchCooperativeKernel(
//         (void*)gpu_bn_inference_nchw_v2,
//         grid,
//         block,
//         args,
//         shBytes,
//         0  // stream
//     );

//     cudaDeviceSynchronize();

//     // Cleanup temporary storage
//     cudaFree(d_cn_sums);
//     cudaFree(d_cn_sq_sums);
//     cudaFree(d_final_mean);
//     cudaFree(d_final_var);
// }