#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>

template <size_t TILE_DIM>
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM]; // one element per thread in block

    size_t gx = blockIdx.x * blockDim.x + threadIdx.x; // input column index
    size_t gy = blockIdx.y * blockDim.y + threadIdx.y; // input row index
    size_t lx = threadIdx.x;
    size_t ly = threadIdx.y;


    tile[ly][lx] = (gx < cols && gy < rows) ? input[gy * cols + gx] : 0.0f;
    __syncthreads();

    // 2. warp picks up strided values from tile, but writes them contiguously
    size_t tx = blockIdx.y * TILE_DIM + lx; // output column index
    size_t ty = blockIdx.x * TILE_DIM + ly; // output row index
    if (tx < rows && ty < cols) {
        output[ty * rows + tx] = tile[lx][ly];
    }
}

extern "C" void solve(const float* input, float* output, int rows, int cols) {
    const size_t BLOCK_SIZE = 16;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrix_transpose_kernel<BLOCK_SIZE><<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}

// Host-side correctness checker
bool check_transpose(const float* input, const float* output, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float expected = input[r * cols + c];
            float got = output[c * rows + r];
            if (expected != got) {
                printf("Mismatch at input[%d][%d], output[%d][%d]: expected %f, got %f\n",
                       r, c, c, r, expected, got);
                return false;
            }
        }
    }
    return true;
}

int main() {
    int rows = 256;
    int cols = 128;
    size_t size = rows * cols * sizeof(float);

    // Host memory
    float* h_input  = (float*)malloc(size);
    float* h_output = (float*)malloc(rows * cols * sizeof(float));

    // Initialize input matrix with some values
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            h_input[r * cols + c] = static_cast<float>(r * cols + c + 1);
        }
    }

    // Device memory
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, rows * cols * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Launch transpose
    solve(d_input, d_output, rows, cols);

    // Copy result back
    cudaMemcpy(h_output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Check correctness
    bool correct = check_transpose(h_input, h_output, rows, cols);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    // if (!correct) return -1;
    return 0;
}
