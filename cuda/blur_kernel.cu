#include "../include/cuda_processing.h"
#include <cuda_runtime.h>

__global__ void gaussianBlurGlobalKernel(const uint8_t* __restrict__ input, uint8_t* __restrict__ output,
                                         int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const int kernel[3][3] = {
            {1, 2, 1},
            {2, 4, 2},
            {1, 2, 1}
        };

        int sum = 0;
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                int ny = min(max(y + ky, 0), height - 1);
                int nx = min(max(x + kx, 0), width - 1);
                sum += input[ny * width + nx] * kernel[ky + 1][kx + 1];
            }
        }
        output[y * width + x] = static_cast<uint8_t>(sum / 16);
    }
}

void launchBlurGlobalKernel(const uint8_t* d_input, uint8_t* d_output, int width, int height) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    gaussianBlurGlobalKernel<<<grid, block>>>(d_input, d_output, width, height);
}