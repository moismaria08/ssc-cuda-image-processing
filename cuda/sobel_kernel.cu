#include "../include/cuda_processing.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void sobelGlobalKernel(const uint8_t* __restrict__ input, uint8_t* __restrict__ output,
                                  int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        output[y * width + x] = 0;
        return;
    }

    const int gx_kernel[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    const int gy_kernel[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    int gx = 0;
    int gy = 0;

    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            uint8_t pixel = input[(y + ky) * width + (x + kx)];
            gx += pixel * gx_kernel[ky + 1][kx + 1];
            gy += pixel * gy_kernel[ky + 1][kx + 1];
        }
    }

    int magnitude = static_cast<int>(sqrtf(static_cast<float>(gx * gx + gy * gy)));
    output[y * width + x] = static_cast<uint8_t>(min(255, magnitude));
}

void launchSobelGlobalKernel(const uint8_t* d_input, uint8_t* d_output, int width, int height) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    sobelGlobalKernel<<<grid, block>>>(d_input, d_output, width, height);
}