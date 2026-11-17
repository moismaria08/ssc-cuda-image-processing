#include "../include/cuda_processing.h"
#include <cuda_runtime.h>

__global__ void rgbToGrayscaleKernel(const uint8_t* __restrict__ input, uint8_t* __restrict__ output,
                                     int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx_in = (y * width + x) * channels;
        int idx_out = y * width + x;

        uint8_t r = input[idx_in];
        uint8_t g = (channels >= 2) ? input[idx_in + 1] : r;
        uint8_t b = (channels >= 3) ? input[idx_in + 2] : r;

        float gray = 0.299f * r + 0.587f * g + 0.114f * b;
        output[idx_out] = static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, gray)));
    }
}

void launchGrayscaleKernel(const uint8_t* d_input, uint8_t* d_output, int width, int height, int channels) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    rgbToGrayscaleKernel<<<grid, block>>>(d_input, d_output, width, height, channels);
}