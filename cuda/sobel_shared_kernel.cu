#include "../include/cuda_processing.h"
#include <cuda_runtime.h>
#include <cmath>

#define TILE_W 16
#define TILE_H 16
#define HALO 1

__global__ void sobelSharedKernel(const uint8_t* __restrict__ input, uint8_t* __restrict__ output,
                                  int width, int height) {
    __shared__ uint8_t tile[TILE_H + 2 * HALO][TILE_W + 2 * HALO];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_W + tx;
    int y = blockIdx.y * TILE_H + ty;

    int cx = min(max(x, 0), width - 1);
    int cy = min(max(y, 0), height - 1);
    tile[ty + HALO][tx + HALO] = input[cy * width + cx];

    if (ty < HALO) {
        int topY = max(y - HALO, 0);
        int botY = min(y + TILE_H, height - 1);
        tile[ty][tx + HALO] = input[topY * width + cx];
        tile[ty + TILE_H + HALO][tx + HALO] = input[botY * width + cx];
    }
    if (tx < HALO) {
        int leftX = max(x - HALO, 0);
        int rightX = min(x + TILE_W, width - 1);
        tile[ty + HALO][tx] = input[cy * width + leftX];
        tile[ty + HALO][tx + TILE_W + HALO] = input[cy * width + rightX];
    }
    if (tx < HALO && ty < HALO) {
        int topY = max(y - HALO, 0);
        int botY = min(y + TILE_H, height - 1);
        int leftX = max(x - HALO, 0);
        int rightX = min(x + TILE_W, width - 1);

        tile[ty][tx] = input[topY * width + leftX];
        tile[ty][tx + TILE_W + HALO] = input[topY * width + rightX];
        tile[ty + TILE_H + HALO][tx] = input[botY * width + leftX];
        tile[ty + TILE_H + HALO][tx + TILE_W + HALO] = input[botY * width + rightX];
    }

    __syncthreads();

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
            uint8_t pixel = tile[ty + HALO + ky][tx + HALO + kx];
            gx += pixel * gx_kernel[ky + 1][kx + 1];
            gy += pixel * gy_kernel[ky + 1][kx + 1];
        }
    }

    int magnitude = static_cast<int>(sqrtf(static_cast<float>(gx * gx + gy * gy)));
    output[y * width + x] = static_cast<uint8_t>(min(255, magnitude));
}

void launchSobelSharedKernel(const uint8_t* d_input, uint8_t* d_output, int width, int height) {
    dim3 block(TILE_W, TILE_H);
    dim3 grid((width + TILE_W - 1) / TILE_W, (height + TILE_H - 1) / TILE_H);
    sobelSharedKernel<<<grid, block>>>(d_input, d_output, width, height);
}