#include "../include/cpu_processing.h"
#include <cmath>
#include <algorithm>

void rgbToGrayscaleCPU(const uint8_t* input, uint8_t* output, int width, int height, int channels) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx_in = (y * width + x) * channels;
            int idx_out = y * width + x;

            uint8_t r = input[idx_in];
            uint8_t g = (channels >= 2) ? input[idx_in + 1] : r;
            uint8_t b = (channels >= 3) ? input[idx_in + 2] : r;

            float gray = 0.299f * r + 0.587f * g + 0.114f * b;
            output[idx_out] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, gray)));
        }
    }
}

void gaussianBlurCPU(const uint8_t* input, uint8_t* output, int width, int height) {
    const int kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int sum = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int ny = std::min(std::max(y + ky, 0), height - 1);
                    int nx = std::min(std::max(x + kx, 0), width - 1);
                    sum += input[ny * width + nx] * kernel[ky + 1][kx + 1];
                }
            }
            output[y * width + x] = static_cast<uint8_t>(sum / 16);
        }
    }
}

void sobelEdgeCPU(const uint8_t* input, uint8_t* output, int width, int height) {
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

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (y == 0 || y == height - 1 || x == 0 || x == width - 1) {
                output[y * width + x] = 0;
                continue;
            }

            int gx = 0, gy = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    uint8_t pixel = input[(y + ky) * width + (x + kx)];
                    gx += pixel * gx_kernel[ky + 1][kx + 1];
                    gy += pixel * gy_kernel[ky + 1][kx + 1];
                }
            }

            int magnitude = static_cast<int>(std::sqrt(static_cast<float>(gx * gx + gy * gy)));
            output[y * width + x] = static_cast<uint8_t>(std::min(255, magnitude));
        }
    }
}