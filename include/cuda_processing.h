#ifndef CUDA_PROCESSING_H
#define CUDA_PROCESSING_H

#include <cstdint>

struct CudaTimings {
    float grayscaleMs = 0.0f;
    float blurGlobalMs = 0.0f;
    float blurSharedMs = 0.0f;
    float sobelGlobalMs = 0.0f;
    float sobelSharedMs = 0.0f;
    float totalPipelineMs = 0.0f;
};

// Lansatoare pentru fiecare kernel individual (lucreaza pe pointeri device d_*)
void launchGrayscaleKernel(const uint8_t* d_input, uint8_t* d_output, int width, int height, int channels);
void launchBlurGlobalKernel(const uint8_t* d_input, uint8_t* d_output, int width, int height);
void launchBlurSharedKernel(const uint8_t* d_input, uint8_t* d_output, int width, int height);
void launchSobelGlobalKernel(const uint8_t* d_input, uint8_t* d_output, int width, int height);
void launchSobelSharedKernel(const uint8_t* d_input, uint8_t* d_output, int width, int height);

// Ruleaza intregul pipeline GPU (alocare VRAM, transfer H2D/D2H, masurare cu cudaEvent)
CudaTimings runCudaPipeline(
    const uint8_t* h_input,
    uint8_t* h_gray,
    uint8_t* h_blurGlobal,
    uint8_t* h_blurShared,
    uint8_t* h_sobelGlobal,
    uint8_t* h_sobelShared,
    int width,
    int height,
    int channels
);

#endif // CUDA_PROCESSING_H