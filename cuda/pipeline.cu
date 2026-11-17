#include "../include/cuda_processing.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "[CUDA EROARE] " << cudaGetErrorString(err)          \
                      << " la " << __FILE__ << ":" << __LINE__ << "\n";       \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

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
) {
    CudaTimings timings;

    size_t numPixels = static_cast<size_t>(width) * height;
    size_t inputBytes = numPixels * channels * sizeof(uint8_t);
    size_t grayBytes = numPixels * sizeof(uint8_t);

    uint8_t *d_input = nullptr, *d_gray = nullptr;
    uint8_t *d_blurGlobal = nullptr, *d_blurShared = nullptr;
    uint8_t *d_sobelGlobal = nullptr, *d_sobelShared = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, inputBytes));
    CUDA_CHECK(cudaMalloc(&d_gray, grayBytes));
    CUDA_CHECK(cudaMalloc(&d_blurGlobal, grayBytes));
    CUDA_CHECK(cudaMalloc(&d_blurShared, grayBytes));
    CUDA_CHECK(cudaMalloc(&d_sobelGlobal, grayBytes));
    CUDA_CHECK(cudaMalloc(&d_sobelShared, grayBytes));

    cudaEvent_t start, stop, totalStart, totalStop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventCreate(&totalStart));
    CUDA_CHECK(cudaEventCreate(&totalStop));

    CUDA_CHECK(cudaEventRecord(totalStart));

    // Transfer Host -> Device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, inputBytes, cudaMemcpyHostToDevice));

    // 1. Grayscale
    CUDA_CHECK(cudaEventRecord(start));
    launchGrayscaleKernel(d_input, d_gray, width, height, channels);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&timings.grayscaleMs, start, stop));

    // 2A. Gaussian Blur (Global Memory)
    CUDA_CHECK(cudaEventRecord(start));
    launchBlurGlobalKernel(d_gray, d_blurGlobal, width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&timings.blurGlobalMs, start, stop));

    // 2B. Gaussian Blur (Shared Memory)
    CUDA_CHECK(cudaEventRecord(start));
    launchBlurSharedKernel(d_gray, d_blurShared, width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&timings.blurSharedMs, start, stop));

    // 3A. Sobel Edge Detection (Global Memory - foloseste iesirea blurShared)
    CUDA_CHECK(cudaEventRecord(start));
    launchSobelGlobalKernel(d_blurShared, d_sobelGlobal, width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&timings.sobelGlobalMs, start, stop));

    // 3B. Sobel Edge Detection (Shared Memory - foloseste iesirea blurShared)
    CUDA_CHECK(cudaEventRecord(start));
    launchSobelSharedKernel(d_blurShared, d_sobelShared, width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&timings.sobelSharedMs, start, stop));

    // Transfer Device -> Host
    CUDA_CHECK(cudaMemcpy(h_gray, d_gray, grayBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_blurGlobal, d_blurGlobal, grayBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_blurShared, d_blurShared, grayBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sobelGlobal, d_sobelGlobal, grayBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sobelShared, d_sobelShared, grayBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(totalStop));
    CUDA_CHECK(cudaEventSynchronize(totalStop));
    CUDA_CHECK(cudaEventElapsedTime(&timings.totalPipelineMs, totalStart, totalStop));

    // Eliberare resurse CUDA
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(totalStart));
    CUDA_CHECK(cudaEventDestroy(totalStop));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_gray));
    CUDA_CHECK(cudaFree(d_blurGlobal));
    CUDA_CHECK(cudaFree(d_blurShared));
    CUDA_CHECK(cudaFree(d_sobelGlobal));
    CUDA_CHECK(cudaFree(d_sobelShared));

    return timings;
}