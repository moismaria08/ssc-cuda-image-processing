#include "../include/cpu_processing.h"
#include "../include/utils.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <cmath>
#include <algorithm>

#ifdef USE_CUDA
#include "../include/cuda_processing.h"
#endif

int maxAbsDiff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    int maxDiff = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        int diff = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
        if (diff > maxDiff) maxDiff = diff;
    }
    return maxDiff;
}

int main(int argc, char** argv) {
    std::string inputPath = "images/input.jpg";
    if (argc > 1) {
        inputPath = argv[1];
    }

    int width = 0, height = 0, channels = 0;
    uint8_t* img = loadImage(inputPath, width, height, channels);

    if (!img && argc <= 1) {
        inputPath = "images/input.png";
        img = loadImage(inputPath, width, height, channels);
    }

    if (!img) {
        std::cerr << "[EROARE] Nu s-a putut incarca imaginea: " << inputPath << "\n";
        return 1;
    }

    std::cout << "Imagine incarcata: " << inputPath
              << " (" << width << "x" << height << ", " << channels << " canale)\n\n";

    size_t numPixels = static_cast<size_t>(width) * height;
    std::vector<uint8_t> grayCPU(numPixels);
    std::vector<uint8_t> blurCPU(numPixels);
    std::vector<uint8_t> sobelCPU(numPixels);

    // 1. Executie CPU (Referinta)
    auto t1 = std::chrono::high_resolution_clock::now();
    rgbToGrayscaleCPU(img, grayCPU.data(), width, height, channels);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeGrayCPU = std::chrono::duration<double, std::milli>(t2 - t1).count();

    auto t3 = std::chrono::high_resolution_clock::now();
    gaussianBlurCPU(grayCPU.data(), blurCPU.data(), width, height);
    auto t4 = std::chrono::high_resolution_clock::now();
    double timeBlurCPU = std::chrono::duration<double, std::milli>(t4 - t3).count();

    auto t5 = std::chrono::high_resolution_clock::now();
    sobelEdgeCPU(blurCPU.data(), sobelCPU.data(), width, height);
    auto t6 = std::chrono::high_resolution_clock::now();
    double timeSobelCPU = std::chrono::duration<double, std::milli>(t6 - t5).count();

    double totalCPU = timeGrayCPU + timeBlurCPU + timeSobelCPU;

    std::cout << "=== Timpi de executie CPU ===\n";
    std::cout << "1. RGB -> Grayscale : " << timeGrayCPU << " ms\n";
    std::cout << "2. Gaussian Blur 3x3: " << timeBlurCPU << " ms\n";
    std::cout << "3. Sobel Edge       : " << timeSobelCPU << " ms\n";
    std::cout << "Total pipeline CPU  : " << totalCPU << " ms\n\n";

    saveImagePNG("demo/output_samples/cpu_gray.png", grayCPU.data(), width, height, 1);
    saveImagePNG("demo/output_samples/cpu_blur.png", blurCPU.data(), width, height, 1);
    saveImagePNG("demo/output_samples/cpu_sobel.png", sobelCPU.data(), width, height, 1);

    const std::string csvFile = "results/benchmark_results.csv";
    appendBenchmarkCSV(csvFile, "CPU_Grayscale", width, height, timeGrayCPU);
    appendBenchmarkCSV(csvFile, "CPU_GaussianBlur", width, height, timeBlurCPU);
    appendBenchmarkCSV(csvFile, "CPU_SobelEdge", width, height, timeSobelCPU);
    appendBenchmarkCSV(csvFile, "CPU_Total", width, height, totalCPU);

#ifdef USE_CUDA
    std::vector<uint8_t> grayGPU(numPixels);
    std::vector<uint8_t> blurGlobalGPU(numPixels);
    std::vector<uint8_t> blurSharedGPU(numPixels);
    std::vector<uint8_t> sobelGlobalGPU(numPixels);
    std::vector<uint8_t> sobelSharedGPU(numPixels);

    CudaTimings gpu = runCudaPipeline(
        img,
        grayGPU.data(),
        blurGlobalGPU.data(),
        blurSharedGPU.data(),
        sobelGlobalGPU.data(),
        sobelSharedGPU.data(),
        width,
        height,
        channels
    );

    std::cout << "=== Timpi de executie GPU (CUDA) ===\n";
    std::cout << "1. RGB -> Grayscale        : " << gpu.grayscaleMs << " ms\n";
    std::cout << "2A. Gaussian Blur (Global) : " << gpu.blurGlobalMs << " ms\n";
    std::cout << "2B. Gaussian Blur (Shared) : " << gpu.blurSharedMs << " ms\n";
    std::cout << "3A. Sobel Edge (Global)    : " << gpu.sobelGlobalMs << " ms\n";
    std::cout << "3B. Sobel Edge (Shared)    : " << gpu.sobelSharedMs << " ms\n";
    std::cout << "Total pipeline GPU (cu H2D/D2H): " << gpu.totalPipelineMs << " ms\n\n";

    std::cout << "=== Validare Corectitudine (Diferenta maxima CPU vs GPU) ===\n";
    std::cout << "Grayscale diff    : " << maxAbsDiff(grayCPU, grayGPU) << "\n";
    std::cout << "Blur Global diff  : " << maxAbsDiff(blurCPU, blurGlobalGPU) << "\n";
    std::cout << "Blur Shared diff  : " << maxAbsDiff(blurCPU, blurSharedGPU) << "\n";
    std::cout << "Sobel Global diff : " << maxAbsDiff(sobelCPU, sobelGlobalGPU) << "\n";
    std::cout << "Sobel Shared diff : " << maxAbsDiff(sobelCPU, sobelSharedGPU) << "\n\n";

    saveImagePNG("demo/output_samples/gpu_gray.png", grayGPU.data(), width, height, 1);
    saveImagePNG("demo/output_samples/gpu_blur_shared.png", blurSharedGPU.data(), width, height, 1);
    saveImagePNG("demo/output_samples/gpu_sobel_shared.png", sobelSharedGPU.data(), width, height, 1);

    appendBenchmarkCSV(csvFile, "GPU_Grayscale", width, height, gpu.grayscaleMs);
    appendBenchmarkCSV(csvFile, "GPU_Blur_Global", width, height, gpu.blurGlobalMs);
    appendBenchmarkCSV(csvFile, "GPU_Blur_Shared", width, height, gpu.blurSharedMs);
    appendBenchmarkCSV(csvFile, "GPU_Sobel_Global", width, height, gpu.sobelGlobalMs);
    appendBenchmarkCSV(csvFile, "GPU_Sobel_Shared", width, height, gpu.sobelSharedMs);
    appendBenchmarkCSV(csvFile, "GPU_Total_Pipeline", width, height, gpu.totalPipelineMs);
#else
    std::cout << "[INFO] Compilat fara suport CUDA (mod CPU-only activ).\n";
#endif

    freeImage(img);
    return 0;
}