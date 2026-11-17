#include "../include/cpu_processing.h"
#include "../include/utils.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <string>

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
    std::vector<uint8_t> grayImg(numPixels);
    std::vector<uint8_t> blurImg(numPixels);
    std::vector<uint8_t> sobelImg(numPixels);

    // 1. Grayscale CPU
    auto t1 = std::chrono::high_resolution_clock::now();
    rgbToGrayscaleCPU(img, grayImg.data(), width, height, channels);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeGray = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // 2. Gaussian Blur CPU
    auto t3 = std::chrono::high_resolution_clock::now();
    gaussianBlurCPU(grayImg.data(), blurImg.data(), width, height);
    auto t4 = std::chrono::high_resolution_clock::now();
    double timeBlur = std::chrono::duration<double, std::milli>(t4 - t3).count();

    // 3. Sobel Edge Detection CPU
    auto t5 = std::chrono::high_resolution_clock::now();
    sobelEdgeCPU(blurImg.data(), sobelImg.data(), width, height);
    auto t6 = std::chrono::high_resolution_clock::now();
    double timeSobel = std::chrono::duration<double, std::milli>(t6 - t5).count();

    std::cout << "=== Timpi de executie CPU (Referinta Secventiala) ===\n";
    std::cout << "1. RGB -> Grayscale : " << timeGray << " ms\n";
    std::cout << "2. Gaussian Blur 3x3: " << timeBlur << " ms\n";
    std::cout << "3. Sobel Edge       : " << timeSobel << " ms\n";
    std::cout << "Total pipeline CPU  : " << (timeGray + timeBlur + timeSobel) << " ms\n\n";

    // Salvare rezultate conform structurii din roadmap
    saveImagePNG("demo/output_samples/cpu_gray.png", grayImg.data(), width, height, 1);
    saveImagePNG("demo/output_samples/cpu_blur.png", blurImg.data(), width, height, 1);
    saveImagePNG("demo/output_samples/cpu_sobel.png", sobelImg.data(), width, height, 1);

    const std::string csvFile = "results/cpu_benchmark.csv";
    appendBenchmarkCSV(csvFile, "CPU_Grayscale", width, height, timeGray);
    appendBenchmarkCSV(csvFile, "CPU_GaussianBlur", width, height, timeBlur);
    appendBenchmarkCSV(csvFile, "CPU_SobelEdge", width, height, timeSobel);

    std::cout << "Salvat in demo/output_samples/ si results/cpu_benchmark.csv\n";

    freeImage(img);
    return 0;
}