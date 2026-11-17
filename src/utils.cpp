#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include "../include/utils.h"
#include <fstream>
#include <filesystem>

uint8_t* loadImage(const std::string& path, int& width, int& height, int& channels) {
    return stbi_load(path.c_str(), &width, &height, &channels, 0);
}

void saveImagePNG(const std::string& path, const uint8_t* data, int width, int height, int channels) {
    stbi_write_png(path.c_str(), width, height, channels, data, width * channels);
}

void freeImage(uint8_t* data) {
    stbi_image_free(data);
}

void appendBenchmarkCSV(const std::string& csvPath, const std::string& stage, int width, int height, double timeMs) {
    bool fileExists = std::filesystem::exists(csvPath);
    std::ofstream out(csvPath, std::ios::app);
    if (!fileExists) {
        out << "Stage,Width,Height,Time_ms\n";
    }
    out << stage << "," << width << "," << height << "," << timeMs << "\n";
}