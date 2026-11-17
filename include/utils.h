#ifndef UTILS_H
#define UTILS_H

#include <cstdint>
#include <string>

uint8_t* loadImage(const std::string& path, int& width, int& height, int& channels);
void saveImagePNG(const std::string& path, const uint8_t* data, int width, int height, int channels);
void freeImage(uint8_t* data);
void appendBenchmarkCSV(const std::string& csvPath, const std::string& stage, int width, int height, double timeMs);

#endif