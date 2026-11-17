#ifndef CPU_PROCESSING_H
#define CPU_PROCESSING_H

#include <cstdint>

void rgbToGrayscaleCPU(const uint8_t* input, uint8_t* output, int width, int height, int channels);
void gaussianBlurCPU(const uint8_t* input, uint8_t* output, int width, int height);
void sobelEdgeCPU(const uint8_t* input, uint8_t* output, int width, int height);

#endif