Referință Secvențială CPU

Implementarea secvențială este scrisă în C++ (src/cpu_processing.cpp) și servește ca referință pentru validarea rezultatelor GPU și măsurarea timpilor de bază.

Algoritmi implementați:
- RGB la Grayscale (rgbToGrayscaleCPU): conversie pe baza formulei Y = 0.299 * R + 0.587 * G + 0.114 * B, complexitate O(W * H).
- Gaussian Blur 3x3 (gaussianBlurCPU): convoluție 2D cu nucleul Gaussian normalizat (suma ponderilor = 16) și replicare la margini, complexitate O(W * H * 9).
- Sobel Edge Detection (sobelEdgeCPU): calcul gradienți orizontali/verticali 3x3 și magnitudine sqrt(Gx^2 + Gy^2), complexitate O(W * H * 9).

Benchmark inițial (imagine test 1152 x 2048 px, g++ 16.2.0 -O2):
- RGB la Grayscale: 3.2465 ms
- Gaussian Blur 3x3: 20.5240 ms
- Sobel Edge Detection: 11.1307 ms
- Total pipeline: 34.9012 ms