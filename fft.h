#ifndef FFT_H
#define FFT_H

#include <cufft.h>

void performFFT(cufftReal* input, cufftComplex* output, int width, int height, int startX, int startY, int windowWidth, int windowHeight);

#endif // FFT_H
