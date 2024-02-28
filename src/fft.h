#ifndef FFT_H
#define FFT_H

#include <cufft.h>

// Declaration for the perform_fft function
cufftComplex* perform_fft(cufftComplex* host_input, int N);
void perform_fft_2d(cufftComplex* host_input, cufftComplex* host_output, int width, int height);

#endif // FFT_H
