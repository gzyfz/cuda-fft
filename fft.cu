#include "fft.h"
#include <cufft.h>
#include <cuda_runtime.h>

// CUDA kernel to prepare data, assuming input is in row-major format
__global__ void prepareData(cufftReal* input, cufftReal* output, int width, int startX, int startY, int windowWidth, int windowHeight) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < windowWidth && idy < windowHeight) {
        int inputIndex = (startY + idy) * width + (startX + idx);
        output[idy * windowWidth + idx] = input[inputIndex];
    }
}

// Wrapper function to call FFT
void performFFT(cufftReal* input, cufftComplex* output, int width, int height, int startX, int startY, int windowWidth, int windowHeight) {
    cufftHandle plan;
    cufftReal* d_input;
    cufftComplex* d_output;

    // Allocate memory on device
    cudaMalloc(&d_input, windowWidth * windowHeight * sizeof(cufftReal));
    cudaMalloc(&d_output, windowWidth * windowHeight * sizeof(cufftComplex));

    // Prepare data
    dim3 blocks((windowWidth + 15) / 16, (windowHeight + 15) / 16);
    dim3 threads(16, 16);
    prepareData << <blocks, threads >> > (input, d_input, width, startX, startY, windowWidth, windowHeight);

    // Create plan and execute FFT
    cufftPlan2d(&plan, windowWidth, windowHeight, CUFFT_R2C);
    cufftExecR2C(plan, d_input, d_output);

    // Copy result back to host
    cudaMemcpy(output, d_output, windowWidth * windowHeight * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_input);
    cudaFree(d_output);
}
