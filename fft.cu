#include "fft.h"
#include <cufft.h>
#include <cuda_runtime.h>

// Perform FFT on the GPU and return the result to host
cufftComplex* perform_fft(cufftComplex* host_input, int N) {
    cufftComplex *device_input, *device_output;

    // Allocate memory on the device
    cudaMalloc(&device_input, N * sizeof(cufftComplex));
    cudaMalloc(&device_output, N * sizeof(cufftComplex));

    // Copy data from host to device
    cudaMemcpy(device_input, host_input, N * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    // Create plan and execute FFT
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    cufftExecC2C(plan, device_input, device_output, CUFFT_FORWARD);

    // Allocate output memory on the host
    cufftComplex* host_output = new cufftComplex[N];

    // Copy result back to host
    cudaMemcpy(host_output, device_output, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    // Cleanup
    cufftDestroy(plan);
    cudaFree(device_input);
    cudaFree(device_output);

    return host_output;
}

void perform_fft_2d(cufftComplex* host_input, cufftComplex* host_output, int width, int height) {
    cufftComplex *device_input, *device_output;
    cudaMalloc(&device_input, width * height * sizeof(cufftComplex));
    cudaMalloc(&device_output, width * height * sizeof(cufftComplex));
    
    cudaMemcpy(device_input, host_input, width * height * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, width, CUFFT_C2C, height); // Plan for batched 1D FFTs
    cufftExecC2C(plan, device_input, device_output, CUFFT_FORWARD);

    cudaMemcpy(host_output, device_output, width * height * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(device_input);
    cudaFree(device_output);
}
