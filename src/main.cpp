#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "fft.h"

namespace py = pybind11;

// Wrapper function for FFT
py::array_t<std::complex<float>> fft(py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> input) {
    py::buffer_info buf = input.request();
    int N = buf.shape[0];

    // Allocate memory on host
    cufftComplex *host_input = new cufftComplex[N];
    
    // Convert input to cufftComplex format
    auto ptr = static_cast<std::complex<float> *>(buf.ptr);
    for (int i = 0; i < N; ++i) {
        host_input[i].x = ptr[i].real();
        host_input[i].y = ptr[i].imag();
    }

    // Perform FFT (function will allocate device memory, copy data, compute FFT, and copy back)
    cufftComplex *host_output = perform_fft(host_input, N);

    // Create output array
    auto result = py::array_t<std::complex<float>>(buf.size);
    py::buffer_info buf_out = result.request();
    auto ptr_out = static_cast<std::complex<float> *>(buf_out.ptr);

    // Copy the result to the output array
    for (int i = 0; i < N; ++i) {
        ptr_out[i] = std::complex<float>(host_output[i].x, host_output[i].y);
    }

    // Cleanup
    delete[] host_input;
    delete[] host_output;

    return result;
}

// Function to apply FFT on each row of a 2D image
py::array_t<std::complex<float>> fft_image(py::array_t<float, py::array::c_style | py::array::forcecast> input) {
    auto buf = input.request();
    if (buf.ndim != 2) throw std::runtime_error("Input should be a 2D array");
    int height = buf.shape[0];
    int width = buf.shape[1];

    // Allocate host memory for input and output
    cufftComplex *host_input = new cufftComplex[width * height];
    cufftComplex *host_output = new cufftComplex[width * height];

    // Prepare input (assume input is already grayscale)
    for (ssize_t i = 0; i < width * height; ++i) {
        host_input[i].x = static_cast<float*>(buf.ptr)[i]; // Real part
        host_input[i].y = 0.0; // Imaginary part is 0 for real input
    }

    // Perform FFT on the image
    perform_fft_2d(host_input, host_output, width, height);

    // Prepare output
    auto result = py::array_t<std::complex<float>>(buf.size);
    auto result_buf = result.request();
    result.resize({height, width}); // Ensure result has the same shape as input

    // Copy the FFT results to the output array
    for (ssize_t i = 0; i < width * height; ++i) {
        static_cast<std::complex<float>*>(result_buf.ptr)[i] = {host_output[i].x, host_output[i].y};
    }

    // Clean up
    delete[] host_input;
    delete[] host_output;

    return result;
}

PYBIND11_MODULE(pyfft, m) {
    m.doc() = "CUDA FFT operations exposed with pybind11";
    m.def("fft", &fft, "Perform FFT on a numpy array of complex numbers using CUDA.");
	m.def("fft_image", &fft_image, "Apply FFT on each row of a 2D numpy array (image) using CUDA.");
}
