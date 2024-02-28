cmake_minimum_required(VERSION 3.18)
project(fft_image LANGUAGES CUDA CXX)
find_package(CUDA)
# Assuming Pybind11 is located in the extern/pybind11 directory
add_subdirectory(extern/pybind11)

# Define your module target
pybind11_add_module(fft_module main.cpp fft.cu)

# Enable separable compilation for CUDA
set_target_properties(fft_module PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link against the Python library and CUDA's cufft library
find_library(CUFFT_LIBRARY cufft HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
target_link_libraries(fft_module PRIVATE pybind11::module ${CUFFT_LIBRARY})



###