cmake_minimum_required(VERSION 3.18)
project(pyfft CUDA CXX)


add_subdirectory(external/pybind11)

# CUDA setup
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Python module
pybind11_add_module(pyfft main.cpp fft.cu)
target_link_libraries(pyfft PRIVATE ${CUDA_LIBRARIES} cufft)

# Specify C++ standard
set_target_properties(pyfft PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 14
)

# Include directories for Pybind11 and CUDA
target_include_directories(pyfft PRIVATE ${PYTHON_INCLUDE_DIRS})
target_include_directories(pyfft PRIVATE ${pybind11_INCLUDE_DIR})
target_include_directories(pyfft PRIVATE ${CUDA_INCLUDE_DIRS})
