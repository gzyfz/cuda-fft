from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pyfft',  
    version='0.1',  
    author='Fangzhou Ye',
    author_email='f1ye@ucsd.edu',
    description='A small package for FFT/2D_FFT using CUDA',
    long_description='This package provides Python bindings to CUDA-accelerated FFT and IFFT operations.',
    ext_modules=[
        CUDAExtension(
            name='pyfft',  # Extension name, used for import in Python
            sources=['main.cpp', 'fft.cu',],  # Source files
            include_dirs=[],  # Any include directories needed
            libraries=['cufft'],  # CUDA cuFFT Library
            library_dirs=[],  # Any additional library directories needed
            extra_compile_args={
                'cxx': [],
                'nvcc': [],  # Additional flags for the NVCC compiler can be specified here
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True),
    }
)
