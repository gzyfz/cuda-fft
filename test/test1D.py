import numpy as np

import os

# Example for adding CUDA and cuDNN directories to the DLL search path
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin"


with os.add_dll_directory(cuda_path):
    # Attempt to load a CUDA/cuDNN-dependent library or module
    import torch
import pyfft
    
def main():
    # Create a sample input signal: a combination of two sine waves
    fs = 8000  # Sampling rate, 8000 Hz
    T = 1.0 / fs  # Sampling interval
    N = 1024  # Number of sample points
    x = np.linspace(0.0, N*T, N, endpoint=False)  # Time vector
    freq1 = 300  # Frequency of the first sine wave
    freq2 = 1200  # Frequency of the second sine wave
    y = 0.5*(np.sin(freq1 * 2.0*np.pi*x) + np.sin(freq2 * 2.0*np.pi*x))  # Sample signal

    # Convert the signal to a complex numpy array (real signal, so imaginary part is 0)
    y_complex = y.astype(np.complex64)

    # Perform FFT using the CUDA-accelerated function
    yf = pyfft.fft(y_complex)

    # For demonstration, print the magnitude of the first few FFT coefficients
    print("Magnitude of the first few FFT coefficients:")
    print(np.abs(yf)[:10])

    # Optionally, plot the input signal and its FFT magnitude
    try:
        import matplotlib.pyplot as plt
        xf = np.fft.fftfreq(N, T)[:N//2]  # FFT sample frequencies
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(x, y)
        plt.title('Input Signal')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')

        plt.subplot(2, 1, 2)
        plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
        plt.title('Magnitude Spectrum of FFT')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('|FFT(y)|')

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib is not installed. Skipping the plot.")

if __name__ == "__main__":
    main()
