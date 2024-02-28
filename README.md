## prerequisites 
# tools 
CUDA toolkit 11.6 \
python >=3.7 <=3.8 \
CMake 
# python packages
numpy \
open-cv 


## set-up
simply run the setup.py to have the package generated and install on your machine.

## usage
```python
    '''
    1D FFT
    '''
    y = 0.5*(np.sin(freq1 * 2.0*np.pi*x) + np.sin(freq2 * 2.0*np.pi*x))  # Sample signal

    # Convert the signal to a complex numpy array (real signal, so imaginary part is 0)
    y_complex = y.astype(np.complex64)

    # Perform FFT using the CUDA-accelerated function
    yf = pyfft.fft(y_complex)


    '''
    Image FFT
    '''
    image = load_image_as_grayscale(image_path)
            
    # Perform FFT on the image
    image_fft = pyfft.fft_image(image_normalized)
```
 ## test
 specify your location of CUDA tool kit in test file and simply run test.sh or test.bat.

