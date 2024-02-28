import cv2
import os
import numpy as np
import sys
cuda_path = sys.argv[1]

with os.add_dll_directory(cuda_path):
    import torch

import pyfft

def load_image_as_grayscale(image_path):
    """Load an image and convert it to grayscale."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image

def display_image(title, image):
    """Display an image using OpenCV."""
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    test_dir = 'test_pic'
    images = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
    for image_path in images:
        image_path = os.path.join(test_dir, image_path)
        # Update this path
        image = load_image_as_grayscale(image_path)
    
        # Normalize the image to range [0, 1] for FFT
        image_normalized = image.astype(np.float32)#/ 255.0
        
        # Perform FFT on the image
        image_fft = pyfft.fft_image(image_normalized)
        
        # Compute magnitude spectrum and scale for visualization
        magnitude_spectrum = np.log(np.abs(image_fft) + 1)
        magnitude_spectrum = np.fft.fftshift(magnitude_spectrum, axes=1)  # Shift zero frequency to center
        magnitude_spectrum = (magnitude_spectrum / magnitude_spectrum.max() * 255).astype(np.uint8)
        
        # Display the original image and its magnitude spectrum
        display_image('Original Image', image)
        display_image('FFT Magnitude Spectrum', magnitude_spectrum)

if __name__ == "__main__":
    main()
