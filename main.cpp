#include <iostream>
#include <opencv2/opencv.hpp>
#include "fft.cu" // Your FFT CUDA header

// Use OpenCV namespace for convenience
using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <Image_Path>" << endl;
        return -1;
    }

    // Load the image
    Mat image = imread(argv[1], IMREAD_GRAYSCALE); // Load as grayscale
    if (image.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    // Convert to float and normalize (preparation for FFT)
    Mat floatImage;
    image.convertTo(floatImage, CV_32F, 1.0 / 255.0);

    // Define the window for FFT (startX, startY, width, height)
    // In a real application, these values could be passed as parameters or selected interactively
    int startX = 10, startY = 10, windowWidth = 100, windowHeight = 100;

    // Placeholder for the result (same size as the input window)
    Mat fftResult(windowHeight, windowWidth, CV_32F);

    // Call your CUDA FFT function
    // You need to implement this function to perform FFT using CUDA on the selected window
    // and store the result in fftResult
    applyFFT(floatImage, fftResult, startX, startY, windowWidth, windowHeight);

    // Convert fftResult to an image format (if necessary) and save or display it
    // This might involve scaling the values to the visible range and converting to uchar
    Mat displayImage;
    fftResult.convertTo(displayImage, CV_8U, 255.0); // Simple normalization to [0,255]
    imshow("FFT Result", displayImage);
    waitKey(0);

    return 0;
}
