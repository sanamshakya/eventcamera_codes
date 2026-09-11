#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Create a solid red 400x400 image (BGR format)
    cv::Mat image(400, 400, CV_8UC3, cv::Scalar(0, 0, 255));

    if (image.empty()) {
        std::cerr << "Error: Could not create image." << std::endl;
        return -1;
    }

    // Save the image to disk
    bool success = cv::imwrite("test_output.jpg", image);

    if (success) {
        std::cout << "OpenCV Test Success! Image saved as test_output.jpg" << std::endl;
    } else {
        std::cerr << "Error: Failed to save the image." << std::endl;
    }

    return 0;
}

