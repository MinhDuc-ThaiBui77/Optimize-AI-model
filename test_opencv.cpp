#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat image = cv::imread("Bida.jpg");
    if (image.empty()) {
        std::cout << "Không thể mở ảnh!\n";
        return -1;
    }
    cv::imshow("Hiển thị ảnh", image);
    cv::waitKey(0);
    return 0;
}
