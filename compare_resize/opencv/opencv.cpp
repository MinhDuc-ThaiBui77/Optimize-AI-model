#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <string>

std::vector<uint8_t> preprocess_image(const std::string& input_path) {
    // Đọc ảnh đầu vào
    cv::Mat img = cv::imread(input_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Could not load image at " << input_path << std::endl;
        return std::vector<uint8_t>();
    }

    // Resize ảnh về 320x320
    cv::Mat img_resized;
    cv::resize(img, img_resized, cv::Size(320, 320), 0, 0, cv::INTER_LINEAR);

    // Chuyển từ BGR sang RGB
    cv::cvtColor(img_resized, img_resized, cv::COLOR_BGR2RGB);

    // Chuyển dữ liệu thành vector<uint8_t>
    std::vector<uint8_t> input_data(img_resized.data, img_resized.data + img_resized.total() * img_resized.elemSize());

    std::cout << "Ảnh đã được chuyển thành Tensor thành công!" << std::endl;
    return input_data;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image_path>" << std::endl;
        return -1;
    }

    std::string input_path = argv[1];
    std::vector<uint8_t> input_data = preprocess_image(input_path);

    // Kiểm tra kích thước vector
    if (!input_data.empty()) {
        std::cout << "Kích thước vector: " << input_data.size() << " bytes" << std::endl;
        std::cout << "Kích thước mong đợi: " << 320 * 320 * 3 << " bytes" << std::endl;
    }

    return 0;
}