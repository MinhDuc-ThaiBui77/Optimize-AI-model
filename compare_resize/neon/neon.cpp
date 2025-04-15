#include <opencv2/opencv.hpp> // Dùng để đọc ảnh
#include <arm_neon.h>
#include <vector>
#include <iostream>
#include <string>
#include <chrono>

// Hàm resize ảnh dùng NEON (bilinear interpolation)
void resize_bilinear_neon(const uint8_t* input, uint8_t* output, int in_width, int in_height, int out_width, int out_height) {
    float scale_x = (float)in_width / out_width;
    float scale_y = (float)in_height / out_height;

    for (int y = 0; y < out_height; y++) {
        float fy = y * scale_y;
        int sy0 = (int)fy;
        int sy1 = sy0 + (sy0 < in_height - 1 ? 1 : 0); // Xử lý biên
        float frac_y = fy - sy0;
        float one_minus_y = 1.0f - frac_y;

        for (int x = 0; x < out_width; x += 4) { // Xử lý 4 pixel cùng lúc
            float fx = x * scale_x;
            int sx0 = (int)fx;
            int sx1 = sx0 + (sx0 < in_width - 1 ? 1 : 0); // Xử lý biên
            float frac_x = fx - sx0;
            float one_minus_x = 1.0f - frac_x;

            // Trọng số bilinear
            float32x4_t w00 = vdupq_n_f32(one_minus_x * one_minus_y);
            float32x4_t w01 = vdupq_n_f32(frac_x * one_minus_y);
            float32x4_t w10 = vdupq_n_f32(one_minus_x * frac_y);
            float32x4_t w11 = vdupq_n_f32(frac_x * frac_y);

            // Tính vị trí pixel trong input
            int idx00 = (sy0 * in_width + sx0) * 3;
            int idx01 = (sy0 * in_width + sx1) * 3;
            int idx10 = (sy1 * in_width + sx0) * 3;
            int idx11 = (sy1 * in_width + sx1) * 3;

            // Kênh B (vì input là BGR)
            uint8x8_t b00_8 = vld1_u8(&input[idx00]);
            uint8x8_t b01_8 = vld1_u8(&input[idx01]);
            uint8x8_t b10_8 = vld1_u8(&input[idx10]);
            uint8x8_t b11_8 = vld1_u8(&input[idx11]);
            float32x4_t b00 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(b00_8))));
            float32x4_t b01 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(b01_8))));
            float32x4_t b10 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(b10_8))));
            float32x4_t b11 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(b11_8))));
            float32x4_t b = vaddq_f32(vaddq_f32(vmulq_f32(b00, w00), vmulq_f32(b01, w01)),
                                      vaddq_f32(vmulq_f32(b10, w10), vmulq_f32(b11, w11)));

            // Kênh G
            uint8x8_t g00_8 = vld1_u8(&input[idx00 + 1]);
            uint8x8_t g01_8 = vld1_u8(&input[idx01 + 1]);
            uint8x8_t g10_8 = vld1_u8(&input[idx10 + 1]);
            uint8x8_t g11_8 = vld1_u8(&input[idx11 + 1]);
            float32x4_t g00 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(g00_8))));
            float32x4_t g01 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(g01_8))));
            float32x4_t g10 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(g10_8))));
            float32x4_t g11 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(g11_8))));
            float32x4_t g = vaddq_f32(vaddq_f32(vmulq_f32(g00, w00), vmulq_f32(g01, w01)),
                                      vaddq_f32(vmulq_f32(g10, w10), vmulq_f32(g11, w11)));

            // Kênh R
            uint8x8_t r00_8 = vld1_u8(&input[idx00 + 2]);
            uint8x8_t r01_8 = vld1_u8(&input[idx01 + 2]);
            uint8x8_t r10_8 = vld1_u8(&input[idx10 + 2]);
            uint8x8_t r11_8 = vld1_u8(&input[idx11 + 2]);
            float32x4_t r00 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(r00_8))));
            float32x4_t r01 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(r01_8))));
            float32x4_t r10 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(r10_8))));
            float32x4_t r11 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(r11_8))));
            float32x4_t r = vaddq_f32(vaddq_f32(vmulq_f32(r00, w00), vmulq_f32(r01, w01)),
                                      vaddq_f32(vmulq_f32(r10, w10), vmulq_f32(r11, w11)));

            // Chuyển đổi từ float về uint8 và lưu (RGB order)
            int out_idx = (y * out_width + x) * 3;
            for (int i = 0; i < 4 && (x + i) < out_width; i++) { // Xử lý biên
                // Lấy từng giá trị từ vector và chuyển về uint8
                uint8_t r_val = static_cast<uint8_t>(vgetq_lane_f32(r, i));
                uint8_t g_val = static_cast<uint8_t>(vgetq_lane_f32(g, i));
                uint8_t b_val = static_cast<uint8_t>(vgetq_lane_f32(b, i));

                // Lưu theo thứ tự RGB
                output[out_idx + i * 3] = r_val;     // R
                output[out_idx + i * 3 + 1] = g_val; // G
                output[out_idx + i * 3 + 2] = b_val; // B
            }
        }
    }
}

std::vector<uint8_t> preprocess_image_neon(const std::string& input_path, int in_width, int in_height, int num_loops) {
    // Đọc ảnh đầu vào bằng OpenCV
    cv::Mat img = cv::imread(input_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Could not load image at " << input_path << std::endl;
        return std::vector<uint8_t>();
    }

    // Đảm bảo kích thước đầu vào đúng
    if (img.cols != in_width || img.rows != in_height) {
        std::cerr << "Error: Input image size does not match specified dimensions (" << in_width << "x" << in_height << ")" << std::endl;
        return std::vector<uint8_t>();
    }

    // Chuẩn bị dữ liệu đầu vào và đầu ra
    std::vector<uint8_t> input_data(img.data, img.data + img.total() * img.elemSize());
    std::vector<uint8_t> output_data(320 * 320 * 3); // 320x320x3

    // Đo thời gian xử lý
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_loops; i++) {
        resize_bilinear_neon(input_data.data(), output_data.data(), in_width, in_height, 320, 320);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double time_ms = duration.count() / 1000.0 / num_loops;

    std::cout << "Average time per loop (NEON): " << time_ms << " ms" << std::endl;
    std::cout << "Ảnh đã được chuyển thành Tensor thành công!" << std::endl;
    return output_data;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_image_path> <input_width> <input_height>" << std::endl;
        return -1;
    }

    std::string input_path = argv[1];
    int in_width = std::stoi(argv[2]);
    int in_height = std::stoi(argv[3]);
    int num_loops = 100; // Lặp 100 lần để đo chính xác

    std::vector<uint8_t> input_data = preprocess_image_neon(input_path, in_width, in_height, num_loops);

    if (!input_data.empty()) {
        std::cout << "Kích thước vector: " << input_data.size() << " bytes" << std::endl;
        std::cout << "Kích thước mong đợi: " << 320 * 320 * 3 << " bytes" << std::endl;
    }
    return 0;
}