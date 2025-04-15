#include <opencv2/opencv.hpp>
#include <arm_neon.h>
#include <vector>
#include <iostream>
#include <string>
#include <chrono>
#include <omp.h>

// Hàm resize ảnh dùng NEON (bilinear interpolation, xử lý 8 pixel)
void resize_bilinear_neon(const uint8_t* input, uint8_t* output, int in_width, int in_height, int out_width, int out_height) {
    float scale_x = (float)in_width / out_width;
    float scale_y = (float)in_height / out_height;

    #pragma omp parallel for
    for (int y = 0; y < out_height; y++) {
        float fy = y * scale_y;
        int sy0 = (int)fy;
        int sy1 = sy0 + (sy0 < in_height - 1 ? 1 : 0);
        float frac_y = fy - sy0;
        float one_minus_y = 1.0f - frac_y;

        for (int x = 0; x < out_width; x += 8) {
            float fx[8];
            int sx0[8], sx1[8];
            float frac_x[8], one_minus_x[8];

            // Tính trọng số cho 8 pixel
            for (int i = 0; i < 8 && (x + i) < out_width; i++) {
                fx[i] = (x + i) * scale_x;
                sx0[i] = (int)fx[i];
                sx1[i] = sx0[i] + (sx0[i] < in_width - 1 ? 1 : 0);
                frac_x[i] = fx[i] - sx0[i];
                one_minus_x[i] = 1.0f - frac_x[i];
            }

            // Tải trọng số vào vector (xử lý 2 nhóm 4 pixel)
            float32x4_t w00_low = { one_minus_x[0] * one_minus_y, one_minus_x[1] * one_minus_y, one_minus_x[2] * one_minus_y, one_minus_x[3] * one_minus_y };
            float32x4_t w01_low = { frac_x[0] * one_minus_y, frac_x[1] * one_minus_y, frac_x[2] * one_minus_y, frac_x[3] * one_minus_y };
            float32x4_t w10_low = { one_minus_x[0] * frac_y, one_minus_x[1] * frac_y, one_minus_x[2] * frac_y, one_minus_x[3] * frac_y };
            float32x4_t w11_low = { frac_x[0] * frac_y, frac_x[1] * frac_y, frac_x[2] * frac_y, frac_x[3] * frac_y };

            float32x4_t w00_high = { (x + 4 < out_width ? one_minus_x[4] * one_minus_y : 0), 
                                     (x + 5 < out_width ? one_minus_x[5] * one_minus_y : 0), 
                                     (x + 6 < out_width ? one_minus_x[6] * one_minus_y : 0), 
                                     (x + 7 < out_width ? one_minus_x[7] * one_minus_y : 0) };
            float32x4_t w01_high = { (x + 4 < out_width ? frac_x[4] * one_minus_y : 0), 
                                     (x + 5 < out_width ? frac_x[5] * one_minus_y : 0), 
                                     (x + 6 < out_width ? frac_x[6] * one_minus_y : 0), 
                                     (x + 7 < out_width ? frac_x[7] * one_minus_y : 0) };
            float32x4_t w10_high = { (x + 4 < out_width ? one_minus_x[4] * frac_y : 0), 
                                     (x + 5 < out_width ? one_minus_x[5] * frac_y : 0), 
                                     (x + 6 < out_width ? one_minus_x[6] * frac_y : 0), 
                                     (x + 7 < out_width ? one_minus_x[7] * frac_y : 0) };
            float32x4_t w11_high = { (x + 4 < out_width ? frac_x[4] * frac_y : 0), 
                                     (x + 5 < out_width ? frac_x[5] * frac_y : 0), 
                                     (x + 6 < out_width ? frac_x[6] * frac_y : 0), 
                                     (x + 7 < out_width ? frac_x[7] * frac_y : 0) };

            // Khai báo các biến ngoài vòng lặp
            float32x4_t r00_low = vdupq_n_f32(0), r01_low = vdupq_n_f32(0), r10_low = vdupq_n_f32(0), r11_low = vdupq_n_f32(0);
            float32x4_t g00_low = vdupq_n_f32(0), g01_low = vdupq_n_f32(0), g10_low = vdupq_n_f32(0), g11_low = vdupq_n_f32(0);
            float32x4_t b00_low = vdupq_n_f32(0), b01_low = vdupq_n_f32(0), b10_low = vdupq_n_f32(0), b11_low = vdupq_n_f32(0);
            float32x4_t r00_high = vdupq_n_f32(0), r01_high = vdupq_n_f32(0), r10_high = vdupq_n_f32(0), r11_high = vdupq_n_f32(0);
            float32x4_t g00_high = vdupq_n_f32(0), g01_high = vdupq_n_f32(0), g10_high = vdupq_n_f32(0), g11_high = vdupq_n_f32(0);
            float32x4_t b00_high = vdupq_n_f32(0), b01_high = vdupq_n_f32(0), b10_high = vdupq_n_f32(0), b11_high = vdupq_n_f32(0);

            // Tải dữ liệu cho 8 pixel
            for (int i = 0; i < 8 && (x + i) < out_width; i++) {
                int idx00 = (sy0 * in_width + sx0[i]) * 3;
                int idx01 = (sy0 * in_width + sx1[i]) * 3;
                int idx10 = (sy1 * in_width + sx0[i]) * 3;
                int idx11 = (sy1 * in_width + sx1[i]) * 3;

                if (i < 4) {
                    uint8x8_t r00_8 = vld1_u8(&input[idx00 + 2]);
                    uint8x8_t r01_8 = vld1_u8(&input[idx01 + 2]);
                    uint8x8_t r10_8 = vld1_u8(&input[idx10 + 2]);
                    uint8x8_t r11_8 = vld1_u8(&input[idx11 + 2]);
                    r00_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(r00_8))));
                    r01_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(r01_8))));
                    r10_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(r10_8))));
                    r11_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(r11_8))));

                    uint8x8_t g00_8 = vld1_u8(&input[idx00 + 1]);
                    uint8x8_t g01_8 = vld1_u8(&input[idx01 + 1]);
                    uint8x8_t g10_8 = vld1_u8(&input[idx10 + 1]);
                    uint8x8_t g11_8 = vld1_u8(&input[idx11 + 1]);
                    g00_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(g00_8))));
                    g01_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(g01_8))));
                    g10_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(g10_8))));
                    g11_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(g11_8))));

                    uint8x8_t b00_8 = vld1_u8(&input[idx00]);
                    uint8x8_t b01_8 = vld1_u8(&input[idx01]);
                    uint8x8_t b10_8 = vld1_u8(&input[idx10]);
                    uint8x8_t b11_8 = vld1_u8(&input[idx11]);
                    b00_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(b00_8))));
                    b01_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(b01_8))));
                    b10_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(b10_8))));
                    b11_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(b11_8))));
                } else {
                    uint8x8_t r00_8 = vld1_u8(&input[idx00 + 2]);
                    uint8x8_t r01_8 = vld1_u8(&input[idx01 + 2]);
                    uint8x8_t r10_8 = vld1_u8(&input[idx10 + 2]);
                    uint8x8_t r11_8 = vld1_u8(&input[idx11 + 2]);
                    r00_high = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(r00_8))));
                    r01_high = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(r01_8))));
                    r10_high = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(r10_8))));
                    r11_high = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(r11_8))));

                    uint8x8_t g00_8 = vld1_u8(&input[idx00 + 1]);
                    uint8x8_t g01_8 = vld1_u8(&input[idx01 + 1]);
                    uint8x8_t g10_8 = vld1_u8(&input[idx10 + 1]);
                    uint8x8_t g11_8 = vld1_u8(&input[idx11 + 1]);
                    g00_high = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(g00_8))));
                    g01_high = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(g01_8))));
                    g10_high = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(g10_8))));
                    g11_high = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(g11_8))));

                    uint8x8_t b00_8 = vld1_u8(&input[idx00]);
                    uint8x8_t b01_8 = vld1_u8(&input[idx01]);
                    uint8x8_t b10_8 = vld1_u8(&input[idx10]);
                    uint8x8_t b11_8 = vld1_u8(&input[idx11]);
                    b00_high = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(b00_8))));
                    b01_high = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(b01_8))));
                    b10_high = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(b10_8))));
                    b11_high = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(b11_8))));
                }
            }

            // Tính giá trị pixel (4 pixel đầu)
            float32x4_t r_low = vaddq_f32(vaddq_f32(vmulq_f32(r00_low, w00_low), vmulq_f32(r01_low, w01_low)),
                                          vaddq_f32(vmulq_f32(r10_low, w10_low), vmulq_f32(r11_low, w11_low)));
            float32x4_t g_low = vaddq_f32(vaddq_f32(vmulq_f32(g00_low, w00_low), vmulq_f32(g01_low, w01_low)),
                                          vaddq_f32(vmulq_f32(g10_low, w10_low), vmulq_f32(g11_low, w11_low)));
            float32x4_t b_low = vaddq_f32(vaddq_f32(vmulq_f32(b00_low, w00_low), vmulq_f32(b01_low, w01_low)),
                                          vaddq_f32(vmulq_f32(b10_low, w10_low), vmulq_f32(b11_low, w11_low)));

            // Tính giá trị pixel (4 pixel sau)
            float32x4_t r_high = vaddq_f32(vaddq_f32(vmulq_f32(r00_high, w00_high), vmulq_f32(r01_high, w01_high)),
                                           vaddq_f32(vmulq_f32(r10_high, w10_high), vmulq_f32(r11_high, w11_high)));
            float32x4_t g_high = vaddq_f32(vaddq_f32(vmulq_f32(g00_high, w00_high), vmulq_f32(g01_high, w01_high)),
                                           vaddq_f32(vmulq_f32(g10_high, w10_high), vmulq_f32(g11_high, w11_high)));
            float32x4_t b_high = vaddq_f32(vaddq_f32(vmulq_f32(b00_high, w00_high), vmulq_f32(b01_high, w01_high)),
                                           vaddq_f32(vmulq_f32(b10_high, w10_high), vmulq_f32(b11_high, w11_high)));

            // Chuyển đổi từ float về uint8 và lưu
            uint8_t r_vals[8], g_vals[8], b_vals[8];
            for (int i = 0; i < 8 && (x + i) < out_width; i++) {
                int out_idx = (y * out_width + (x + i)) * 3;
                if (i < 4) {
                    r_vals[i] = static_cast<uint8_t>(vgetq_lane_f32(r_low, i));
                    g_vals[i] = static_cast<uint8_t>(vgetq_lane_f32(g_low, i));
                    b_vals[i] = static_cast<uint8_t>(vgetq_lane_f32(b_low, i));
                } else {
                    r_vals[i] = static_cast<uint8_t>(vgetq_lane_f32(r_high, i - 4));
                    g_vals[i] = static_cast<uint8_t>(vgetq_lane_f32(g_high, i - 4));
                    b_vals[i] = static_cast<uint8_t>(vgetq_lane_f32(b_high, i - 4));
                }
                output[out_idx] = r_vals[i];     // R
                output[out_idx + 1] = g_vals[i]; // G
                output[out_idx + 2] = b_vals[i]; // B
            }
        }
    }
}

std::vector<uint8_t> preprocess_image_neon(const std::string& input_path, int in_width, int in_height, int num_loops) {
    cv::Mat img = cv::imread(input_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Could not load image at " << input_path << std::endl;
        return std::vector<uint8_t>();
    }

    if (img.cols != in_width || img.rows != in_height) {
        std::cerr << "Error: Input image size does not match specified dimensions (" << in_width << "x" << in_height << ")" << std::endl;
        return std::vector<uint8_t>();
    }

    std::vector<uint8_t> input_data(img.data, img.data + img.total() * img.elemSize());
    std::vector<uint8_t> output_data(320 * 320 * 3);

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
    int num_loops = 100;

    std::vector<uint8_t> input_data = preprocess_image_neon(input_path, in_width, in_height, num_loops);

    if (!input_data.empty()) {
        std::cout << "Kích thước vector: " << input_data.size() << " bytes" << std::endl;
        std::cout << "Kích thước mong đợi: " << 320 * 320 * 3 << " bytes" << std::endl;
    }
    return 0;
}