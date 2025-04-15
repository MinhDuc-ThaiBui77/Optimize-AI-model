#include <tensorflow/c/c_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

// Hàm giải phóng bộ nhớ Tensor
void NoOpDeallocator(void* data, size_t a, void* b) {}

int main() {
    std::string image_path = "Bida.jpg";  
    cv::Mat img = cv::imread(image_path);
    
    if (img.empty()) {
        std::cerr << " Lỗi: Không thể đọc ảnh từ " << image_path << std::endl;
        return -1;
    }

    int original_width = img.cols;
    int original_height = img.rows;

    // Resize ảnh về kích thước chuẩn 320x320
    cv::Mat img_resized;
    cv::resize(img, img_resized, cv::Size(320, 320));
    cv::cvtColor(img_resized, img_resized, cv::COLOR_BGR2RGB);//Chuyen anh tu BGR thanh RGB vi OpenCV Đọc ảnh khác với Tensorflow

    std::vector<uint8_t> input_data(img_resized.data, img_resized.data + img_resized.total() * img_resized.elemSize());
    // dùng để chuyển đổi dữ liệu ảnh từ OpenCV cv::Mat thành một mảng dữ liệu 1D (vector) có thể đưa vào TensorFlow.
    std::cout << " Ảnh đã được chuyển thành Tensor thành công!" << std::endl;

    //  Load Model TensorFlow
    TF_Status* status = TF_NewStatus();
    TF_Graph* graph = TF_NewGraph();
    TF_SessionOptions* sess_opts = TF_NewSessionOptions();
    const char* tags[] = {"serve"};
    TF_Session* session = TF_LoadSessionFromSavedModel(sess_opts, nullptr, 
        "model/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model", 
        tags, 1, graph, nullptr, status);

    if (TF_GetCode(status) != TF_OK) {
        std::cerr << " Lỗi khi load model: " << TF_Message(status) << std::endl;
        TF_DeleteStatus(status);
        return -1;
    }
    std::cout << " Model loaded successfully!" << std::endl;

    // Tạo tensor đầu vào
    int64_t dims[] = {1, 320, 320, 3};
    TF_Tensor* input_tensor = TF_NewTensor(TF_UINT8, dims, 4, input_data.data(), input_data.size(), NoOpDeallocator, nullptr);
    if (!input_tensor) {
        std::cerr << " Lỗi: Không thể tạo Tensor đầu vào!" << std::endl;
        return -1;
    }

    // Lấy input & output từ model
    TF_Operation* input_op = TF_GraphOperationByName(graph, "serving_default_input_tensor");
    TF_Operation* output_scores = TF_GraphOperationByName(graph, "StatefulPartitionedCall");
    TF_Operation* output_boxes = TF_GraphOperationByName(graph, "StatefulPartitionedCall");
    TF_Operation* output_classes = TF_GraphOperationByName(graph, "StatefulPartitionedCall");
    TF_Operation* output_num_detections = TF_GraphOperationByName(graph, "StatefulPartitionedCall");

    TF_Output inputs[] = {{input_op, 0}};
    TF_Output outputs[] = {
        {output_scores, 4},  // detection_scores
        {output_boxes, 1},   // detection_boxes
        {output_classes, 2}, // detection_classes
        {output_num_detections, 5} // num_detections
    };

    if (!input_op || !output_scores || !output_boxes || !output_classes || !output_num_detections) {
        std::cerr << " Không tìm thấy input/output trong graph!" << std::endl;
        return -1;
    }

    // Chạy mô hình
    TF_Tensor* output_tensors[4] = {nullptr, nullptr, nullptr, nullptr};

    TF_SessionRun(session, nullptr, 
                  inputs, &input_tensor, 1, 
                  outputs, output_tensors, 4, 
                  nullptr, 0, nullptr, status);

    if (TF_GetCode(status) != TF_OK) {
        std::cerr << " Lỗi khi chạy mô hình: " << TF_Message(status) << std::endl;
        return -1;
    }
    std::cout << " Inference thành công!" << std::endl;

    // Đọc kết quả
    if (output_tensors[0] && output_tensors[1] && output_tensors[2] && output_tensors[3]) {
        float* scores = static_cast<float*>(TF_TensorData(output_tensors[0]));
        float* boxes = static_cast<float*>(TF_TensorData(output_tensors[1]));
        float* classes = static_cast<float*>(TF_TensorData(output_tensors[2]));
        float* num_detections = static_cast<float*>(TF_TensorData(output_tensors[3]));

        int valid_detections = static_cast<int>(num_detections[0]);  // Số lượng vật thể thực tế
        std::cout << "🔹 Tổng số vật thể phát hiện: " << valid_detections << std::endl;

        int person_count = 0;
        for (int i = 0; i < valid_detections; i++) {
            if (scores[i] >= 0.5 && static_cast<int>(classes[i]) == 1) {  // Chỉ lấy lớp "Person"
                person_count++;
                std::cout << " Người " << person_count 
                          << " - Score: " << scores[i] << std::endl;

                // Lấy tọa độ bounding box
                float ymin = boxes[i * 4] * original_height;
                float xmin = boxes[i * 4 + 1] * original_width;
                float ymax = boxes[i * 4 + 2] * original_height;
                float xmax = boxes[i * 4 + 3] * original_width;

                std::cout << "    Bounding Box: [" 
                          << ymin << ", " << xmin << ", " 
                          << ymax << ", " << xmax << "]" << std::endl;

                // Vẽ bounding box trên ảnh
                cv::rectangle(img, cv::Point(xmin, ymin), cv::Point(xmax, ymax), 
                              cv::Scalar(0, 255, 0), 2);

                // Hiển thị nhãn "Person" với độ tin cậy
                std::string label = "Person: " + std::to_string(scores[i]).substr(0, 4);
                int baseLine;
                cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                cv::rectangle(img, cv::Point(xmin, ymin - textSize.height - 5), 
                              cv::Point(xmin + textSize.width, ymin), 
                              cv::Scalar(0, 255, 0), cv::FILLED);
                cv::putText(img, label, cv::Point(xmin, ymin - 2), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
            }
        }

        std::cout << " Tổng số người phát hiện: " << person_count << std::endl;

        // Lưu ảnh kết quả
        cv::imwrite("output.jpg", img);
        std::cout << " Ảnh kết quả đã được lưu thành output.jpg" << std::endl;

        // Hiển thị ảnh
        cv::imshow("Detection Result", img);
        cv::waitKey(0);
    } else {
        std::cerr << " Lỗi: Không nhận được output tensor!" << std::endl;
    }

    // Giải phóng bộ nhớ
    TF_DeleteTensor(input_tensor);
    for (int i = 0; i < 4; i++) {
        TF_DeleteTensor(output_tensors[i]);
    }
    TF_DeleteSession(session, status);
    TF_DeleteGraph(graph);
    TF_DeleteSessionOptions(sess_opts);
    TF_DeleteStatus(status);

    return 0;
}
