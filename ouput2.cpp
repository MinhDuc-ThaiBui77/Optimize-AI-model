#include <tensorflow/c/c_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

void NoOpDeallocator(void* data, size_t a, void* b) {}

void resizeImage(cv::Mat& img, int width, int height) {
    cv::resize(img, img, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
}

int main() {
    std::string image_path = "Bida.jpg";
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Lỗi: Không thể đọc ảnh từ " << image_path << std::endl;
        return -1;
    }
    
    int original_width = img.cols;
    int original_height = img.rows;
    cv::Mat img_resized;
    cv::resize(img, img_resized, cv::Size(320, 320));
    cv::cvtColor(img_resized, img_resized, cv::COLOR_BGR2RGB);
    std::vector<uint8_t> input_data(img_resized.data, img_resized.data + img_resized.total() * img_resized.elemSize());
    
    TF_Status* status = TF_NewStatus();
    TF_Graph* graph = TF_NewGraph();
    TF_SessionOptions* sess_opts = TF_NewSessionOptions();
    const char* tags[] = {"serve"};
    TF_Session* session = TF_LoadSessionFromSavedModel(sess_opts, nullptr, 
        "model/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model", 
        tags, 1, graph, nullptr, status);

    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "Lỗi khi load model: " << TF_Message(status) << std::endl;
        return -1;
    }
    
    int64_t dims[] = {1, 320, 320, 3};
    TF_Tensor* input_tensor = TF_NewTensor(TF_UINT8, dims, 4, input_data.data(), input_data.size(), NoOpDeallocator, nullptr);
    
    TF_Operation* input_op = TF_GraphOperationByName(graph, "serving_default_input_tensor");
    TF_Operation* output_scores = TF_GraphOperationByName(graph, "StatefulPartitionedCall");
    TF_Operation* output_boxes = TF_GraphOperationByName(graph, "StatefulPartitionedCall");
    TF_Operation* output_classes = TF_GraphOperationByName(graph, "StatefulPartitionedCall");
    TF_Operation* output_num_detections = TF_GraphOperationByName(graph, "StatefulPartitionedCall");

    TF_Output inputs[] = {{input_op, 0}};
    TF_Output outputs[] = {
        {output_scores, 4},
        {output_boxes, 1},
        {output_classes, 2},
        {output_num_detections, 5}
    };
    
    TF_Tensor* output_tensors[4] = {nullptr, nullptr, nullptr, nullptr};
    TF_SessionRun(session, nullptr, 
                  inputs, &input_tensor, 1, 
                  outputs, output_tensors, 4, 
                  nullptr, 0, nullptr, status);

    float* scores = static_cast<float*>(TF_TensorData(output_tensors[0]));
    float* boxes = static_cast<float*>(TF_TensorData(output_tensors[1]));
    float* classes = static_cast<float*>(TF_TensorData(output_tensors[2]));
    float* num_detections = static_cast<float*>(TF_TensorData(output_tensors[3]));

    int valid_detections = static_cast<int>(num_detections[0]);
    
    for (int i = 0; i < valid_detections; i++) {
        if (scores[i] >= 0.5 && static_cast<int>(classes[i]) == 1) {
            float ymin = boxes[i * 4] * original_height;
            float xmin = boxes[i * 4 + 1] * original_width;
            float ymax = boxes[i * 4 + 2] * original_height;
            float xmax = boxes[i * 4 + 3] * original_width;

            cv::rectangle(img, cv::Point(xmin, ymin), cv::Point(xmax, ymax), 
                          cv::Scalar(0, 255, 0), 2);
            std::string label = "Person: " + std::to_string(scores[i]).substr(0, 4);
            cv::putText(img, label, cv::Point(xmin, ymin - 2), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
    }

    // Resize ảnh về 320x460
    resizeImage(img, 320, 460);
    
    // Lưu ảnh
    std::string outputPath = "output_test3.jpg";
    cv::imwrite(outputPath, img);
    std::cout << "Ảnh kết quả đã được lưu tại: " << outputPath << std::endl;
    
    // Hiển thị ảnh
    cv::imshow("Detection Result", img);
    cv::waitKey(0);
    
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

