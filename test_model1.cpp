#include <tensorflow/c/c_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>


//Ở chương trình này thì input truyền vào đang là một ảnh rgb màu trắng, ảnh này tự tạo bằng  


void NoOpDeallocator(void* data, size_t a, void* b) {}

int main() {
    std::cout << "TensorFlow Version: " << TF_Version() << std::endl;

    // 🔹 1. Khởi tạo môi trường TensorFlow
    TF_Status* status = TF_NewStatus();
    TF_Graph* graph = TF_NewGraph();
    TF_SessionOptions* sess_opts = TF_NewSessionOptions();
    TF_Buffer* run_opts = nullptr;
    const char* tags[] = {"serve"};  // Thêm tag-set "serve"
    TF_Session* session = TF_LoadSessionFromSavedModel(sess_opts, run_opts, "model/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model", tags, 1, graph, nullptr, status);

    if (TF_GetCode(status) != TF_OK) {
        std::cerr << " Lỗi khi load model: " << TF_Message(status) << std::endl;
        return -1;
    }
    std::cout << " Model loaded successfully!" << std::endl;

    // 🔹 2. Tạo tensor đầu vào (giả sử input shape là [1, 320, 320, 3])
    int64_t dims[] = {1, 320, 320, 3};  
    //std::vector<float> input_data(320 * 320 * 3, 1.0f); // Dữ liệu đầu vào toàn 1.0
    //TF_Tensor* input_tensor = TF_NewTensor(TF_UINT8, dims, 4, input_data.data(), input_data.size() * sizeof(float), NoOpDeallocator, nullptr);
    std::vector<uint8_t> input_data(320 * 320 * 3, 255);  // Giá trị pixel 0-255
    TF_Tensor* input_tensor = TF_NewTensor(TF_UINT8, dims, 4, input_data.data(), input_data.size(), NoOpDeallocator, nullptr);
    // 🔹 3. Tìm input & output trong graph


    TF_Operation* input_op = TF_GraphOperationByName(graph, "serving_default_input_tensor");
    TF_Operation* output_op = TF_GraphOperationByName(graph, "StatefulPartitionedCall:5");

    if (!input_op) {
        std::cerr << "Không tìm thấy tensor đầu vào: serving_default_input_tensor!" << std::endl;
    }
    if (!output_op) {
        std::cerr << "Không tìm thấy tensor đầu ra: StatefulPartitionedCall!" << std::endl;
    }





    if (!input_op || !output_op) {
        std::cerr << " Không tìm thấy input/output trong graph!" << std::endl;
        return -1;
    }

    // 🔹 4. Chạy mô hình
    TF_Output inputs[] = {{input_op, 0}};
    TF_Output outputs[] = {{output_op, 0}};
    TF_Tensor* output_tensor = nullptr;

    TF_SessionRun(session, nullptr, 
                  inputs, &input_tensor, 1, 
                  outputs, &output_tensor, 1, 
                  nullptr, 0, nullptr, status);

    if (TF_GetCode(status) != TF_OK) {
        std::cerr << " Lỗi khi chạy mô hình: " << TF_Message(status) << std::endl;
        return -1;
    }
    std::cout << " Inference thành công!" << std::endl;

    // 🔹 5. Lấy kết quả từ tensor đầu ra
    float* output_data = static_cast<float*>(TF_TensorData(output_tensor));
    std::cout << "🔹 Giá trị đầu ra đầu tiên: " << output_data[0] << std::endl;

    // 🔹 6. Giải phóng bộ nhớ
    TF_DeleteTensor(input_tensor);
    TF_DeleteTensor(output_tensor);
    TF_DeleteSession(session, status);
    TF_DeleteGraph(graph);
    TF_DeleteSessionOptions(sess_opts);
    TF_DeleteStatus(status);

    return 0;
}
