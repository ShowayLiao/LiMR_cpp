#include <iostream> 
#include <vector> 
#include <algorithm> 
#include <opencv2/opencv.hpp> 
#include "engine/TrtEngine.h" 
#include "pipeline/FrameTask.h" 
#include "pipeline/Preprocessor.h" 
#include "common/CudaMemory.hpp" 

int main() { 
    std::cout << "=== Single Frame Validation Test (Fixed) ===" << std::endl; 

    trt::TrtEngine engine;
    Preprocessor preprocessor(256, 256); 

    // [Step 1] Loading model 
    // Make sure your path is correct 
    if (!engine.load("input/LiMR_merged.onnx", trt::Precision::FP16)) { 
        std::cerr << "Failed to load model" << std::endl; 
        return -1; 
    } 
    std::cout << "[Step 1] Model loaded successfully" << std::endl; 

    // [Step 2] Loading test image 
    cv::Mat testImage = cv::imread("input/IMG_9255.png"); 
    if (testImage.empty()) { 
        std::cerr << "Failed to load test image" << std::endl; 
        return -1; 
    } 
    std::cout << "[Step 2] Test image loaded: " << testImage.cols << "x" << testImage.rows << std::endl; 

    // [Step 3] Creating FrameTask & Buffers 
    // Simulate Task structure, in actual project this is managed by FrameTask constructor or Pool 
    auto task = std::make_shared<pipeline::FrameTask>(); 
    task->frame_id = 0; 
    task->original_image = testImage; 

    // 分配显存 (根据模型实际需求，假设 Batch=1) 
    size_t img_pixels = 256 * 256; 
    task->d_input = make_device_buffer(3 * img_pixels * sizeof(float)); 
    
    // 注意：这里需要根据模型真实的输出节点名称来获取大小，或者手动指定 
    // 如果 TrtEngine 实现了 getBufferSize，最好用那个 
    task->d_pred_score = make_device_buffer(sizeof(float)); 
    task->d_pred_label = make_device_buffer(sizeof(bool)); 
    task->d_anomaly_map = make_device_buffer(img_pixels * sizeof(float)); 
    // task->d_pred_mask = make_device_buffer(img_pixels * sizeof(bool)); // 如果需要 

    cudaStream_t stream; 
    cudaStreamCreate(&stream); 

    // [Step 4] Preprocessing (GPU-based) 
    // Use Preprocessor to perform preprocessing on GPU
    preprocessor.process(testImage, task->d_input.get(), stream);
    cudaStreamSynchronize(stream);
    std::cout << "[Step 4] Preprocessing completed (GPU-based)" << std::endl; 

    // [Step 5] Running inference 
    // 注意：这里假设 TrtEngine 已经更新为支持 infer(ptr, batch_size) 的新版接口 
    engine.infer(task->d_input.get(), 1); 
    std::cout << "[Step 5] Inference completed" << std::endl; 

    // [Step 6] Copying results to host 
    // 直接从引擎获取结果，避免通过 PostProcessor
    float* d_score = (float*)engine.getBuffer("pred_score");
    bool* d_label = (bool*)engine.getBuffer("pred_label");
    float* d_map = (float*)engine.getBuffer("anomaly_map");

    if (!d_score || !d_label || !d_map) {
        std::cerr << "Failed to get engine output buffers" << std::endl;
        return -1;
    }

    float score;
    bool label;
    std::vector<float> anomaly_map(256 * 256);

    cudaMemcpy(&score, d_score, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&label, d_label, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(anomaly_map.data(), d_map, 256 * 256 * sizeof(float), cudaMemcpyDeviceToHost);

    // [Step 7] Verifying results 
    std::cout << "[Step 7] Results:" << std::endl; 
    std::cout << "  Pred Score: " << score << std::endl; 
    std::cout << "  Pred Label: " << (label ? "Anomaly" : "Normal") << std::endl; 
    
    // Simple statistics 
    float min_val = *std::min_element(anomaly_map.begin(), anomaly_map.end()); 
    float max_val = *std::max_element(anomaly_map.begin(), anomaly_map.end()); 
    std::cout << "  Anomaly Map range: [" << min_val << ", " << max_val << "]" << std::endl; 

    // [Step 8] Saving visualization (使用 applyColorMap) 
    cv::Mat map_gray(256, 256, CV_8UC1); 
    for (int i = 0; i < img_pixels; i++) { 
        // 归一化到 0-255，注意处理边界 
        float val = anomaly_map[i]; 
        // 如果 output 本身就是 0-1，直接乘 
        // 如果 output 是 logits，可能需要 sigmoid 或 min-max normalization 
        // 这里假设是 0-1 分数 
        val = std::max(0.0f, std::min(1.0f, val)); 
        map_gray.data[i] = static_cast<uint8_t>(val * 255); 
    } 

    cv::Mat heatmap_vis; 
    cv::applyColorMap(map_gray, heatmap_vis, cv::COLORMAP_JET); // 使用标准的 JET 配色 
    cv::imwrite("output_heatmap_fixed.png", heatmap_vis); 
    std::cout << "[Step 8] Heatmap saved to output_heatmap_fixed.png" << std::endl; 

    cudaStreamDestroy(stream); 
    return 0; 
}