#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "pipeline/InputThread.h"
#include "pipeline/InferenceThread.h"
#include "engine/TrtEngine.h"

int main() {
    std::cout << "=== Inference Thread Test ===" << std::endl;

    // Create input and output queues
    SafeQueue<pipeline::FrameTaskPtr> input_queue;
    SafeQueue<pipeline::FrameTaskPtr> output_queue;

    // Load TensorRT engine
    trt::TrtEngine engine;
    std::string model_path = "input/LiMR_merged.onnx";
    if (!engine.load(model_path, trt::Precision::FP16)) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return -1;
    }
    std::cout << "Model loaded successfully" << std::endl;

    // Create inference thread
    pipeline::InferenceThread inference_thread(input_queue, output_queue, &engine);

    // Start inference thread
    inference_thread.start();
    std::cout << "Inference thread started" << std::endl;

    // Load test image
    cv::Mat test_image = cv::imread("input/IMG_9255.png");
    if (test_image.empty()) {
        std::cerr << "Failed to load test image" << std::endl;
        return -1;
    }
    std::cout << "Test image loaded: " << test_image.cols << "x" << test_image.rows << std::endl;

    // Create and push test tasks
    int num_tasks = 5;
    for (int i = 0; i < num_tasks; i++) {
        auto task = std::make_shared<pipeline::FrameTask>();
        task->frame_id = i;
        task->original_image = test_image;

        // Allocate GPU memory
        size_t img_pixels = 256 * 256;
        task->d_input = make_device_buffer(3 * img_pixels * sizeof(float));
        task->d_pred_score = make_device_buffer(sizeof(float));
        task->d_pred_label = make_device_buffer(sizeof(bool));
        task->d_anomaly_map = make_device_buffer(img_pixels * sizeof(float));
        task->d_dynamic_mask = make_device_buffer(img_pixels * sizeof(uint8_t));

        input_queue.push(task);
        std::cout << "Pushed task " << i << " to input queue" << std::endl;
    }

    // Wait for tasks to complete
    std::cout << "Waiting for inference to complete..." << std::endl;
    
    // Collect results
    int completed_tasks = 0;
    auto start_time = std::chrono::steady_clock::now();
    
    while (completed_tasks < num_tasks && 
           std::chrono::steady_clock::now() - start_time < std::chrono::seconds(30)) {
        pipeline::FrameTaskPtr task;
        if (output_queue.try_pop(task)) {
            completed_tasks++;
            std::cout << "Received completed task " << task->frame_id << std::endl;
            
            if (task->is_valid) {
                std::cout << "  Task is valid" << std::endl;
                std::cout << "  Score buffer size: " << task->h_pred_score.size() << std::endl;
                std::cout << "  Label buffer size: " << task->h_pred_label.size() << std::endl;
                std::cout << "  Anomaly map size: " << task->h_anomaly_map.size() << std::endl;
            } else {
                std::cerr << "  Task is invalid" << std::endl;
            }
        }
        
        // Short sleep to avoid busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Stop inference thread
    inference_thread.stop();
    std::cout << "Inference thread stopped" << std::endl;

    // Summary
    std::cout << "Test completed. Processed " << completed_tasks << " out of " << num_tasks << " tasks." << std::endl;

    return 0;
}