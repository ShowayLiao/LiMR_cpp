#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "pipeline/InputThread.h"

int main() {
    std::cout << "=== Input Thread Test ===" << std::endl;

    // Create InputThread with test video or camera
    std::string video_source = "input/blade.avi";
    pipeline::InputThread input_thread(video_source, 256, 256);

    // Start input thread
    input_thread.start();
    std::cout << "Input thread started" << std::endl;

    // Get input queue
    auto& input_queue = input_thread.getInputQueue();

    // Test run for 10 seconds
    auto start_time = std::chrono::steady_clock::now();
    int task_count = 0;

    while (std::chrono::steady_clock::now() - start_time < std::chrono::seconds(10)) {
        // Try to get task from queue
        pipeline::FrameTaskPtr task;
        if (input_queue.try_pop(task)) {
            task_count++;
            std::cout << "Received task " << task->frame_id 
                      << " - Image size: " << task->original_image.cols << "x" << task->original_image.rows << std::endl;
            
            // Verify GPU memory allocation
            if (task->d_input && task->d_pred_score && task->d_pred_label && task->d_anomaly_map) {
                std::cout << "  GPU memory allocated successfully" << std::endl;
            } else {
                std::cerr << "  ERROR: GPU memory not allocated" << std::endl;
            }
        }
        
        // Short sleep to avoid busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Stop input thread
    input_thread.stop();
    std::cout << "Input thread stopped" << std::endl;

    std::cout << "Test completed. Received " << task_count << " tasks." << std::endl;

    return 0;
}