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
            
            // InputThread owns only capture-stage data; inference buffers belong to InferenceThread.
            if (!task->original_image.empty() && task->d_original_image) {
                std::cout << "  Capture-stage GPU buffer allocated successfully" << std::endl;
            } else {
                std::cerr << "  ERROR: capture-stage data was not allocated" << std::endl;
                return 1;
            }
        }
        
        // Short sleep to avoid busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Stop input thread
    input_thread.stop();
    std::cout << "Input thread stopped" << std::endl;

    // A failed open must still leave a joinable thread that stop()/destruction can reclaim.
    {
        pipeline::InputThread invalid_input("input/does-not-exist.avi", 256, 256);
        invalid_input.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        invalid_input.stop();
    }

    std::cout << "Test completed. Received " << task_count << " tasks." << std::endl;

    return 0;
}
