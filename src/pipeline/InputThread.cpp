#include "pipeline/InputThread.h"
#include <iostream>
#include <chrono>
#include "pipeline/Pipeline.h"

namespace pipeline {

InputThread::InputThread(const std::string& source, int width, int height, Pipeline* pipeline)
    : source_(source),
      width_(width),
      height_(height),
      running_(false),
      frame_id_(0),
      pipeline_(pipeline) {
}

InputThread::~InputThread() {
    stop();
}

void InputThread::start() {
    if (!running_) {
        running_ = true;
        thread_ = std::thread(&InputThread::run, this);
    }
}

void InputThread::stop() {
    if (running_) {
        running_ = false;
        if (thread_.joinable()) {
            thread_.join();
        }
        if (cap_.isOpened()) {
            cap_.release();
        }
        input_queue_.shutdown();
    }
}

bool InputThread::isRunning() const {
    return running_;
}

SafeQueue<FrameTaskPtr>& InputThread::getInputQueue() {
    return input_queue_;
}

void InputThread::run() {
    std::cout << "[InputThread] Starting with source: " << source_ << std::endl;

    // Open video source
    if (!cap_.open(source_)) {
        std::cerr << "[InputThread] Failed to open video source: " << source_ << std::endl;
        running_ = false;
        return;
    }

    std::cout << "[InputThread] Video source opened successfully" << std::endl;

    cv::Mat frame;
    size_t img_pixels = width_ * height_;

    while (running_) {
        // Read frame
        if (!cap_.read(frame)) {
            std::cerr << "[InputThread] Failed to read frame, restarting..." << std::endl;
            // Try to reopen video source
            cap_.release();
            if (!cap_.open(source_)) {
                std::cerr << "[InputThread] Failed to reopen video source" << std::endl;
                break;
            }
            continue;
        }

        // Get task object (prefer from pool, create new if none available)
        FrameTaskPtr task;
        if (pipeline_) {
            task = pipeline_->get_empty_task();
            task->frame_id = frame_id_++;
            task->original_image = frame;
            
            // 检查是否需要分配显存（首次使用时）
            if (!task->d_input) {
                task->d_input = make_device_buffer(3 * img_pixels * sizeof(float));
            }
            if (!task->d_pred_score) {
                task->d_pred_score = make_device_buffer(sizeof(float));
            }
            if (!task->d_pred_label) {
                task->d_pred_label = make_device_buffer(sizeof(bool));
            }
            if (!task->d_anomaly_map) {
                task->d_anomaly_map = make_device_buffer(img_pixels * sizeof(float));
            }
            if (!task->d_dynamic_mask) {
                task->d_dynamic_mask = make_device_buffer(img_pixels * sizeof(uint8_t));
            }
        } else {
            // 如果没有pipeline，使用原来的方式创建任务
            task = std::make_shared<FrameTask>();
            task->frame_id = frame_id_++;
            task->original_image = frame;

            // Allocate GPU memory
            task->d_input = make_device_buffer(3 * img_pixels * sizeof(float));
            task->d_pred_score = make_device_buffer(sizeof(float));
            task->d_pred_label = make_device_buffer(sizeof(bool));
            task->d_anomaly_map = make_device_buffer(img_pixels * sizeof(float));
            task->d_dynamic_mask = make_device_buffer(img_pixels * sizeof(uint8_t));
        }

        // Push to input queue
        input_queue_.push(task);

        // Control frame rate
        std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS
    }

    std::cout << "[InputThread] Stopped" << std::endl;
}

} // namespace pipeline