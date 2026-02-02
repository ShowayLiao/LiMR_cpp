#include "pipeline/InputThread.h"
#include <iostream>
#include <chrono>
#include "pipeline/Pipeline.h"
#include <cuda_runtime.h>

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
        // Reset the queue before starting
        input_queue_.reset();
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
        // Don't shutdown the queue here, just reset it
        input_queue_.reset();
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
    bool opened = false;
    
    // Try to open as camera index first (if source is a number)
    try {
        int camera_index = std::stoi(source_);
        opened = cap_.open(camera_index);
        if (opened) {
            std::cout << "[InputThread] Opened camera with index: " << camera_index << std::endl;
        }
    } catch (const std::invalid_argument& e) {
        // Not a number, try to open as video file
        opened = cap_.open(source_);
        if (opened) {
            std::cout << "[InputThread] Opened video file: " << source_ << std::endl;
        }
    } catch (const std::out_of_range& e) {
        // Number out of range, try to open as video file
        opened = cap_.open(source_);
        if (opened) {
            std::cout << "[InputThread] Opened video file: " << source_ << std::endl;
        }
    }
    
    if (!opened) {
        std::cerr << "[InputThread] Failed to open video source: " << source_ << std::endl;
        running_ = false;
        return;
    }

    std::cout << "[InputThread] Video source opened successfully" << std::endl;

    cv::Mat frame;
    size_t img_pixels = width_ * height_;

    while (running_) {
        if (input_queue_.size() > 3) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        // Read frame with timeout
        if (!cap_.read(frame)) {
            std::cerr << "[InputThread] Failed to read frame, restarting..." << std::endl;
            // Try to reopen video source
            cap_.release();
            if (!running_) {
                break;
            }
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
            if (!task) continue; // 如果池子里没任务了，跳过此帧
            
            task->frame_id = frame_id_++;
            task->original_image = frame;
            task->target_width = width_;
            task->target_height = height_;
            
            // Upload original image to GPU memory
            size_t original_img_bytes = frame.cols * frame.rows * 3 * sizeof(uint8_t);
            if (!task->d_original_image) {
                task->d_original_image = make_device_buffer(original_img_bytes);
            }
            cudaMemcpyAsync(task->d_original_image.get(), frame.data, original_img_bytes, cudaMemcpyHostToDevice);
            
            // 注意：不再需要分配中间推理缓冲区，这些现在由InferenceThread管理
        } else {
            // 如果没有pipeline，使用原来的方式创建任务
            task = std::make_shared<FrameTask>();
            task->frame_id = frame_id_++;
            task->original_image = frame;
            task->target_width = width_;
            task->target_height = height_;

            // Upload original image to GPU memory
            size_t original_img_bytes = frame.cols * frame.rows * 3 * sizeof(uint8_t);
            task->d_original_image = make_device_buffer(original_img_bytes);
            cudaMemcpyAsync(task->d_original_image.get(), frame.data, original_img_bytes, cudaMemcpyHostToDevice);

            // 注意：不再需要分配中间推理缓冲区，这些现在由InferenceThread管理
        }

        // Push to input queue
        input_queue_.push(task);

        // Control frame rate
        // std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS
    }

    std::cout << "[InputThread] Stopped" << std::endl;
}

} // namespace pipeline