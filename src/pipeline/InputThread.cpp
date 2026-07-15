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
    running_ = false;
    if (thread_.joinable()) {
        thread_.join();
    }
    if (cap_.isOpened()) {
        cap_.release();
    }
    // Don't shutdown the queue here, just reset it.
    input_queue_.reset();
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
    while (running_) {
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
            // VideoCapture may reuse frame storage on the next read. A queued task must own
            // the host image until preprocessing has consumed it.
            task->original_image = frame.clone();
            task->target_width = width_;
            task->target_height = height_;
            
            // Upload original image to GPU memory
            size_t original_img_bytes = frame.cols * frame.rows * 3 * sizeof(uint8_t);
            if (!task->ensureOriginalImageBuffer(original_img_bytes)) {
                std::cerr << "[InputThread] Invalid original-image buffer size" << std::endl;
                pipeline_->return_task(task);
                continue;
            }
            const cudaError_t uploadStatus = cudaMemcpyAsync(
                task->d_original_image.get(), task->original_image.data, original_img_bytes, cudaMemcpyHostToDevice);
            if (uploadStatus != cudaSuccess) {
                std::cerr << "[InputThread] Failed to upload original image: "
                          << cudaGetErrorString(uploadStatus) << std::endl;
                task->is_valid = false;
                pipeline_->return_task(task);
                continue;
            }
            
            // 注意：不再需要分配中间推理缓冲区，这些现在由InferenceThread管理
        } else {
            // 如果没有pipeline，使用原来的方式创建任务
            task = std::make_shared<FrameTask>();
            task->frame_id = frame_id_++;
            task->original_image = frame.clone();
            task->target_width = width_;
            task->target_height = height_;

            // Upload original image to GPU memory
            size_t original_img_bytes = frame.cols * frame.rows * 3 * sizeof(uint8_t);
            if (!task->ensureOriginalImageBuffer(original_img_bytes)) {
                std::cerr << "[InputThread] Invalid original-image buffer size" << std::endl;
                continue;
            }
            const cudaError_t uploadStatus = cudaMemcpyAsync(
                task->d_original_image.get(), task->original_image.data, original_img_bytes, cudaMemcpyHostToDevice);
            if (uploadStatus != cudaSuccess) {
                std::cerr << "[InputThread] Failed to upload original image: "
                          << cudaGetErrorString(uploadStatus) << std::endl;
                continue;
            }

            // 注意：不再需要分配中间推理缓冲区，这些现在由InferenceThread管理
        }

        // Keep latency bounded. Any task displaced by a newer frame is returned to the pool.
        std::optional<FrameTaskPtr> discarded = input_queue_.push_latest(std::move(task));
        if (discarded && pipeline_) {
            pipeline_->return_task(std::move(*discarded));
        }

        // Control frame rate
        // std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS
    }

    std::cout << "[InputThread] Stopped" << std::endl;
}

} // namespace pipeline
