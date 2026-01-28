#include "pipeline/InferenceThread.h"
#include <iostream>
#include <ctime>
#include "pipeline/FrameTask.h"

namespace pipeline {

InferenceThread::InferenceThread(
    SafeQueue<FrameTaskPtr>& input_queue,
    SafeQueue<FrameTaskPtr>& output_queue,
    trt::TrtEngine* engine
) : 
    input_queue_(input_queue),
    output_queue_(output_queue),
    engine_(engine),
    preprocessor_(256, 256),
    running_(false)
{
    cudaStreamCreate(&stream_);
}

InferenceThread::~InferenceThread() {
    stop();
    cudaStreamDestroy(stream_);
}

void InferenceThread::start() {
    if (!running_) {
        std::cout << "[InferenceThread] Starting thread..." << std::endl;
        running_ = true;
        thread_ = std::thread(&InferenceThread::run, this);
        std::cout << "[InferenceThread] Thread created and started" << std::endl;
    }
}

void InferenceThread::stop() {
    if (running_) {
        running_ = false;
        // Don't shutdown the queue here, let the thread finish processing
        if (thread_.joinable()) {
            thread_.join();
        }
        std::cout << "[InferenceThread] Stopped" << std::endl;
    }
}

bool InferenceThread::isRunning() const {
    return running_;
}

void InferenceThread::setThreshold(float threshold) {
    threshold_ = threshold;
    std::cout << "[InferenceThread] Threshold set to: " << threshold << std::endl;
}

void InferenceThread::run() {
    std::cout << "[InferenceThread] Running..." << std::endl;

    while (running_) {
        FrameTaskPtr task;
        input_queue_.pop(task);
        
        // Check if task is valid (shutdown returns default-constructed task)
        if (!task) {
            // Queue was shutdown
            break;
        }
        
        try {
            // Record start time
            task->start_time = static_cast<double>(std::clock()) / CLOCKS_PER_SEC;

            // Step 1: Preprocessing (GPU-based)
            preprocessor_.process(task->original_image, task->d_input.get(), stream_);

            // Step 2: Running inference
            engine_->infer(task->d_input.get(), 1);

            // Step 3: Postprocessing (GPU-based)
            postprocessor_.process(engine_, task, threshold_, stream_);

            // Record end time
            task->end_time = static_cast<double>(std::clock()) / CLOCKS_PER_SEC;

            // Step 4: Push completed task to output queue
            output_queue_.push(task);

        } catch (const std::exception& e) {
            std::cerr << "[InferenceThread] Error processing task " << task->frame_id << ": " << e.what() << std::endl;
            task->is_valid = false;
            output_queue_.push(task);
        }
    }

    std::cout << "[InferenceThread] Exiting run loop" << std::endl;
}

} // namespace pipeline