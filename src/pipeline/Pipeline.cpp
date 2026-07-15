#include "pipeline/Pipeline.h"
#include <thread>
#include <chrono>

namespace pipeline {

Pipeline::Pipeline(const std::string& video_source, int width, int height, trt::TrtEngine* engine, bool skip_normalization)
    : input_thread_(video_source, width, height, this),
      inference_thread_(input_thread_.getInputQueue(), output_queue_, task_pool_, engine, width, height, skip_normalization),
      engine_(engine),
      running_(false) {
    for (int i = 0; i < 20; ++i) {
        auto task = std::make_shared<FrameTask>();
        task_pool_.push(task);
    }
}

Pipeline::~Pipeline() {
    stop();
}

void Pipeline::start() {
    if (!running_) {
        output_queue_.reset();
        std::cout << "[Pipeline] Starting threads..." << std::endl;
        input_thread_.start();
        std::cout << "[Pipeline] InputThread started" << std::endl;
        inference_thread_.start();
        std::cout << "[Pipeline] InferenceThread started" << std::endl;
        running_ = true;
        std::cout << "[Pipeline] Pipeline started successfully" << std::endl;
    }
}

void Pipeline::stop() {
    if (running_) {
        inference_thread_.stop();
        input_thread_.stop();
        running_ = false;
    }
}

bool Pipeline::isRunning() const {
    return running_;
}

SafeQueue<FrameTaskPtr>& Pipeline::getOutputQueue() {
    return output_queue_;
}

void Pipeline::setThreshold(float threshold) {
    inference_thread_.setThreshold(threshold);
    std::cout << "[Pipeline] Threshold set to: " << threshold << std::endl;
}

FrameTaskPtr Pipeline::get_empty_task()
{
    FrameTaskPtr task;
    if (task_pool_.try_pop(task))
    {
        task->reset();
        return task;
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    if (task_pool_.try_pop(task))
    {
        task->reset();
        return task;
    }


    return nullptr;
}

void Pipeline::return_task(FrameTaskPtr task) {
    task_pool_.push(task);
}

} // namespace pipeline
