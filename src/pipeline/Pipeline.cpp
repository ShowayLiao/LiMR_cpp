#include "pipeline/Pipeline.h"

namespace pipeline {

Pipeline::Pipeline(const std::string& video_source, int width, int height, trt::TrtEngine* engine)
    : input_thread_(video_source, width, height, this),
      inference_thread_(input_thread_.getInputQueue(), output_queue_, engine),
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
        input_thread_.stop();
        inference_thread_.stop();
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

FrameTaskPtr Pipeline::get_empty_task() {
    FrameTaskPtr task;
    if (task_pool_.try_pop(task)) {
        task->reset();
        return task;
    } else {

        return std::make_shared<FrameTask>();
    }
}

void Pipeline::return_task(FrameTaskPtr task) {
    task_pool_.push(task);
}

} // namespace pipeline