#pragma once

#include <atomic>

#include "pipeline/InputThread.h"
#include "pipeline/InferenceThread.h"
#include "engine/TrtEngine.h"
#include "common/SafeQueue.hpp"
#include "pipeline/FrameTask.h"

namespace pipeline {

class Pipeline {
public:
    Pipeline(const std::string& video_source,
             int render_width, int render_height,
             trt::TrtEngine* engine,
             bool skip_normalization = false);
    ~Pipeline();

    void start();
    void stop();
    bool isRunning() const;

    SafeQueue<FrameTaskPtr>& getOutputQueue();

    void setThreshold(float threshold);

    FrameTaskPtr get_empty_task();
    void return_task(FrameTaskPtr task);

private:
    InputThread input_thread_;
    SafeQueue<FrameTaskPtr> output_queue_{1};
    SafeQueue<FrameTaskPtr> task_pool_; 
    InferenceThread inference_thread_;
    trt::TrtEngine* engine_;
    std::atomic<bool> running_{false};
};

} // namespace pipeline
