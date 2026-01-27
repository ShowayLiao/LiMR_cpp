#pragma once

#include <thread>
#include <atomic>
#include "pipeline/FrameTask.h"
#include "pipeline/Preprocessor.h"
#include "pipeline/PostProcessor.h"
#include "engine/TrtEngine.h"
#include "common/SafeQueue.hpp"

namespace pipeline {

class InferenceThread {
public:
    InferenceThread(
        SafeQueue<FrameTaskPtr>& input_queue,
        SafeQueue<FrameTaskPtr>& output_queue,
        trt::TrtEngine* engine
    );
    ~InferenceThread();

    void start();
    void stop();
    bool isRunning() const;

private:
    void run();

    SafeQueue<FrameTaskPtr>& input_queue_;
    SafeQueue<FrameTaskPtr>& output_queue_;
    trt::TrtEngine* engine_;
    
    Preprocessor preprocessor_;
    PostProcessor postprocessor_;
    
    std::thread thread_;
    std::atomic<bool> running_;
    
    cudaStream_t stream_;
};

} // namespace pipeline