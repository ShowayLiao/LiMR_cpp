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
        trt::TrtEngine* engine,
        int render_width = 256,
        int render_height = 256,
        bool skip_normalization = false
    );
    ~InferenceThread();

    void start();
    void stop();
    bool isRunning() const;
    void setThreshold(float threshold);

private:
    void run();

    SafeQueue<FrameTaskPtr>& input_queue_;
    SafeQueue<FrameTaskPtr>& output_queue_;
    trt::TrtEngine* engine_;
    int render_width_;
    int render_height_;
    
    // Model dimensions
    int model_input_width_;
    int model_input_height_;
    int model_output_width_;
    int model_output_height_;
    
    // Workspace buffers
    DeviceBuffer m_d_trt_input = nullptr;
    DeviceBuffer m_d_raw_anomaly_map = nullptr;
    DeviceBuffer m_d_raw_mask = nullptr;
    
    std::unique_ptr<Preprocessor> preprocessor_;
    std::unique_ptr<PostProcessor> postprocessor_;
    bool running_;
    std::thread thread_;
    cudaStream_t stream_;
    float threshold_ = 0.5f;
    bool skip_normalization_;
};

} // namespace pipeline