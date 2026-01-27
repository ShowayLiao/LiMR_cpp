#pragma once

#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>
#include "pipeline/FrameTask.h"
#include "common/SafeQueue.hpp"
#include "common/CudaMemory.hpp"

namespace pipeline {

class Pipeline; // 前向声明

class InputThread {
public:
    InputThread(const std::string& source, int width, int height, Pipeline* pipeline = nullptr);
    ~InputThread();

    void start();
    void stop();
    bool isRunning() const;

    SafeQueue<FrameTaskPtr>& getInputQueue();

private:
    void run();

    std::string source_;
    int width_;
    int height_;
    cv::VideoCapture cap_;
    std::thread thread_;
    std::atomic<bool> running_;
    SafeQueue<FrameTaskPtr> input_queue_;
    long frame_id_;
    Pipeline* pipeline_; // Pipeline指针，用于获取任务池
};

} // namespace pipeline
