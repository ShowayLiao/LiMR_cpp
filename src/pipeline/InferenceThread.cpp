#include "pipeline/InferenceThread.h"
#include <iostream>
#include <ctime>
#include "pipeline/FrameTask.h"
#include "common/CudaMemory.hpp"

namespace pipeline {

InferenceThread::InferenceThread(
    SafeQueue<FrameTaskPtr>& input_queue,
    SafeQueue<FrameTaskPtr>& output_queue,
    trt::TrtEngine* engine,
    int render_width,
    int render_height,
    bool skip_normalization
) : 
    input_queue_(input_queue),
    output_queue_(output_queue),
    engine_(engine),
    render_width_(render_width),
    render_height_(render_height),
    running_(false),
    skip_normalization_(skip_normalization)
{
    // Get model input shape
    nvinfer1::Dims input_shape = engine_->getInputShape();
    model_input_width_ = input_shape.d[2];
    model_input_height_ = input_shape.d[3];
    
    // Get model output shape
    nvinfer1::Dims output_shape = engine_->getOutputShape();
    model_output_width_ = output_shape.d[2];
    model_output_height_ = output_shape.d[3];
    
    std::cout << "[InferenceThread] Model input size: " << model_input_width_ << "x" << model_input_height_ << std::endl;
    std::cout << "[InferenceThread] Model output size: " << model_output_width_ << "x" << model_output_height_ << std::endl;
    std::cout << "[InferenceThread] Render size: " << render_width_ << "x" << render_height_ << std::endl;
    
    // Allocate workspace buffers
    size_t input_size = 3 * model_input_width_ * model_input_height_ * sizeof(float);
    size_t output_size = model_output_width_ * model_output_height_ * sizeof(float);
    size_t mask_size = model_output_width_ * model_output_height_ * sizeof(uint8_t);
    
    m_d_trt_input = make_device_buffer(input_size);
    m_d_raw_anomaly_map = make_device_buffer(output_size);
    m_d_raw_mask = make_device_buffer(mask_size);
    
    // Initialize preprocessor with model input size
    preprocessor_ = std::make_unique<Preprocessor>(model_input_width_, model_input_height_, skip_normalization_);
    
    // Initialize postprocessor with model output size and render size
    postprocessor_ = std::make_unique<PostProcessor>(model_output_width_, model_output_height_, render_width_, render_height_);
    
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
        // Shutdown the queue to wake up any waiting threads
        input_queue_.shutdown();
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
        // [DEBUG] 标记进入等待状态
        std::cout << "[InferenceThread] Waiting for task from queue..." << std::endl;
        
        FrameTaskPtr task;
        input_queue_.pop(task);
        
        if (!task) break;

        try {
            // std::cout << "[InferenceThread] Processing Task ID: " << task->frame_id << std::endl;
            
            // Record start time
            task->start_time = static_cast<double>(std::clock()) / CLOCKS_PER_SEC;

            // Step 1: Preprocessing (GPU-based) - Resize and normalize
            preprocessor_->process(task->original_image, m_d_trt_input.get(), stream_);

            // [DEBUG] 推理开始前的最后标记
            std::cout << "[InferenceThread] Starting TensorRT Inference..." << std::endl;
            engine_->infer(m_d_trt_input.get(), 1);

            // Step 3: Get output buffers from engine
            float* d_score = (float*)engine_->getBuffer("pred_score");
            bool* d_label = (bool*)engine_->getBuffer("pred_label");
            float* d_map = (float*)engine_->getBuffer("anomaly_map");

            // Copy output to our workspace buffers
            size_t anomaly_map_size = model_output_width_ * model_output_height_ * sizeof(float);
            cudaMemcpyAsync(m_d_raw_anomaly_map.get(), d_map, anomaly_map_size, cudaMemcpyDeviceToDevice, stream_);

            // Step 4: Postprocessing (GPU-based)
            // Ensure output buffers are ready
            task->ensureOutputBuffers();
            
            // Process and write back results
            postprocessor_->process(engine_, task, threshold_, stream_, 
                                   static_cast<float*>(m_d_raw_anomaly_map.get()), 
                                   task->target_width, task->target_height);

            // Record end time
            task->end_time = static_cast<double>(std::clock()) / CLOCKS_PER_SEC;

            // [DEBUG] 完成标记
            // 移除输出信息
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