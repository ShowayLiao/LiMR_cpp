#include "pipeline/InferenceThread.h"
#include "pipeline/AnomalyMapShape.h"
#include <iostream>
#include <stdexcept>
#include <ctime>
#include <limits>
#include "pipeline/FrameTask.h"
#include "common/CudaMemory.hpp"

namespace pipeline {

InferenceThread::InferenceThread(
    SafeQueue<FrameTaskPtr>& input_queue,
    SafeQueue<FrameTaskPtr>& output_queue,
    SafeQueue<FrameTaskPtr>& task_pool,
    trt::TrtEngine* engine,
    int render_width,
    int render_height,
    bool skip_normalization
) : 
    input_queue_(input_queue),
    output_queue_(output_queue),
    task_pool_(task_pool),
    engine_(engine),
    render_width_(render_width),
    render_height_(render_height),
    running_(false),
    skip_normalization_(skip_normalization)
{
    if (!engine_) {
        throw std::invalid_argument("InferenceThread requires a TensorRT engine");
    }

    const trt::Binding* input_binding = nullptr;
    for (const auto& item : engine_->getBindings()) {
        if (item.second.isInput) {
            if (input_binding) throw std::runtime_error("Multiple TensorRT inputs are not supported");
            input_binding = &item.second;
        }
    }
    if (!input_binding || input_binding->type != nvinfer1::DataType::kFLOAT ||
        input_binding->dims.nbDims != 4 || input_binding->dims.d[0] != 1 || input_binding->dims.d[1] != 3) {
        throw std::runtime_error("TensorRT input contract must be FLOAT [1,3,H,W]");
    }

    const trt::Binding* map_binding = engine_->getTensorInfo("anomaly_map");
    const trt::Binding* score_binding = engine_->getTensorInfo("pred_score");
    const trt::Binding* label_binding = engine_->getTensorInfo("pred_label");
    const auto anomaly_map_shape = map_binding
        ? parse_anomaly_map_shape(map_binding->dims)
        : std::nullopt;
    if (!map_binding || map_binding->type != nvinfer1::DataType::kFLOAT || !anomaly_map_shape) {
        throw std::runtime_error(
            "TensorRT anomaly_map contract must be FLOAT [1,1,H,W] or [1,H,W]");
    }
    if (!score_binding || score_binding->type != nvinfer1::DataType::kFLOAT ||
        score_binding->size < sizeof(float)) {
        throw std::runtime_error("TensorRT pred_score contract must contain at least one FLOAT value");
    }
    if (!label_binding || label_binding->type != nvinfer1::DataType::kBOOL ||
        label_binding->size < sizeof(uint8_t)) {
        throw std::runtime_error("TensorRT pred_label contract must contain at least one BOOL value");
    }

    // Get model input shape
    const nvinfer1::Dims input_shape = input_binding->dims;
    if (input_shape.nbDims != 4 || input_shape.d[2] <= 0 || input_shape.d[3] <= 0 ||
        input_shape.d[2] > std::numeric_limits<int>::max() ||
        input_shape.d[3] > std::numeric_limits<int>::max()) {
        throw std::runtime_error("TensorRT returned an invalid NCHW input shape");
    }
    model_input_height_ = static_cast<int>(input_shape.d[2]);
    model_input_width_ = static_cast<int>(input_shape.d[3]);
    
    // Get model output shape
    model_output_height_ = anomaly_map_shape->height;
    model_output_width_ = anomaly_map_shape->width;
    
    std::cout << "[InferenceThread] Model input size: " << model_input_width_ << "x" << model_input_height_ << std::endl;
    std::cout << "[InferenceThread] Model output size: " << model_output_width_ << "x" << model_output_height_ << std::endl;
    std::cout << "[InferenceThread] Render size: " << render_width_ << "x" << render_height_ << std::endl;
    
    // Allocate workspace buffers
    const size_t expected_input_size = 3U * static_cast<size_t>(model_input_width_) *
        static_cast<size_t>(model_input_height_) * sizeof(float);
    trt_input_bytes_ = engine_->getInputSize();
    if (trt_input_bytes_ != expected_input_size) {
        throw std::runtime_error("TensorRT input binding size does not match the NCHW float preprocessor output");
    }
    const size_t output_size = anomaly_map_shape->elements * sizeof(float);
    if (map_binding->size != output_size) {
        throw std::runtime_error("TensorRT anomaly_map byte size does not match its FLOAT shape");
    }
    anomaly_map_bytes_ = map_binding->size;
    const size_t mask_size = static_cast<size_t>(model_output_width_) *
        static_cast<size_t>(model_output_height_) * sizeof(uint8_t);
    
    m_d_trt_input = make_device_buffer(trt_input_bytes_);
    m_d_raw_anomaly_map = make_device_buffer(output_size);
    m_d_raw_mask = make_device_buffer(mask_size);
    
    // Initialize preprocessor with model input size
    preprocessor_ = std::make_unique<Preprocessor>(model_input_width_, model_input_height_, skip_normalization_);
    
    // Initialize postprocessor with model output size and render size
    postprocessor_ = std::make_unique<PostProcessor>(model_output_width_, model_output_height_, render_width_, render_height_);
    
    const cudaError_t stream_status = cudaStreamCreate(&stream_);
    if (stream_status != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to create inference CUDA stream: ") +
                                 cudaGetErrorString(stream_status));
    }
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
        
        FrameTaskPtr task;
        if (!input_queue_.pop(task) || !task) break;

        try {
            // std::cout << "[InferenceThread] Processing Task ID: " << task->frame_id << std::endl;
            
            // Record start time
            task->start_time = static_cast<double>(std::clock()) / CLOCKS_PER_SEC;

            // Step 1: Preprocessing (GPU-based) - Resize and normalize
            preprocessor_->process(task->original_image, m_d_trt_input.get(), stream_);

            // Preprocessing and TensorRT are enqueued on stream_, preserving their order
            // without a CPU-side synchronization point.
            if (engine_->getInputSize() != trt_input_bytes_) {
                throw std::runtime_error("TensorRT input binding size changed after pipeline creation");
            }

            // [DEBUG] 推理开始前的最后标记
            engine_->infer(m_d_trt_input.get(), 1, stream_);

            // Step 3: Get output buffers from engine
            float* d_map = static_cast<float*>(engine_->getBuffer("anomaly_map"));
            const trt::Binding* map_info = engine_->getTensorInfo("anomaly_map");
            const auto current_map_shape = map_info
                ? parse_anomaly_map_shape(map_info->dims)
                : std::nullopt;
            if (!d_map || !map_info || map_info->type != nvinfer1::DataType::kFLOAT ||
                !current_map_shape || current_map_shape->height != model_output_height_ ||
                current_map_shape->width != model_output_width_ ||
                map_info->size != anomaly_map_bytes_) {
                throw std::runtime_error("TensorRT anomaly_map binding changed or is unavailable");
            }

            // Copy output to our workspace buffers
            const cudaError_t map_copy_status = cudaMemcpyAsync(
                m_d_raw_anomaly_map.get(), d_map, anomaly_map_bytes_, cudaMemcpyDeviceToDevice, stream_);
            if (map_copy_status != cudaSuccess) {
                throw std::runtime_error(std::string("Failed to copy TensorRT anomaly_map: ") +
                                         cudaGetErrorString(map_copy_status));
            }

            // Step 4: Postprocessing (GPU-based)
            // Ensure output buffers are ready
            if (!task->ensureOutputBuffers()) {
                throw std::runtime_error("Invalid render output dimensions");
            }
            
            // Process and write back results
            postprocessor_->process(engine_, task, threshold_.load(), stream_,
                                   static_cast<float*>(m_d_raw_anomaly_map.get()), 
                                   task->target_width, task->target_height);

            // Record end time
            task->end_time = static_cast<double>(std::clock()) / CLOCKS_PER_SEC;

            // [DEBUG] 完成标记
            // 移除输出信息
            if (auto discarded = output_queue_.push_latest(std::move(task))) {
                task_pool_.push(std::move(*discarded));
            }

        } catch (const std::exception& e) {
            std::cerr << "[InferenceThread] Error processing task " << task->frame_id << ": " << e.what() << std::endl;
            task->is_valid = false;
            if (auto discarded = output_queue_.push_latest(std::move(task))) {
                task_pool_.push(std::move(*discarded));
            }
        }
    }

    std::cout << "[InferenceThread] Exiting run loop" << std::endl;
}

} // namespace pipeline
