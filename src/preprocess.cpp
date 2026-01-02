#include "preprocess.h"
#include "utils.cuh"
#include <stdexcept>

namespace preprocess {

Preprocessor::Preprocessor()
        : inputWidth_(0),
          inputHeight_(0),
          type_("F32"),
          scale_(1.0f / 255.0f),
          mean_{0.485f, 0.456f, 0.406f},
          std_{0.229f, 0.224f, 0.225f},
          outputBuffer_(nullptr),
          outputBufferSize_(0) {
        // Create CUDA stream for asynchronous operations
        cudaStreamCreate(&stream_);
    }

bool Preprocessor::init(int inputWidth, int inputHeight, float scale, float mean, const std::string& type) {
    try {
        inputWidth_ = inputWidth;
        inputHeight_ = inputHeight;
        type_ = type;
        scale_ = scale;
        // Use ImageNet standard mean and std for RGB
        mean_[0] = 0.485f;
        mean_[1] = 0.456f;
        mean_[2] = 0.406f;
        std_[0] = 0.229f;
        std_[1] = 0.224f;
        std_[2] = 0.225f;
        
        // Calculate output buffer size based on data type
        size_t elemSize = (type_ == "F16") ? sizeof(__half) : sizeof(float);
        outputBufferSize_ = 1 * 3 * inputWidth_ * inputHeight_ * elemSize;
        
        // Allocate output buffer on GPU
        cudaMalloc(&outputBuffer_, outputBufferSize_);
        
        // Initialize GPU mats based on data type
        if (type_ == "F32") {
            gpuBlob_.create(1, 3 * inputWidth_ * inputHeight_, CV_32F);
        } else if (type_ == "F16") {
            gpuBlob_.create(1, 3 * inputWidth_ * inputHeight_, CV_16F);
        } else {
            throw std::runtime_error("Unsupported data type: " + type_);
        }
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

Preprocessor::~Preprocessor() {
    // Free allocated GPU memory
    if (outputBuffer_ != nullptr) {
        cudaFree(outputBuffer_);
    }
    
    // Destroy CUDA stream
    cudaStreamDestroy(stream_);
}

void Preprocessor::preprocess(const cv::Mat& inputFrame, void*& outputGpuPtr) {
    // Upload input frame to GPU
    gpuFrame_.upload(inputFrame);
    
    // Convert from BGR to RGB
    cv::cuda::cvtColor(gpuFrame_, gpuFrame_, cv::COLOR_BGR2RGB);
    
    // Resize the image
    cv::cuda::resize(gpuFrame_, gpuResized_, cv::Size(inputWidth_, inputHeight_), 0, 0, cv::INTER_LINEAR);
    
    // Convert data type and normalize
    gpuResized_.convertTo(gpuResized_, CV_32F);
    
    if (type_ == "F16") {
        cv::cuda::convertFp16(gpuResized_, gpuConverted_);
    } else if (type_ == "F32") {
        gpuConverted_ = gpuResized_;
    } else {
        throw std::runtime_error("Unsupported data type: " + type_);
    }
    
    // Normalize the image
    cv::cuda::multiply(gpuConverted_, cv::Scalar(scale_), gpuConverted_);
    cv::cuda::subtract(gpuConverted_, cv::Scalar(mean_[0], mean_[1], mean_[2]), gpuConverted_);
    cv::cuda::divide(gpuConverted_, cv::Scalar(std_[0], std_[1], std_[2]), gpuConverted_);
    
    // Convert from HWC to CHW format
    ConvertToBlob(gpuConverted_, gpuBlob_, stream_, type_);
    
    // Copy processed data to output buffer
    cudaMemcpyAsync(outputBuffer_, gpuBlob_.data, outputBufferSize_, cudaMemcpyDeviceToDevice, stream_);
    
    // Synchronize stream to ensure all operations are completed
    cudaStreamSynchronize(stream_);
    
    // Set output pointer
    outputGpuPtr = outputBuffer_;
}

} // namespace preprocess