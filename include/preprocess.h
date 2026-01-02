#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

namespace preprocess {

class Preprocessor {
public:
    Preprocessor();
    ~Preprocessor();
    
    bool init(int inputWidth, int inputHeight, float scale, float mean, const std::string& type = "F32");
    void preprocess(const cv::Mat& inputFrame, void*& outputGpuPtr);
    
private:
    int inputWidth_;
    int inputHeight_;
    std::string type_;
    cudaStream_t stream_;
    float scale_;
    float mean_[3];
    float std_[3];
    
    // GPU memory buffers
    cv::cuda::GpuMat gpuFrame_;
    cv::cuda::GpuMat gpuResized_;
    cv::cuda::GpuMat gpuConverted_;
    cv::cuda::GpuMat gpuNormalized_;
    cv::cuda::GpuMat gpuBlob_;
    void* outputBuffer_;
    size_t outputBufferSize_;
};

} // namespace preprocess

#endif // PREPROCESS_H