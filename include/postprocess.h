#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <string>
#include <vector>

namespace postprocess {

struct AnomalyResult {
    float* anomalyMap; // Pointer to GPU memory containing the anomaly map
    float maxScore;    // Maximum anomaly score
    cv::Point maxLoc;  // Location of maximum anomaly score
};

class Postprocessor {
public:
    Postprocessor();
    ~Postprocessor();
    
    bool init(int outputHeight = 256, int outputWidth = 256, const std::string& type = "F32");
    void postprocess(float* d_outputA, float* d_anomalyMap, int outputWidth, int outputHeight);
    void postprocess(float* d_outputA, float* d_outputB, float* d_anomalyMap, int outputWidth, int outputHeight);
    void process_and_accumulate(float* student_feature, float* teacher_feature, float* output_anomaly_map, int feature_c, int feature_h, int feature_w, int out_h, int out_w);
    
    AnomalyResult getAnomalyResult() const;
    
private:
    int outputHeight_;
    int outputWidth_;
    int numChannels_;
    int featureHeight_;
    int featureWidth_;
    std::string type_;
    cudaStream_t stream_;
    
    AnomalyResult result_;
    
    // Helper functions
    torch::Tensor calculateCosineSimilarity(const torch::Tensor& studentTensor, const torch::Tensor& teacherTensor);
    torch::Tensor resizeTensor(const torch::Tensor& tensor, int height, int width);
    
public:
    // Method to dynamically configure feature map dimensions
    void setOutputShape(int numChannels, int featureHeight, int featureWidth);
};

} // namespace postprocess

#endif // POSTPROCESS_H