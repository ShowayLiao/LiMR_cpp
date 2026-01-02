#include "postprocess.h"
#include <stdexcept>
#include <numeric>
#include <half.h>

namespace postprocess {

Postprocessor::Postprocessor()
    : outputHeight_(0),
      outputWidth_(0),
      numChannels_(512),
      featureHeight_(14),
      featureWidth_(14),
      type_("F32") {
    
    // Create CUDA stream
    cudaStreamCreate(&stream_);
    
    // Initialize result
    result_.anomalyMap = nullptr;
    result_.maxScore = 0.0f;
    result_.maxLoc = cv::Point(0, 0);
}

Postprocessor::~Postprocessor() {
    // Destroy CUDA stream
    cudaStreamDestroy(stream_);
}

bool Postprocessor::init(int outputHeight, int outputWidth, const std::string& type) {
    try {
        outputHeight_ = outputHeight;
        outputWidth_ = outputWidth;
        type_ = type;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Postprocessor init failed: " << e.what() << std::endl;
        return false;
    }
}

void Postprocessor::postprocess(float* d_outputA, float* d_anomalyMap, int outputWidth, int outputHeight) {
    // Configure tensor type based on input
    auto tensorType = (type_ == "F16") ? torch::kFloat16 : torch::kFloat32;
    torch::TensorOptions options = torch::TensorOptions().dtype(tensorType).device(torch::kCUDA);
    
    // For simplicity, we'll create a dummy anomaly map based on the output
    // In a real scenario, you would process the output tensor(s) here
    torch::Tensor outputTensor = torch::from_blob(
        d_outputA,
        {1, 1, outputHeight, outputWidth}, // Assuming output is BCHW format
        options
    );
    
    // Create anomaly map (for this example, we'll just use the output directly)
    torch::Tensor anomalyMap = outputTensor.squeeze(0).squeeze(0);
    
    // Ensure anomaly map has the correct dimensions (H x W)
    if (anomalyMap.dim() != 2) {
        throw std::runtime_error("Anomaly map must be 2D");
    }
    
    // Ensure anomaly map has the correct size
    if (anomalyMap.size(0) != outputHeight || anomalyMap.size(1) != outputWidth) {
        // Resize if dimensions don't match
        anomalyMap = resizeTensor(anomalyMap, outputHeight, outputWidth);
        anomalyMap = anomalyMap.squeeze(0); // Remove channel dimension added by resizeTensor
    }
    
    // Convert to float32 to ensure consistent memory layout for d_anomalyMap
    anomalyMap = anomalyMap.to(torch::kFloat32);
    
    // Ensure tensor is contiguous (critical for correct memory layout)
    if (!anomalyMap.is_contiguous()) {
        anomalyMap = anomalyMap.contiguous();
    }
    
    // Copy result to output GPU pointer (always float32)
    cudaMemcpyAsync(d_anomalyMap, anomalyMap.data_ptr<float>(),
                   outputHeight * outputWidth * sizeof(float),
                   cudaMemcpyDeviceToDevice, stream_);
    
    // Synchronize stream
    cudaStreamSynchronize(stream_);
    
    // Update result
    result_.anomalyMap = d_anomalyMap;
    
    // Calculate max score and location (this would require copying data to CPU)
    // For now, we'll leave it as 0 since it's not critical for the core functionality
    // and would add overhead
}

void Postprocessor::postprocess(float* d_outputA, float* d_outputB, float* d_anomalyMap, int outputWidth, int outputHeight) {
    // Configure tensor type based on input
    auto tensorType = (type_ == "F16") ? torch::kFloat16 : torch::kFloat32;
    const size_t elem_size = (tensorType == torch::kFloat16) ? sizeof(half_float::half) : sizeof(float);
    torch::TensorOptions options = torch::TensorOptions().dtype(tensorType).device(torch::kCUDA);
    
    // Create temporary anomaly map
    torch::Tensor TempAnomalyMap = torch::zeros({1, 1, outputHeight, outputWidth}, options);
    
    // Create tensors from output buffers using dynamically configured feature map dimensions
    // These values can be set using setOutputShape() based on actual model output
    torch::Tensor outputTensorA = torch::from_blob(
        d_outputA,
        {1, numChannels_, featureHeight_, featureWidth_}, // BCHW format
        options
    );
    
    torch::Tensor outputTensorB = torch::from_blob(
        d_outputB,
        {1, numChannels_, featureHeight_, featureWidth_}, // BCHW format
        options
    );
    
    // Calculate cosine similarity between the two outputs
    torch::Tensor similarity = calculateCosineSimilarity(outputTensorA, outputTensorB);
    
    // similarity is already anomaly score (1 - cosine_similarity)
    torch::Tensor temp = similarity;
    temp = temp.to(tensorType); // Convert back to desired type
    
    // Resize the tensor to the output size
    temp = torch::unsqueeze(temp, 1);
    std::vector<int64_t> resize_size = {outputHeight, outputWidth};
    temp = torch::nn::functional::interpolate(
        temp,
        torch::nn::functional::InterpolateFuncOptions()
            .size(resize_size)
            .mode(torch::kBilinear)
            .align_corners(true)
    );
    
    // Accumulate to temporary anomaly map
    TempAnomalyMap.add_(temp);
    
    // Squeeze the temporary anomaly map
    TempAnomalyMap = TempAnomalyMap.squeeze(0).squeeze(0);
    
    // Ensure tensor is contiguous (critical for correct memory layout)
    if (!TempAnomalyMap.is_contiguous()) {
        TempAnomalyMap = TempAnomalyMap.contiguous();
    }
    
    // Convert to float32 to ensure consistent memory layout for d_anomalyMap
    TempAnomalyMap = TempAnomalyMap.to(torch::kFloat32);
    
    // Copy result to output GPU pointer (always float32)
    cudaMemcpyAsync(d_anomalyMap, TempAnomalyMap.data_ptr<float>(),
                   outputHeight * outputWidth * sizeof(float),
                   cudaMemcpyDeviceToDevice, stream_);
    
    // Synchronize stream
    cudaStreamSynchronize(stream_);
    
    // Update result
    result_.anomalyMap = d_anomalyMap;
    
    // Calculate max score and location (this would require copying data to CPU)
    // For now, we'll leave it as 0 since it's not critical for the core functionality
    // and would add overhead
}

AnomalyResult Postprocessor::getAnomalyResult() const {
    return result_;
}

void Postprocessor::setOutputShape(int numChannels, int featureHeight, int featureWidth) {
    numChannels_ = numChannels;
    featureHeight_ = featureHeight;
    featureWidth_ = featureWidth;
}

torch::Tensor Postprocessor::calculateCosineSimilarity(const torch::Tensor& studentTensor, const torch::Tensor& teacherTensor) {
    // Convert to float32 for calculation if needed
    torch::Tensor studentFloat = (studentTensor.dtype() == torch::kFloat16) ? 
        studentTensor.to(torch::kFloat32) : studentTensor;
    torch::Tensor teacherFloat = (teacherTensor.dtype() == torch::kFloat16) ? 
        teacherTensor.to(torch::kFloat32) : teacherTensor;
    
    // Calculate cosine similarity
    torch::Tensor similarity = torch::nn::functional::cosine_similarity(
        studentFloat, teacherFloat,
        torch::nn::functional::CosineSimilarityFuncOptions().dim(1)
    );
    
    // Convert back to desired type
    torch::Tensor anomaly = 1.0f - similarity;
    return anomaly.to(studentTensor.dtype());
}

torch::Tensor Postprocessor::resizeTensor(const torch::Tensor& tensor, int height, int width) {
    // Add channel dimension if needed
    torch::Tensor temp = torch::unsqueeze(tensor, 1);
    
    // Resize tensor
    std::vector<int64_t> output_size = {height, width};
    torch::Tensor resized = torch::nn::functional::interpolate(
        temp,
        torch::nn::functional::InterpolateFuncOptions()
            .size(output_size)
            .mode(torch::kBilinear)
            .align_corners(true)
    );
    
    return resized;
}

void Postprocessor::process_and_accumulate(float* student_feature, float* teacher_feature, float* output_anomaly_map, int feature_c, int feature_h, int feature_w, int out_h, int out_w) {
    // Configure tensor type based on input
    auto tensorType = (type_ == "F16") ? torch::kFloat16 : torch::kFloat32;
    const size_t elem_size = (tensorType == torch::kFloat16) ? sizeof(half_float::half) : sizeof(float);
    torch::TensorOptions options = torch::TensorOptions().dtype(tensorType).device(torch::kCUDA);
    
    // Create tensors from the input feature maps
    torch::Tensor studentTensor = torch::from_blob(
        student_feature,
        {1, feature_c, feature_h, feature_w}, // BCHW format
        options
    );
    
    torch::Tensor teacherTensor = torch::from_blob(
        teacher_feature,
        {1, feature_c, feature_h, feature_w}, // BCHW format
        options
    );
    
    // Calculate cosine similarity between student and teacher features
    torch::Tensor similarity = calculateCosineSimilarity(studentTensor, teacherTensor);
    
    // similarity is already anomaly score (1 - cosine_similarity)
    torch::Tensor temp = similarity;
    temp = temp.to(tensorType); // Convert back to desired type
    
    // Resize the tensor to the output size
    temp = torch::unsqueeze(temp, 1);
    std::vector<int64_t> resize_size = {out_h, out_w};
    temp = torch::nn::functional::interpolate(
        temp,
        torch::nn::functional::InterpolateFuncOptions()
            .size(resize_size)
            .mode(torch::kBilinear)
            .align_corners(true)
    );
    
    // Convert output_anomaly_map to a tensor for in-place addition
    torch::Tensor anomalyMapTensor = torch::from_blob(
        output_anomaly_map,
        {1, 1, out_h, out_w}, // BCHW format
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
    );
    
    // Accumulate the result to the output anomaly map
    anomalyMapTensor.add_(temp.to(torch::kFloat32));
    
    // Synchronize CUDA stream to ensure all operations are completed
    cudaStreamSynchronize(stream_);
}

} // namespace postprocess