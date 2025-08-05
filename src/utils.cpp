# include "utils.h"
# include <iostream>
# include <fstream>
# include <string>
# include "NvInfer.h"
# include "NvOnnxParser.h"
# include <opencv2/opencv.hpp>
# include <torch/torch.h>
# include <stdint.h>

class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

void build_engine(const std::string& onnxModelPath, 
                  const std::string& engineFilePath, 
                  int batchSize, 
                  int inputWidth, 
                  int inputHeight, 
                  int outputWidth, 
                  int outputHeight) {
    
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    if (!parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        throw std::runtime_error("Failed to parse ONNX model: " + onnxModelPath);
    }

    auto config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30); // Set workspace size to 1GB
    
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{batchSize, 3, inputHeight, inputWidth});
    config->addOptimizationProfile(profile);

    nvinfer1::IHostMemory* engineData = builder->buildSerializedNetwork(*network, *config);

    if (!engineData) {
        throw std::runtime_error("Failed to build serialized network from ONNX model: " + onnxModelPath);
    }

    std::ofstream engineFile(engineFilePath, std::ios::binary);

    if (!engineFile) {
        throw std::runtime_error("Failed to open engine file for writing: " + engineFilePath);
    }
    engineFile.write(static_cast<const char*>(engineData->data()), engineData->size());
    engineFile.close();

    // Clean up resources
    delete engineData;
    delete parser;
    delete network;
    delete builder;
    delete config;

}


void cal_anomaly_map(std::vector<torch::Tensor>& StuOutputTensors, 
                     std::vector<torch::Tensor>& TeaOutputTensors, 
                     cv::Mat*& anomalyMap,
                     const int batchSize,
                     const int outputWidth,
                     const int outputHeight) {
    torch::Tensor TempOut = torch::zeros({batchSize,1,outputHeight,outputWidth});
    for (int i = 0;i<StuOutputTensors.size();i++){
        if (StuOutputTensors[i].sizes() != TeaOutputTensors[i].sizes()) {
            throw std::runtime_error("Output tensors from student and teacher models must have the same shape.");
        }
        // Calculate the absolute difference between student and teacher outputs
        auto temp = torch::nn::functional::cosine_similarity(StuOutputTensors[0], 
                                                            TeaOutputTensors[0],
                                                            torch::nn::functional::CosineSimilarityFuncOptions().dim(1));
        temp = torch::unsqueeze(temp, 1); // Add a channel dimension
        temp = torch::nn::functional::interpolate(temp, 
                                                torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({outputWidth})).mode(torch::kBilinear).align_corners(true));       
        TempOut = TempOut + temp;                                      
    }
    TempOut.squeeze(0); // Remove the channel dimension
    
    
}


