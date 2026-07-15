#pragma once

#include <string>
#include <vector>
#include <map>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <filesystem>
#include "engine/InputShape.h"

namespace trt {

enum class Precision {
    FP32,
    FP16,
    INT8
};

struct Binding {
    int index = -1;
    std::string name;
    size_t size = 0;      // Bytes required by the active execution-context shape.
    size_t capacity = 0;  // Bytes allocated at buffer; may exceed size after a shape shrink.
    nvinfer1::Dims dims{};
    nvinfer1::DataType type = nvinfer1::DataType::kFLOAT;
    bool isInput = false;
    void* buffer = nullptr;
};

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

class TrtEngine {
public:
    TrtEngine();
    ~TrtEngine();

    bool load(const std::string& modelPath, Precision precision = Precision::FP16,
              int dynamic_input_height = 224, int dynamic_input_width = 224);

    // Configure the active NCHW input size for a dynamic engine before creating a pipeline.
    bool setInputShape(int height, int width);
    bool isInputSpatialDynamic() const;
    static bool calculateTensorBytes(const nvinfer1::Dims& dims,
                                     nvinfer1::DataType type,
                                     size_t& bytes);
    static std::filesystem::path getEngineCachePath(const std::filesystem::path& onnxPath,
                                                    Precision precision,
                                                    int profileHeight,
                                                    int profileWidth);

    void infer(void* inputData, int batchSize = 1, cudaStream_t stream = nullptr);

    void* getBuffer(const std::string& name);
    const Binding* getTensorInfo(const std::string& name) const;

    nvinfer1::ICudaEngine* getEngine() const;
    nvinfer1::IExecutionContext* getContext() const;
    cudaStream_t getStream() const;

    const std::map<std::string, Binding>& getBindings() const;
    int getNbBindings() const;
    const char* getBindingName(int index) const;
    bool isOutput(int index) const;

    size_t getInputSize() const;
    size_t getOutputSize() const;
    nvinfer1::Dims getInputShape() const;
    nvinfer1::Dims getOutputShape() const;

private:
    bool loadFromPlan(const std::string& enginePath);
    bool buildFromOnnx(const std::string& onnxPath, const std::string& enginePath,
                       Precision precision, int dynamic_input_height, int dynamic_input_width);
    bool saveEngine(const std::string& enginePath, const void* data, size_t size);
    std::vector<char> readBinaryFile(const std::string& path);
    bool allocateBuffers(int maxBatchSize);
    bool updateBindingBuffers();
    void clearLoadedEngine();
    void freeBuffers();
    static size_t getElementSize(nvinfer1::DataType type);

    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_;

    std::map<std::string, Binding> bindings_;
    std::string input_name_;
    bool input_spatial_dynamic_ = false;
    int max_batch_size_;
    bool initialized_;
};

} // namespace trt
