#pragma once

#include <string>
#include <vector>
#include <map>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <filesystem>

namespace trt {

enum class Precision {
    FP32,
    FP16,
    INT8
};

struct Binding {
    int index;
    std::string name;
    size_t size;
    nvinfer1::Dims dims;
    nvinfer1::DataType type;
    bool isInput;
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

    bool load(const std::string& modelPath, Precision precision = Precision::FP16);

    void infer(void* inputData, int batchSize = 1);

    void* getBuffer(const std::string& name);

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
    bool buildFromOnnx(const std::string& onnxPath, const std::string& enginePath, Precision precision);
    bool saveEngine(const std::string& enginePath, const void* data, size_t size);
    std::vector<char> readBinaryFile(const std::string& path);
    void allocateBuffers(int maxBatchSize);
    void freeBuffers();
    size_t getElementSize(nvinfer1::DataType type);

    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_;

    std::map<std::string, Binding> bindings_;
    std::string input_name_;
    int max_batch_size_;
    bool initialized_;
};

} // namespace trt
