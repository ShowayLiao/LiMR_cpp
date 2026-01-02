#ifndef INFERENCE_H
#define INFERENCE_H

#include <NvInfer.h>
#include <string>
#include <vector>
#include <../samples/common/logger.h>

namespace inference {

class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();
    
    bool init(const std::string& modelPath, int batchSize = 1);
    void infer(void* inputGpuPtr);
    
    nvinfer1::ICudaEngine* getEngine() const;
    nvinfer1::IExecutionContext* getContext() const;
    
    size_t getInputSize() const;
    size_t getOutputSize() const;
    nvinfer1::Dims getOutputShape() const;
    
    const std::vector<void*>& getBuffers() const;
    int getNbBindings() const;
    const char* getBindingName(int index) const;
    bool isOutput(int index) const;
    
private:
    sample::Logger logger_;
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    
    std::vector<void*> buffers_;
    cudaStream_t stream_;
    
    int batchSize_;
    size_t inputSize_;
    size_t outputSize_;
    
    // Helper function to calculate tensor volume
    size_t getVolume(const nvinfer1::Dims& dims);
};

} // namespace inference

#endif // INFERENCE_H