#include "inference.h"
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace inference {

InferenceEngine::InferenceEngine()
    : runtime_(nullptr),
      engine_(nullptr),
      context_(nullptr),
      batchSize_(1),
      inputSize_(0),
      outputSize_(0) {
    
    // Create CUDA stream
    cudaStreamCreate(&stream_);
}

InferenceEngine::~InferenceEngine() {
    // Free GPU memory buffers
    for (void* buffer : buffers_) {
        if (buffer != nullptr) {
            cudaFree(buffer);
        }
    }
    
    // Destroy execution context
    if (context_ != nullptr) {
        delete context_;
    }
    
    // Destroy engine
    if (engine_ != nullptr) {
        delete engine_;
    }
    
    // Destroy runtime
    if (runtime_ != nullptr) {
        delete runtime_;
    }
    
    // Destroy CUDA stream
    cudaStreamDestroy(stream_);
}

bool InferenceEngine::init(const std::string& modelPath, int batchSize) {
    try {
        batchSize_ = batchSize;
        
        // Create runtime
        runtime_ = nvinfer1::createInferRuntime(logger_);
        if (!runtime_) {
            throw std::runtime_error("Failed to create TensorRT runtime");
        }
        
        // Load and deserialize engine
        std::ifstream modelFile(modelPath, std::ios::binary | std::ios::ate);
        if (!modelFile.is_open()) {
            throw std::runtime_error("Failed to open engine file: " + modelPath);
        }
        
        size_t modelSize = modelFile.tellg();
        modelFile.seekg(0, std::ios::beg);
        
        std::vector<char> modelData(modelSize);
        modelFile.read(modelData.data(), modelSize);
        
        engine_ = runtime_->deserializeCudaEngine(modelData.data(), modelSize);
        if (!engine_) {
            throw std::runtime_error("Failed to deserialize engine");
        }
        
        // Create execution context
        context_ = engine_->createExecutionContext();
        if (!context_) {
            throw std::runtime_error("Failed to create execution context");
        }
        
        // Set input shape
        context_->setInputShape(engine_->getIOTensorName(0), nvinfer1::Dims4(batchSize_, 3, 224, 224));
        
        // Allocate memory for input and output tensors
        buffers_.resize(engine_->getNbIOTensors());
        
        for (int i = 0; i < engine_->getNbIOTensors(); i++) {
            const char* tensorName = engine_->getIOTensorName(i);
            nvinfer1::Dims shape = context_->getTensorShape(tensorName);
            size_t size = getVolume(shape) * sizeof(float);
            
            cudaMalloc(&buffers_[i], size);
            context_->setTensorAddress(tensorName, buffers_[i]);
            
            // Store input and output sizes
            if (engine_->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT) {
                inputSize_ = size;
            } else {
                outputSize_ = size;
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "InferenceEngine init failed: " << e.what() << std::endl;
        return false;
    }
}

void InferenceEngine::infer(void* inputGpuPtr) {
    // Copy input data to engine input buffer
    cudaMemcpyAsync(buffers_[0], inputGpuPtr, inputSize_, cudaMemcpyDeviceToDevice, stream_);
    
    // Execute inference
    context_->enqueueV3(stream_);
    
    // Synchronize stream to ensure all operations are completed
    cudaStreamSynchronize(stream_);
}

const std::vector<void*>& InferenceEngine::getBuffers() const {
    return buffers_;
}

int InferenceEngine::getNbBindings() const {
    return engine_->getNbIOTensors();
}

const char* InferenceEngine::getBindingName(int index) const {
    return engine_->getIOTensorName(index);
}

bool InferenceEngine::isOutput(int index) const {
    const char* tensorName = getBindingName(index);
    return engine_->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kOUTPUT;
}

nvinfer1::ICudaEngine* InferenceEngine::getEngine() const {
    return engine_;
}

nvinfer1::IExecutionContext* InferenceEngine::getContext() const {
    return context_;
}

size_t InferenceEngine::getInputSize() const {
    return inputSize_;
}

size_t InferenceEngine::getOutputSize() const {
    return outputSize_;
}

nvinfer1::Dims InferenceEngine::getOutputShape() const {
    for (int i = 0; i < engine_->getNbIOTensors(); i++) {
        const char* tensorName = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kOUTPUT) {
            return context_->getTensorShape(tensorName);
        }
    }
    return nvinfer1::Dims(); // Return empty dims if no output tensor found
}

size_t InferenceEngine::getVolume(const nvinfer1::Dims& dims) {
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<size_t>());
}

} // namespace inference