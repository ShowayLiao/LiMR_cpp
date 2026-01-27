#include "engine/TrtEngine.h"
#include <NvOnnxParser.h>
#include <numeric>
#include <stdexcept>

namespace trt {

TrtEngine::TrtEngine()
    : runtime_(nullptr),
      engine_(nullptr),
      context_(nullptr),
      stream_(nullptr),
      max_batch_size_(1),
      initialized_(false) {
    
    cudaStreamCreate(&stream_);
}

TrtEngine::~TrtEngine() {
    freeBuffers();
    
    if (context_) {
        context_.reset();
    }
    
    if (engine_) {
        engine_.reset();
    }
    
    if (runtime_) {
        runtime_.reset();
    }
    
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

bool TrtEngine::load(const std::string& modelPath, Precision precision) {
    try {
        std::filesystem::path path(modelPath);
        std::string extension = path.extension().string();
        
        if (extension == ".engine") {
            return loadFromPlan(modelPath);
        } else if (extension == ".onnx") {
            std::string enginePath = (path.parent_path() / (path.stem().string() + ".engine")).string();
            
            if (std::filesystem::exists(enginePath)) {
                return loadFromPlan(enginePath);
            } else {
                return buildFromOnnx(modelPath, enginePath, precision);
            }
        } else {
            throw std::runtime_error("Unsupported model format: " + extension);
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        return false;
    }
}

bool TrtEngine::loadFromPlan(const std::string& planPath) {
    try {
        std::vector<char> modelData = readBinaryFile(planPath);
        if (modelData.empty()) {
            throw std::runtime_error("Failed to read plan file: " + planPath);
        }
        
        runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) {
            throw std::runtime_error("Failed to create TensorRT runtime");
        }
        
        engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(modelData.data(), modelData.size())
        );
        
        if (!engine_) {
            throw std::runtime_error("Failed to deserialize engine");
        }
        
        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!context_) {
            throw std::runtime_error("Failed to create execution context");
        }
        
        allocateBuffers(max_batch_size_);
        initialized_ = true;
        
        std::cout << "[INFO] Engine loaded from: " << planPath << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load plan: " << e.what() << std::endl;
        return false;
    }
}

bool TrtEngine::buildFromOnnx(const std::string& onnxPath, const std::string& enginePath, Precision precision) {
    try {
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
        if (!builder) {
            throw std::runtime_error("Failed to create builder");
        }
        
        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        if (!network) {
            throw std::runtime_error("Failed to create network");
        }
        
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config) {
            throw std::runtime_error("Failed to create builder config");
        }
        
        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger_));
        if (!parser) {
            throw std::runtime_error("Failed to create ONNX parser");
        }
        
        if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
            throw std::runtime_error("Failed to parse ONNX file: " + onnxPath);
        }
        
        switch (precision) {
            case Precision::FP16:
                if (!builder->platformHasFastFp16()) {
                    std::cout << "[WARNING] Platform does not support fast FP16, falling back to FP32" << std::endl;
                } else {
                    config->setFlag(nvinfer1::BuilderFlag::kFP16);
                }
                break;
            case Precision::INT8:
                if (!builder->platformHasFastInt8()) {
                    std::cout << "[WARNING] Platform does not support fast INT8, falling back to FP16" << std::endl;
                    config->setFlag(nvinfer1::BuilderFlag::kFP16);
                } else {
                    config->setFlag(nvinfer1::BuilderFlag::kINT8);
                }
                break;
            case Precision::FP32:
            default:
                break;
        }
        
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30);
        
        auto profile = builder->createOptimizationProfile();
        if (!profile) {
            throw std::runtime_error("Failed to create optimization profile");
        }
        
        auto input = network->getInput(0);
        if (!input) {
            throw std::runtime_error("Failed to get input tensor");
        }
        
        auto input_dims = input->getDimensions();
        
        if (input_dims.nbDims == 4) {
            int32_t min_batch = 1, opt_batch = 1, max_batch = 1;
            int32_t min_c = input_dims.d[1], opt_c = input_dims.d[1], max_c = input_dims.d[1];
            int32_t min_h = 256, opt_h = 256, max_h = 256;
            int32_t min_w = 256, opt_w = 256, max_w = 256;
            
            if (input_dims.d[0] == -1) {
                min_batch = 1;
                opt_batch = 1;
                max_batch = 1;
            }
            if (input_dims.d[1] == -1) {
                min_c = 3;
                opt_c = 3;
                max_c = 3;
            }
            if (input_dims.d[2] == -1) {
                min_h = 256;
                opt_h = 256;
                max_h = 256;
            }
            if (input_dims.d[3] == -1) {
                min_w = 256;
                opt_w = 256;
                max_w = 256;
            }
            
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, 
                                   nvinfer1::Dims4{min_batch, min_c, min_h, min_w});
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, 
                                   nvinfer1::Dims4{opt_batch, opt_c, opt_h, opt_w});
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, 
                                   nvinfer1::Dims4{max_batch, max_c, max_h, max_w});
            
            config->addOptimizationProfile(profile);
            std::cout << "[INFO] Optimization profile set for dynamic input" << std::endl;
        }
        
        auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
        if (!plan) {
            throw std::runtime_error("Failed to build serialized network");
        }
        
        if (!saveEngine(enginePath, plan->data(), plan->size())) {
            throw std::runtime_error("Failed to save engine to: " + enginePath);
        }
        
        runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) {
            throw std::runtime_error("Failed to create TensorRT runtime");
        }
        
        engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(plan->data(), plan->size())
        );
        
        if (!engine_) {
            throw std::runtime_error("Failed to deserialize engine");
        }
        
        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!context_) {
            throw std::runtime_error("Failed to create execution context");
        }
        
        allocateBuffers(max_batch_size_);
        initialized_ = true;
        
        std::cout << "[INFO] Engine built from ONNX and saved to: " << enginePath << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to build engine from ONNX: " << e.what() << std::endl;
        return false;
    }
}

bool TrtEngine::saveEngine(const std::string& planPath, const void* data, size_t size) {
    try {
        std::ofstream file(planPath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + planPath);
        }
        
        file.write(static_cast<const char*>(data), size);
        file.close();
        
        std::cout << "[INFO] Engine saved to: " << planPath << " (" << size << " bytes)" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to save engine: " << e.what() << std::endl;
        return false;
    }
}

std::vector<char> TrtEngine::readBinaryFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return {};
    }
    
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    
    return buffer;
}

size_t TrtEngine::getElementSize(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kBOOL: return 1;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kINT32: return 4;
        default: return 4;
    }
}

void TrtEngine::allocateBuffers(int maxBatchSize) {
    for (int i = 0; i < engine_->getNbIOTensors(); i++) {
        const char* name = engine_->getIOTensorName(i);
        nvinfer1::Dims dims = engine_->getTensorShape(name);
        nvinfer1::DataType type = engine_->getTensorDataType(name);

        size_t vol = 1;
        for (int d = 0; d < dims.nbDims; d++) {
            if (dims.d[d] == -1) {
                vol *= maxBatchSize;
            } else {
                vol *= dims.d[d];
            }
        }
        
        Binding b;
        b.index = i;
        b.name = name;
        b.type = type;
        b.dims = dims;
        b.size = vol * getElementSize(type);
        b.isInput = (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT);
        
        cudaMalloc(&b.buffer, b.size);
        context_->setTensorAddress(name, b.buffer);
        
        bindings_[name] = b;
        
        if (b.isInput) {
            input_name_ = name;
        }
    }
}

void TrtEngine::freeBuffers() {
    for (auto& pair : bindings_) {
        if (pair.second.buffer != nullptr) {
            cudaFree(pair.second.buffer);
        }
    }
    bindings_.clear();
}

void TrtEngine::infer(void* inputData, int batchSize) {
    if (!initialized_) {
        throw std::runtime_error("Engine not initialized");
    }
    
    nvinfer1::Dims dims = bindings_[input_name_].dims;
    if (dims.nbDims > 0 && dims.d[0] == -1) {
        dims.d[0] = batchSize;
        context_->setInputShape(input_name_.c_str(), dims);
    }
    
    auto& in_b = bindings_[input_name_];
    size_t current_size = batchSize * 3 * 256 * 256 * sizeof(float);
    cudaMemcpyAsync(in_b.buffer, inputData, current_size, cudaMemcpyDeviceToDevice, stream_);
    
    context_->enqueueV3(stream_);
    cudaStreamSynchronize(stream_);
}

void* TrtEngine::getBuffer(const std::string& name) {
    auto it = bindings_.find(name);
    if (it != bindings_.end()) {
        return it->second.buffer;
    }
    return nullptr;
}

nvinfer1::ICudaEngine* TrtEngine::getEngine() const {
    return engine_.get();
}

nvinfer1::IExecutionContext* TrtEngine::getContext() const {
    return context_.get();
}

cudaStream_t TrtEngine::getStream() const {
    return stream_;
}

const std::map<std::string, Binding>& TrtEngine::getBindings() const {
    return bindings_;
}

int TrtEngine::getNbBindings() const {
    return engine_->getNbIOTensors();
}

const char* TrtEngine::getBindingName(int index) const {
    return engine_->getIOTensorName(index);
}

bool TrtEngine::isOutput(int index) const {
    const char* tensorName = getBindingName(index);
    return engine_->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kOUTPUT;
}

size_t TrtEngine::getInputSize() const {
    auto it = bindings_.find(input_name_);
    if (it != bindings_.end()) {
        return it->second.size;
    }
    return 0;
}

size_t TrtEngine::getOutputSize() const {
    size_t totalSize = 0;
    for (const auto& pair : bindings_) {
        if (!pair.second.isInput) {
            totalSize += pair.second.size;
        }
    }
    return totalSize;
}

nvinfer1::Dims TrtEngine::getInputShape() const {
    auto it = bindings_.find(input_name_);
    if (it != bindings_.end()) {
        return it->second.dims;
    }
    return nvinfer1::Dims();
}

nvinfer1::Dims TrtEngine::getOutputShape() const {
    for (const auto& pair : bindings_) {
        if (!pair.second.isInput) {
            return pair.second.dims;
        }
    }
    return nvinfer1::Dims();
}

} // namespace trt
