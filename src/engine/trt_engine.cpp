#include "engine/TrtEngine.h"
#include <NvOnnxParser.h>
#include <array>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace trt {

namespace {

const char* precisionTag(Precision precision) {
    switch (precision) {
        case Precision::FP32: return "fp32";
        case Precision::FP16: return "fp16";
        case Precision::INT8: return "int8";
    }
    return "unknown";
}

std::string fileIdentityToken(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open model while computing cache identity: " + path.string());
    }

    // FNV-1a is used as a deterministic cache identity, not as a security primitive.
    // Hashing the contents avoids accepting a stale engine when timestamps are copied
    // or preserved by deployment/package tools.
    uint64_t hash = UINT64_C(14695981039346656037);
    constexpr uint64_t prime = UINT64_C(1099511628211);
    // Keep the buffer comfortably below the default Windows thread stack limit.
    std::array<char, 64 * 1024> block{};
    while (input) {
        input.read(block.data(), static_cast<std::streamsize>(block.size()));
        const std::streamsize count = input.gcount();
        for (std::streamsize i = 0; i < count; ++i) {
            hash ^= static_cast<unsigned char>(block[static_cast<size_t>(i)]);
            hash *= prime;
        }
    }
    if (!input.eof()) {
        throw std::runtime_error("Failed to read model while computing cache identity: " + path.string());
    }

    std::ostringstream token;
    token << std::hex << std::setfill('0') << std::setw(16) << hash;
    return token.str();
}

} // namespace

TrtEngine::TrtEngine()
    : runtime_(nullptr),
      engine_(nullptr),
      context_(nullptr),
      stream_(nullptr),
      max_batch_size_(1),
      initialized_(false) {
    
    const cudaError_t streamStatus = cudaStreamCreate(&stream_);
    if (streamStatus != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to create TensorRT CUDA stream: ") +
                                 cudaGetErrorString(streamStatus));
    }
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

bool TrtEngine::load(const std::string& modelPath, Precision precision,
                     int dynamic_input_height, int dynamic_input_width) {
    try {
        if (dynamic_input_height <= 0 || dynamic_input_width <= 0) {
            throw std::invalid_argument("TensorRT profile dimensions must be positive");
        }
        std::filesystem::path path(modelPath);
        std::string extension = path.extension().string();
        
        if (extension == ".engine") {
            if (!loadFromPlan(modelPath)) return false;
            if (isInputSpatialDynamic() && !setInputShape(dynamic_input_height, dynamic_input_width)) {
                clearLoadedEngine();
                return false;
            }
            return true;
        } else if (extension == ".onnx") {
            const std::string enginePath = getEngineCachePath(
                path, precision, dynamic_input_height, dynamic_input_width).string();
            const bool cacheIsFresh = std::filesystem::exists(enginePath) &&
                std::filesystem::last_write_time(enginePath) >= std::filesystem::last_write_time(path);

            if (cacheIsFresh && loadFromPlan(enginePath)) {
                if (!isInputSpatialDynamic() || setInputShape(dynamic_input_height, dynamic_input_width)) {
                    return true;
                }
                std::cout << "[INFO] Cached engine does not support requested input shape; rebuilding" << std::endl;
                clearLoadedEngine();
            }
            return buildFromOnnx(modelPath, enginePath, precision, dynamic_input_height, dynamic_input_width);
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
        clearLoadedEngine();
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
        
        if (!allocateBuffers(max_batch_size_)) {
            throw std::runtime_error("Failed to allocate TensorRT binding buffers");
        }
        nvinfer1::Dims inputDims = engine_->getTensorShape(input_name_.c_str());
        input_spatial_dynamic_ = inputDims.nbDims == 4 && (inputDims.d[2] == -1 || inputDims.d[3] == -1);
        if (!input_spatial_dynamic_ && !updateBindingBuffers()) {
            throw std::runtime_error("Failed to resolve TensorRT binding shapes");
        }
        initialized_ = true;
        
        std::cout << "[INFO] Engine loaded from: " << planPath << std::endl;
        return true;
    } catch (const std::exception& e) {
        clearLoadedEngine();
        std::cerr << "Failed to load plan: " << e.what() << std::endl;
        return false;
    }
}

bool TrtEngine::buildFromOnnx(const std::string& onnxPath, const std::string& enginePath,
                              Precision precision, int dynamic_input_height, int dynamic_input_width) {
    try {
        clearLoadedEngine();
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
            int64_t min_batch = 1, opt_batch = 1, max_batch = 1;
            int64_t min_c = input_dims.d[1], opt_c = input_dims.d[1], max_c = input_dims.d[1];
            int64_t min_h = input_dims.d[2], opt_h = input_dims.d[2], max_h = input_dims.d[2];
            int64_t min_w = input_dims.d[3], opt_w = input_dims.d[3], max_w = input_dims.d[3];
            
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
                min_h = 224;
                opt_h = dynamic_input_height;
                max_h = 448;
            }
            if (input_dims.d[3] == -1) {
                min_w = 224;
                opt_w = dynamic_input_width;
                max_w = 448;
            }
            
            const bool minSet = profile->setDimensions(
                input->getName(), nvinfer1::OptProfileSelector::kMIN,
                nvinfer1::Dims4{min_batch, min_c, min_h, min_w});
            const bool optSet = profile->setDimensions(
                input->getName(), nvinfer1::OptProfileSelector::kOPT,
                nvinfer1::Dims4{opt_batch, opt_c, opt_h, opt_w});
            const bool maxSet = profile->setDimensions(
                input->getName(), nvinfer1::OptProfileSelector::kMAX,
                nvinfer1::Dims4{max_batch, max_c, max_h, max_w});
            if (!minSet || !optSet || !maxSet || config->addOptimizationProfile(profile) < 0) {
                throw std::runtime_error("Failed to configure TensorRT optimization profile");
            }
            std::cout << "[INFO] Optimization profile configured" << std::endl;
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
        
        if (!allocateBuffers(max_batch_size_)) {
            throw std::runtime_error("Failed to allocate TensorRT binding buffers");
        }
        nvinfer1::Dims parsedInputDims = engine_->getTensorShape(input_name_.c_str());
        input_spatial_dynamic_ = parsedInputDims.nbDims == 4 &&
            (parsedInputDims.d[2] == -1 || parsedInputDims.d[3] == -1);
        if (input_spatial_dynamic_ && !setInputShape(dynamic_input_height, dynamic_input_width)) {
            throw std::runtime_error("Failed to configure dynamic ONNX input shape");
        }
        if (!input_spatial_dynamic_ && !updateBindingBuffers()) {
            throw std::runtime_error("Failed to resolve TensorRT binding shapes");
        }
        initialized_ = true;
        
        std::cout << "[INFO] Engine built from ONNX and saved to: " << enginePath << std::endl;
        return true;
    } catch (const std::exception& e) {
        clearLoadedEngine();
        std::cerr << "Failed to build engine from ONNX: " << e.what() << std::endl;
        return false;
    }
}

bool TrtEngine::saveEngine(const std::string& planPath, const void* data, size_t size) {
    try {
        if (!data || size == 0 || size > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
            throw std::runtime_error("Invalid serialized engine buffer size");
        }
        std::ofstream file(planPath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + planPath);
        }
        
        file.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        if (!file.good()) {
            throw std::runtime_error("Failed while writing engine file: " + planPath);
        }
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
    
    const std::streampos endPosition = file.tellg();
    if (endPosition <= 0 || endPosition > std::numeric_limits<std::streamsize>::max()) return {};
    const size_t size = static_cast<size_t>(endPosition);
    file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(size);
    file.read(buffer.data(), static_cast<std::streamsize>(size));
    if (!file) return {};
    
    return buffer;
}

size_t TrtEngine::getElementSize(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kBOOL: return 1;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kUINT8: return 1;
        case nvinfer1::DataType::kFP8: return 1;
        case nvinfer1::DataType::kBF16: return 2;
        case nvinfer1::DataType::kINT64: return 8;
        case nvinfer1::DataType::kINT4: return 0;
        default: return 0;
    }
}

bool TrtEngine::calculateTensorBytes(const nvinfer1::Dims& dims,
                                     nvinfer1::DataType type,
                                     size_t& bytes) {
    bytes = 0;
    if (dims.nbDims <= 0) return false;

    size_t volume = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) return false;
        const size_t extent = static_cast<size_t>(dims.d[i]);
        if (volume > std::numeric_limits<size_t>::max() / extent) return false;
        volume *= extent;
    }

    const size_t elementSize = getElementSize(type);
    if (elementSize == 0 || volume > std::numeric_limits<size_t>::max() / elementSize) return false;
    bytes = volume * elementSize;
    return true;
}

bool TrtEngine::allocateBuffers(int maxBatchSize) {
    (void)maxBatchSize;
    for (int i = 0; i < engine_->getNbIOTensors(); i++) {
        const char* name = engine_->getIOTensorName(i);
        nvinfer1::Dims dims = engine_->getTensorShape(name);
        nvinfer1::DataType type = engine_->getTensorDataType(name);

        Binding b;
        b.index = i;
        b.name = name;
        b.type = type;
        b.dims = dims;
        b.isInput = (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT);

        bindings_[name] = b;
        if (b.isInput) {
            input_name_ = name;
        }
    }

    return !input_name_.empty();
}

std::filesystem::path TrtEngine::getEngineCachePath(const std::filesystem::path& onnxPath,
                                                    Precision precision,
                                                    int profileHeight,
                                                    int profileWidth) {
    const std::string filename = onnxPath.stem().string() + "." + precisionTag(precision) + "." +
        std::to_string(profileHeight) + "x" + std::to_string(profileWidth) + ".onnx-" +
        fileIdentityToken(onnxPath) + ".engine";
    return onnxPath.parent_path() / filename;
}

bool TrtEngine::updateBindingBuffers() {
    struct PendingBindingUpdate {
        Binding* binding;
        nvinfer1::Dims dims;
        size_t requiredSize;
        void* replacement = nullptr;
    };

    std::vector<PendingBindingUpdate> updates;
    updates.reserve(bindings_.size());

    for (auto& pair : bindings_) {
        Binding& binding = pair.second;
        nvinfer1::Dims dims = context_->getTensorShape(binding.name.c_str());
        size_t requiredSize = 0;
        if (!calculateTensorBytes(dims, binding.type, requiredSize)) {
            std::cerr << "Unresolved or invalid TensorRT shape for tensor: " << binding.name << std::endl;
            return false;
        }
        updates.push_back(PendingBindingUpdate{&binding, dims, requiredSize});
    }

    for (auto& update : updates) {
        if (update.binding->capacity >= update.requiredSize) continue;

        const cudaError_t allocationStatus = cudaMalloc(&update.replacement, update.requiredSize);
        if (allocationStatus != cudaSuccess) {
            std::cerr << "Failed to grow TensorRT buffer for " << update.binding->name << ": "
                      << cudaGetErrorString(allocationStatus) << std::endl;
            for (const auto& allocated : updates) {
                if (allocated.replacement) cudaFree(allocated.replacement);
            }
            return false;
        }
    }

    for (auto& update : updates) {
        Binding& binding = *update.binding;
        void* targetBuffer = update.replacement ? update.replacement : binding.buffer;
        if (!targetBuffer || !context_->setTensorAddress(binding.name.c_str(), targetBuffer)) {
            std::cerr << "Failed to update TensorRT binding address for tensor: " << binding.name << std::endl;
            for (auto& pending : updates) {
                if (pending.replacement) cudaFree(pending.replacement);
                pending.replacement = nullptr;
            }
            return false;
        }
        if (update.replacement) {
            void* previousBuffer = binding.buffer;
            binding.buffer = update.replacement;
            update.replacement = nullptr;
            binding.capacity = update.requiredSize;
            if (previousBuffer) cudaFree(previousBuffer);
        }
        binding.size = update.requiredSize;
        binding.dims = update.dims;
    }

    return true;
}

void TrtEngine::clearLoadedEngine() {
    freeBuffers();
    context_.reset();
    engine_.reset();
    runtime_.reset();
    input_name_.clear();
    input_spatial_dynamic_ = false;
    initialized_ = false;
}

void TrtEngine::freeBuffers() {
    for (auto& pair : bindings_) {
        if (pair.second.buffer != nullptr) {
            cudaFree(pair.second.buffer);
        }
    }
    bindings_.clear();
}

void TrtEngine::infer(void* inputData, int batchSize, cudaStream_t stream) {
    if (!initialized_) {
        throw std::runtime_error("Engine not initialized");
    }
    
    const auto inputIt = bindings_.find(input_name_);
    if (inputIt == bindings_.end()) {
        throw std::runtime_error("TensorRT input binding is missing");
    }
    Binding& in_b = inputIt->second;
    if (in_b.type != nvinfer1::DataType::kFLOAT || in_b.dims.nbDims != 4 ||
        in_b.dims.d[0] != batchSize || in_b.dims.d[1] != 3) {
        throw std::runtime_error("TensorRT input must be a concrete NCHW float tensor with three channels");
    }
    if (batchSize != 1) {
        throw std::runtime_error("Changing batch size after engine initialization is not supported");
    }
    if (!inputData || !in_b.buffer || in_b.size == 0) {
        throw std::runtime_error("TensorRT input buffer is not ready");
    }
    cudaStream_t executionStream = stream ? stream : stream_;
    const cudaError_t copyStatus = cudaMemcpyAsync(
        in_b.buffer, inputData, in_b.size, cudaMemcpyDeviceToDevice, executionStream);
    if (copyStatus != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to copy TensorRT input: ") + cudaGetErrorString(copyStatus));
    }

    if (!context_->enqueueV3(executionStream)) {
        throw std::runtime_error("TensorRT enqueueV3 failed");
    }
}

bool TrtEngine::setInputShape(int height, int width) {
    if (!engine_ || !context_ || height <= 0 || width <= 0) return false;

    if (input_spatial_dynamic_ &&
        !isSupportedDynamicSpatialShape(InputShape{1, 3, height, width})) {
        return false;
    }

    nvinfer1::Dims dims = engine_->getTensorShape(input_name_.c_str());
    if (dims.nbDims != 4) return false;
    if (dims.d[0] != -1 && dims.d[0] != 1) return false;
    if (dims.d[1] != -1 && dims.d[1] != 3) return false;
    if (dims.d[2] != -1 && dims.d[2] != height) return false;
    if (dims.d[3] != -1 && dims.d[3] != width) return false;

    dims.d[0] = 1;
    dims.d[1] = 3;
    dims.d[2] = height;
    dims.d[3] = width;
    if (!context_->setInputShape(input_name_.c_str(), dims)) return false;
    return updateBindingBuffers();
}

bool TrtEngine::isInputSpatialDynamic() const {
    return input_spatial_dynamic_;
}

void* TrtEngine::getBuffer(const std::string& name) {
    auto it = bindings_.find(name);
    if (it != bindings_.end()) {
        return it->second.buffer;
    }
    return nullptr;
}

const Binding* TrtEngine::getTensorInfo(const std::string& name) const {
    const auto it = bindings_.find(name);
    return it == bindings_.end() ? nullptr : &it->second;
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
    auto anomalyMap = bindings_.find("anomaly_map");
    if (anomalyMap != bindings_.end()) {
        return anomalyMap->second.dims;
    }
    for (const auto& pair : bindings_) {
        if (!pair.second.isInput) {
            return pair.second.dims;
        }
    }
    return nvinfer1::Dims();
}

} // namespace trt
