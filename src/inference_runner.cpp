#include "inference_runner.h"
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

/**
 * Constructor
 */
InferenceRunner::InferenceRunner(InferenceMode mode) 
    : mode_(mode),
      engineA_(nullptr),
      engineB_(nullptr),
      inputWidth_(0),
      inputHeight_(0),
      outputWidth_(0),
      outputHeight_(0),
      scale_(0.0f),
      mean_(0.0f),
      d_inputBuffer_(nullptr),
      d_anomalyMap_(nullptr),
      inputBufferSize_(0),
      anomalyMapSize_(0) {
    
}

/**
 * Destructor
 */
InferenceRunner::~InferenceRunner() {
    freeGPUBuffers();
}

/**
 * Initialize inference runner
 */
bool InferenceRunner::init(const std::string& enginePathA, 
                          const std::string& enginePathB, 
                          int inputWidth, 
                          int inputHeight,
                          int outputWidth,
                          int outputHeight,
                          float scale,
                          float mean) {
    try {
        // Save parameters
        inputWidth_ = inputWidth;
        inputHeight_ = inputHeight;
        outputWidth_ = outputWidth;
        outputHeight_ = outputHeight;
        scale_ = scale;
        mean_ = mean;
        
        // Initialize preprocessor
        if (!preprocessor_.init(inputWidth, inputHeight, scale, mean)) {
            throw std::runtime_error("Preprocessor initialization failed");
        }
        
        // Initialize engine A first to get output shape
        engineA_ = std::make_unique<InferenceEngine>();
        if (!engineA_->init(enginePathA)) {
            throw std::runtime_error("Engine A initialization failed");
        }
        
        // Initialize postprocessor
        postprocessor_.init(outputHeight, outputWidth);
        
        // Get output shape from engine A and configure postprocessor with feature map dimensions
        if (mode_ == InferenceMode::DUAL_ENGINE) {
            nvinfer1::Dims outputDims = engineA_->getOutputShape();
            if (outputDims.nbDims == 4) {
                int numChannels = outputDims.d[1];
                int featureHeight = outputDims.d[2];
                int featureWidth = outputDims.d[3];
                postprocessor_.setOutputShape(numChannels, featureHeight, featureWidth);
            }
        }
        
        // Calculate input buffer size
        inputBufferSize_ = 3 * inputWidth * inputHeight * sizeof(float);
        
        // Initialize engine B in dual-engine mode
        if (mode_ == InferenceMode::DUAL_ENGINE) {
            if (enginePathB.empty()) {
                throw std::runtime_error("Engine B path is required in dual-engine mode");
            }
            
            engineB_ = std::make_unique<InferenceEngine>();
            if (!engineB_->init(enginePathB)) {
                throw std::runtime_error("Engine B initialization failed");
            }
        }
        
        // Calculate anomaly map buffer size
        if (mode_ == InferenceMode::SINGLE_ENGINE) {
            // In single-engine mode, get output dimensions from engineA
            nvinfer1::Dims outputDims = engineA_->getOutputShape();
            if (outputDims.nbDims == 4) {
                // (B, C, H, W) format, assuming C=1 for anomaly map
                int outputH = outputDims.d[2];
                int outputW = outputDims.d[3];
                anomalyMapSize_ = outputH * outputW * sizeof(float);
                // Update output dimensions for heatmap generation
                outputHeight_ = outputH;
                outputWidth_ = outputW;
                std::cout << "[INFO] Single-engine mode: Using engine output dimensions (H: " << outputH << ", W: " << outputW << ")" << std::endl;
            } else {
                // Fallback to provided dimensions if output shape is unexpected
                anomalyMapSize_ = outputWidth * outputHeight * sizeof(float);
                std::cout << "[INFO] Single-engine mode: Using fallback dimensions (H: " << outputHeight << ", W: " << outputWidth << ")" << std::endl;
            }
        } else {
            // In dual-engine mode, use provided dimensions
            anomalyMapSize_ = outputWidth * outputHeight * sizeof(float);
            std::cout << "[INFO] Dual-engine mode: Using provided dimensions (H: " << outputHeight << ", W: " << outputWidth << ")" << std::endl;
        }
        std::cout << "[INFO] Anomaly map buffer size: " << anomalyMapSize_ << " bytes" << std::endl;
        
        // Allocate GPU memory buffers
        cudaError_t cudaStatus;
        cudaStatus = cudaMalloc(&d_inputBuffer_, inputBufferSize_);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("Failed to allocate input buffer on GPU");
        }
        
        cudaStatus = cudaMalloc(&d_anomalyMap_, anomalyMapSize_);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("Failed to allocate anomaly map buffer on GPU");
        }
        
        std::cout << "InferenceRunner initialization succeeded" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "InferenceRunner initialization failed: " << e.what() << std::endl;
        freeGPUBuffers();
        return false;
    }
}

/**
 * Execute inference
 */
cv::Mat InferenceRunner::infer(const cv::Mat& inputImage) {
    cv::Mat heatmap;
    
    try {
        if (!engineA_) {
            throw std::runtime_error("InferenceRunner not initialized");
        }
        
        totalStart_ = std::chrono::high_resolution_clock::now();
        
        // 1. Preprocessing
        preprocessStart_ = std::chrono::high_resolution_clock::now();
        
        // Debug: Check input image before preprocessing (commented out)
        // double imgMin, imgMax;
        // cv::minMaxLoc(inputImage, &imgMin, &imgMax);
        // std::cout << "[DEBUG] Input Image - Min: " << imgMin << ", Max: " << imgMax << ", Channels: " << inputImage.channels() << std::endl;
        
        preprocessor_.preprocess(inputImage, d_inputBuffer_);
        
        // Debug: Check a few values from the preprocessed GPU buffer (commented out)
        // float* h_debugBuffer = new float[10];
        // cudaMemcpy(h_debugBuffer, d_inputBuffer_, 10 * sizeof(float), cudaMemcpyDeviceToHost);
        // std::cout << "[DEBUG] Preprocessed Input (first 10 values): ";
        // for (int i = 0; i < 10; ++i) {
        //     std::cout << h_debugBuffer[i] << " ";
        // }
        // std::cout << std::endl;
        // delete[] h_debugBuffer;
        
        preprocessEnd_ = std::chrono::high_resolution_clock::now();
        
        // 2. Inference
        inferStart_ = std::chrono::high_resolution_clock::now();
        
        // Execute inference
        if (mode_ == InferenceMode::DUAL_ENGINE) {
            // Dual-engine mode: execute both engines with the same input
            if (!engineB_) {
                throw std::runtime_error("Engine B not initialized in dual-engine mode");
            }
            
            // Execute both engines with the same input
            engineA_->infer(d_inputBuffer_);
            engineB_->infer(d_inputBuffer_);
            
            // Postprocessing: Loop through all output layers and accumulate results
            postprocessStart_ = std::chrono::high_resolution_clock::now();
            
            // Clear the anomaly map buffer before accumulation
            cudaMemset(d_anomalyMap_, 0, anomalyMapSize_);
            
            // Get buffers from both engines
            const auto& buffersA = engineA_->getBuffers();
            const auto& buffersB = engineB_->getBuffers();
            
            // Loop through all bindings to find outputs
            for (int i = 0; i < engineA_->getNbBindings(); i++) {
                if (engineA_->isOutput(i)) {
                    // Get tensor shape
                    const char* tensorName = engineA_->getBindingName(i);
                    nvinfer1::Dims shape = engineA_->getEngine()->getTensorShape(tensorName);
                    
                    // Extract dimensions (B, C, H, W)
                    int feature_c = shape.d[1];
                    int feature_h = shape.d[2];
                    int feature_w = shape.d[3];
                    
                    // std::cout << "[DEBUG] Processing layer: " << tensorName 
                    //           << " (C: " << feature_c << ", H: " << feature_h << ", W: " << feature_w << ")" << std::endl;
                    
                    // Process and accumulate results from this layer
                    postprocessor_.process_and_accumulate(
                        static_cast<float*>(buffersA[i]),
                        static_cast<float*>(buffersB[i]),
                        static_cast<float*>(d_anomalyMap_),
                        feature_c, feature_h, feature_w,
                        outputHeight_, outputWidth_);
                }
            }
            
            postprocessEnd_ = std::chrono::high_resolution_clock::now();
            
        } else if (mode_ == InferenceMode::SINGLE_ENGINE) {
            // Single-engine mode: execute inference and get result directly
            engineA_->infer(d_inputBuffer_);
            
            // Get output bindings from engineA
            const auto& buffers = engineA_->getBuffers();
            int outputIndex = -1;
            
            // Find the first output binding
            for (int i = 0; i < engineA_->getNbBindings(); i++) {
                if (engineA_->isOutput(i)) {
                    outputIndex = i;
                    break; // Assume only one output (Anomaly Map)
                }
            }
            
            if (outputIndex >= 0) {
                // Directly copy result from engine output to d_anomalyMap_
                cudaMemcpyAsync(d_anomalyMap_, buffers[outputIndex], anomalyMapSize_, cudaMemcpyDeviceToDevice);
                cudaStreamSynchronize(0); // Ensure copy completes
            } else {
                throw std::runtime_error("No output binding found in single-engine mode");
            }
        }
        
        inferEnd_ = std::chrono::high_resolution_clock::now();
        
        // 3. Generate heatmap
        heatmap = generateHeatmapFromAnomalyMap(static_cast<float*>(d_anomalyMap_));
        
        totalEnd_ = std::chrono::high_resolution_clock::now();
        
    } catch (const std::exception& e) {
        std::cerr << "Inference failed: " << e.what() << std::endl;
        heatmap = cv::Mat(); // Return empty Mat to indicate failure
    }
    
    return heatmap;
}

/**
 * Save heatmap
 */
bool InferenceRunner::saveHeatmap(const cv::Mat& heatmap, const std::string& path) const {
    if (heatmap.empty()) {
        std::cerr << "Cannot save empty heatmap" << std::endl;
        return false;
    }
    
    if (!cv::imwrite(path, heatmap)) {
        std::cerr << "Failed to save heatmap to " << path << std::endl;
        return false;
    }
    
    std::cout << "Heatmap saved to " << path << std::endl;
    return true;
}

/**
 * Print timing statistics
 */
void InferenceRunner::printTimingStats() const {
    auto preprocessTime = std::chrono::duration_cast<std::chrono::milliseconds>(preprocessEnd_ - preprocessStart_).count();
    auto inferTime = std::chrono::duration_cast<std::chrono::milliseconds>(inferEnd_ - inferStart_).count();
    auto postprocessTime = std::chrono::duration_cast<std::chrono::milliseconds>(postprocessEnd_ - postprocessStart_).count();
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd_ - totalStart_).count();
    
    std::cout << "Timing Statistics:" << std::endl;
    std::cout << "  Preprocess: " << preprocessTime << " ms" << std::endl;
    std::cout << "  Inference: " << inferTime << " ms" << std::endl;
    std::cout << "  Postprocess: " << postprocessTime << " ms" << std::endl;
    std::cout << "  Total: " << totalTime << " ms" << std::endl;
    std::cout << "  FPS: " << (totalTime > 0 ? 1000.0 / totalTime : 0.0) << std::endl;
}

/**
 * Free GPU memory buffers
 */
void InferenceRunner::freeGPUBuffers() {
    if (d_inputBuffer_) {
        cudaFree(d_inputBuffer_);
        d_inputBuffer_ = nullptr;
    }
    
    if (d_anomalyMap_) {
        cudaFree(d_anomalyMap_);
        d_anomalyMap_ = nullptr;
    }
}

/**
 * Generate heatmap from anomaly map
 */
cv::Mat InferenceRunner::generateHeatmapFromAnomalyMap(float* d_anomalyMap) {
    // Copy GPU anomaly map data to CPU
    cv::Mat anomalyMap(outputHeight_, outputWidth_, CV_32F);
    
    // Ensure correct memory copy size
    size_t expectedBytes = outputHeight_ * outputWidth_ * sizeof(float);
    cudaMemcpy(anomalyMap.data, d_anomalyMap, expectedBytes, cudaMemcpyDeviceToHost);
    
    // Debug: Check raw anomaly map values (commented out)
    double minVal, maxVal;
    cv::minMaxLoc(anomalyMap, &minVal, &maxVal);
    // std::cout << "[DEBUG] Raw AnomalyMap - Min: " << minVal << ", Max: " << maxVal << std::endl;
    
    // Normalize to 0-255 with safety check
    cv::Mat normalizedMap;
    cv::Mat heatmap;
    
    if (maxVal > 0) {
        // Normalize only if there are non-zero values
        // Invert the normalization range to fix color mapping: 0->255 (red), 255->0 (blue)
        cv::normalize(anomalyMap, normalizedMap, 255, 0, cv::NORM_MINMAX, CV_8U);
        
        // Generate heatmap
        cv::applyColorMap(normalizedMap, heatmap, cv::COLORMAP_JET);
    } else {
        // If all values are zero, return blue image and warn
        std::cerr << "[WARNING] Anomaly Map is empty! All values are zero." << std::endl;
        heatmap = cv::Mat(outputHeight_, outputWidth_, CV_8UC3, cv::Scalar(255, 0, 0)); // Blue image
    }
    
    return heatmap;
}