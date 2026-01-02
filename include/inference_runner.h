#ifndef INFERENCE_RUNNER_H
#define INFERENCE_RUNNER_H

#include "preprocess.h"
#include "inference.h"
#include "postprocess.h"
#include "app_config.h"
#include <chrono>
#include <string>
#include <memory>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using preprocess::Preprocessor;
using inference::InferenceEngine;
using postprocess::Postprocessor;

/**
 * Inference runner class, used to coordinate preprocessing, inference and postprocessing modules
 * Supports single-engine and dual-engine inference modes
 */
class InferenceRunner {
public:
    /**
     * Constructor
     * @param mode Inference mode
     */
    explicit InferenceRunner(InferenceMode mode);
    
    /**
     * Destructor
     */
    ~InferenceRunner();
    
    /**
     * Initialize inference runner
     * @param enginePathA Path to Engine A
     * @param enginePathB Path to Engine B (required in dual-engine mode)
     * @param inputWidth Input image width
     * @param inputHeight Input image height
     * @param outputWidth Output image width
     * @param outputHeight Output image height
     * @param scale Image scaling factor
     * @param mean Image mean value
     * @return Whether initialization succeeded
     */
    bool init(const std::string& enginePathA, 
              const std::string& enginePathB = "", 
              int inputWidth = 256, 
              int inputHeight = 256,
              int outputWidth = 256,
              int outputHeight = 256,
              float scale = 1.0f / 255.0f,
              float mean = 0.5f);
    
    /**
     * Execute inference
     * @param inputImage Input image
     * @return Generated heatmap
     */
    cv::Mat infer(const cv::Mat& inputImage);
    
    /**
     * Save heatmap
     * @param heatmap Heatmap to save
     * @param path Save path
     * @return Whether saving succeeded
     */
    bool saveHeatmap(const cv::Mat& heatmap, const std::string& path) const;
    
    /**
     * Print timing statistics
     */
    void printTimingStats() const;

private:
    InferenceMode mode_;                      // Inference mode
    Preprocessor preprocessor_;                // Preprocessor
    std::unique_ptr<InferenceEngine> engineA_; // Engine A (used in both single and dual engine modes)
    std::unique_ptr<InferenceEngine> engineB_; // Engine B (only used in dual engine mode)
    Postprocessor postprocessor_;              // Postprocessor
    
    int inputWidth_;                          // Input width
    int inputHeight_;                         // Input height
    int outputWidth_;                         // Output width
    int outputHeight_;                        // Output height
    float scale_;                             // Scaling factor
    float mean_;                              // Mean value
    
    // GPU memory buffers
    void* d_inputBuffer_;                     // Input buffer
    void* d_anomalyMap_;                      // Anomaly map buffer
    
    // Buffer sizes
    size_t inputBufferSize_;                  // Input buffer size
    size_t anomalyMapSize_;                   // Anomaly map buffer size
    
    // Timing variables
    std::chrono::high_resolution_clock::time_point preprocessStart_; // Preprocessing start time
    std::chrono::high_resolution_clock::time_point preprocessEnd_;   // Preprocessing end time
    std::chrono::high_resolution_clock::time_point inferStart_;      // Inference start time
    std::chrono::high_resolution_clock::time_point inferEnd_;        // Inference end time
    std::chrono::high_resolution_clock::time_point postprocessStart_;// Postprocessing start time
    std::chrono::high_resolution_clock::time_point postprocessEnd_;  // Postprocessing end time
    std::chrono::high_resolution_clock::time_point totalStart_;      // Total start time
    std::chrono::high_resolution_clock::time_point totalEnd_;        // Total end time
    
    /**
     * Free GPU memory buffers
     */
    void freeGPUBuffers();
    
    /**
     * Generate heatmap from anomaly map
     * @param d_anomalyMap GPU pointer to anomaly map
     * @return Heatmap
     */
    cv::Mat generateHeatmapFromAnomalyMap(float* d_anomalyMap);
};

#endif // INFERENCE_RUNNER_H