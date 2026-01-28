# include "app_config.h"
# include <iostream>
# include <fstream>
# include <cctype>

// Helper function: Check if string is a number (camera index)
static bool is_number(const std::string& str) {
    if (str.empty()) return false;
    for (char c : str) {
        if (!std::isdigit(c)) return false;
    }
    return true;
}

AppConfig::AppConfig(const std::string& videoSource, 
           const std::string& enginePathA, 
           const std::string& enginePathB,
           const std::string& imgSource, 
           const std::string& type,
           InferenceMode inferenceMode,
           int batchSize, 
           int inputWidth, 
           int inputHeight, 
           int outputWidth, 
           int outputHeight,
           float heatmapAlpha,
           int colormapType)
        : videoSource(videoSource), 
          enginePathA(enginePathA), 
          enginePathB(enginePathB),
          imgSource(imgSource), 
          type(type),
          inferenceMode(inferenceMode),
          batchSize(batchSize), 
          inputWidth(inputWidth), 
          inputHeight(inputHeight), 
          outputWidth(outputWidth), 
          outputHeight(outputHeight),
          heatmapAlpha(heatmapAlpha),
          colormapType(colormapType) {
            // Only check video source file existence if it's not a camera index (number)
            if (!videoSource.empty() && !is_number(videoSource) && !std::ifstream(videoSource).good()) {
                throw std::runtime_error("Video source file does not exist: " + videoSource);
            }
            if (!enginePathA.empty() && !std::ifstream(enginePathA).good()) {
                throw std::runtime_error("Engine A file does not exist: " + enginePathA);
            }
            if (inferenceMode == InferenceMode::DUAL_ENGINE && !enginePathB.empty() && !std::ifstream(enginePathB).good()) {
                throw std::runtime_error("Engine B file does not exist: " + enginePathB);
            }
          }