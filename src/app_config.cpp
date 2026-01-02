# include "app_config.h"
# include <iostream>
# include <fstream>

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
            if (!videoSource.empty() && !std::ifstream(videoSource).good()) {
                throw std::runtime_error("Video source file does not exist: " + videoSource);
            }
            if (!enginePathA.empty() && !std::ifstream(enginePathA).good()) {
                throw std::runtime_error("Engine A file does not exist: " + enginePathA);
            }
            if (inferenceMode == InferenceMode::DUAL_ENGINE && !enginePathB.empty() && !std::ifstream(enginePathB).good()) {
                throw std::runtime_error("Engine B file does not exist: " + enginePathB);
            }
          }