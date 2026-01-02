#ifndef APP_CONFIG_H
#define APP_CONFIG_H

#include <string>
// 尽量不在头文件中 include iostream，除非必要
// #include <iostream> 

/**
 * Inference Mode Enum
 */
enum class InferenceMode {
    SINGLE_ENGINE,  // Single Engine Mode
    DUAL_ENGINE     // Dual Engine Mode
};

class AppConfig {
public:
    AppConfig(const std::string& videoSource = "", 
              const std::string& enginePathA = "", 
              const std::string& enginePathB = "",
              const std::string& imgSource = "",  // Note: Should be empty string, not space " "
              const std::string& type = "F32",
              InferenceMode inferenceMode = InferenceMode::SINGLE_ENGINE,
              int batchSize = 1, 
              int inputWidth = 224, 
              int inputHeight = 224, 
              int outputWidth = 224, 
              int outputHeight = 224,
              float heatmapAlpha = 0.5f,
              int colormapType = 2); // 2 corresponds to cv::COLORMAP_JET

    std::string videoSource;
    std::string enginePathA;
    std::string enginePathB;
    std::string imgSource;
    std::string type;           // "F32" or "INT8"
    InferenceMode inferenceMode;
    int batchSize;
    int inputWidth;
    int inputHeight;
    int outputWidth;
    int outputHeight;
    
    // Heatmap parameters
    float heatmapAlpha;
    int colormapType;
};

#endif // APP_CONFIG_H