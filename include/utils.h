# ifndef UTILS_H
# define UTILS_H

# include <iostream>
# include <fstream>
# include <string>
# include "NvInfer.h"
# include <opencv2/opencv.hpp>
# include <torch/torch.h>

void build_engine(const std::string& onnxModelPath, 
                  const std::string& engineFilePath, 
                  int batchSize = 1, 
                  int inputWidth = 224, 
                  int inputHeight = 224, 
                  int outputWidth = 224, 
                  int outputHeight = 224);

void cal_anomaly_map(std::vector<torch::Tensor>& StuOutputTensors, 
                     std::vector<torch::Tensor>& TeaOutputTensors, 
                     torch::Tensor& anomalyMap);

typedef union {
    struct {
        uint16_t mantissa : 10;  // 10位尾数
        uint16_t exponent : 5;   // 5位指数
        uint16_t sign     : 1;    // 1位符号
    } bits;
    uint16_t raw_value;          // 16位原始值
} fp16;


# endif