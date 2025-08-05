# include "config.h"
# include <iostream>
# include <fstream>

Config::Config(const std::string& videoSource, 
           const std::string& StuModelPath, 
           const std::string& TeaModelPath,
           const std::string& imgSource, 
           const std::string& type,
           int batchSize, 
           int inputWidth, 
           int inputHeight, 
           int outputWidth, 
           int outputHeight)
        : videoSource(videoSource), 
          StuModelPath(StuModelPath), 
          TeaModelPath(TeaModelPath),
          imgSource(imgSource), 
          type(type),
          batchSize(batchSize), 
          inputWidth(inputWidth), 
          inputHeight(inputHeight), 
          outputWidth(outputWidth), 
          outputHeight(outputHeight) {
            if(!std::ifstream(videoSource).good()) {
                throw std::runtime_error("Video source file does not exist: " + videoSource);
            }
            if (!std::ifstream(StuModelPath).good()) {
                throw std::runtime_error("Student model file does not exist: " + StuModelPath);
            }
            if (!std::ifstream(TeaModelPath).good()) {
                throw std::runtime_error("Teacher model file does not exist: " + TeaModelPath);
            }


          }

