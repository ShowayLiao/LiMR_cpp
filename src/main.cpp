#include <iostream>
#include <fstream>
#include "video_thread.h"
#include "config.h"
#include "utils.h"
#include <fstream>
#include <opencv2/opencv.hpp>


int main(){
    
    // set the video and model path

    try
    {   
        std::cout<<cv::getBuildInformation() << std::endl; // Print OpenCV build information
        std::cout<<cv::cuda::getCudaEnabledDeviceCount() << " CUDA devices available." << std::endl; // Print number of CUDA devices available
        const std::string videoSource = "../../input/blade.avi"; // Path to the video file
        const std::string StuModelPath = "../../input/LiMR_student_16.engine"; // Path to the student model ONNX file
        const std::string TeaModelPath = "../../input/LiMR_teacher_16.engine"; // Path to the teacher model ONNX file
        const std::string StuOnnxPath = "../../input/LiMR_student.onnx"; // Path to the student model ONNX file
        const std::string TeaOnnxPath = "../../input/LiMR_teacher.onnx"; // Path to the teacher model ONNX file
        // const std::string imgSource = "../../input/IMG_9260.png "; // Path to the image file, if any
        const std::string imgSource = " "; // Path to the image file, if any
        const std::string type = "F32"; // Data type for the model, can be "F32" or "INT8"
        // if you use code 'config.set_flag(trt.BuilderFlag.FP16) ' to build the model, you should also set type to "F32"
        // because the input and output nodes of model is still FP32, only the internal computation is in FP16

        Config NormalConfig(videoSource, 
                    StuModelPath, 
                    TeaModelPath,
                    imgSource,
                    type);

        // check if the engine files exist, if not, build them
        if(!std::ifstream(StuModelPath).good()) {
            std::cout << "Building student model engine..." << std::endl;
            build_engine(StuOnnxPath, 
                        StuModelPath, 
                        NormalConfig.batchSize,
                        NormalConfig.inputWidth,
                        NormalConfig.inputHeight,
                        NormalConfig.outputWidth,
                        NormalConfig.outputHeight);
            
        } else {
            std::cout << "Student model engine already exists." << std::endl;
        }

        if (!std::ifstream(TeaModelPath).good()) {
            std::cout << "Building teacher model engine..." << std::endl;
            build_engine(TeaOnnxPath, 
                        TeaModelPath, 
                        NormalConfig.batchSize,
                        NormalConfig.inputWidth,
                        NormalConfig.inputHeight,
                        NormalConfig.outputWidth,
                        NormalConfig.outputHeight);
        } else {
            std::cout << "Teacher model engine already exists." << std::endl;
        }
        

        VideoThread::VideoCaptureThread videoThread(NormalConfig);
        videoThread.start();

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        system("pause");
        return 1;
    }



}
