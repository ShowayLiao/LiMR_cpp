# ifndef PIPELINE_H
# define PIPELINE_H

# include<iostream>
# include<opencv2/opencv.hpp>
# include "NvInfer.h"
# include "config.h"
# include <../samples/common/logger.h>
# include <torch/torch.h>
# include <opencv2/cudaarithm.hpp>



namespace pipeline {

    void Transform(const cv::Mat& inputFrame, cv::cuda::GpuMat& OutputFrame,const cv::Size& size,const std::string& type = "F32");
    void reverse_transform(const cv::Mat& inputFrame, cv::Mat& outputFrame);
    void centerCrop(const cv::Mat& inputFrame, cv::Mat& outputFrame, int size);


    class Pipeline {
    public:
        Pipeline(const Config& videoConfig);

        void inference(cv::Mat& inputFrame, cv::Mat& outputFrame);
        nvinfer1::ICudaEngine* load_model(const std::string& modelPath);
        void allocate_memory(nvinfer1::ICudaEngine*& Engine,
                            nvinfer1::IExecutionContext*& StuContext,
                                std::vector<void*>& buffer);
        // void get_output_tensors(nvinfer1::IExecutionContext*& Context,
        //                         nvinfer1::ICudaEngine*& Engine,
        //                         cudaStream_t& stream,  
        //                         std::vector<void*>& buffer,
        //                         std::vector<torch::Tensor> outputTensors);
        void cal_anomaly_map(cv::Mat& anomaly_map);
        


    protected:
        Config config;
        sample::Logger my_logger;
        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(my_logger);
        // nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(*logger);
        nvinfer1::ICudaEngine* StuEngine = nullptr;
        nvinfer1::ICudaEngine* TeaEngine = nullptr;
        nvinfer1::IExecutionContext* StuContext = nullptr;
        nvinfer1::IExecutionContext* TeaContext = nullptr;
        std::vector<void*> StuBuffer;
        std::vector<void*> TeaBuffer;
        cudaStream_t StuStream;
        cudaStream_t TeaStream;
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;
        bool graphInitialized = false; // 标记图是否已初始化
        cv::cuda::HostMem hostMem;
        
    };

} // namespace pipeline





# endif // PIPELINE_H