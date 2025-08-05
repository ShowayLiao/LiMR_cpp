# include "config.h"
# include <vector>
# include "NvInfer.h"
# include <../samples/common/logger.h>
# include <fstream>
# include <numeric>
# include "pipeline.h"
# include <torch/torch.h>
# include <opencv2/opencv.hpp>
# include <opencv2/cudaimgproc.hpp>
# include <opencv2/cudaarithm.hpp>
# include <opencv2/cudawarping.hpp>
# include <utils.cuh>
# include "half.h"






namespace pipeline {


    // ------------------------Transform functions------------------------
    void Transform(const cv::Mat& inputFrame, cv::cuda::GpuMat& OutputFrame,const cv::Size& size,const std::string& type) {
        // ----------------deal in turns in cpu------------------

        // cv::Mat outputFrame_1;
        // // // Example transformation: Convert to grayscale
        // cv::cvtColor(inputFrame, outputFrame_1, cv::COLOR_BGR2RGB);
        
        // cv::resize(outputFrame_1, outputFrame_1, size); // Resize 
        // outputFrame_1.convertTo(outputFrame_1, CV_32F); 
        // outputFrame_1 = outputFrame_1 / 255.0; // Normalize to [0, 1] range
        // cv::subtract(outputFrame_1, cv::Scalar(0.485, 0.456, 0.406), outputFrame_1); // Subtract mean
        // cv::divide(outputFrame_1, cv::Scalar(0.229, 0.224, 0.225), outputFrame_1); // Divide by std deviation
        // outputFrame_1 = cv::dnn::blobFromImage(outputFrame_1); // Convert to blob
        // float* data_ptr_1 = outputFrame_1.ptr<float>(0);

        // OutputFrame = outputFrame_1.clone(); // Copy the processed data to output Mat

        // OutputFrame.upload(outputFrame_1); // Upload to GPU memory


        // float* data_ptr_1 = outputFrame_1.ptr<float>(0);



        // -------------------deal by cv::dnn (faster)---------------

        // 速度较快
        // auto inputFrame_1 = inputFrame.clone(); // 确保输入图像不被修改
        // auto OutputFrame_1 = cv::dnn::blobFromImage(
        //     inputFrame_1, 
        //     1.0 / 255.0,                     // 归一化因子
        //     size,                             // 目标尺寸
        //     cv::Scalar(0.485, 0.456, 0.406),  // ImageNet均值 (RGB顺序)
        //     true,                             // swapRB=true (BGR→RGB)
        //     false,                            // 不裁剪
        //     CV_32F                            // 输出浮点型
        // );

        // float* data_ptr_1 = outputFrame_1.ptr<float>(0);
        // // 访问 [0][0][0][0]（首个元素）
        // float first_element_1 = data_ptr_1[0];

        // std::vector<double> stdDev{0.229, 0.224, 0.225}; 
        
        // // 手动除以标准差（blobFromImage不支持）
        // float* data = reinterpret_cast<float*>(OutputFrame_1.data);
        // const int channels = 3;
        // const int area = size.area();
        // for (int c = 0; c < channels; ++c) {
        //     cv::Mat channel(1, area, CV_32F, data + c * area);
        //     channel /= stdDev[c];  // stdDev = {0.229, 0.224, 0.225}
        // }

        // OutputFrame = OutputFrame_1.clone(); // 将处理后的数据复制到输出 Mat

        

        //----------------------deal by cv::cuda(fastest)--------------------------


        cv::cuda::GpuMat gpuFrame, gpuResized;
        cv::cuda::HostMem pinnedFrame(inputFrame);
        gpuFrame.upload(pinnedFrame.createMatHeader());

        // GPU accelerated color conversion and resizing
        cv::cuda::cvtColor(gpuFrame, gpuFrame, cv::COLOR_BGR2RGB);


        cv::cuda::resize(gpuFrame, gpuResized, size,0, 0, cv::INTER_LINEAR); // GPU上缩放图像

        cv::cuda::GpuMat gpuTypeConverted;
        gpuResized.convertTo(gpuResized, CV_32F);
        if (type=="F16") cv::cuda::convertFp16(gpuResized, gpuTypeConverted); // 转换为FP16
        else if (type=="F32") gpuTypeConverted = gpuResized; // 转换为FP32
        else throw std::runtime_error("Unsupported data type: " + type);


        // normalize the image
        cv::cuda::multiply(gpuTypeConverted, cv::Scalar(1.0/255.0), gpuTypeConverted);
        cv::cuda::subtract(gpuTypeConverted, cv::Scalar(0.485, 0.456, 0.406), gpuTypeConverted);
        cv::cuda::divide(gpuTypeConverted, cv::Scalar(0.229, 0.224, 0.225), gpuTypeConverted);


        // transform to NCHW format
        ConvertToBlob(gpuTypeConverted, OutputFrame,0,type); 


        // for debugging
        // cv::cuda::HostMem hostMem;                           // 创建锁页内存对象
        // hostMem.create(OutputFrame.size(), OutputFrame.type()); // 分配与 GPU 数据匹配的内存
        // OutputFrame.download(hostMem);                       // 高效下载到锁页内存
        // cv::Mat cpuFrame = hostMem.createMatHeader();

        // float * data_ptr = cpuFrame.ptr<float>(0); // 获取数据指针
        // // // 访问 [0][0][0][0]（首个元素）
        // float first_element = data_ptr[0]; // 获取首个元素的值

        // std::cout<< "First element in GPU output: " << first_element << std::endl;


    }


    void reverse_transform(const cv::Mat& inputFrame, cv::Mat& outputFrame) {

        std::vector<cv::Mat> imageBlobs; // 直接使用输入的 blob
        cv::dnn::imagesFromBlob(inputFrame, imageBlobs); // 将 blob 转换为图像格式
        cv::Mat imageBlob = imageBlobs[0]; // 获取第一个图像 blob

        // 反归一化
        cv::Scalar mean(0.485, 0.456, 0.406);
        cv::Scalar std(0.229, 0.224, 0.225);
        cv::multiply(imageBlob, std, imageBlob); // x * std
        cv::add(imageBlob, mean, imageBlob);     // x + mean

        // 缩放并转换数据类型
        cv::Mat restoredImage;
        imageBlob.convertTo(restoredImage, CV_32FC3, 255.0); // 缩放到 [0,255]
        restoredImage.convertTo(restoredImage, CV_8UC3);      // 转为 8 位整型

        // 通道顺序转换 (RGB → BGR)
        if (restoredImage.channels() == 3) {
            cv::cvtColor(restoredImage, restoredImage, cv::COLOR_RGB2BGR);
        }

        outputFrame = restoredImage.clone();
    }

    // modified from https://blog.csdn.net/qq_44908396/article/details/143855329
    // in our code, we only use resize instead of centerCrop
    void centerCrop(const cv::Mat& inputFrame, cv::Mat& outputFrame, int size) {
       
        int inputWidth = inputFrame.cols;
        int inputHeight = inputFrame.rows;
    
 
        int startX = (inputWidth - size) / 2;
        int startY = (inputHeight - size) / 2;
    

        if (startX < 0) startX = 0;
        if (startY < 0) startY = 0;
        if (startX + size > inputWidth) size = inputWidth - startX;
        if (startY + size > inputHeight) size = inputHeight - startY;

        cv::Rect cropRegion(startX, startY, size, size);
        cv::Mat tempFrame = inputFrame.clone();
        outputFrame = tempFrame(cropRegion).clone(); // Crop the region from the input frame

    }





    // ------------------------Pipeline class------------------------
    Pipeline::Pipeline(const Config& videoConfig)
            : config(videoConfig),
            StuEngine(load_model(config.StuModelPath)),
            TeaEngine(load_model(config.TeaModelPath)),
            StuContext(StuEngine->createExecutionContext()),
            TeaContext(TeaEngine->createExecutionContext()){
                
                // --------set input shape for the models----------
                StuContext->setInputShape(StuEngine->getIOTensorName(0), nvinfer1::Dims4(config.batchSize,3, config.inputHeight, config.inputWidth));
                TeaContext->setInputShape(TeaEngine->getIOTensorName(0), nvinfer1::Dims4(config.batchSize,3, config.inputHeight, config.inputWidth));

                // ---------allocate memory for input and output tensors----------
                StuBuffer.resize(StuEngine->getNbIOTensors());
                allocate_memory(StuEngine, StuContext,StuBuffer);
                TeaBuffer.resize(TeaEngine->getNbIOTensors());
                allocate_memory(TeaEngine, TeaContext,TeaBuffer);

                //--------- create CUDA streams for asynchronous execution------
                cudaStreamCreate(&StuStream); // Create CUDA stream for student model
                cudaStreamCreate(&TeaStream); // Create CUDA stream for teacher model

              };


    void Pipeline::inference(cv::Mat& inputFrame, cv::Mat& outputFrame) {
        // ---------preprocess the input frame----------
        // time clock
        auto total_start = std::chrono::high_resolution_clock::now();
        auto t1 = std::chrono::high_resolution_clock::now();

        cv::cuda::GpuMat preprocessedFrame; // Use GpuMat for GPU processing
        if (config.type == "F32") preprocessedFrame.create(1,3*config.inputWidth*config.inputHeight,CV_32F); // Create GpuMat for preprocessed frame
        else if (config.type == "F16") preprocessedFrame.create(1,3*config.inputWidth*config.inputHeight,CV_16F); // Create GpuMat for preprocessed frame
        else throw std::runtime_error("Unsupported data type: " + config.type);

        // preprocessedFrame.create(1,3*config.inputWidth*config.inputHeight,CV_32F);

        // cv::Mat preprocessedFrame;

        Transform(inputFrame, preprocessedFrame, cv::Size(config.inputWidth, config.inputHeight),config.type); // Transform the input frame


        // ----------inference the models----------------
        
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "preprocess time: " 
                << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                << " ms" << std::endl;
        t1 = std::chrono::high_resolution_clock::now();

        // copy the preprocessed frame to GPU memory
        cudaMemcpyAsync(StuBuffer[0], preprocessedFrame.data, 
                        preprocessedFrame.size().area() * preprocessedFrame.elemSize(), cudaMemcpyDeviceToDevice, StuStream);
        // cudaMemcpyAsync(StuBuffer[0], preprocessedFrame.data, 
        //         hostMem.size().area() * hostMem.elemSize(), cudaMemcpyHostToDevice, StuStream);
        StuContext->enqueueV3(StuStream);
        cudaMemcpyAsync(TeaBuffer[0], preprocessedFrame.data, 
                        preprocessedFrame.size().area() * preprocessedFrame.elemSize(), cudaMemcpyDeviceToDevice, TeaStream);
        // cudaMemcpyAsync(TeaBuffer[0], preprocessedFrame.data, 
        //         hostMem.size().area() * hostMem.elemSize(), cudaMemcpyHostToDevice, TeaStream);
        TeaContext->enqueueV3(TeaStream);

        
        cudaStreamSynchronize(StuStream); // Wait for student model inference to complete
        cudaStreamSynchronize(TeaStream); // Wait for teacher model inference to complete

        //---------------post process----------------

        t2 = std::chrono::high_resolution_clock::now();
        std::cout << "CUDA time: " 
                << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                << " ms" << std::endl;
    
        // Copy the output from GPU memory to CPU memory
        // std::cout<< "Copying output from GPU to CPU..." << std::endl;
        std::vector<torch::Tensor> StuOutputTensors;
        std::vector<torch::Tensor> TeaOutputTensors;

        cv::Mat anomaly_map;

        if (config.type == "F32") anomaly_map = cv::Mat::zeros(config.outputHeight, config.outputWidth, CV_32FC1); // Initialize anomaly map
        else anomaly_map = cv::Mat::zeros(config.outputHeight, config.outputWidth, CV_16FC1);

        t1 = std::chrono::high_resolution_clock::now();

        // cudaDeviceSynchronize();
        cal_anomaly_map(anomaly_map); // Calculate the anomaly map

        outputFrame = anomaly_map.clone(); // Copy the anomaly map to output frame

        t2 = std::chrono::high_resolution_clock::now();
        std::cout << "post process time: " 
                << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                << " ms" << std::endl;

        auto total_end = std::chrono::high_resolution_clock::now();
        std::cout << "total time: " 
                << std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count()
                << " ms" << std::endl;

    }



    nvinfer1::ICudaEngine* Pipeline::load_model(const std::string& modelPath) {
        
        // open engine file
        std::ifstream ModelFile(modelPath, std::ios::binary | std::ios::ate);
        if (!ModelFile.is_open()) throw std::runtime_error("File open failed");

        // copy the file to memory
        size_t ModelSize = ModelFile.tellg(); // Get the size of the file
        if (ModelSize < 0) throw std::runtime_error("Failed to get file size");
        ModelFile.seekg(0,std::ios::beg); // Reset the pointer to the beginning of the file

        std::vector<char> trtModel(ModelSize); // Allocate memory for the model data
        ModelFile.read(trtModel.data(),ModelSize); // Read the entire file into memory
        
        
        // Deserialize the model
        nvinfer1::ICudaEngine* Engine = Pipeline::runtime->deserializeCudaEngine(trtModel.data(), ModelSize);
        if (!Engine) throw std::runtime_error("Failed to deserialize engine");
        return Engine; // Return a cloned pointer to the engine

    }

    void Pipeline::allocate_memory(nvinfer1::ICudaEngine*& Engine,
                                nvinfer1::IExecutionContext*& Context,
                                std::vector<void*>& buffer) {
        
        // Get the number of input and output tensors
        auto getVolume = [](const nvinfer1::Dims& d) {
            return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int>());
        };
        
        for(int idx = 0;idx < Engine->getNbIOTensors();idx++){
            auto tensor_name = Engine->getIOTensorName(idx);
            auto shape = Context->getTensorShape(tensor_name);
            size_t size = getVolume(shape) * sizeof(float); // Assuming float32 type
            auto err = cudaMalloc(&buffer[idx], size); // Allocate memory on GPU  
            Context->setTensorAddress(tensor_name, buffer[idx]); // Set the tensor address in the context    
        }

    }

    void Pipeline::cal_anomaly_map(cv::Mat& anomaly_map) {
        if (StuBuffer.size() != TeaBuffer.size()) {
            throw std::runtime_error("Output buffers size mismatch");
        }

        // 1. configuration of tensor type(support F16 and F32)
        auto tensorType = (config.type == "F16") ? torch::kFloat16 : torch::kFloat32;
        const size_t elem_size = (tensorType == torch::kFloat16) ? sizeof(half_float::half) : sizeof(float);
        torch::TensorOptions options = torch::TensorOptions().dtype(tensorType).device(torch::kCUDA);

        // 2. create temporary anomaly map
        torch::Tensor TempAnomalyMap = torch::zeros({config.batchSize, 1, config.outputHeight, config.outputWidth}, options);

        // 3. calculate anomaly map
        for (int i = 1; i < StuBuffer.size(); i++) {
            auto tensor_name_stu = StuEngine->getIOTensorName(i);
            auto tensor_name_tea = TeaEngine->getIOTensorName(i);
            auto shape = StuContext->getTensorShape(tensor_name_stu);
            
        // 4. create tensors from buffers(zero copy or deep copy)
        #ifdef USE_DIRECT_BLOB
            void* stu_address = const_cast<void*>(StuContext->getTensorAddress(tensor_name_stu));
            torch::Tensor StuTensor = torch::from_blob(stu_address, {shape.d[0], shape.d[1], shape.d[2], shape.d[3]}, options);
            void* tea_address = const_cast<void*>(TeaContext->getTensorAddress(tensor_name_tea));
            torch::Tensor TeaTensor = torch::from_blob(tea_address, {shape.d[0], shape.d[1], shape.d[2], shape.d[3]}, options);
        #else
            // deep copy
            torch::Tensor StuTensor = torch::from_blob(StuBuffer[i], {shape.d[0], shape.d[1], shape.d[2], shape.d[3]}, options).clone();
            torch::Tensor TeaTensor = torch::from_blob(TeaBuffer[i], {shape.d[0], shape.d[1], shape.d[2], shape.d[3]}, options).clone();
        #endif

            // 5. use FP32 for cosine similarity calculation (to avoid precision loss)
            torch::Tensor temp;
            {
                // 5.1 converte to FP32 if necessary
                torch::Tensor StuFloat = (tensorType == torch::kFloat16) ? 
                    StuTensor.to(torch::kFloat32) : StuTensor;
                torch::Tensor TeaFloat = (tensorType == torch::kFloat16) ? 
                    TeaTensor.to(torch::kFloat32) : TeaTensor;

                // 5.2 calculate cosine similarity
                temp = 1 - torch::nn::functional::cosine_similarity(
                    StuFloat, TeaFloat, 
                    torch::nn::functional::CosineSimilarityFuncOptions().dim(1)
                );
                temp = temp.to(tensorType); // turn into desired type
            }

            // 6. resize the tensor to the output size
            temp = torch::unsqueeze(temp, 1);
            std::vector<int64_t> output_size = {config.outputHeight, config.outputWidth};
            temp = torch::nn::functional::interpolate(
                temp, 
                torch::nn::functional::InterpolateFuncOptions()
                    .size(output_size)
                    .mode(torch::kBilinear)
                    .align_corners(true)
            );
            TempAnomalyMap.add_(temp); // In-place addition to the temporary anomaly map
        }

        // 7. squeeze and copy to CPU
        TempAnomalyMap = TempAnomalyMap.squeeze(0).squeeze(0).cpu();
        const size_t expected_bytes = TempAnomalyMap.numel() * elem_size;
        
        if (anomaly_map.total() * anomaly_map.elemSize() != expected_bytes) {
            throw std::runtime_error("cv::Mat memory size mismatch");
        }
        
        // 8. copy the anomaly map to cv::Mat
        if (anomaly_map.type() == CV_32F || tensorType == torch::kFloat32) {
            std::memcpy(anomaly_map.data, TempAnomalyMap.data_ptr(), expected_bytes);
        } 
        else if (anomaly_map.type() == CV_16F && tensorType == torch::kFloat16) {
            // Convert FP16 to half_float::half and copy
            cudaMemcpy(anomaly_map.data, TempAnomalyMap.data_ptr(), 
                    expected_bytes, cudaMemcpyDeviceToHost);
        }
    }


};