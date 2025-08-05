# include<opencv2/opencv.hpp>
# include<cuda_runtime.h>
# include<opencv2/core/cuda.hpp>
# include "utils.cuh"
# include "utils.h"
# include <cuda_fp16.h>


// // CUDA内核：HWC转NCHW (支持非连续内存)
// __global__ void HWC2CHW(const float* src, float* dst, int H, int W, int C, size_t src_step) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     const int total = H * W * C;
//     // 使用网格跨步循环处理所有像素
//     for (; idx < total; idx += blockDim.x * gridDim.x) {
//         int c = idx / (H * W);      // 计算通道维度
//         int sp = idx % (H * W);      // 空间位置索引
//         int h = sp / W;             // 行坐标
//         int w = sp % W;             // 列坐标
        
//         // 修正：使用传入的src_step计算源索引 (字节步长转浮点数步长)
//         size_t src_index = h * (src_step / sizeof(float)) + w * C + c;
//         dst[idx] = src[src_index];  // 直接写入目标位置
//     }
// }

// void ConvertToBlob(const cv::cuda::GpuMat& gpuResized, cv::cuda::GpuMat& outputBlob, cudaStream_t stream = 0,const std::string type) {
//     // CV_Assert(gpuResized.type() == CV_32FC3);
    
//     // 1. 强制确保输入内存连续
//     cv::cuda::GpuMat continuousInput;
//     if (!gpuResized.isContinuous()) {
//         gpuResized.copyTo(continuousInput); // 克隆非连续输入
//     } else {
//         continuousInput = gpuResized;       // 直接引用连续输入
//     }

//     const int H = continuousInput.rows;
//     const int W = continuousInput.cols;
//     const int C = 3;
    
//     // 2. 准备输出内存 (连续1x(C*H*W)矩阵)
//     if (type == "F32") outputBlob.create(1, H * W * C, CV_32F);
//     else if (type == "F16") outputBlob.create(1, H * W * C, CV_16F);
//     else {
//         throw std::runtime_error("Unsupported type: " + type);
//     }
//     outputBlob.setTo(0); // 显式初始化防止未定义值

//     // 3. 获取跨步参数（关键修正）
//     const size_t src_step = continuousInput.step; // 获取字节跨步
//     const float* src = continuousInput.ptr<float>(0);
//     float* dst = outputBlob.ptr<float>(0);
    
//     // 4. 配置内核参数
//     const int totalPixels = H * W * C;
//     const int blockSize = 256;
//     const int gridSize = (totalPixels + blockSize - 1) / blockSize;

//     // 5. 启动内核（传递src_step参数）
//     HWC2CHW<<<gridSize, blockSize, 0, stream>>>(src, dst, H, W, C, src_step);
    
//     // 6. 错误检查（同步流并验证状态）
//     cudaError_t err = cudaStreamSynchronize(stream);
//     if (err != cudaSuccess) {
//         fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
//     }
// }

template<typename T>
__global__ void HWC2CHW_Kernel(const void* src, void* dst, int H, int W, int C, size_t src_step) {
    T* src_ptr = (T*)src;
    T* dst_ptr = (T*)dst;
    const int total = H * W * C;
    const size_t src_step_elem = src_step / sizeof(T);  // 按元素计算的步长
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total; 
         idx += blockDim.x * gridDim.x) 
    {
        const int c = idx / (H * W);      // 目标通道索引
        const int sp = idx % (H * W);      // 空间位置
        const int h = sp / W;             // 行
        const int w = sp % W;             // 列
        
        // 计算源数据位置（考虑内存对齐）
        const size_t src_index = h * src_step_elem + w * C + c;
        dst_ptr[idx] = src_ptr[src_index];
    }
}

// 主机函数：支持FP32/FP16的转换
void ConvertToBlob(const cv::cuda::GpuMat& input, 
                   cv::cuda::GpuMat& output, 
                   cudaStream_t stream , 
                   const std::string& type ) 
{
    // 1. 确保输入内存连续
    cv::cuda::GpuMat continuousInput;
    if (!input.isContinuous()) {
        input.clone().copyTo(continuousInput);
    } else {
        continuousInput = input;
    }

    const int H = continuousInput.rows;
    const int W = continuousInput.cols;
    const int C = continuousInput.channels();

    // 2. 准备输出内存（优化内存分配）
    const int outType = (type == "F16") ? CV_16F : CV_32F;
    output.create(1, H * W * C, outType);
    
    // 3. 内核参数配置
    const int totalElements = H * W * C;
    const int blockSize = 256;  // 最优线程块大小
    const int gridSize = (totalElements + blockSize - 1) / blockSize;

    // 4. 动态选择内核类型
    if (outType == CV_32F) {
        HWC2CHW_Kernel<float><<<gridSize, blockSize, 0, stream>>>(
            continuousInput.ptr(), output.ptr<float>(), 
            H, W, C, continuousInput.step
        );
    } else if (outType == CV_16F) {
        HWC2CHW_Kernel<__half><<<gridSize, blockSize, 0, stream>>>(
            continuousInput.ptr(), output.ptr<__half>(), 
            H, W, C, continuousInput.step
        );
    }

    // 5. 异步错误检查（非阻塞）
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}