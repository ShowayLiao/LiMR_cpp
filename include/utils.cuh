# ifndef UTILS_CUH
# define UTILS_CUH
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>



__global__ void HWC2CHW_Kernel(const void* src, void* dst, int H, int W, int C, size_t src_step);
void ConvertToBlob(const cv::cuda::GpuMat& input, 
                   cv::cuda::GpuMat& output, 
                   cudaStream_t stream = 0, 
                   const std::string& type = "F32");

# endif // UTILS_CUH