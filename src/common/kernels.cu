#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ float clamp(float v, float min, float max) { return fmaxf(min, fminf(v, max)); }

// [修改] 返回类型改为 uchar4
__device__ uchar4 floatToJet(float v) {
    uchar4 c;
    float r = clamp(1.5f - fabsf(4.0f * v - 3.0f), 0.0f, 1.0f);
    float g = clamp(1.5f - fabsf(4.0f * v - 2.0f), 0.0f, 1.0f);
    float b = clamp(1.5f - fabsf(4.0f * v - 1.0f), 0.0f, 1.0f);
    
    c.x = (unsigned char)(r * 255.0f);
    c.y = (unsigned char)(g * 255.0f);
    c.z = (unsigned char)(b * 255.0f);
    c.w = 255; // [关键] Alpha 通道设为 255 (不透明)
    return c;
}

// [修改] output 指针改为 uchar4*
__global__ void applyColorMapKernel(float* map, uchar4* output, int w, int h) {
    int idx = (blockIdx.y * blockDim.y + threadIdx.y) * w + (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < w * h) {
        float val = map[idx];
        if (val < 0.0f) val = 0.0f; else if (val > 1.0f) val = 1.0f;
        output[idx] = floatToJet(val);
    }
}

// [修改] 接口强制转换
void launchApplyColorMap(float* d_map, void* d_out_rgba, int w, int h, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    // 强制转换为 uchar4*
    applyColorMapKernel<<<grid, block, 0, stream>>>(d_map, (uchar4*)d_out_rgba, w, h);
}