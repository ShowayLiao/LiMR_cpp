#include <cuda_runtime.h>
#include <cstdint>

__global__ void layoutConvertNormKernel(uint8_t* src, float* dst, int w, int h, bool skip_normalization) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h) {
        int src_idx = (y * w + x) * 3;
        
        if (skip_normalization) {
            // Only do layout conversion, no normalization
            dst[0 * h * w + y * w + x] = (float)src[src_idx + 0];
            dst[1 * h * w + y * w + x] = (float)src[src_idx + 1];
            dst[2 * h * w + y * w + x] = (float)src[src_idx + 2];
        } else {
            // Do both layout conversion and normalization
            float scale = 1.0f / 255.0f;
            float mean_r = 0.485f;
            float mean_g = 0.456f;
            float mean_b = 0.406f;
            float std_r = 0.229f;
            float std_g = 0.224f;
            float std_b = 0.225f;
            
            dst[0 * h * w + y * w + x] = ((float)src[src_idx + 0] * scale - mean_r) / std_r;
            dst[1 * h * w + y * w + x] = ((float)src[src_idx + 1] * scale - mean_g) / std_g;
            dst[2 * h * w + y * w + x] = ((float)src[src_idx + 2] * scale - mean_b) / std_b;
        }
    }
}

void launchLayoutConvertNorm(uint8_t* src, float* dst, int w, int h, cudaStream_t stream, bool skip_normalization) {
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    
    layoutConvertNormKernel<<<grid, block, 0, stream>>>(src, dst, w, h, skip_normalization);
}
