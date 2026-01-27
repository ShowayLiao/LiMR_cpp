#include <cuda_runtime.h>

__global__ void postprocessKernel(float* src, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h) {
        int idx = y * w + x;
        src[idx] = fmaxf(0.0f, src[idx]);
    }
}

void launchPostprocess(float* src, int w, int h, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    
    postprocessKernel<<<grid, block, 0, stream>>>(src, w, h);
}
