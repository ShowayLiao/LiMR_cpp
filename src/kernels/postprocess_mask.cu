#include <cuda_runtime.h>

__global__ void generateMaskFromMapKernel(float* d_map, uint8_t* d_out_mask, int w, int h, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h) {
        int idx = y * w + x;
        float val = d_map[idx];
        d_out_mask[idx] = (val > threshold) ? 1 : 0;
    }
}

void launchGenerateMaskFromMap(float* d_map, uint8_t* d_out_mask, int w, int h, float threshold, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    generateMaskFromMapKernel<<<grid, block, 0, stream>>>(d_map, d_out_mask, w, h, threshold);
}
