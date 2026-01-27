#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void maskToSurfaceKernel(uint8_t* d_mask, cudaSurfaceObject_t surface, int w, int h, uchar4 color) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h) {
        int idx = y * w + x;
        uchar4 pixel;

        if (d_mask[idx] == 1) {
            pixel = color;
        } else {
            pixel.x = 0;
            pixel.y = 0;
            pixel.z = 0;
            pixel.w = 0;
        }

        surf2Dwrite(pixel, surface, x * sizeof(uchar4), y);
    }
}

void launchMaskToSurface(uint8_t* d_mask, cudaSurfaceObject_t surface, int w, int h, uchar4 color, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    maskToSurfaceKernel<<<grid, block, 0, stream>>>(d_mask, surface, w, h, color);
}
