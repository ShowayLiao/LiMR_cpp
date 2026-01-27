#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void heatmapToSurfaceKernel(float* d_map, cudaSurfaceObject_t surface, int w, int h, float min_val, float max_val) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h) {
        int idx = y * w + x;
        float val = d_map[idx];

        float normalized = (val - min_val) / (max_val - min_val + 1e-6f);
        normalized = fmaxf(0.0f, fminf(1.0f, normalized));

        uchar4 color;
        color.x = (unsigned char)(normalized * 255);
        color.y = (unsigned char)((1.0f - normalized) * 255);
        color.z = 0;
        color.w = 255;

        surf2Dwrite(color, surface, x * sizeof(uchar4), y);
    }
}

void launchHeatmapToSurface(float* d_map, cudaSurfaceObject_t surface, int w, int h, float min_val, float max_val, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    heatmapToSurfaceKernel<<<grid, block, 0, stream>>>(d_map, surface, w, h, min_val, max_val);
}
