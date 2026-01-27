#pragma once

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

struct uchar4 {
    unsigned char x;
    unsigned char y;
    unsigned char z;
    unsigned char w;
};

void launchLayoutConvertNorm(uint8_t* src, float* dst, int w, int h, cudaStream_t stream);
void launchPostprocess(float* src, int w, int h, cudaStream_t stream);
void launchHeatmapToSurface(float* d_map, cudaSurfaceObject_t surface, int w, int h, float min_val, float max_val, cudaStream_t stream);
void launchGenerateMaskFromMap(float* d_map, uint8_t* d_out_mask, int w, int h, float threshold, cudaStream_t stream);
void launchMaskToSurface(uint8_t* d_mask, cudaSurfaceObject_t surface, int w, int h, uchar4 color, cudaStream_t stream);
