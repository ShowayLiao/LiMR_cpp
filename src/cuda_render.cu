// ========================================================================
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h> 
#endif
// ========================================================================

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <GL/gl.h>
#include <iostream>
#include "cuda_render.h"

// ==========================================
// Kernel 1: Float -> Jet Colormap (RGBA)
// ==========================================
__device__ void valToJet(float v, unsigned char& r, unsigned char& g, unsigned char& b) {
    if (v < 0.25f) { r = 0; g = (unsigned char)(255 * 4 * v); b = 255; }
    else if (v < 0.5f) { r = 0; g = 255; b = (unsigned char)(255 * (1 + 4 * (0.25f - v))); }
    else if (v < 0.75f) { r = (unsigned char)(255 * 4 * (v - 0.5f)); g = 255; b = 0; }
    else { r = 255; g = (unsigned char)(255 * (1 + 4 * (0.75f - v))); b = 0; }
}

__global__ void render_heatmap_kernel(float* d_input, uchar4* d_output_texture, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float val = d_input[idx]; 
    
    // 【核心修复】增加饱和截断 (Saturate)
    // 如果 val > 1.0，强制设为 1.0；如果 val < 0.0，强制设为 0.0
    // 这样 >1.0 的高热力区就会保持纯红，不会回卷变成其他颜色
    val = fminf(fmaxf(val, 0.0f), 1.0f);
    
    unsigned char r, g, b;
    valToJet(val, r, g, b);
    d_output_texture[idx] = make_uchar4(r, g, b, 255);
}

// ==========================================
// Kernel 2: BGR (OpenCV) -> RGBA (OpenGL)
// ==========================================
// 输入依然是 uchar3 (因为OpenCV是BGR)，但输出必须是 uchar4
__global__ void render_frame_kernel(uchar3* d_input, uchar4* d_output_texture, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    uchar3 bgr = d_input[idx];
    // 交换 B/R 并加上 Alpha
    d_output_texture[idx] = make_uchar4(bgr.z, bgr.y, bgr.x, 255); 
}

// ==========================================
// C++ Wrapper 实现
// ==========================================
CudaInteropTexture::CudaInteropTexture() 
    : texture_id(0), cuda_res(nullptr), d_buffer(nullptr), d_rgba_buffer(nullptr), width(0), height(0) {}

CudaInteropTexture::~CudaInteropTexture() {
    cleanup();
}

void CudaInteropTexture::cleanup() {
    if (cuda_res) { cudaGraphicsUnregisterResource(cuda_res); cuda_res = nullptr; }
    if (texture_id) { glDeleteTextures(1, &texture_id); texture_id = 0; }
    if (d_buffer) { cudaFree(d_buffer); d_buffer = nullptr; }
    if (d_rgba_buffer) { cudaFree(d_rgba_buffer); d_rgba_buffer = nullptr; } // 清理新buffer
}

void CudaInteropTexture::init(int w, int h) {
    if (texture_id != 0) return;
    this->width = w; this->height = h;

    // 1. 创建 OpenGL 纹理 (改用 GL_RGBA)
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // 重点：申请 GL_RGBA 显存
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    // 2. 注册给 CUDA
    cudaGraphicsGLRegisterImage(&cuda_res, texture_id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
}

// 渲染热力图
void CudaInteropTexture::upload_and_render_heatmap(const float* cpu_ptr, size_t size_bytes) {
    if (!cuda_res) return;

    // 分配输入 buffer
    if (!d_buffer) cudaMalloc(&d_buffer, size_bytes);
    cudaMemcpy(d_buffer, cpu_ptr, size_bytes, cudaMemcpyHostToDevice);

    // 分配输出 buffer (uchar4)
    // 【修复】不再使用 static，防止冲突
    if (!d_rgba_buffer) cudaMalloc(&d_rgba_buffer, width * height * sizeof(uchar4));

    cudaGraphicsMapResources(1, &cuda_res, 0);
    cudaArray_t viewCudaArray;
    cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, cuda_res, 0, 0);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    // 运行 Kernel
    render_heatmap_kernel<<<grid, block>>>((float*)d_buffer, (uchar4*)d_rgba_buffer, width, height);

    // 拷贝 uchar4 数据到 Array
    cudaMemcpy2DToArray(viewCudaArray, 0, 0, d_rgba_buffer, width * sizeof(uchar4), width * sizeof(uchar4), height, cudaMemcpyDeviceToDevice);

    cudaGraphicsUnmapResources(1, &cuda_res, 0);
}

// 渲染原图
void CudaInteropTexture::upload_and_render_frame(const unsigned char* cpu_ptr, size_t size_bytes) {
    if (!cuda_res) return;

    if (!d_buffer) cudaMalloc(&d_buffer, size_bytes);
    cudaMemcpy(d_buffer, cpu_ptr, size_bytes, cudaMemcpyHostToDevice);

    if (!d_rgba_buffer) cudaMalloc(&d_rgba_buffer, width * height * sizeof(uchar4));

    cudaGraphicsMapResources(1, &cuda_res, 0);
    cudaArray_t viewCudaArray;
    cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, cuda_res, 0, 0);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    // 输入是 uchar3 (BGR), 输出是 uchar4 (RGBA)
    render_frame_kernel<<<grid, block>>>((uchar3*)d_buffer, (uchar4*)d_rgba_buffer, width, height);

    cudaMemcpy2DToArray(viewCudaArray, 0, 0, d_rgba_buffer, width * sizeof(uchar4), width * sizeof(uchar4), height, cudaMemcpyDeviceToDevice);

    cudaGraphicsUnmapResources(1, &cuda_res, 0);
}