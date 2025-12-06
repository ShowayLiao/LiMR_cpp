#pragma once

// ========================================================================
// 必须放在最顶端！
// ========================================================================
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
// ========================================================================

#include <GL/gl.h>
#include <cuda_runtime.h>

class CudaInteropTexture {
public:
    // ... (构造、析构、init 等保持不变) ...
    CudaInteropTexture();
    ~CudaInteropTexture();
    void init(int w, int h);
    void cleanup();
    GLuint getTextureID() const { return texture_id; }
    void upload_and_render_heatmap(const float* cpu_ptr, size_t size_bytes);
    void upload_and_render_frame(const unsigned char* cpu_ptr, size_t size_bytes);

private:
    GLuint texture_id;
    struct cudaGraphicsResource* cuda_res;
    
    void* d_buffer;      // 接收 CPU 数据的 Buffer (Float 或 uchar3)
    void* d_rgba_buffer; // [新增] 渲染结果 Buffer (uchar4) - 解决 static 冲突
    
    int width, height;
};