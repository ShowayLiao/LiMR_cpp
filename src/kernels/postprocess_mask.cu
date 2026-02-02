#include <cuda_runtime.h>
#include <stdint.h>

// New Kernel: Resize and convert BGR to RGBA
__global__ void resizeBGRToRGBAKernel(uint8_t* d_src, int src_w, int src_h, uchar4* d_dst, int dst_w, int dst_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < dst_w && y < dst_h) {
        // Simple nearest neighbor interpolation
        int srcX = x * src_w / dst_w;
        int srcY = y * src_h / dst_h;
        
        // Boundary protection
        if (srcX >= src_w) srcX = src_w - 1;
        if (srcY >= src_h) srcY = src_h - 1;

        // Source data is BGR (3 channels)
        int srcIdx = (srcY * src_w + srcX) * 3;
        int dstIdx = y * dst_w + x;
        
        uchar4 val;
        // OpenCV is BGR, OpenGL needs RGB
        val.x = d_src[srcIdx + 2]; // R <- BGR[2]
        val.y = d_src[srcIdx + 1]; // G <- BGR[1]
        val.z = d_src[srcIdx];     // B <- BGR[0]
        val.w = 255;                // Alpha
        
        d_dst[dstIdx] = val;
    }
}

// Host
void launchResizeBGRToRGBA(uint8_t* d_src, int src_w, int src_h, uchar4* d_dst, int dst_w, int dst_h, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
    resizeBGRToRGBAKernel<<<grid, block, 0, stream>>>(d_src, src_w, src_h, d_dst, dst_w, dst_h);
}

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

__global__ void generateOverlayFromMaskKernel(uint8_t* d_mask, uchar4* d_out_overlay, int w, int h, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h) {
        int idx = y * w + x;
        uint8_t mask_val = d_mask[idx];
        
        // Create semi-transparent red overlay
        uchar4 overlay;
        if (mask_val > 0) {
            // Red color with alpha based on mask value and threshold
            overlay.x = 255;     // Red channel
            overlay.y = 0;       // Green channel
            overlay.z = 0;       // Blue channel
            overlay.w = static_cast<unsigned char>(255 * (mask_val / 1.0f) * threshold);  // Alpha channel
        } else {
            // Fully transparent for background
            overlay.x = 0;
            overlay.y = 0;
            overlay.z = 0;
            overlay.w = 0;
        }
        
        d_out_overlay[idx] = overlay;
    }
}

void launchGenerateOverlayFromMask(uint8_t* d_mask, uchar4* d_out_overlay, int w, int h, float threshold, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    generateOverlayFromMaskKernel<<<grid, block, 0, stream>>>(d_mask, d_out_overlay, w, h, threshold);
}

__global__ void combineImageWithOverlayKernel(uchar4* d_original, uchar4* d_overlay, uchar4* d_out_combined, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h) {
        int idx = y * w + x;
        
        uchar4 orig = d_original[idx];
        uchar4 over = d_overlay[idx];
        
        float a = over.w / 255.0f;
        float inv_a = 1.0f - a;
        
        uchar4 comb;
        comb.x = (unsigned char)(orig.x * inv_a + over.x * a);
        comb.y = (unsigned char)(orig.y * inv_a + over.y * a);
        comb.z = (unsigned char)(orig.z * inv_a + over.z * a);
        comb.w = 255;
        
        d_out_combined[idx] = comb;
    }
}

void launchCombineImageWithOverlay(uchar4* d_original, uchar4* d_overlay, uchar4* d_out_combined, int w, int h, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    combineImageWithOverlayKernel<<<grid, block, 0, stream>>>(d_original, d_overlay, d_out_combined, w, h);
}

__global__ void fillSolidColorKernel(uint8_t* d_dst, int w, int h, uint8_t r, uint8_t g, uint8_t b) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < w && y < h) {
        int idx = (y * w + x) * 3;
        d_dst[idx] = b;     // B channel
        d_dst[idx + 1] = g; // G channel
        d_dst[idx + 2] = r; // R channel
    }
}

void launchFillSolidColor(uint8_t* d_dst, int w, int h, uint8_t r, uint8_t g, uint8_t b, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    fillSolidColorKernel<<<grid, block, 0, stream>>>(d_dst, w, h, r, g, b);
}

__global__ void resizeImageKernel(uint8_t* d_src, int src_w, int src_h, uint8_t* d_dst, int dst_w, int dst_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < dst_w && y < dst_h) {
        // Calculate corresponding position in source image
        float src_x_f = static_cast<float>(x) * src_w / dst_w;
        float src_y_f = static_cast<float>(y) * src_h / dst_h;
        int src_x = static_cast<int>(src_x_f);
        int src_y = static_cast<int>(src_y_f);
        
        // Clamp to avoid out-of-bounds access
        if (src_x < 0) src_x = 0;
        if (src_x >= src_w) src_x = src_w - 1;
        if (src_y < 0) src_y = 0;
        if (src_y >= src_h) src_y = src_h - 1;
        
        // Get source pixel (BGR format)
        int src_idx = (src_y * src_w + src_x) * 3;
        int dst_idx = (y * dst_w + x) * 3;
        
        // Copy pixel values
        d_dst[dst_idx] = d_src[src_idx];     // B channel
        d_dst[dst_idx + 1] = d_src[src_idx + 1]; // G channel
        d_dst[dst_idx + 2] = d_src[src_idx + 2]; // R channel
    }
}

void launchResizeImage(uint8_t* d_src, int src_w, int src_h, uint8_t* d_dst, int dst_w, int dst_h, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

    resizeImageKernel<<<grid, block, 0, stream>>>(d_src, src_w, src_h, d_dst, dst_w, dst_h);
}

__global__ void resizeImageRGBAKernel(uint8_t* d_src, int src_w, int src_h, uint8_t* d_dst, int dst_w, int dst_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < dst_w && y < dst_h) {
        // Calculate corresponding position in source image
        float src_x_f = static_cast<float>(x) * src_w / dst_w;
        float src_y_f = static_cast<float>(y) * src_h / dst_h;
        int src_x = static_cast<int>(src_x_f);
        int src_y = static_cast<int>(src_y_f);
        
        // Clamp to avoid out-of-bounds access
        if (src_x < 0) src_x = 0;
        if (src_x >= src_w) src_x = src_w - 1;
        if (src_y < 0) src_y = 0;
        if (src_y >= src_h) src_y = src_h - 1;
        
        // Get source pixel (RGBA format)
        int src_idx = (src_y * src_w + src_x) * 4;
        int dst_idx = (y * dst_w + x) * 4;
        
        // Copy pixel values
        d_dst[dst_idx] = d_src[src_idx];     // R channel
        d_dst[dst_idx + 1] = d_src[src_idx + 1]; // G channel
        d_dst[dst_idx + 2] = d_src[src_idx + 2]; // B channel
        d_dst[dst_idx + 3] = d_src[src_idx + 3]; // A channel
    }
}

// void launchResizeImageRGBA(uint8_t* d_src, int src_w, int src_h, uint8_t* d_dst, int dst_w, int dst_h, cudaStream_t stream) {
//     dim3 block(16, 16);
//     dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

//     resizeImageRGBAKernel<<<grid, block, 0, stream>>>(d_src, src_w, src_h, d_dst, dst_w, dst_h);
// }


__global__ void resizeImageRGBAKernel_Optimized(uchar4* d_src, int src_w, int src_h, 
                                                uchar4* d_dst, int dst_w, int dst_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < dst_w && y < dst_h) {


        float scale_x = (float)src_w / (float)dst_w;
        float scale_y = (float)src_h / (float)dst_h;
        
       
        int src_x = (int)(x * scale_x);
        int src_y = (int)(y * scale_y);
        
      
        src_x = min(max(src_x, 0), src_w - 1);
        src_y = min(max(src_y, 0), src_h - 1);
        

        int src_idx = src_y * src_w + src_x;
        int dst_idx = y * dst_w + x;
        
        // d_dst[dst_idx] = d_src[src_idx]; 
        uchar4 pixel = d_src[src_idx];
        // 保持原始的 alpha 值，不要硬编码为 255
        d_dst[dst_idx] = pixel;
    }
}


void launchResizeImageRGBA(uint8_t* d_src, int src_w, int src_h, 
                           uint8_t* d_dst, int dst_w, int dst_h, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);


    resizeImageRGBAKernel_Optimized<<<grid, block, 0, stream>>>(
        (uchar4*)d_src, src_w, src_h, 
        (uchar4*)d_dst, dst_w, dst_h
    );
}
