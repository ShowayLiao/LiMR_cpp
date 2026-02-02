#include "pipeline/PostProcessor.h"
#include "common/CudaMemory.hpp"
#include <stdexcept>
#include <cuda_runtime.h>
#include <npp.h>
#include <pipeline/FrameTask.h>

extern void launchGenerateMaskFromMap(float* d_map, uint8_t* d_mask, int w, int h, float threshold, cudaStream_t stream);
extern void launchApplyColorMap(float* d_map, void* d_out_rgb, int w, int h, cudaStream_t stream);
extern void launchGenerateOverlayFromMask(uint8_t* d_mask, uchar4* d_out_overlay, int w, int h, float threshold, cudaStream_t stream);
extern void launchCombineImageWithOverlay(uchar4* d_original, uchar4* d_overlay, uchar4* d_out_combined, int w, int h, cudaStream_t stream);
extern void launchResizeImage(uint8_t* d_src, int src_w, int src_h, uint8_t* d_dst, int dst_w, int dst_h, cudaStream_t stream);
extern void launchResizeImageRGBA(uint8_t* d_src, int src_w, int src_h, uint8_t* d_dst, int dst_w, int dst_h, cudaStream_t stream);
extern void launchResizeBGRToRGBA(uint8_t* d_src, int src_w, int src_h, uchar4* d_dst, int dst_w, int dst_h, cudaStream_t stream);

namespace pipeline {

PostProcessor::PostProcessor(int width, int height) : width_(width), height_(height), render_width_(width), render_height_(height) {
}

PostProcessor::PostProcessor(int model_width, int model_height, int render_width, int render_height) 
    : width_(model_width), height_(model_height), render_width_(render_width), render_height_(render_height) {
    std::cout << "[PostProcessor] Model output size: " << model_width << "x" << model_height << std::endl;
    std::cout << "[PostProcessor] Render size: " << render_width << "x" << render_height << std::endl;
}

PostProcessor::~PostProcessor() {
}

void PostProcessor::ensureWorkspace(int model_pixels, int render_pixels) {
    // Check if model-sized buffers need allocation
    if (allocated_size_model_ < model_pixels) {
        // Allocate or reallocate model-sized buffers
        d_workspace_mask_ = make_device_buffer(model_pixels * sizeof(uint8_t));
        d_workspace_overlay_model_ = make_device_buffer(model_pixels * 4 * sizeof(uint8_t));
        d_workspace_heatmap_model_ = make_device_buffer(model_pixels * 4 * sizeof(uint8_t));
        allocated_size_model_ = model_pixels;
        // std::cout << "[PostProcessor] Allocated model workspace: " << model_pixels << " pixels" << std::endl;
    }
    
    // Check if render-sized buffers need allocation
    if (allocated_size_render_ < render_pixels) {
        // Allocate or reallocate render-sized buffers
        d_workspace_overlay_resized_ = make_device_buffer(render_pixels * 4 * sizeof(uint8_t));
        d_workspace_original_resized_ = make_device_buffer(render_pixels * 4 * sizeof(uint8_t));
        allocated_size_render_ = render_pixels;
        // std::cout << "[PostProcessor] Allocated render workspace: " << render_pixels << " pixels" << std::endl;
    }
}

void PostProcessor::process(trt::TrtEngine* engine, FrameTaskPtr task, float threshold, cudaStream_t stream, float* d_raw_anomaly_map, int render_width, int render_height) {
    // Get target size from task
    int rw = task->target_width;
    int rh = task->target_height;
    
    // Calculate pixel counts
    int model_pixels = width_ * height_;
    int render_pixels = rw * rh;
    
    // Ensure workspace buffers are properly sized
    ensureWorkspace(model_pixels, render_pixels);
    
    // Ensure output buffers are properly sized
    task->ensureOutputBuffers();
    
    float* d_score = (float*)engine->getBuffer("pred_score");
    bool* d_label = (bool*)engine->getBuffer("pred_label");

    // Ensure pointers are valid
    if (!d_score || !d_label || !d_raw_anomaly_map) {
        throw std::runtime_error("Failed to get engine output buffers");
    }

    // Fill task result fields
    task->h_pred_score.resize(1);
    task->h_pred_label.resize(1);
    task->h_anomaly_map.resize(width_ * height_);
    
    // Copy with correct sizes
    cudaMemcpyAsync(task->h_pred_score.data(), d_score, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(task->h_pred_label.data(), d_label, sizeof(uint8_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(task->h_anomaly_map.data(), d_raw_anomaly_map, width_ * height_ * sizeof(float), cudaMemcpyDeviceToHost, stream);
    
    // Wait for memcpy to complete before accessing data
    cudaStreamSynchronize(stream);
    
    // Print first 5 elements of anomaly map
    std::cout << "[Debug] First 5 elements of anomaly map: " << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "Element " << i << ": " << task->h_anomaly_map[i] << std::endl;
    }
    
    // Print 5 elements from the middle of the anomaly map
    int middle_idx = (width_ * height_) / 2;
    std::cout << "[Debug] 5 elements from middle of anomaly map: " << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "Element " << middle_idx + i << ": " << task->h_anomaly_map[middle_idx + i] << std::endl;
    }

    // Use workspace buffers instead of allocating new ones
    uint8_t* d_mask = (uint8_t*)d_workspace_mask_.get();

    // Generate mask from anomaly map
    launchGenerateMaskFromMap(d_raw_anomaly_map, d_mask, width_, height_, threshold, stream);

    // Use workspace buffer for overlay (model size)
    uchar4* d_overlay = (uchar4*)d_workspace_overlay_model_.get();
    launchGenerateOverlayFromMask(d_mask, d_overlay, width_, height_, threshold, stream);

    // Use workspace buffer for heatmap (model size)
    // Apply color map to anomaly map
    launchApplyColorMap(d_raw_anomaly_map, d_workspace_heatmap_model_.get(), width_, height_, stream);

    // Resize heatmap to render size and write to task's final buffer
    launchResizeImageRGBA(
        (uint8_t*)d_workspace_heatmap_model_.get(),
        width_, height_,
        (uint8_t*)task->d_final_heatmap.get(),
        task->target_width, task->target_height,
        stream
    );

    // Resize overlay to render size using workspace buffer
    launchResizeImageRGBA(
        (uint8_t*)d_workspace_overlay_model_.get(),
        width_, height_,
        (uint8_t*)d_workspace_overlay_resized_.get(),
        task->target_width, task->target_height,
        stream
    );

    // Create resized original image for overlay using workspace buffer
    int src_width = task->original_image.cols;
    int src_height = task->original_image.rows;
    
    // Call custom kernel to resize BGR to RGBA
    launchResizeBGRToRGBA(
        (uint8_t*)task->d_original_image.get(),
        task->original_image.cols, task->original_image.rows,
        (uchar4*)d_workspace_original_resized_.get(),
        rw, rh,
        stream
    );

    // Combine resized original image with overlay and write to task's final buffer
    launchCombineImageWithOverlay(
        (uchar4*)d_workspace_original_resized_.get(),
        (uchar4*)d_workspace_overlay_resized_.get(),
        (uchar4*)task->d_final_overlay.get(),
        rw,
        rh,
        stream
    );

    cudaStreamSynchronize(stream);
}

} // namespace pipeline