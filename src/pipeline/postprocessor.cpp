#include "pipeline/PostProcessor.h"
#include "common/CudaMemory.hpp"
#include <stdexcept>
#include <cuda_runtime.h>
#include <npp.h>
#include <pipeline/FrameTask.h>
#include <limits>
#include <string>
#include "kernels/HeatmapKernels.h"

extern void launchGenerateMaskFromMap(float* d_map, uint8_t* d_mask, int w, int h, float threshold, cudaStream_t stream);
extern void launchGenerateOverlayFromMask(uint8_t* d_mask, uchar4* d_out_overlay, int w, int h, float threshold, cudaStream_t stream);
extern void launchCombineImageWithOverlay(uchar4* d_original, uchar4* d_overlay, uchar4* d_out_combined, int w, int h, cudaStream_t stream);
extern void launchResizeImage(uint8_t* d_src, int src_w, int src_h, uint8_t* d_dst, int dst_w, int dst_h, cudaStream_t stream);
extern void launchResizeImageRGBA(uint8_t* d_src, int src_w, int src_h, uint8_t* d_dst, int dst_w, int dst_h, cudaStream_t stream);
extern void launchResizeBGRToRGBA(uint8_t* d_src, int src_w, int src_h, uchar4* d_dst, int dst_w, int dst_h, cudaStream_t stream);

namespace pipeline {

namespace {

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

size_t checkedPixelCount(int width, int height, const char* label) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument(std::string(label) + " dimensions must be positive");
    }
    const size_t w = static_cast<size_t>(width);
    const size_t h = static_cast<size_t>(height);
    if (w > std::numeric_limits<size_t>::max() / h) {
        throw std::overflow_error(std::string(label) + " pixel count overflow");
    }
    return w * h;
}

} // namespace

PostProcessor::PostProcessor(int width, int height) : width_(width), height_(height), render_width_(width), render_height_(height) {
}

PostProcessor::PostProcessor(int model_width, int model_height, int render_width, int render_height) 
    : width_(model_width), height_(model_height), render_width_(render_width), render_height_(render_height) {
    std::cout << "[PostProcessor] Model output size: " << model_width << "x" << model_height << std::endl;
    std::cout << "[PostProcessor] Render size: " << render_width << "x" << render_height << std::endl;
}

PostProcessor::~PostProcessor() {
}

void PostProcessor::ensureWorkspace(size_t model_pixels, size_t render_pixels) {
    if (model_pixels > std::numeric_limits<size_t>::max() / 4U ||
        render_pixels > std::numeric_limits<size_t>::max() / 4U) {
        throw std::overflow_error("Postprocessor RGBA workspace size overflow");
    }
    // Check if model-sized buffers need allocation
    if (allocated_size_model_ < model_pixels) {
        // Allocate or reallocate model-sized buffers
        d_workspace_mask_ = make_device_buffer(model_pixels * sizeof(uint8_t));
        d_workspace_overlay_model_ = make_device_buffer(model_pixels * 4 * sizeof(uint8_t));
        d_workspace_heatmap_model_ = make_device_buffer(model_pixels * 4 * sizeof(uint8_t));
        heatmap_reduction_workspace_bytes_ = getHeatmapReductionWorkspaceSize(model_pixels);
        if (heatmap_reduction_workspace_bytes_ == 0) {
            throw std::runtime_error("Failed to determine heatmap reduction workspace size");
        }
        d_heatmap_reduction_workspace_ = make_device_buffer(heatmap_reduction_workspace_bytes_);
        d_heatmap_min_ = make_device_buffer(sizeof(float));
        d_heatmap_max_ = make_device_buffer(sizeof(float));
        d_heatmap_invalid_ = make_device_buffer(sizeof(int));
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
    if (!engine || !task || !d_raw_anomaly_map) {
        throw std::invalid_argument("PostProcessor requires valid engine, task, and anomaly map pointers");
    }
    // Get target size from task
    int rw = task->target_width;
    int rh = task->target_height;
    
    // Calculate pixel counts
    const size_t model_pixels = checkedPixelCount(width_, height_, "Model output");
    const size_t render_pixels = checkedPixelCount(rw, rh, "Render output");
    if (render_width != rw || render_height != rh) {
        throw std::invalid_argument("Render dimensions do not match the frame task");
    }
    
    // Ensure workspace buffers are properly sized
    ensureWorkspace(model_pixels, render_pixels);
    
    // Ensure output buffers are properly sized
    if (!task->ensureOutputBuffers()) {
        throw std::runtime_error("Invalid output buffer dimensions");
    }
    
    float* d_score = static_cast<float*>(engine->getBuffer("pred_score"));
    void* d_label = engine->getBuffer("pred_label");
    const trt::Binding* score_info = engine->getTensorInfo("pred_score");
    const trt::Binding* label_info = engine->getTensorInfo("pred_label");
    const trt::Binding* map_info = engine->getTensorInfo("anomaly_map");

    // Ensure pointers are valid
    if (!d_score || !d_label || !score_info || !label_info || !map_info ||
        score_info->type != nvinfer1::DataType::kFLOAT ||
        label_info->type != nvinfer1::DataType::kBOOL ||
        map_info->type != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error("Failed to get engine output buffers");
    }

    nvinfer1::Dims scalar_dims{};
    scalar_dims.nbDims = 1;
    scalar_dims.d[0] = 1;
    size_t score_element_bytes = 0;
    size_t label_element_bytes = 0;
    size_t expected_map_bytes = 0;
    if (!trt::TrtEngine::calculateTensorBytes(scalar_dims, score_info->type, score_element_bytes) ||
        !trt::TrtEngine::calculateTensorBytes(scalar_dims, label_info->type, label_element_bytes) ||
        !trt::TrtEngine::calculateTensorBytes(map_info->dims, map_info->type, expected_map_bytes) ||
        expected_map_bytes != map_info->size || expected_map_bytes != model_pixels * sizeof(float)) {
        throw std::runtime_error("TensorRT output byte contract is inconsistent");
    }

    // Fill task result fields
    task->h_pred_score.resize(1);
    task->h_pred_label.resize(1);
    task->h_anomaly_map.resize(model_pixels);
    
    // Copy with correct sizes
    checkCuda(cudaMemcpyAsync(task->h_pred_score.data(), d_score, score_element_bytes,
                              cudaMemcpyDeviceToHost, stream), "Copy pred_score");
    checkCuda(cudaMemcpyAsync(task->h_pred_label.data(), d_label, label_element_bytes,
                              cudaMemcpyDeviceToHost, stream), "Copy pred_label");
    checkCuda(cudaMemcpyAsync(task->h_anomaly_map.data(), d_raw_anomaly_map, expected_map_bytes,
                              cudaMemcpyDeviceToHost, stream), "Copy anomaly_map");

    // Use workspace buffers instead of allocating new ones
    uint8_t* d_mask = (uint8_t*)d_workspace_mask_.get();

    // Generate mask from anomaly map
    launchGenerateMaskFromMap(d_raw_anomaly_map, d_mask, width_, height_, threshold, stream);
    checkCuda(cudaPeekAtLastError(), "Generate anomaly mask");

    // Use workspace buffer for overlay (model size)
    uchar4* d_overlay = (uchar4*)d_workspace_overlay_model_.get();
    launchGenerateOverlayFromMask(d_mask, d_overlay, width_, height_, threshold, stream);
    checkCuda(cudaPeekAtLastError(), "Generate anomaly overlay");

    // Use workspace buffer for heatmap (model size)
    // Apply color map to anomaly map
    checkCuda(launchApplyColorMapAutoRange(
                  d_raw_anomaly_map, d_workspace_heatmap_model_.get(), width_, height_,
                  d_heatmap_reduction_workspace_.get(), heatmap_reduction_workspace_bytes_,
                  static_cast<float*>(d_heatmap_min_.get()), static_cast<float*>(d_heatmap_max_.get()),
                  static_cast<int*>(d_heatmap_invalid_.get()), stream),
              "Generate normalized heatmap");
    int invalid_heatmap_value = 0;
    checkCuda(cudaMemcpyAsync(&invalid_heatmap_value, d_heatmap_invalid_.get(), sizeof(int),
                              cudaMemcpyDeviceToHost, stream), "Validate anomaly_map values");

    // Resize heatmap to render size and write to task's final buffer
    launchResizeImageRGBA(
        (uint8_t*)d_workspace_heatmap_model_.get(),
        width_, height_,
        (uint8_t*)task->d_final_heatmap.get(),
        task->target_width, task->target_height,
        stream
    );
    checkCuda(cudaPeekAtLastError(), "Resize heatmap");

    // Resize overlay to render size using workspace buffer
    launchResizeImageRGBA(
        (uint8_t*)d_workspace_overlay_model_.get(),
        width_, height_,
        (uint8_t*)d_workspace_overlay_resized_.get(),
        task->target_width, task->target_height,
        stream
    );
    checkCuda(cudaPeekAtLastError(), "Resize overlay");

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
    checkCuda(cudaPeekAtLastError(), "Resize source image");

    // Combine resized original image with overlay and write to task's final buffer
    launchCombineImageWithOverlay(
        (uchar4*)d_workspace_original_resized_.get(),
        (uchar4*)d_workspace_overlay_resized_.get(),
        (uchar4*)task->d_final_overlay.get(),
        rw,
        rh,
        stream
    );
    checkCuda(cudaPeekAtLastError(), "Combine source image and overlay");

    checkCuda(cudaStreamSynchronize(stream), "Synchronize postprocessing stream");
    if (invalid_heatmap_value != 0) {
        throw std::runtime_error("TensorRT anomaly_map contains NaN or Inf");
    }
}

} // namespace pipeline
