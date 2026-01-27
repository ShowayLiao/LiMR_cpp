#include "pipeline/PostProcessor.h"
#include "common/CudaMemory.hpp"
#include <stdexcept>

extern void launchGenerateMaskFromMap(float* d_map, uint8_t* d_mask, int w, int h, float threshold, cudaStream_t stream);
extern void launchApplyColorMap(float* d_map, void* d_out_rgb, int w, int h, cudaStream_t stream);

namespace pipeline {

PostProcessor::PostProcessor() {
}

PostProcessor::~PostProcessor() {
}

void PostProcessor::process(trt::TrtEngine* engine, FrameTaskPtr task, float threshold, cudaStream_t stream) {
    float* d_score = (float*)engine->getBuffer("pred_score");
    bool* d_label = (bool*)engine->getBuffer("pred_label");
    float* d_map = (float*)engine->getBuffer("anomaly_map");

    // Ensure pointers are valid
    if (!d_score || !d_label || !d_map) {
        throw std::runtime_error("Failed to get engine output buffers");
    }

    // Fill task result fields
    task->h_pred_score.resize(1);
    task->h_pred_label.resize(1);
    task->h_anomaly_map.resize(256 * 256);
    
    // Copy with correct sizes
    cudaMemcpyAsync(task->h_pred_score.data(), d_score, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(task->h_pred_label.data(), d_label, sizeof(uint8_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(task->h_anomaly_map.data(), d_map, 256 * 256 * sizeof(float), cudaMemcpyDeviceToHost, stream);

    if (!task->d_dynamic_mask) {
        task->d_dynamic_mask = make_device_buffer(256 * 256 * sizeof(uint8_t));
    }
    uint8_t* d_custom_mask = (uint8_t*)task->d_dynamic_mask.get();

    


    launchGenerateMaskFromMap(d_map, d_custom_mask, 256, 256, threshold, stream);

    if (!task->d_heatmap_gpu) {
        task->d_heatmap_gpu = make_device_buffer(256 * 256 * 4 * sizeof(uint8_t)); 
    }
    
    // 调用 Kernel
    launchApplyColorMap(d_map, task->d_heatmap_gpu.get(), 256, 256, stream);

    cudaStreamSynchronize(stream);
}

} // namespace pipeline