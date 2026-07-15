#pragma once

#include "pipeline/FrameTask.h"
#include "engine/TrtEngine.h"
#include "common/CudaMemory.hpp"

namespace pipeline {

class PostProcessor {
public:
    PostProcessor(int width = 256, int height = 256);
    PostProcessor(int model_width, int model_height, int render_width, int render_height);
    ~PostProcessor();

    void process(trt::TrtEngine* engine, FrameTaskPtr task, float threshold, cudaStream_t stream, float* d_raw_anomaly_map, int render_width, int render_height);

private:
    int width_;
    int height_;
    int render_width_;
    int render_height_;
    std::vector<float> h_score_buf_;
    std::vector<uint8_t> h_label_buf_;
    
    // Workspace buffers for reuse
    DeviceBuffer d_workspace_mask_;
    DeviceBuffer d_workspace_overlay_model_;
    DeviceBuffer d_workspace_heatmap_model_;
    DeviceBuffer d_workspace_overlay_resized_;
    DeviceBuffer d_workspace_original_resized_;
    DeviceBuffer d_heatmap_reduction_workspace_;
    DeviceBuffer d_heatmap_min_;
    DeviceBuffer d_heatmap_max_;
    DeviceBuffer d_heatmap_invalid_;
    size_t heatmap_reduction_workspace_bytes_ = 0;
    size_t allocated_size_model_ = 0;
    size_t allocated_size_render_ = 0;
    
    void ensureWorkspace(size_t model_pixels, size_t render_pixels);
};

}
