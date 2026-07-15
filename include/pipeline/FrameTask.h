#pragma once

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "../common/CudaMemory.hpp"
#include "pipeline/FrameBufferLayout.h"

namespace pipeline {

struct AnomalyPeak {
    float score;
    int x;
    int y;
    int class_id;
};

struct FrameTask {
    long frame_id;
    double timestamp;
    double start_time;
    double end_time;
    bool is_valid;

    cv::Mat original_image;
    DeviceBuffer d_original_image;

    // Final output buffers (frontend resolution)
    DeviceBuffer d_final_heatmap;
    DeviceBuffer d_final_overlay;

    // Target resolution for rendering
    int target_width;
    int target_height;

    // Allocated buffer sizes
    int allocated_width;
    int allocated_height;
    size_t original_image_capacity_bytes;
    size_t output_buffer_capacity_bytes;

    std::vector<float> h_pred_score;
    std::vector<uint8_t> h_pred_label;
    std::vector<float> h_anomaly_map;
    std::vector<uint8_t> h_pred_mask;

    unsigned int gl_heatmap_tex_id;
    unsigned int gl_mask_tex_id;
    unsigned int gl_overlay_tex_id;
    unsigned int gl_combined_tex_id;

    std::vector<AnomalyPeak> peaks;

    FrameTask() : 
        frame_id(0),
        timestamp(0.0),
        start_time(0.0),
        end_time(0.0),
        is_valid(true),
        d_original_image(nullptr),
        d_final_heatmap(nullptr),
        d_final_overlay(nullptr),
        target_width(0),
        target_height(0),
        allocated_width(0),
        allocated_height(0),
        original_image_capacity_bytes(0),
        output_buffer_capacity_bytes(0),
        gl_heatmap_tex_id(0),
        gl_mask_tex_id(0),
        gl_overlay_tex_id(0),
        gl_combined_tex_id(0)
    {}

    void reset() {
        frame_id = 0;
        timestamp = 0.0;
        start_time = 0.0;
        end_time = 0.0;
        is_valid = true;
        peaks.clear();
        h_pred_score.clear();
        h_pred_label.clear();
        h_anomaly_map.clear();
        h_pred_mask.clear();
        
        // Reset target resolution
        target_width = 0;
        target_height = 0;
        
        // Keep allocation metadata: pooled tasks may safely reuse larger buffers.
    }

    bool ensureOriginalImageBuffer(size_t required_bytes) {
        if (required_bytes == 0) return false;

        if (!d_original_image || original_image_capacity_bytes < required_bytes) {
            d_original_image = make_device_buffer(required_bytes);
            original_image_capacity_bytes = required_bytes;
        }
        return true;
    }

    bool hasOutputBufferCapacity() const {
        size_t required_bytes = 0;
        return tryGetFrameBufferBytes(target_width, target_height, 4, required_bytes) &&
            d_final_heatmap && d_final_overlay && output_buffer_capacity_bytes >= required_bytes;
    }

    bool ensureOutputBuffers() {
        size_t required_bytes = 0;
        if (!tryGetFrameBufferBytes(target_width, target_height, 4, required_bytes)) return false;

        if (!d_final_heatmap || !d_final_overlay || output_buffer_capacity_bytes < required_bytes) {
            d_final_heatmap = make_device_buffer(required_bytes);
            d_final_overlay = make_device_buffer(required_bytes);
            output_buffer_capacity_bytes = required_bytes;
        }

        allocated_width = target_width;
        allocated_height = target_height;
        return true;
    }
};

using FrameTaskPtr = std::shared_ptr<FrameTask>;

} // namespace pipeline
