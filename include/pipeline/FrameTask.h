#pragma once

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "../common/CudaMemory.hpp"

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
        
        // Reset buffer size records
        allocated_width = 0;
        allocated_height = 0;
    }

    void ensureOutputBuffers() {
        // Skip if target size not set
        if (target_width <= 0 || target_height <= 0) return;

        // Check if buffers need resizing
        if (!d_final_heatmap || allocated_width != target_width || allocated_height != target_height) {
            
            // 4-channel RGBA size
            size_t size = target_width * target_height * 4 * sizeof(uint8_t);

            // std::cout << "[FrameTask] Resizing output buffers to " << target_width << "x" << target_height << std::endl;
            
            d_final_heatmap = make_device_buffer(size);
            d_final_overlay = make_device_buffer(size);
            
            // Update allocated size records
            allocated_width = target_width;
            allocated_height = target_height;
        }
    }
};

using FrameTaskPtr = std::shared_ptr<FrameTask>;

} // namespace pipeline