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
    bool is_valid = true;

    cv::Mat original_image;

    DeviceBuffer d_input = nullptr;

    DeviceBuffer d_pred_score = nullptr;

    DeviceBuffer d_pred_label = nullptr;

    DeviceBuffer d_anomaly_map = nullptr;

    DeviceBuffer d_pred_mask = nullptr;

    DeviceBuffer d_dynamic_mask = nullptr;
    DeviceBuffer d_heatmap_gpu = nullptr;

    std::vector<float> h_pred_score;
    std::vector<uint8_t> h_pred_label;
    std::vector<float> h_anomaly_map;
    std::vector<uint8_t> h_pred_mask;

    unsigned int gl_heatmap_tex_id = 0;
    unsigned int gl_mask_tex_id = 0;

    cv::Mat heatmap_vis;
    std::vector<AnomalyPeak> peaks;

    void reset() {
        frame_id = 0;
        timestamp = 0.0;
        start_time = 0.0;
        end_time = 0.0;
        is_valid = true;
        peaks.clear();
    }
};

using FrameTaskPtr = std::shared_ptr<FrameTask>;

} // namespace pipeline
