#pragma once

// Include Config header file
#include "app_config.h"

// Include standard library headers
#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include other project headers
#include "cuda_render.h"
#include "inference_runner.h"

namespace VideoThread {
    class VideoCaptureThread {
    public:
        VideoCaptureThread(const AppConfig& videoConfig);
        ~VideoCaptureThread();
        void stop();

        bool update(float threshold);
        std::vector<cv::Rect> getDefectRects() const { return defect_rects; }

        // [Modified] Directly return interop Texture ID
        unsigned int getResultFrameTexture() const { return cuda_tex_frame.getTextureID(); }
        unsigned int getResultHeatmapTexture() const { return cuda_tex_heatmap.getTextureID(); }
        
        bool isOpened() const { return cap.isOpened(); }

    private:
        cv::VideoCapture cap;
        std::string source;
        AppConfig config;
        std::unique_ptr<InferenceRunner> runner;

        // [Added] Two interop texture objects
        CudaInteropTexture cuda_tex_frame;   // Original image
        CudaInteropTexture cuda_tex_heatmap; // Heatmap

        std::vector<cv::Rect> defect_rects;
    };
}