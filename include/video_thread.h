#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "config.h"
#include "pipeline.h"
#include "cuda_render.h" // [新增]

namespace VideoThread {
    class VideoCaptureThread {
    public:
        VideoCaptureThread(const Config& videoConfig);
        ~VideoCaptureThread();
        void stop();

        bool update(float threshold);
        std::vector<cv::Rect> getDefectRects() const { return defect_rects; }

        // [修改] 直接返回互操作的 Texture ID
        unsigned int getResultFrameTexture() const { return cuda_tex_frame.getTextureID(); }
        unsigned int getResultHeatmapTexture() const { return cuda_tex_heatmap.getTextureID(); }
        
        bool isOpened() const { return cap.isOpened(); }

    private:
        cv::VideoCapture cap;
        std::string source;
        Config config;
        pipeline::Pipeline pipeline;

        // [新增] 两个互操作纹理对象
        CudaInteropTexture cuda_tex_frame;   // 原图
        CudaInteropTexture cuda_tex_heatmap; // 热力图

        std::vector<cv::Rect> defect_rects;
    };
}