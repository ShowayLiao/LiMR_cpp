#ifndef VIDEO_THREAD_H
#define VIDEO_THREAD_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include "config.h"
#include "pipeline.h"

namespace VideoThread {
    class VideoCaptureThread {
    public:
        VideoCaptureThread(const Config& videoConfig);
        ~VideoCaptureThread();

        // [修改] 不再需要 start() 开启线程，而是由外部控制
        // void start(); 
        void stop(); // 释放资源

        // [新增] 处理一帧数据的核心函数 (供 main 循环调用)
        // 返回 true 表示成功读取并处理了一帧
        bool update(); 

        // [新增] 获取用于 UI 显示的图像
        cv::Mat getResultFrame() const { return display_frame; }
        cv::Mat getResultOverlay() const { return display_overlay; }
        
        // [新增] 用于控制参数（如果需要 UI 调整阈值等）
        // void setThreshold(float t);

        bool isOpened() const { return cap.isOpened(); }

    private:
        // 内部辅助函数
        void visualizeAnomalyMap(const cv::Mat& anomaly_map, cv::Mat& heatmap, const cv::Mat& frame, cv::Mat& overlay);

        cv::VideoCapture cap;
        std::string source;
        Config config;
        pipeline::Pipeline pipeline;

        // [新增] 缓存当前的显示结果，供 ImGui 读取
        cv::Mat current_frame;
        cv::Mat display_frame;   // 用于显示的原始帧 (resize后)
        cv::Mat display_overlay; // 用于显示的异常热力图
    };
}
#endif