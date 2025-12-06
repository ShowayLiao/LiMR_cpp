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
        bool update(float threshold); 

        // [新增] 获取用于 UI 显示的图像
        // [新增] 获取纯热力图的接口
        cv::Mat getResultHeatmap() const { return display_heatmap; }
        cv::Mat getResultFrame() const { return display_frame; }
        cv::Mat getResultOverlay() const { return display_overlay; }
        
        
        // [新增] 用于控制参数（如果需要 UI 调整阈值等）
        // void setThreshold(float t);

        bool isOpened() const { return cap.isOpened(); }

    private:
        // 内部辅助函数
        // [修改] 可视化函数的签名，增加一个输出参数 pure_heatmap
        void visualizeAnomalyMap(const cv::Mat& anomaly_map, 
                                 cv::Mat& pure_heatmap,  // 输出：纯热力图
                                 const cv::Mat& frame, 
                                 cv::Mat& final_overlay, // 输出：最终叠加图
                                 float threshold);

        cv::VideoCapture cap;
        std::string source;
        Config config;
        pipeline::Pipeline pipeline;

        // [新增] 缓存当前的显示结果，供 ImGui 读取
        cv::Mat current_frame;
        cv::Mat display_frame;   // 用于显示的原始帧 (resize后)
        // [新增] 用于显示的纯热力图 (RGB)
        cv::Mat display_heatmap; 
        cv::Mat display_overlay; // 最终结果图 (RGB)
    };
}
#endif