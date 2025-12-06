#include "video_thread.h"

namespace VideoThread {

    VideoCaptureThread::VideoCaptureThread(const Config& videoConfig)
        : source(videoConfig.videoSource), config(videoConfig), pipeline(config) {
        
        if(config.imgSource == " ") {
             cap.open(source);
             if (!cap.isOpened()) {
                 std::cerr << "Error: Could not open video source " << source << std::endl;
             }
        }
        // 如果是图片模式，逻辑稍有不同，这里为了简化演示，主要适配视频流
    }

    VideoCaptureThread::~VideoCaptureThread() {
        stop();
    }

    void VideoCaptureThread::stop() {
        if (cap.isOpened()) {
            cap.release();
        }
        // cv::destroyAllWindows(); // [删除] ImGui 不需要这个
    }

    // [核心重构] 处理单帧逻辑
    bool VideoCaptureThread::update() {
        cv::Mat frame;
        
        // 1. 读取数据
        if (config.imgSource != " ") {
            // 图片模式：每次读同一张图 (或者你可以做个标志位只读一次)
            frame = cv::imread(config.imgSource);
            if (frame.empty()) return false;
        } else {
            // 视频模式
            if (!cap.isOpened() || !cap.read(frame)) {
                // 视频播放结束，可以选择循环播放
                cap.set(cv::CAP_PROP_POS_FRAMES, 0); 
                if(!cap.read(frame)) return false; 
            }
        }

        // 2. 推理
        cv::Mat anomaly_map;
        pipeline.inference(frame, anomaly_map);

        // 3. 统一 Resize (为了 UI 显示整齐)
        // 注意：这里建议直接 resize 到 config.outputHeight，或者由 UI 决定缩放
        cv::resize(frame, frame, cv::Size(448, 448)); 

        // 4. 生成可视化结果
        cv::Mat heatmap, overlay;
        visualizeAnomalyMap(anomaly_map, heatmap, frame, overlay);

        // 5. 将结果存入类成员变量，供 UI 读取
        // ImGui 需要 RGB 格式，OpenCV 默认是 BGR，这里可以顺便转一下
        cv::cvtColor(frame, this->display_frame, cv::COLOR_BGR2RGB);
        cv::cvtColor(overlay, this->display_overlay, cv::COLOR_BGR2RGB);

        return true;
    }

    void VideoCaptureThread::visualizeAnomalyMap(const cv::Mat& anomaly_map, cv::Mat& heatmap,
                                                 const cv::Mat& original_frame, cv::Mat& overlay) {
        cv::Mat nomalized_map;
        if (anomaly_map.type() != CV_32FC1) {
            anomaly_map.convertTo(nomalized_map, CV_32FC1);
        } else {
            nomalized_map = anomaly_map.clone();
        }
        
        cv::multiply(nomalized_map, 255.0, nomalized_map);
        nomalized_map.convertTo(heatmap, CV_8UC1);

        cv::applyColorMap(heatmap, heatmap, cv::COLORMAP_JET);
        cv::resize(heatmap, heatmap, original_frame.size());
        cv::addWeighted(original_frame, 0.5, heatmap, 0.5, 0, overlay);
    }

} // namespace VideoThread