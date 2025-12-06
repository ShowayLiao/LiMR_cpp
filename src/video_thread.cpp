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
    bool VideoCaptureThread::update(float threshold) {
        cv::Mat frame;
        
        // 1. 读取帧 (逻辑不变)
        if (config.imgSource != " ") {
            frame = cv::imread(config.imgSource);
            if (frame.empty()) return false;
        } else {
            if (!cap.isOpened() || !cap.read(frame)) {
                cap.set(cv::CAP_PROP_POS_FRAMES, 0); 
                if(!cap.read(frame)) return false; 
            }
        }

        // 2. 推理
        cv::Mat anomaly_map;
        pipeline.inference(frame, anomaly_map);

        // 3. Resize
        cv::resize(frame, frame, cv::Size(448, 448)); 

        // 注意这里传入的是类的成员变量用于接收结果
        cv::Mat temp_heatmap, temp_overlay;
        visualizeAnomalyMap(anomaly_map, temp_heatmap, frame, temp_overlay, threshold);

        // 5. [修改] 将三张图都转换为 RGB 供 ImGui 显示
        cv::cvtColor(frame, this->display_frame, cv::COLOR_BGR2RGB);
        // 新增：转换热力图
        cv::cvtColor(temp_heatmap, this->display_heatmap, cv::COLOR_BGR2RGB);
        cv::cvtColor(temp_overlay, this->display_overlay, cv::COLOR_BGR2RGB);

        return true;
    }

    // [核心修改] 可视化逻辑重构
    void VideoCaptureThread::visualizeAnomalyMap(const cv::Mat& anomaly_map, 
                                                 cv::Mat& pure_heatmap, // 输出1
                                                 const cv::Mat& original_frame, 
                                                 cv::Mat& final_overlay, // 输出2
                                                 float threshold) {
        // --- A. 准备基础数据 ---
        cv::Mat norm_map_8u, norm_map_float;
        if (anomaly_map.type() != CV_32FC1) anomaly_map.convertTo(norm_map_float, CV_32FC1);
        else norm_map_float = anomaly_map.clone();
        
        cv::multiply(norm_map_float, 255.0, norm_map_float);
        norm_map_float.convertTo(norm_map_8u, CV_8UC1);

        // --- B. 生成【纯热力图】 ---
        cv::applyColorMap(norm_map_8u, pure_heatmap, cv::COLORMAP_JET);
        cv::resize(pure_heatmap, pure_heatmap, original_frame.size());

        // --- C. 生成【框选缺陷在原图上】 ---
        // 1. 复制一份原图作为底图
        final_overlay = original_frame.clone();

        // // 2. 计算二值化掩膜
        cv::Mat binary_mask;
        double thresh_val = threshold * 255.0;
        cv::threshold(norm_map_8u, binary_mask, thresh_val, 255, cv::THRESH_BINARY);
        cv::resize(binary_mask, binary_mask, original_frame.size(), 0, 0, cv::INTER_NEAREST);

        // 3. 查找并绘制轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            if (cv::contourArea(contour) < 50) continue; // 过滤噪点
            cv::Rect rect = cv::boundingRect(contour);
            // 在原图底图上画红框
            cv::rectangle(final_overlay, rect, cv::Scalar(0, 0, 255), 2);
        }

        // (可选：如果你希望最终结果图里也带一点点半透明热力，取消下面这行的注释)
        // cv::addWeighted(final_overlay, 0.7, pure_heatmap, 0.3, 0, final_overlay);
    }

} // namespace VideoThread