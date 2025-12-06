#include "video_thread.h"

namespace VideoThread {

    VideoCaptureThread::VideoCaptureThread(const Config& videoConfig)
        : source(videoConfig.videoSource), config(videoConfig), pipeline(config) {
        if(config.imgSource == " ") {
             cap.open(source);
        }
        
        // [新增] 初始化 CUDA 互操作纹理
        // 假设宽高是 448x448，如果不是，请用 config.inputWidth
        cuda_tex_frame.init(config.inputWidth, config.inputWidth);
        cuda_tex_heatmap.init(config.inputWidth, config.inputWidth);
    }

    VideoCaptureThread::~VideoCaptureThread() { stop(); }
    void VideoCaptureThread::stop() { if (cap.isOpened()) cap.release(); }

    bool VideoCaptureThread::update(float threshold) {
        cv::Mat frame, output_anomaly;
        
        // 1. 读取数据
        if (config.imgSource != " ") {
            frame = cv::imread(config.imgSource);
            if (frame.empty()) return false;
        } else {
            if (!cap.isOpened() || !cap.read(frame)) {
                cap.set(cv::CAP_PROP_POS_FRAMES, 0); 
                if(!cap.read(frame)) return false; 
            }
        }

        // 2. 统一 Resize (Pipeline 需要 448)
        // 即使 pipeline 内部也会 resize，但我们需要这张图来显示
        cv::resize(frame, frame, cv::Size(config.inputWidth, config.inputWidth));

        // 3. 推理 (output_anomaly 是 float 类型)
        pipeline.inference(frame, output_anomaly);

        // ==========================================
        // 4. [优化核心] 渲染热力图 (Float -> GPU -> Texture)
        // ==========================================
        // output_anomaly 是 32F 单通道
        size_t heatmap_size = output_anomaly.cols * output_anomaly.rows * sizeof(float);
        cuda_tex_heatmap.upload_and_render_heatmap((float*)output_anomaly.data, heatmap_size);

        // ==========================================
        // 5. [优化核心] 渲染原图 (BGR -> GPU -> RGB Texture)
        // ==========================================
        // frame 是 BGR 8U 3通道
        size_t frame_size = frame.cols * frame.rows * 3 * sizeof(unsigned char);
        cuda_tex_frame.upload_and_render_frame(frame.data, frame_size);

        // ==========================================
        // 6. 计算缺陷框 (依然在 CPU 做，这部分开销极小)
        // ==========================================
        // 为了 findContours，我们需要一个小的二值图。这个操作很快，可以保留在 CPU
        cv::Mat binary_mask;
        cv::Mat norm_map_8u;
        
        // 简单的 float->8u 转换，用于 CPU 轮廓计算
        // 注意：这里不需要 applyColorMap 了，只需要二值化
        output_anomaly.convertTo(norm_map_8u, CV_8UC1, 255.0); 
        
        double thresh_val = threshold * 255.0;
        cv::threshold(norm_map_8u, binary_mask, thresh_val, 255, cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        this->defect_rects.clear();
        for (const auto& contour : contours) {
            if (cv::contourArea(contour) < 50) continue;
            this->defect_rects.push_back(cv::boundingRect(contour));
        }
        
        // 注意：这里我们不再生成 display_overlay 图像了
        // 因为前端会复用 tex_frame 并把红框画在上面 (上一条优化)

        return true;
    }
}