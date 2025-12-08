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

        // 3. [修改] 推理
        float* d_anomaly_map_gpu = nullptr; // 用于接收 GPU 指针
        pipeline.inference(frame, d_anomaly_map_gpu);

        // ==========================================
        // 4. 渲染热力图 (全 GPU 链路)
        // ==========================================
        // d_anomaly_map_gpu 现在已经在显存里了
        // CudaInteropTexture 现在的 upload_... 接收 GPU 指针并执行 D2D 拷贝
        size_t heatmap_size = config.inputWidth * config.inputWidth * sizeof(float);
        cuda_tex_heatmap.upload_and_render_heatmap(d_anomaly_map_gpu, heatmap_size);

        // ==========================================
        // 5. 渲染原图 (暂维持原状)
        // ==========================================
        // 你的 inference 里还是用的 cpu Mat inputFrame，所以这里还是上传 CPU frame
        size_t frame_size = frame.cols * frame.rows * 3 * sizeof(unsigned char);
        cuda_tex_frame.upload_and_render_frame(frame.data, frame_size);

        // ==========================================
        // 6. 缺陷框计算 (需要下载一小部分数据回 CPU)
        // ==========================================
        // 因为 findContours 必须在 CPU 跑，我们需要把热力图下载回来一小部分做二值化
        // 或者，更聪明的方法：写一个 CUDA Kernel 做二值化，只下载 8-bit 的 mask (体积是 float 的 1/4)
        // 这里简单演示直接下载 float 的情况：
        
        static cv::Mat cpu_anomaly_map(config.inputWidth, config.inputWidth, CV_32FC1);
        cudaMemcpy(cpu_anomaly_map.data, d_anomaly_map_gpu, heatmap_size, cudaMemcpyDeviceToHost);
        
        // 转换到 8U 做 findContours
        cv::Mat norm_map_8u, binary_mask;
        cpu_anomaly_map.convertTo(norm_map_8u, CV_8UC1, 255.0);
        
        double thresh_val = threshold * 255.0;
        cv::threshold(norm_map_8u, binary_mask, thresh_val, 255, cv::THRESH_BINARY);
        
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        this->defect_rects.clear();
        for (const auto& contour : contours) {
            if (cv::contourArea(contour) < 50) continue;
            this->defect_rects.push_back(cv::boundingRect(contour));
        }

        return true;
    }
}