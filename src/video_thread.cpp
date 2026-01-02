#include "video_thread.h"
#include "app_config.h"
#include "inference_runner.h"

namespace VideoThread {

    VideoCaptureThread::VideoCaptureThread(const AppConfig& videoConfig)
        : source(videoConfig.videoSource), config(videoConfig) {
        // Initialize InferenceRunner
        runner = std::make_unique<InferenceRunner>(videoConfig.inferenceMode);
        runner->init(videoConfig.enginePathA, videoConfig.enginePathB, 
                    videoConfig.inputWidth, videoConfig.inputHeight,
                    videoConfig.outputWidth, videoConfig.outputHeight);
        if(config.imgSource == " ") {
             cap.open(source);
        }
        
        // [Added] Initialize CUDA interop textures
        cuda_tex_frame.init(config.inputWidth, config.inputHeight);
        cuda_tex_heatmap.init(config.outputWidth, config.outputHeight);
    }

    VideoCaptureThread::~VideoCaptureThread() { stop(); }
    void VideoCaptureThread::stop() { if (cap.isOpened()) cap.release(); }

    bool VideoCaptureThread::update(float threshold) {
        cv::Mat frame, output_anomaly;
        
        // 1. Read data
        if (config.imgSource != " ") {
            frame = cv::imread(config.imgSource);
            if (frame.empty()) return false;
        } else {
            if (!cap.isOpened() || !cap.read(frame)) {
                cap.set(cv::CAP_PROP_POS_FRAMES, 0); 
                if(!cap.read(frame)) return false; 
            }
        }

        // 2. Uniform Resize (Pipeline requires 448)
        // Even though the pipeline also does resize internally, we need this image for display
        cv::resize(frame, frame, cv::Size(config.inputWidth, config.inputWidth));

        // 3. Inference - using InferenceRunner
        cv::Mat heatmap = runner->infer(frame);

        // ==========================================
        // 4. Render heatmap
        // ==========================================
        // Upload CPU heatmap to texture
        size_t heatmap_size = heatmap.cols * heatmap.rows * 3 * sizeof(unsigned char);
        cuda_tex_heatmap.upload_and_render_frame(heatmap.data, heatmap_size);

        // ==========================================
        // 5. Render original image (keep as is)
        // ==========================================
        // Your inference still uses CPU Mat inputFrame, so upload CPU frame here
        size_t frame_size = frame.cols * frame.rows * 3 * sizeof(unsigned char);
        cuda_tex_frame.upload_and_render_frame(frame.data, frame_size);

        // ==========================================
        // 6. Defect box calculation
        // ==========================================
        // Calculate defect boxes using heatmap returned by InferenceRunner
        
        // Convert heatmap to grayscale
        cv::Mat gray_heatmap;
        cv::cvtColor(heatmap, gray_heatmap, cv::COLOR_BGR2GRAY);
        
        // Binarization processing
        cv::Mat binary_mask;
        double thresh_val = threshold * 255.0;
        cv::threshold(gray_heatmap, binary_mask, thresh_val, 255, cv::THRESH_BINARY);
        
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