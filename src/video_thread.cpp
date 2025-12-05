# include "video_thread.h"
# include "config.h"
# include "pipeline.h"
#include "shared_state.h"

namespace VideoThread {
    //-----------define class VideoCaptureThread---------------
    VideoCaptureThread::VideoCaptureThread(const Config& videoConfig)
        : source(videoConfig.videoSource), running(false),config(videoConfig),pipeline(config) {
        cap.open(source);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video source " << source << std::endl;
        }

    }

    // Destructor to release resources
    VideoCaptureThread::~VideoCaptureThread() {
        stop();
    }

    void VideoCaptureThread::stop() {
        running = false;
        cap.release();
        if (cap.isOpened()) {
            std::cerr << "Error: Could not release video source " << source << std::endl;
        }
        cv::destroyAllWindows();
    }

    // Start the video capture thread
    void VideoCaptureThread::start() {
        if (!running) {
            running = true;
            if(config.imgSource == " ") {
                // std::thread(&VideoCaptureThread::run, this).detach(); // Start video capture thread
                run();
            } else {
                // std::thread(&VideoCaptureThread::run_img, this).detach(); // Start image capture thread
                run_img();
            }
        }
    }

    // void VideoCaptureThread::run(){
    //     cv::Mat frame;
    //     cv::Mat anomaly_map;
    //     cv::Mat heatmap;
    //     cv::Mat overlay;
    //     while (running) {
    //         if (!cap.read(frame)) {
    //             std::cerr << "Error: Could not read frame from video source " << source << std::endl;
    //             break;
    //         }
    //         // std::cout<<frame.channels()<<" channels"<<std::endl;

    //         pipeline.inference(frame, anomaly_map);

    //         cv::resize(frame, frame, cv::Size(448, 448));

    //         visualizeAnomalyMap(anomaly_map, heatmap,frame,overlay);

    //         cv::imshow("Video Frame", frame);
    //         cv::imshow("Anomaly Map", overlay);
            
    //         if (cv::waitKey(1) >= 0) { // Wait for 30 ms or until a key is pressed
    //             break;
    //         }
    //     }
    //     stop();

    // }
    void VideoCaptureThread::run(){
        cv::Mat frame, anomaly_map, heatmap, overlay;
        
        // FPS 计算相关变量
        auto lastTime = std::chrono::high_resolution_clock::now();
        int frameCount = 0;

        while (running) {
            auto startInferTime = std::chrono::high_resolution_clock::now(); // 计时开始

            if (!cap.read(frame)) {
                // 循环播放逻辑
                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                continue;
            }

            pipeline.inference(frame, anomaly_map);
            
            // 计算耗时
            auto endInferTime = std::chrono::high_resolution_clock::now();
            long duration = std::chrono::duration_cast<std::chrono::milliseconds>(endInferTime - startInferTime).count();
            g_processTime = duration;

            // 更新异常分数 (假设 anomaly_map 的最大值代表异常程度，具体看你业务逻辑)
            double minVal, maxVal;
            cv::minMaxLoc(anomaly_map, &minVal, &maxVal);
            g_anomalyScore = (float)maxVal;

            cv::resize(frame, frame, cv::Size(448, 448));
            visualizeAnomalyMap(anomaly_map, heatmap, frame, overlay);

            // === WebUI 更新逻辑 ===
            {
                cv::Mat combined;
                cv::hconcat(frame, overlay, combined); 
                std::lock_guard<std::mutex> lock(g_webMutex);
                g_webFrame = combined.clone();
            }

            // === ❌ 移除或注释掉本地弹窗 ===
            // cv::imshow("Video Frame", frame);
            // cv::imshow("Anomaly Map", overlay);
            // cv::waitKey(1); 
            
            // FPS 计算
            frameCount++;
            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = now - lastTime;
            if (diff.count() >= 1.0) {
                g_fps = frameCount / diff.count();
                frameCount = 0;
                lastTime = now;
            }
            
            // 简单的线程休眠防止跑飞
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        stop();
    }

    void VideoCaptureThread::run_img(){
        cv::Mat frame;
        cv::Mat anomaly_map;
        cv::Mat heatmap;
        cv::Mat overlay;

        frame = cv::imread(config.imgSource);

        pipeline.inference(frame, anomaly_map);

        cv::resize(frame, frame, cv::Size(config.outputHeight, config.outputHeight));

        visualizeAnomalyMap(anomaly_map, heatmap,frame,overlay);


        // cv::imshow("Video Frame", frame);
        // cv::imshow("Anomaly Map", overlay);

        cv::waitKey(0); // Wait indefinitely until a key is pressed
        
        stop();

    }



    bool VideoCaptureThread::isRunning() const {
        return running;
    }

    void VideoCaptureThread::visualizeAnomalyMap(const cv::Mat& anomaly_map, cv::Mat& heatmap,
                                                 const cv::Mat& original_frame,cv::Mat & overlay) {
        // Normalize the anomaly map to [0, 255] for visualization
        // cv::Mat normalized_map;
        // cv::normalize(anomaly_map, normalized_map, 0, 255, cv::NORM_MINMAX);
        // Clip values to [0, 1]

        cv::Mat nomalized_map;

        if (anomaly_map.type() != CV_32FC1) {
            anomaly_map.convertTo(nomalized_map, CV_32FC1); // Convert to float and normalize
        }
        else nomalized_map = anomaly_map.clone();
        
        cv::multiply(nomalized_map, 255.0, nomalized_map); // Scale to [0, 255]
        nomalized_map.convertTo(heatmap, CV_8UC1);

        // Apply a colormap for better visualization
        cv::applyColorMap(heatmap, heatmap, cv::COLORMAP_JET);
        cv::resize(heatmap, heatmap, original_frame.size()); // Resize heatmap to match the original frame size
        cv::addWeighted(original_frame, 0.5, heatmap, 0.5, 0,overlay); // Blend the heatmap with the original frame
    }



} // namespace VideoThread