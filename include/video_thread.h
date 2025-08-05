# ifndef VIDEO_THREAD_H
# define VIDEO_THREAD_H

# include <opencv2/opencv.hpp>
# include <iostream>
# include "config.h"
# include "pipeline.h"

namespace VideoThread {
    class VideoCaptureThread {
    public:
        VideoCaptureThread(const Config& videoConfig);
        ~VideoCaptureThread();

        void start();
        void stop();
        bool isRunning() const;
        void visualizeAnomalyMap(const cv::Mat& anomaly_map, cv::Mat& heatmap,const cv::Mat& frame,cv::Mat & overlay);
        


    private:
        void run();
        void run_img();
        cv::VideoCapture cap;
        std::string source;
        bool running;
        Config config;
        pipeline::Pipeline pipeline;
        
    };


}



# endif