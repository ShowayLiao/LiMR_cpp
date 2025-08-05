# include "video_thread.h"
# include "config.h"
# include "pipeline.h"


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

    void VideoCaptureThread::run(){
        cv::Mat frame;
        cv::Mat anomaly_map;
        cv::Mat heatmap;
        cv::Mat overlay;
        while (running) {
            if (!cap.read(frame)) {
                std::cerr << "Error: Could not read frame from video source " << source << std::endl;
                break;
            }
            // std::cout<<frame.channels()<<" channels"<<std::endl;

            pipeline.inference(frame, anomaly_map);

            cv::resize(frame, frame, cv::Size(448, 448));

            visualizeAnomalyMap(anomaly_map, heatmap,frame,overlay);

            cv::imshow("Video Frame", frame);
            cv::imshow("Anomaly Map", overlay);
            
            if (cv::waitKey(1) >= 0) { // Wait for 30 ms or until a key is pressed
                break;
            }
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


        cv::imshow("Video Frame", frame);
        cv::imshow("Anomaly Map", overlay);

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