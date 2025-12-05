#define _CRT_SECURE_NO_WARNINGS 

// ================= 系统配置宏 =================
#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00 
#endif
#pragma comment(lib, "ws2_32.lib")
#endif

// ================= 头文件 =================
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <memory> 
#include <vector>
#include <string>
#include <fstream>  // 新增：用于读取 index.html 文件
#include <sstream>  // 新增：用于将文件转为字符串

// 项目头文件
#include "video_thread.h"
#include "config.h"
#include "utils.h"
#include "shared_state.h"

// 第三方库
#include <opencv2/opencv.hpp>
#include "httplib.h" 

// ================= 全局变量 =================
cv::Mat g_webFrame;
std::mutex g_webMutex;
std::atomic<float> g_anomalyScore{0.0f};
std::atomic<float> g_fps{0.0f};
std::atomic<long> g_processTime{0};
std::atomic<bool> g_isInferencing{false};

std::unique_ptr<VideoThread::VideoCaptureThread> g_videoThreadPtr = nullptr;

// ================= 主程序 =================
void server_thread(int port) {
    httplib::Server svr;

    // [核心修改] 1. 首页：从文件读取 HTML，而不是使用硬编码的字符串
    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        std::ifstream file("index.html"); // 打开同级目录下的 index.html
        if (file) {
            std::stringstream buffer;
            buffer << file.rdbuf(); // 读取文件全部内容
            res.set_content(buffer.str(), "text/html");
        } else {
            // 如果找不到文件，返回简单的错误提示
            res.set_content("<h1>Error: index.html not found!</h1><p>请确保 index.html 文件在程序运行目录下。</p>", "text/html");
        }
    });

    // 2. 获取实时数据
    svr.Get("/api/stats", [](const httplib::Request&, httplib::Response& res) {
        char json[128];
        snprintf(json, sizeof(json), "{\"fps\": %.1f, \"score\": %.4f, \"time\": %ld}", 
                g_fps.load(), g_anomalyScore.load(), g_processTime.load());
        res.set_content(json, "application/json");
    });

    // 3. 启动/重载
    svr.Post("/api/start", [](const httplib::Request& req, httplib::Response& res) {
        std::cout << "[WebUI] Receive Start Command..." << std::endl;
        
        std::string videoSrc = req.get_param_value("videoSource");
        std::string stuPath = req.get_param_value("stuPath");
        std::string teaPath = req.get_param_value("teaPath");
        
        if (videoSrc.empty()) videoSrc = "../../input/blade.avi";
        if (stuPath.empty()) stuPath = "../../input/LiMR_student_16.engine";
        
        if (g_videoThreadPtr) {
            g_videoThreadPtr->stop();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            g_videoThreadPtr.reset();
        }

        Config DynamicConfig(videoSrc, stuPath, teaPath, " ", "F32");
        
        try {
            g_videoThreadPtr = std::make_unique<VideoThread::VideoCaptureThread>(DynamicConfig);
            g_videoThreadPtr->start();
            g_isInferencing = true;
            std::cout << "[WebUI] Inference Started with video: " << videoSrc << std::endl;
            res.set_content("OK", "text/plain");
        } catch (const std::exception& e) {
            std::cerr << "[Error] " << e.what() << std::endl;
            res.status = 500;
            res.set_content("Error starting thread", "text/plain");
        }
    });

    // 4. 停止
    svr.Get("/api/stop", [](const httplib::Request&, httplib::Response& res) {
        if (g_videoThreadPtr) {
            g_videoThreadPtr->stop();
            g_isInferencing = false;
        }
        res.set_content("Stopped", "text/plain");
    });

    // 5. 视频流
    svr.Get("/video", [](const httplib::Request&, httplib::Response& res) {
        std::cout << "[Video] Client connected!" << std::endl; // 调试 1：有链接进来吗？
        res.set_content_provider(
            "multipart/x-mixed-replace; boundary=frame",
            [](size_t offset, httplib::DataSink &sink) {
                while (true) {
                    // 如果推理没开，就休息
                    if (!g_isInferencing) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        continue;
                    }

                    std::vector<uchar> buf;
                    bool gotFrame = false;
                    cv::Mat smallFrame;

                    // ================= 核心修改：使用 try_lock =================
                    // 尝试上锁。如果 AI 线程正在写数据，try_lock 会失败返回 false。
                    // 此时我们直接跳过这一帧，绝不让 AI 线程等待！
                    std::unique_lock<std::mutex> lock(g_webMutex, std::try_to_lock);
                    
                    if (lock.owns_lock()) {
                        if (!g_webFrame.empty()) {
                            // 缩小图片尺寸！这是降低延迟的关键！
                            // 如果原图很大，imencode 会吃掉大量 CPU，导致推理变慢
                            if (g_webFrame.cols > 640) {
                                float scale = 640.0f / g_webFrame.cols;
                                cv::resize(g_webFrame, smallFrame, cv::Size(), scale, scale);
                            } else {
                                g_webFrame.copyTo(smallFrame);
                            }
                            gotFrame = true;
                        }
                    } else {
                        // 锁被 AI 占用中，Web 端主动退让
                        std::this_thread::sleep_for(std::chrono::milliseconds(5)); 
                        continue; 
                    }
                    // 锁在这里自动释放，AI 线程自由了

                    if (gotFrame && !smallFrame.empty()) {
                        // 编码：质量调低到 50，进一步省 CPU
                        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 50};
                        if (cv::imencode(".jpg", smallFrame, buf, params)) {
                            
                            // 调试 2：如果有这行日志，说明图片已经生成好了
                            // std::cout << "[Video] Sending frame size: " << buf.size() << std::endl; 

                            std::string head = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + 
                                               std::to_string(buf.size()) + "\r\n\r\n";
                            
                            // 发送数据
                            if (!sink.write(head.c_str(), head.size())) break; // 发送失败则断开
                            if (!sink.write(reinterpret_cast<const char*>(buf.data()), buf.size())) break;
                            if (!sink.write("\r\n", 2)) break;
                        }
                    } else {
                        // 如果没取到帧（可能是空的），打印一下警告
                        // std::cerr << "[Video] Empty frame!" << std::endl;
                    }
                    
                    // 控制网页刷新率 30FPS，给 CPU 喘息时间
                    std::this_thread::sleep_for(std::chrono::milliseconds(33)); 
                }
                std::cout << "[Video] Client disconnected." << std::endl;
                return true; 
            }
        );
    });

    std::cout << ">>> WebUI Ready at http://localhost:" << port << std::endl;
    svr.listen("0.0.0.0", port);
}

int main() {
    std::cout << cv::getBuildInformation() << std::endl;
    try {
        std::cout << cv::cuda::getCudaEnabledDeviceCount() << " CUDA devices available." << std::endl;
    } catch (...) {
        std::cout << "CUDA check failed or no CUDA device." << std::endl;
    }
    server_thread(8080);
    return 0;
}