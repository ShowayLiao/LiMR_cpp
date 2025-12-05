#pragma once
#include <opencv2/opencv.hpp>
#include <mutex>
#include <atomic>

// 图像共享
extern cv::Mat g_webFrame;
extern std::mutex g_webMutex;

// 实时指标 (用于在网页上显示)
extern std::atomic<float> g_anomalyScore; // 当前异常分数
extern std::atomic<float> g_fps;          // 当前 FPS
extern std::atomic<long> g_processTime;   // 处理耗时 (ms)

// 控制标志
extern std::atomic<bool> g_isInferencing; // 是否正在推理中