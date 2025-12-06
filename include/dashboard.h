#pragma once
#include <memory>
#include <vector>
#include <string>
#include "imgui.h"
#include "video_thread.h"
#include "config.h"
#include "utils.h" // 包含 build_engine 等声明

class Dashboard {
public:
    Dashboard();
    ~Dashboard();

    // 每帧调用此函数进行绘制和逻辑处理
    void Render(int display_w, int display_h);

    // 辅助：处理纹理更新
    void UpdateTextures();

private:
    // --- 内部绘制方法 ---
    void SetupStyle();
    void DrawSidePanel(float width, float height);
    void DrawMainView(float start_x, float width, float height);
    
    // --- 核心业务对象 (智能指针管理) ---
    std::unique_ptr<Config> appConfig;
    std::unique_ptr<VideoThread::VideoCaptureThread> videoProcessor;

    // --- UI 状态变量 ---
    bool is_initialized = false;
    bool is_running = false;
    
    // 纹理句柄
    unsigned int tex_frame = 0;   // 原始图
    // [新增] 纯热力图纹理
    unsigned int tex_heatmap = 0; 
    unsigned int tex_overlay = 0; // 最终结果图

    // --- 输入缓存 (ImGui 需要 char*) ---
    char video_path[256] = "../../input/blade.avi";
    char stu_onnx_path[256] = "../../input/LiMR_student.onnx";
    char tea_onnx_path[256] = "../../input/LiMR_teacher.onnx";
    char stu_engine_path[256] = "../../input/LiMR_student_16.engine";
    char tea_engine_path[256] = "../../input/LiMR_teacher_16.engine";
    
    int current_precision_idx = 0;
    const char* precision_items[2] = { "F32", "F16 "};

    float defect_threshold = 0.5f;
};