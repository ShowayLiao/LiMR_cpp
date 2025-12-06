#include "dashboard.h"
#include <iostream>
#include <fstream>
#include <GLFW/glfw3.h> // 如果需要 OpenGL 函数

// 辅助函数：更新纹理 (你可以放在 utils.h 里，这里为了方便直接写)
static void update_texture_internal(const cv::Mat& mat, unsigned int& texture_id) {
    if (mat.empty()) return;
    if (texture_id == 0) glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mat.cols, mat.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, mat.data);
}

Dashboard::Dashboard() {
    SetupStyle(); // 初始化时自动应用样式
}

Dashboard::~Dashboard() {
    if (tex_frame) glDeleteTextures(1, &tex_frame);
    // [新增]
    if (tex_heatmap) glDeleteTextures(1, &tex_heatmap);
    if (tex_overlay) glDeleteTextures(1, &tex_overlay);
}

void Dashboard::SetupStyle() {
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 8.0f;
    style.FrameRounding = 6.0f;
    style.WindowPadding = ImVec2(15, 15);
    style.ItemSpacing = ImVec2(10, 10);
    
    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg] = ImVec4(0.12f, 0.12f, 0.14f, 1.00f);
    colors[ImGuiCol_Button]   = ImVec4(0.22f, 0.44f, 0.75f, 1.00f);
    colors[ImGuiCol_Header]   = ImVec4(0.20f, 0.22f, 0.25f, 1.00f);
    colors[ImGuiCol_Text]     = ImVec4(0.90f, 0.90f, 0.92f, 1.00f);
}

// 2. UpdateTextures 中增加热力图的上传
void Dashboard::UpdateTextures() {
    if (is_initialized && is_running && videoProcessor) {
        if (videoProcessor->update(defect_threshold)) {
            update_texture_internal(videoProcessor->getResultFrame(), tex_frame);
            // [新增] 上传热力图
            update_texture_internal(videoProcessor->getResultHeatmap(), tex_heatmap);
            update_texture_internal(videoProcessor->getResultOverlay(), tex_overlay);
        }
    }
}

void Dashboard::Render(int display_w, int display_h) {
    // 更新后端逻辑
    UpdateTextures();

    float sideBarWidth = 350.0f;

    // 绘制两部分
    DrawSidePanel(sideBarWidth, (float)display_h);
    DrawMainView(sideBarWidth, (float)display_w - sideBarWidth, (float)display_h);
}

void Dashboard::DrawSidePanel(float width, float height) {
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(width, height));
    
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
    ImGui::Begin("ControlPanel", nullptr, flags);

    ImGui::TextColored(ImVec4(0.22f, 0.64f, 1.0f, 1.0f), "LiMR DETECTOR");
    ImGui::Separator();
    ImGui::Spacing();

    // 状态
    ImGui::Text("STATUS: "); ImGui::SameLine();
    if (is_initialized) ImGui::TextColored(ImVec4(0, 1, 0, 1), "READY");
    else ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "CONFIGURING");
    ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    ImGui::Separator();

    // 配置区
    if (is_initialized) ImGui::BeginDisabled();
    ImGui::Text("Configuration");
    ImGui::InputText("Video", video_path, 256);
    ImGui::InputText("Stu ONNX", stu_onnx_path, 256);
    ImGui::InputText("Tea ONNX", tea_onnx_path, 256);
    ImGui::InputText("Stu Engine", stu_engine_path, 256);
    ImGui::InputText("Tea Engine", tea_engine_path, 256);
    // ... 其他输入框 ...
    ImGui::Combo("Precision", &current_precision_idx, precision_items, 2);

    if (is_initialized) ImGui::EndDisabled();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Inference Parameters");
    
    // 滑块：范围 0.0 到 1.0
    ImGui::SliderFloat("Threshold", &defect_threshold, 0.0f, 1.0f, "Conf: %.2f");
    // 添加一句解释
    if (ImGui::IsItemHovered()) 
        ImGui::SetTooltip("Adjust sensitivity for defect contours");

    ImGui::Spacing();
    ImGui::Separator();

    // ImGui::Spacing();
    
    // 按钮逻辑
    float btnH = 45.0f;
    if (!is_initialized) {
        if (ImGui::Button("INITIALIZE SYSTEM", ImVec2(-1, btnH))) {
            try {
                // 这里调用你的 backend 初始化逻辑
                std::string type_str = precision_items[current_precision_idx];
                appConfig = std::unique_ptr<Config>(new Config(video_path, stu_engine_path, tea_engine_path, " ", type_str));
                
                // 检查 engine (简化版，实际可用你的 build_engine 函数)
                if (!std::ifstream(stu_engine_path).good()) {
                    build_engine(stu_onnx_path, stu_engine_path, appConfig->batchSize, appConfig->inputWidth, appConfig->inputHeight, appConfig->outputWidth, appConfig->outputHeight);
                }
                if (!std::ifstream(tea_engine_path).good()) {
                     build_engine(tea_onnx_path, tea_engine_path, appConfig->batchSize, appConfig->inputWidth, appConfig->inputHeight, appConfig->outputWidth, appConfig->outputHeight);
                }

                videoProcessor = std::unique_ptr<VideoThread::VideoCaptureThread>(new VideoThread::VideoCaptureThread(*appConfig));
                if (videoProcessor->isOpened()) {
                    is_initialized = true;
                    is_running = true;
                }
            } catch (const std::exception& e) {
                std::cerr << e.what() << std::endl;
            }
        }
    } else {
        if (is_running) {
            if (ImGui::Button("PAUSE", ImVec2(width * 0.45f, btnH))) is_running = false;
        } else {
            if (ImGui::Button("RESUME", ImVec2(width * 0.45f, btnH))) is_running = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("RESET", ImVec2(width * 0.45f, btnH))) {
            is_running = false;
            is_initialized = false;
            videoProcessor.reset();
            appConfig.reset();
        }
    }

    ImGui::End();
}

void Dashboard::DrawMainView(float start_x, float width, float height) {
    ImGui::SetNextWindowPos(ImVec2(start_x, 0));
    ImGui::SetNextWindowSize(ImVec2(width, height));
    
    // 背景设黑
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.08f, 0.08f, 1.0f));
    ImGui::Begin("ViewPanel", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
    // ImGui::PopStyleColor();

    // ImGui::Begin("ViewPanel", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
    // ImGui::PopStyleColor();

    // [修改判断条件] 检查三个纹理是否都准备好了
    if (tex_frame != 0 && tex_overlay != 0 && tex_heatmap != 0) {
        // --- 自适应三栏布局计算 ---
        float pad = 15.0f; // 图片之间的间距
        // 总宽度减去左、中1、中2、右四个间隙
        float availW = width - pad * 4; 
        // 分成三份
        float targetW = availW / 3.0f;
        float targetH = targetW; // 假设 1:1 比例

        // 高度限制检查
        if (targetH > height - 80) { 
            targetH = height - 80;
            targetW = targetH;
        }

        // 整体居中计算
        float totalContentWidth = targetW * 3 + pad * 2;
        float cursorX = (width - totalContentWidth) / 2.0f;
        float cursorY = (height - targetH) / 2.0f;
        if (cursorX < pad) cursorX = pad;
        if (cursorY < pad) cursorY = pad;

        ImGui::SetCursorPos(ImVec2(cursorX, cursorY));

        // --- 绘制三张图 ---
        
        // 1. 原始图
        ImGui::BeginGroup();
        ImGui::Text("Original Input");
        ImGui::Image((void*)(intptr_t)tex_frame, ImVec2(targetW, targetH));
        ImGui::EndGroup();

        ImGui::SameLine(0, pad);

        // 2. [新增] 纯热力图
        ImGui::BeginGroup();
        ImGui::Text("Heatmap View");
        ImGui::Image((void*)(intptr_t)tex_heatmap, ImVec2(targetW, targetH));
        ImGui::EndGroup();

        ImGui::SameLine(0, pad);

        // 3. 最终结果图（红框在原图上）
        ImGui::BeginGroup();
        ImGui::Text("Defect Detection");
        ImGui::Image((void*)(intptr_t)tex_overlay, ImVec2(targetW, targetH));
        ImGui::EndGroup();

    } else {
        // 显示等待文字
        const char* txt = "WAITING FOR SIGNAL...";
        ImVec2 txtSize = ImGui::CalcTextSize(txt);
        ImGui::SetCursorPos(ImVec2((width - txtSize.x) / 2, (height - txtSize.y) / 2));
        ImGui::TextColored(ImVec4(0.5, 0.5, 0.5, 1), txt);
    }

    ImGui::End();
    ImGui::PopStyleColor();
}