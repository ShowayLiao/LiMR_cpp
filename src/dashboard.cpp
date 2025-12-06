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
    // 智能指针会自动释放 backend，这里可以手动清理纹理
    if (tex_frame) glDeleteTextures(1, &tex_frame);
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

void Dashboard::UpdateTextures() {
    if (is_initialized && is_running && videoProcessor) {
        if (videoProcessor->update()) {
            update_texture_internal(videoProcessor->getResultFrame(), tex_frame);
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
    ImGui::PopStyleColor();

    if (tex_frame != 0 && tex_overlay != 0) {
        // 自适应布局计算
        float pad = 20.0f;
        float availW = width - pad * 3; // 减去间隙
        float targetW = availW / 2.0f;
        float targetH = targetW; // 1:1 比例

        if (targetH > height - 100) { // 高度限制
            targetH = height - 100;
            targetW = targetH;
        }

        // 居中
        float cursorX = (width - (targetW * 2 + pad)) / 2.0f;
        if (cursorX < 0) cursorX = 0;
        ImGui::SetCursorPos(ImVec2(cursorX, (height - targetH) / 2.0f));

        // 画图
        ImGui::BeginGroup();
        ImGui::Text("Original");
        ImGui::Image((void*)(intptr_t)tex_frame, ImVec2(targetW, targetH));
        ImGui::EndGroup();

        ImGui::SameLine(0, pad);

        ImGui::BeginGroup();
        ImGui::Text("Result");
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
}