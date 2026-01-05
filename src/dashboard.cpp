#include "dashboard.h"
#include "model_loader.h"
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
    // if (tex_overlay) glDeleteTextures(1, &tex_overlay);
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
            // [修改] 直接拿 GPU Texture ID
            tex_frame   = videoProcessor->getResultFrameTexture();
            tex_heatmap = videoProcessor->getResultHeatmapTexture();
            
            // Overlay 实际上就是 Frame，我们直接复用 ID 即可
            tex_overlay = tex_frame;
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
    
    // 推理模式选择
    ImGui::Combo("Inference Mode", &inference_mode_idx, inference_mode_items, 2);
    
    // Engine路径输入
    ImGui::InputText("Engine Path A", engine_path_a, 256);
    if (inference_mode_idx == 1) { // Dual Engine模式
        ImGui::InputText("Engine Path B", engine_path_b, 256);
    }
    
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
                
                // 根据选择的模式设置推理模式
                InferenceMode mode = (inference_mode_idx == 0) ? InferenceMode::SINGLE_ENGINE : InferenceMode::DUAL_ENGINE;
                
                // 创建模型加载器
                ModelLoader modelLoader("./.cache");
                
                try {
                    // 使用默认参数处理模型文件
                    int defaultWidth = 224;
                    int defaultHeight = 224;
                    int defaultBatchSize = 1;
                    
                    // 处理模型文件A
                    std::string processedEngineA = modelLoader.processModel(engine_path_a, 
                                                                          defaultWidth, 
                                                                          defaultHeight, 
                                                                          defaultBatchSize,
                                                                          type_str);
                    
                    // 处理模型文件B（如果是双引擎模式）
                    std::string processedEngineB = engine_path_b;
                    if (mode == InferenceMode::DUAL_ENGINE && strlen(engine_path_b) > 0) {
                        processedEngineB = modelLoader.processModel(engine_path_b, 
                                                                    defaultWidth, 
                                                                    defaultHeight, 
                                                                    defaultBatchSize,
                                                                    type_str);
                    }
                    
                    // 创建配置对象，使用处理后的模型路径
                    appConfig = std::unique_ptr<AppConfig>(new AppConfig(video_path, processedEngineA, processedEngineB, " ", type_str, mode));
                    
                    // 验证处理后的模型文件是否存在
                    if (!std::ifstream(processedEngineA).good()) {
                        throw ModelLoaderException("Processed engine A file not found: " + processedEngineA);
                    }
                    if (mode == InferenceMode::DUAL_ENGINE && !processedEngineB.empty() && !std::ifstream(processedEngineB).good()) {
                        throw ModelLoaderException("Processed engine B file not found: " + processedEngineB);
                    }

                    videoProcessor = std::unique_ptr<VideoThread::VideoCaptureThread>(new VideoThread::VideoCaptureThread(*appConfig));
                } catch (const ModelLoaderException& e) {
                    std::cerr << "[ERROR] Model processing failed: " << e.what() << std::endl;
                    ImGui::OpenPopup("Error");
                    return;
                } catch (const std::exception& e) {
                    std::cerr << "[ERROR] Initialization failed: " << e.what() << std::endl;
                    ImGui::OpenPopup("Error");
                    return;
                }
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
        
        // Apply & Reload 按钮
        ImGui::Spacing();
        if (ImGui::Button("Apply & Reload", ImVec2(-1, btnH))) {
            try {
                // 停止当前视频线程
                is_running = false;
                
                // 更新配置
                std::string type_str = precision_items[current_precision_idx];
                InferenceMode mode = (inference_mode_idx == 0) ? InferenceMode::SINGLE_ENGINE : InferenceMode::DUAL_ENGINE;
                
                // 创建模型加载器
                ModelLoader modelLoader("./.cache");
                
                try {
                    // 使用默认参数处理模型文件
                    int defaultWidth = 224;
                    int defaultHeight = 224;
                    int defaultBatchSize = 1;
                    
                    // 处理模型文件A
                    std::string processedEngineA = modelLoader.processModel(engine_path_a, 
                                                                          defaultWidth, 
                                                                          defaultHeight, 
                                                                          defaultBatchSize,
                                                                          type_str);
                    
                    // 处理模型文件B（如果是双引擎模式）
                    std::string processedEngineB = engine_path_b;
                    if (mode == InferenceMode::DUAL_ENGINE && strlen(engine_path_b) > 0) {
                        processedEngineB = modelLoader.processModel(engine_path_b, 
                                                                    defaultWidth, 
                                                                    defaultHeight, 
                                                                    defaultBatchSize,
                                                                    type_str);
                    }
                    
                    // 创建新的配置对象，使用处理后的模型路径
                    appConfig = std::unique_ptr<AppConfig>(new AppConfig(video_path, processedEngineA, processedEngineB, " ", type_str, mode));
                    
                    // 验证处理后的模型文件是否存在
                    if (!std::ifstream(processedEngineA).good()) {
                        throw ModelLoaderException("Processed engine A file not found: " + processedEngineA);
                    }
                    if (mode == InferenceMode::DUAL_ENGINE && !processedEngineB.empty() && !std::ifstream(processedEngineB).good()) {
                        throw ModelLoaderException("Processed engine B file not found: " + processedEngineB);
                    }
                    
                    // 重新初始化视频处理器
                    videoProcessor = std::unique_ptr<VideoThread::VideoCaptureThread>(new VideoThread::VideoCaptureThread(*appConfig));
                    if (videoProcessor->isOpened()) {
                        is_running = true;
                    }
                } catch (const ModelLoaderException& e) {
                    std::cerr << "[ERROR] Model processing failed: " << e.what() << std::endl;
                    ImGui::OpenPopup("Error");
                    return;
                } catch (const std::exception& e) {
                    std::cerr << "[ERROR] Initialization failed: " << e.what() << std::endl;
                    ImGui::OpenPopup("Error");
                    return;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error applying new configuration: " << e.what() << std::endl;
            }
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
        // ImGui::BeginGroup();
        // ImGui::Text("Defect Detection");
        // ImGui::Image((void*)(intptr_t)tex_overlay, ImVec2(targetW, targetH));
        // ImGui::EndGroup();

        // ImGui::SameLine(0, pad); 
        ImGui::BeginGroup();
        ImGui::Text("Defect Detection");

        // [步骤 1] 获取当前图片在屏幕上的起始绝对坐标
        ImVec2 p_min = ImGui::GetCursorScreenPos();

        // [步骤 2] 绘制底图
        ImGui::Image((void*)(intptr_t)tex_overlay, ImVec2(targetW, targetH));

        // [步骤 3] 获取画笔，绘制红框
        auto rects = videoProcessor->getDefectRects();
        
        if (!rects.empty() && appConfig) { // 确保 appConfig 存在
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            
            // 【核心修复：计算缩放比例】
            // 获取推理时使用的基准分辨率 (后端是基于这个分辨率算的坐标)
            // 我们从配置对象中读取这些值，确保通用性
            float baseW = (float)appConfig->inputWidth;  // 例如 448.0f
            float baseH = (float)appConfig->inputHeight; // 例如 448.0f

            // 防止除以零的保护措施
            if (baseW > 0 && baseH > 0) {
                 // 计算缩放因子： (当前显示的宽高 / 基准宽高)
                float scale_x = targetW / baseW;
                float scale_y = targetH / baseH;

                for (const auto& rect : rects) {
                    // 【核心修复：应用缩放】
                    // 屏幕绝对坐标 X = 图片起始X + (原始坐标X * 缩放因子X)
                    float x = p_min.x + rect.x * scale_x;
                    float y = p_min.y + rect.y * scale_y;
                    float w = rect.width * scale_x;
                    float h = rect.height * scale_y;

                    // 绘制矩形
                    draw_list->AddRect(
                        ImVec2(x, y),          // 左上角
                        ImVec2(x + w, y + h),  // 右下角
                        IM_COL32(255, 0, 0, 255), // 红色
                        0.0f, 0, 2.0f // 无圆角，线宽2.0
                    );
                }
            }
        }
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