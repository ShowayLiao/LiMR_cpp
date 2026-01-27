#include <GLFW/glfw3.h> // If OpenGL functions are needed
#include "dashboard.h"
#include "model_loader.h"
#include <iostream>
#include <fstream>

// Helper function: Update texture (you can put this in utils.h, but for convenience it's written here directly)
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
    SetupStyle(); // Apply style automatically during initialization
    resources_initialized = false;
}

void Dashboard::InitResources() {
    if (!resources_initialized) {
        // Generate texture IDs
        glGenTextures(1, &tex_frame);
        glBindTexture(GL_TEXTURE_2D, tex_frame);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 256, 256, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        
        glGenTextures(1, &tex_heatmap);
        glBindTexture(GL_TEXTURE_2D, tex_heatmap);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        // Key point: format is changed to GL_RGBA here
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 256, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);
        
        // Register resource (keep unchanged)
        cudaGraphicsGLRegisterImage(&cuda_res_heatmap, tex_heatmap, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
        
        // [Added] Register as CUDA resource
        cudaGraphicsGLRegisterImage(&cuda_res_heatmap, tex_heatmap, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
        
        glGenTextures(1, &tex_overlay);
        glBindTexture(GL_TEXTURE_2D, tex_overlay);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 256, 256, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        
        // Print texture IDs to ensure they are not 0
        // std::cout << "[Dashboard] Texture IDs generated: " << std::endl;
        // std::cout << "  - tex_frame: " << tex_frame << std::endl;
        // std::cout << "  - tex_heatmap: " << tex_heatmap << std::endl;
        // std::cout << "  - tex_overlay: " << tex_overlay << std::endl;
        
        // Initialize with 256x256 black background texture
        // Create black background data
        std::vector<unsigned char> black_background(256 * 256 * 3, 0);
        
        // Update black background to textures
        glBindTexture(GL_TEXTURE_2D, tex_frame);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 256, 256, 0, GL_RGB, GL_UNSIGNED_BYTE, black_background.data());
        
        glBindTexture(GL_TEXTURE_2D, tex_heatmap);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 256, 256, 0, GL_RGB, GL_UNSIGNED_BYTE, black_background.data());
        
        glBindTexture(GL_TEXTURE_2D, tex_overlay);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 256, 256, 0, GL_RGB, GL_UNSIGNED_BYTE, black_background.data());
        
        std::cout << "[Dashboard] Initialized textures with black background" << std::endl;
        
        resources_initialized = true;
        std::cout << "[Dashboard] Resources initialized" << std::endl;
    }
}

Dashboard::~Dashboard() {
    if (cuda_res_heatmap) {
        cudaGraphicsUnregisterResource(cuda_res_heatmap);
        cuda_res_heatmap = nullptr;
    }
    if (tex_frame) glDeleteTextures(1, &tex_frame);
    // [新增]
    if (tex_heatmap) glDeleteTextures(1, &tex_heatmap);
    if (tex_overlay) glDeleteTextures(1, &tex_overlay);
}

void Dashboard::UpdateData(const pipeline::FrameTaskPtr& task) {
    if (!task || !task->is_valid) return;
    
    // Save current task data
    current_task = task;
    
    // If task contains valid data, update textures
    if (!task->original_image.empty()) {
        // Update original image texture
        glBindTexture(GL_TEXTURE_2D, tex_frame);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, task->original_image.cols, task->original_image.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, task->original_image.data);
    }
    
    // Update heatmap texture (using CUDA-OpenGL Interop to update directly from device memory)
    if (task->d_heatmap_gpu) {
        cudaGraphicsMapResources(1, &cuda_res_heatmap, 0);
        cudaArray_t tex_array;
        cudaGraphicsSubResourceGetMappedArray(&tex_array, cuda_res_heatmap, 0, 0);
        
        // [Critical fix] Both Pitch and Width are 256 * 4
        cudaMemcpy2DToArray(
            tex_array, 0, 0,
            task->d_heatmap_gpu.get(),
            256 * 4 * sizeof(uint8_t), // Pitch (bytes per row in source data)
            256 * 4 * sizeof(uint8_t), // Width (bytes to copy)
            256,                       // Height
            cudaMemcpyDeviceToDevice
        );
        
        cudaGraphicsUnmapResources(1, &cuda_res_heatmap, 0);
    }
    
    // Update overlay texture (using detection results)
    // Prefer to use heatmap_vis as detection result
    if (!task->heatmap_vis.empty()) {
        // Update overlay texture
        glBindTexture(GL_TEXTURE_2D, tex_overlay);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, task->heatmap_vis.cols, task->heatmap_vis.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, task->heatmap_vis.data);
    } else if (!task->original_image.empty()) {
        // If no detection result, fallback to original image
        glBindTexture(GL_TEXTURE_2D, tex_overlay);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, task->original_image.cols, task->original_image.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, task->original_image.data);
    }
    
    // For CUDA Interop mode, texture content is updated by PostProcessor, no need to call glTexImage2D here
    // Just need to ensure texture ID is valid
}

void Dashboard::GetTextureIDs(unsigned int& tex_frame_out, unsigned int& tex_heatmap_out, unsigned int& tex_mask_out) {
    tex_frame_out = tex_frame;
    tex_heatmap_out = tex_heatmap;
    tex_mask_out = tex_overlay;
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

// 2. Add heatmap upload in UpdateTextures
void Dashboard::UpdateTextures() {
    if (is_initialized && is_running && pipeline) {
        // Get output queue from pipeline
        auto& output_queue = pipeline->getOutputQueue();
        
        // Try to get a task from the output queue
        pipeline::FrameTaskPtr task;
        if (output_queue.try_pop(task)) {
            if (task->is_valid) {
                // Use new UpdateData method to update textures
                UpdateData(task);
            }
        }
    }
}

void Dashboard::Render(int display_w, int display_h) {
    // Save current task for recycling after update
    pipeline::FrameTaskPtr old_task = current_task;
    
    // Update backend logic
    UpdateTextures();

    // Print texture IDs to ensure they are not 0
    // static int render_count = 0;
    // if (render_count % 30 == 0) { // Print every 30 frames
    //     std::cout << "[Dashboard] Rendering frame " << render_count << std::endl;
    //     std::cout << "  - tex_frame: " << tex_frame << std::endl;
    //     std::cout << "  - tex_heatmap: " << tex_heatmap << std::endl;
    //     std::cout << "  - tex_overlay: " << tex_overlay << std::endl;
    // }
    // render_count++;

    float sideBarWidth = 350.0f;

    // Draw two parts
    DrawSidePanel(sideBarWidth, (float)display_h);
    DrawMainView(sideBarWidth, (float)display_w - sideBarWidth, (float)display_h);
    
    // Recycle old task back to pool
    if (old_task && pipeline) {
        pipeline->return_task(old_task);
    }
}

void Dashboard::DrawSidePanel(float width, float height) {
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(width, height));
    
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
    ImGui::Begin("ControlPanel", nullptr, flags);

    ImGui::TextColored(ImVec4(0.22f, 0.64f, 1.0f, 1.0f), "LiMR DETECTOR");
    ImGui::Separator();
    ImGui::Spacing();

    // Status
    ImGui::Text("STATUS: "); ImGui::SameLine();
    if (is_initialized) ImGui::TextColored(ImVec4(0, 1, 0, 1), "READY");
    else ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "CONFIGURING");
    
    // Display single task processing time
    if (current_task) {
        double processing_time = current_task->end_time - current_task->start_time;
        ImGui::Text("Processing Time: %.3f ms", processing_time);
    } else {
        ImGui::Text("Processing Time: -- ms");
    }
    
    ImGui::Separator();

    // Configuration section
    if (is_initialized) ImGui::BeginDisabled();
    ImGui::Text("Configuration");
    ImGui::InputText("Video", video_path, 256);
    
    // Engine path input
    ImGui::InputText("Model Path", engine_path_a, 256);
    
    ImGui::Combo("Precision", &current_precision_idx, precision_items, 2);

    if (is_initialized) ImGui::EndDisabled();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Inference Parameters");
    
    // Slider: range 0.0 to 1.0
    ImGui::SliderFloat("Threshold", &defect_threshold, 0.0f, 1.0f, "Conf: %.2f");
    // Add an explanation
    if (ImGui::IsItemHovered()) 
        ImGui::SetTooltip("Adjust sensitivity for defect contours");

    ImGui::Spacing();
    ImGui::Separator();

    // Button logic
    float btnH = 45.0f;
    if (!is_initialized) {
        if (ImGui::Button("INITIALIZE SYSTEM", ImVec2(-1, btnH))) {
            try {
                // Call your backend initialization logic here
                std::string type_str = precision_items[current_precision_idx];
                trt::Precision precision = (type_str == "F16 ") ? trt::Precision::FP16 : trt::Precision::FP32;
                
                // Create configuration object
                appConfig = std::unique_ptr<AppConfig>(new AppConfig(video_path, engine_path_a, "", " ", type_str, InferenceMode::SINGLE_ENGINE));
                
                // Create and load TensorRT engine
                engine = std::make_unique<trt::TrtEngine>();
                if (!engine->load(engine_path_a, precision)) {
                    std::cerr << "Failed to load engine: " << engine_path_a << std::endl;
                    return;
                }
                
                // Create pipeline
                std::cout << "[Dashboard] Creating pipeline with video source: " << video_path << std::endl;
                pipeline = std::make_unique<pipeline::Pipeline>(video_path, 256, 256, engine.get());
                
                // Start pipeline
                std::cout << "[Dashboard] Starting pipeline..." << std::endl;
                pipeline->start();
                std::cout << "[Dashboard] Pipeline started successfully" << std::endl;
                
                is_initialized = true;
                is_running = true;
                std::cout << "[Dashboard] System initialized successfully" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Initialization failed: " << e.what() << std::endl;
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
            pipeline.reset();
            engine.reset();
            appConfig.reset();
            
            // Reset textures
            tex_frame = 0;
            tex_heatmap = 0;
            tex_overlay = 0;
        }
        
        // Apply & Reload button
        ImGui::Spacing();
        if (ImGui::Button("Apply & Reload", ImVec2(-1, btnH))) {
            try {
                // Stop current pipeline
                is_running = false;
                
                // Update configuration
                std::string type_str = precision_items[current_precision_idx];
                trt::Precision precision = (type_str == "F16 ") ? trt::Precision::FP16 : trt::Precision::FP32;
                
                // Update configuration object
                appConfig = std::unique_ptr<AppConfig>(new AppConfig(video_path, engine_path_a, "", " ", type_str, InferenceMode::SINGLE_ENGINE));
                
                // Recreate and load TensorRT engine
                engine = std::make_unique<trt::TrtEngine>();
                if (!engine->load(engine_path_a, precision)) {
                    std::cerr << "Failed to load engine: " << engine_path_a << std::endl;
                    return;
                }
                
                // Recreate pipeline
                pipeline = std::make_unique<pipeline::Pipeline>(video_path, 256, 256, engine.get());
                
                // Start pipeline
                pipeline->start();
                
                is_running = true;
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
    
    // Set background to black
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.08f, 0.08f, 1.0f));
    ImGui::Begin("ViewPanel", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
    // ImGui::PopStyleColor();

    // ImGui::Begin("ViewPanel", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
    // ImGui::PopStyleColor();

    // [Modified condition] Check if all three textures are ready
    if (tex_frame != 0 && tex_overlay != 0 && tex_heatmap != 0) {
        // --- Adaptive three-column layout calculation ---
        float pad = 15.0f; // Spacing between images
        // Total width minus left, middle1, middle2, right four gaps
        float availW = width - pad * 4; 
        // Divide into three parts
        float targetW = availW / 3.0f;
        float targetH = targetW; // Assume 1:1 ratio
        
        // Print layout calculation dimensions
        // std::cout << "[Dashboard] Layout calculation: " << std::endl;
        // std::cout << "  - View width: " << width << std::endl;
        // std::cout << "  - View height: " << height << std::endl;
        // std::cout << "  - Available width: " << availW << std::endl;
        // std::cout << "  - Target width: " << targetW << std::endl;
        // std::cout << "  - Target height: " << targetH << std::endl;

        // Height limit check
        if (targetH > height - 80) { 
            targetH = height - 80;
            targetW = targetH;
        }

        // Overall center calculation
        float totalContentWidth = targetW * 3 + pad * 2;
        float cursorX = (width - totalContentWidth) / 2.0f;
        float cursorY = (height - targetH) / 2.0f;
        if (cursorX < pad) cursorX = pad;
        if (cursorY < pad) cursorY = pad;

        ImGui::SetCursorPos(ImVec2(cursorX, cursorY));

        // --- Draw three images ---
        
        // 1. Original image
        ImGui::BeginGroup();
        ImGui::Text("Original Input");
        ImGui::Image((void*)(intptr_t)tex_frame, ImVec2(targetW, targetH));
        ImGui::EndGroup();

        ImGui::SameLine(0, pad);

        // 2. [Added] Pure heatmap
        ImGui::BeginGroup();
        ImGui::Text("Heatmap View");
        ImGui::Image((void*)(intptr_t)tex_heatmap, ImVec2(targetW, targetH));
        ImGui::EndGroup();

        ImGui::SameLine(0, pad);

        // 3. Final result image (red boxes on original image)
        // ImGui::BeginGroup();
        // ImGui::Text("Defect Detection");
        // ImGui::Image((void*)(intptr_t)tex_overlay, ImVec2(targetW, targetH));
        // ImGui::EndGroup();

        // ImGui::SameLine(0, pad); 
        ImGui::BeginGroup();
        ImGui::Text("Defect Detection");

        // [Step 1] Get the starting absolute coordinates of the current image on the screen
        ImVec2 p_min = ImGui::GetCursorScreenPos();

        // [Step 2] Draw the base image
        ImGui::Image((void*)(intptr_t)tex_overlay, ImVec2(targetW, targetH));

        // [Step 3] Get the drawing pen and draw red boxes
        // TODO: Get defect rectangles from the task object
        // For now, we'll just draw a dummy rectangle
        if (appConfig) { // Ensure appConfig exists
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            
            // 【Core fix: Calculate scaling ratio】
            // Get the base resolution used during inference (backend calculates coordinates based on this resolution)
            // We read these values from the configuration object to ensure generality
            float baseW = 256.0f;  // Fixed to 256x256
            float baseH = 256.0f; // Fixed to 256x256

            // Protection against division by zero
            if (baseW > 0 && baseH > 0) {
                // Calculate scaling factors: (current display width/height / base width/height)
                float scale_x = targetW / baseW;
                float scale_y = targetH / baseH;

                // Get detection boxes from current task data
                if (current_task && current_task->is_valid) {
                    // Iterate through peaks data and draw detection boxes
                    for (const auto& peak : current_task->peaks) {
                        // Assume peak's x, y are center coordinates, create a rectangle centered at this point
                        int rect_width = 30; // Rectangle width
                        int rect_height = 30; // Rectangle height
                        int rect_x = peak.x - rect_width / 2;
                        int rect_y = peak.y - rect_height / 2;
                        
                        // Calculate screen coordinates
                        float x = p_min.x + rect_x * scale_x;
                        float y = p_min.y + rect_y * scale_y;
                        float w = rect_width * scale_x;
                        float h = rect_height * scale_y;

                        // Draw rectangle
                        draw_list->AddRect(
                            ImVec2(x, y),          // Top-left corner
                            ImVec2(x + w, y + h),  // Bottom-right corner
                            IM_COL32(255, 0, 0, 255), // Red color
                            0.0f, 0, 2.0f // No rounded corners, line width 2.0
                        );
                    }
                }
            }
        }
        ImGui::EndGroup();

    } else {
        // Display waiting text
        const char* txt = "WAITING FOR SIGNAL...";
        ImVec2 txtSize = ImGui::CalcTextSize(txt);
        ImGui::SetCursorPos(ImVec2((width - txtSize.x) / 2, (height - txtSize.y) / 2));
        ImGui::TextColored(ImVec4(0.5, 0.5, 0.5, 1), txt);
    }

    ImGui::End();
    ImGui::PopStyleColor();
}