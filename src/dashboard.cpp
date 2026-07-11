#include <GLFW/glfw3.h> // If OpenGL functions are needed
#include <GL/gl.h> // Include OpenGL headers for GL_BGR
#include "dashboard.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <thread>
#include "pipeline/Preprocessor.h"
#include "pipeline/Postprocessor.h"
#include "common/CudaMemory.hpp"
#include "pipeline/Pipeline.h"

#ifndef GL_BGR
#define GL_BGR 0x80E0
#endif

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

// Input type enumeration
enum class InputType {
    UNKNOWN,
    IMAGE,
    VIDEO,
    CAMERA
};

// Helper function: Detect input type
static InputType detect_input_type(const std::string& input) {
    // Check if input is a binary digit (0 or 1) for camera
    if (input == "0" || input == "1") {
        return InputType::CAMERA;
    }
    
    // Check if input is an image file
    std::string lower_input = input;
    std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
    
    std::vector<std::string> image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"};
    for (const auto& ext : image_extensions) {
        if (lower_input.find(ext) != std::string::npos) {
            return InputType::IMAGE;
        }
    }
    
    // Check if input is a video file
    std::vector<std::string> video_extensions = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".flv"};
    for (const auto& ext : video_extensions) {
        if (lower_input.find(ext) != std::string::npos) {
            return InputType::VIDEO;
        }
    }
    
    return InputType::UNKNOWN;
}

// Helper function: Process image input
static bool process_image_input(const std::string& image_path, cv::Mat& out_image) {
    out_image = cv::imread(image_path);
    return !out_image.empty();
}

// Helper function: Check if input is a valid video file
static bool is_valid_video_file(const std::string& video_path) {
    cv::VideoCapture cap(video_path);
    bool isValid = cap.isOpened();
    if (isValid) {
        cap.release();
    }
    return isValid;
}

// Helper function: Check if input is a valid camera index
static bool is_valid_camera(int camera_index) {
    cv::VideoCapture cap(camera_index);
    bool isValid = cap.isOpened();
    if (isValid) {
        cap.release();
    }
    return isValid;
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
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        
        glGenTextures(1, &tex_heatmap);
        glBindTexture(GL_TEXTURE_2D, tex_heatmap);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        // Key point: format is changed to GL_RGBA here
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);
        
        // Register as CUDA resource
        cudaGraphicsGLRegisterImage(&cuda_res_heatmap, tex_heatmap, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
        
        glGenTextures(1, &tex_overlay);
        glBindTexture(GL_TEXTURE_2D, tex_overlay);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        // Change format to GL_RGBA for overlay with alpha channel
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);
        
        // Register overlay texture as CUDA resource
        cudaGraphicsGLRegisterImage(&cuda_res_overlay, tex_overlay, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
        
        glGenTextures(1, &tex_combined);
        glBindTexture(GL_TEXTURE_2D, tex_combined);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        // Format is GL_RGBA for combined image with overlay
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);
        
        // Register combined texture as CUDA resource
        cudaGraphicsGLRegisterImage(&cuda_res_combined, tex_combined, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
        
        // Print texture IDs to ensure they are not 0
        // std::cout << "[Dashboard] Texture IDs generated: " << std::endl;
        // std::cout << "  - tex_frame: " << tex_frame << std::endl;
        // std::cout << "  - tex_heatmap: " << tex_heatmap << std::endl;
        // std::cout << "  - tex_overlay: " << tex_overlay << std::endl;
        
        // Initialize with black background texture
        // Create RGB black buffer for tex_frame
        std::vector<unsigned char> black_background(width * height * 3, 0);
        
        // Create RGBA black buffer for other textures
        std::vector<unsigned char> black_rgba(width * height * 4, 0);
        
        // Update black background to textures
        // tex_frame uses RGB format
        glBindTexture(GL_TEXTURE_2D, tex_frame);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, black_background.data());
        
        // tex_heatmap uses RGBA format - use glTexSubImage2D to preserve format
        glBindTexture(GL_TEXTURE_2D, tex_heatmap);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, black_rgba.data());
        
        // tex_overlay uses RGBA format - use glTexSubImage2D to preserve format
        glBindTexture(GL_TEXTURE_2D, tex_overlay);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, black_rgba.data());
        
        // tex_combined uses RGBA format - use glTexSubImage2D to preserve format
        glBindTexture(GL_TEXTURE_2D, tex_combined);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, black_rgba.data());
        
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
    if (cuda_res_overlay) {
        cudaGraphicsUnregisterResource(cuda_res_overlay);
        cuda_res_overlay = nullptr;
    }
    if (cuda_res_combined) {
        cudaGraphicsUnregisterResource(cuda_res_combined);
        cuda_res_combined = nullptr;
    }
    if (tex_frame) glDeleteTextures(1, &tex_frame);
    // [新增]
    if (tex_heatmap) glDeleteTextures(1, &tex_heatmap);
    if (tex_overlay) glDeleteTextures(1, &tex_overlay);
    if (tex_combined) glDeleteTextures(1, &tex_combined);
}

void Dashboard::UpdateData(const pipeline::FrameTaskPtr& task) {
    if (!task || !task->is_valid) return;
    
    current_task = task;
    
    if (task->end_time > task->start_time) {
        recent_processing_time = task->end_time - task->start_time;
    }
    
    if (!task->original_image.empty()) {
        glBindTexture(GL_TEXTURE_2D, tex_frame);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, task->original_image.cols, task->original_image.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, task->original_image.data);
    }
    
    if (task->d_final_heatmap) {
        cudaStreamSynchronize(0);
        cudaGraphicsMapResources(1, &cuda_res_heatmap, 0);
        cudaArray_t tex_array;
        cudaGraphicsSubResourceGetMappedArray(&tex_array, cuda_res_heatmap, 0, 0);
        cudaMemcpy2DToArray(tex_array, 0, 0, task->d_final_heatmap.get(), task->target_width * 4 * sizeof(uint8_t), task->target_width * 4 * sizeof(uint8_t), task->target_height, cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &cuda_res_heatmap, 0);
    }
    
    if (task->d_final_overlay) {
        cudaStreamSynchronize(0);
        cudaGraphicsMapResources(1, &cuda_res_overlay, 0);
        cudaArray_t tex_array;
        cudaGraphicsSubResourceGetMappedArray(&tex_array, cuda_res_overlay, 0, 0);
        cudaMemcpy2DToArray(tex_array, 0, 0, task->d_final_overlay.get(), task->target_width * 4 * sizeof(uint8_t), task->target_width * 4 * sizeof(uint8_t), task->target_height, cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &cuda_res_overlay, 0);
    } else if (!task->original_image.empty()) {
        glBindTexture(GL_TEXTURE_2D, tex_overlay);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, task->original_image.cols, task->original_image.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, task->original_image.data);
    }
    
    if (task->d_final_overlay) {
        cudaStreamSynchronize(0);
        cudaGraphicsMapResources(1, &cuda_res_combined, 0);
        cudaArray_t tex_array;
        cudaGraphicsSubResourceGetMappedArray(&tex_array, cuda_res_combined, 0, 0);
        cudaMemcpy2DToArray(tex_array, 0, 0, task->d_final_overlay.get(), task->target_width * 4 * sizeof(uint8_t), task->target_width * 4 * sizeof(uint8_t), task->target_height, cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &cuda_res_combined, 0);
    } else if (!task->original_image.empty()) {
        glBindTexture(GL_TEXTURE_2D, tex_combined);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, task->original_image.cols, task->original_image.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, task->original_image.data);
    }
    
    glFlush();
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
    // For image input (is_running == false), still update textures if we have a current task
    // This ensures the image is displayed even after processing is complete
    if (is_initialized && !is_running && current_task) {
        // Update textures with current task data
        UpdateData(current_task);
    }
}

void Dashboard::Render(int display_w, int display_h) {
    pipeline::FrameTaskPtr prev_task = current_task;
    
    UpdateTextures();

    float sideBarWidth = 350.0f;

    DrawSidePanel(sideBarWidth, (float)display_h);
    DrawMainView(sideBarWidth, (float)display_w - sideBarWidth, (float)display_h);
    
    if (prev_task && prev_task != current_task && pipeline) {
        pipeline->return_task(prev_task);
    }
}

void Dashboard::DrawSidePanel(float panel_width, float panel_height) {
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(panel_width, panel_height));
    
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
    ImGui::Begin("ControlPanel", nullptr, flags);

    ImGui::TextColored(ImVec4(0.22f, 0.64f, 1.0f, 1.0f), "AnomaRT");
    ImGui::Separator();
    ImGui::Spacing();

    // Status
    ImGui::Text("STATUS: "); ImGui::SameLine();
    if (is_initialized) ImGui::TextColored(ImVec4(0, 1, 0, 1), "READY");
    else ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "CONFIGURING");
    
    // Display processing time (using recent value to avoid flickering)
    if (recent_processing_time > 0) {
        ImGui::Text("Processing Time: %.3f ms", recent_processing_time * 1000.0);
    } else {
        ImGui::Text("Processing Time: -- ms");
    }
    
    ImGui::Separator();

    // Configuration section
    ImGui::Text("Configuration");
    ImGui::InputText("Video", video_path, 256);
    
    // Engine path input
    ImGui::InputText("Model Path", engine_path_a, 256);
    
    ImGui::Combo("Precision", &current_precision_idx, precision_items, 2);

    // Resolution selection
    ImGui::Combo("Resolution", &current_resolution_idx, resolution_items, 4);
    
    // Update width and height based on selected resolution
    if (current_resolution_idx == 0) { // 256x256
        this->width = 256;
        this->height = 256;
    } else if (current_resolution_idx == 1) { // 512x512
        this->width = 512;
        this->height = 512;
    } else if (current_resolution_idx == 2) { // 640x480
        this->width = 640;
        this->height = 480;
    } else if (current_resolution_idx == 3) { // 1024x768
        this->width = 1024;
        this->height = 768;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Inference Parameters");
    
    // Slider: range 0.0 to 1.0
    if (ImGui::SliderFloat("Threshold", &defect_threshold, 0.0f, 1.0f, "Conf: %.2f")) {
        // Update threshold in pipeline when slider value changes
        if (pipeline && is_initialized) {
            pipeline->setThreshold(defect_threshold);
            std::cout << "[Dashboard] Threshold updated to: " << defect_threshold << std::endl;
        }
    }
    // Add an explanation
    if (ImGui::IsItemHovered()) 
        ImGui::SetTooltip("Adjust sensitivity for defect contours");
    
    // Anomalib mode checkbox
    if (ImGui::Checkbox("Anomalib Mode", &use_anomalib_mode_)) {
        std::cout << "[Dashboard] Anomalib mode: " << (use_anomalib_mode_ ? "enabled" : "disabled") << std::endl;
    }
    // Add an explanation
    if (ImGui::IsItemHovered()) 
        ImGui::SetTooltip("Skip normalization in preprocessing");

    ImGui::Spacing();
    ImGui::Separator();

    // Button logic
    float btnH = 45.0f;
    if (!is_initialized) {
        if (ImGui::Button("INITIALIZE SYSTEM", ImVec2(-1, btnH))) {
            try {
                // [Safety Check] If textures exist but size doesn't match selected resolution, clear them
                static int tex_width_tracker = 0;
                static int tex_height_tracker = 0;

                // Check if re-allocation is needed
                if (resources_initialized && (tex_width_tracker != this->width || tex_height_tracker != this->height)) {
                    std::cout << "[Dashboard] Resolution mismatch detected during init. Cleaning up old resources..." << std::endl;

                    // 1. MUST Unregister CUDA resources first!
                    if (cuda_res_heatmap) { cudaGraphicsUnregisterResource(cuda_res_heatmap); cuda_res_heatmap = nullptr; }
                    if (cuda_res_overlay) { cudaGraphicsUnregisterResource(cuda_res_overlay); cuda_res_overlay = nullptr; }
                    if (cuda_res_combined) { cudaGraphicsUnregisterResource(cuda_res_combined); cuda_res_combined = nullptr; }

                    // [FIX] Ensure CUDA is completely done with resources before GL deletes them
                    cudaDeviceSynchronize();

                    // 2. Then delete GL textures
                    if (tex_frame) { glDeleteTextures(1, &tex_frame); tex_frame = 0; }
                    if (tex_heatmap) { glDeleteTextures(1, &tex_heatmap); tex_heatmap = 0; }
                    if (tex_overlay) { glDeleteTextures(1, &tex_overlay); tex_overlay = 0; }
                    if (tex_combined) { glDeleteTextures(1, &tex_combined); tex_combined = 0; }
                    
                    // 3. Force synchronization to be safe
                    glFinish();
                    
                    resources_initialized = false;
                }

                // Call your backend initialization logic here
                std::string type_str = precision_items[current_precision_idx];
                trt::Precision precision = (type_str == "F16 ") ? trt::Precision::FP16 : trt::Precision::FP32;
                
                // Detect input type
                std::string input_str = video_path;
                InputType input_type = detect_input_type(input_str);
                
                std::cout << "[Dashboard] Input type detected: ";
                switch (input_type) {
                    case InputType::IMAGE:
                        std::cout << "IMAGE" << std::endl;
                        break;
                    case InputType::VIDEO:
                        std::cout << "VIDEO" << std::endl;
                        break;
                    case InputType::CAMERA:
                        std::cout << "CAMERA" << std::endl;
                        break;
                    default:
                        std::cout << "UNKNOWN" << std::endl;
                        break;
                }
                
                // Handle different input types
                if (input_type == InputType::IMAGE) {
                    // Process image input
                    cv::Mat image;
                    if (process_image_input(input_str, image)) {
                        std::cout << "[Dashboard] Image loaded successfully: " << image.cols << "x" << image.rows << std::endl;
                        
                        // Create configuration object
                        appConfig = std::unique_ptr<AppConfig>(new AppConfig(video_path, engine_path_a, "", " ", type_str, InferenceMode::SINGLE_ENGINE));
                        
                        // Create and load TensorRT engine
                        engine = std::make_unique<trt::TrtEngine>();
                        if (!engine->load(engine_path_a, precision)) {
                            std::cerr << "Failed to load engine: " << engine_path_a << std::endl;
                            return;
                        }
                        
                        // Initialize resources if not already initialized
                        if (!resources_initialized) {
                            InitResources();
                            // Track the allocated size
                            tex_width_tracker = this->width;
                            tex_height_tracker = this->height;
                        }
                        
                        // For image input, use the same pipeline approach as video
                        // Create pipeline
                        pipeline = std::make_unique<pipeline::Pipeline>(video_path, static_cast<int>(this->width), static_cast<int>(this->height), engine.get(), use_anomalib_mode_);
                        
                        // Start pipeline
                        pipeline->start();
                        
                        // Set threshold
                        pipeline->setThreshold(defect_threshold);
                        std::cout << "[Dashboard] Threshold set to: " << defect_threshold << std::endl;
                        
                        // Give the pipeline some time to process the image
                        std::this_thread::sleep_for(std::chrono::milliseconds(500));
                        
                        // Try to get the processed task from the output queue
                        pipeline::FrameTaskPtr task;
                        auto& output_queue = pipeline->getOutputQueue();
                        
                        // Wait for a short time to get the processed task
                        for (int i = 0; i < 10; i++) {
                            if (output_queue.try_pop(task)) {
                                if (task->is_valid) {
                                    // Update textures with the processed image
                                    UpdateData(task);
                                    break;
                                }
                            }
                            std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        }
                        
                        // Stop the pipeline since we only need to process one image
                        pipeline->stop();
                        
                        // Set initialized flag
                        is_initialized = true;
                        is_running = false; // Image processing is one-time
                        std::cout << "[Dashboard] Image processed successfully" << std::endl;
                    } else {
                        std::cerr << "[Dashboard] Failed to load image: " << input_str << std::endl;
                        return;
                    }
                } else if (input_type == InputType::VIDEO || input_type == InputType::CAMERA) {
                    // For video or camera input, use the existing pipeline
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
                    pipeline = std::make_unique<pipeline::Pipeline>(video_path, static_cast<int>(this->width), static_cast<int>(this->height), engine.get(), use_anomalib_mode_);
                    
                    // Initialize resources if not already initialized
                    if (!resources_initialized) {
                        InitResources();
                        // Track the allocated size
                        tex_width_tracker = this->width;
                        tex_height_tracker = this->height;
                    }
                    
                    // Start pipeline
                    std::cout << "[Dashboard] Starting pipeline..." << std::endl;
                    pipeline->start();
                    std::cout << "[Dashboard] Pipeline started successfully" << std::endl;
                    
                    // Set threshold
                    pipeline->setThreshold(defect_threshold);
                    std::cout << "[Dashboard] Threshold set to: " << defect_threshold << std::endl;
                    
                    is_initialized = true;
                    is_running = true;
                    std::cout << "[Dashboard] System initialized successfully" << std::endl;
                } else {
                    std::cerr << "[Dashboard] Unknown input type: " << input_str << std::endl;
                    return;
                }
            } catch (const std::exception& e) {
                std::cerr << "Initialization failed: " << e.what() << std::endl;
            }
        }
    } else {
        if (is_running) {
            if (ImGui::Button("PAUSE", ImVec2(150, btnH))) {
                is_running = false;
                if (pipeline) {
                    pipeline->stop();
                    std::cout << "[Dashboard] Pipeline paused" << std::endl;
                }
            }
        } else {
            if (ImGui::Button("RESUME", ImVec2(150, btnH))) {
                is_running = true;
                if (pipeline) {
                    pipeline->start();
                    std::cout << "[Dashboard] Pipeline resumed" << std::endl;
                }
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("RESET", ImVec2(150, btnH))) {
            // Stop pipeline first before destroying it
            if (pipeline) {
                pipeline->stop();
                // Give some time for threads to stop properly
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            is_running = false;
            is_initialized = false;
            pipeline.reset();
            engine.reset();
            appConfig.reset();
            
            // Reset CUDA resources
            if (cuda_res_heatmap) {
                cudaGraphicsUnregisterResource(cuda_res_heatmap);
                cuda_res_heatmap = nullptr;
            }
            if (cuda_res_overlay) {
                cudaGraphicsUnregisterResource(cuda_res_overlay);
                cuda_res_overlay = nullptr;
            }
            if (cuda_res_combined) {
                cudaGraphicsUnregisterResource(cuda_res_combined);
                cuda_res_combined = nullptr;
            }

            // Release OpenGL textures
            if (tex_frame) glDeleteTextures(1, &tex_frame);
            if (tex_heatmap) glDeleteTextures(1, &tex_heatmap);
            if (tex_overlay) glDeleteTextures(1, &tex_overlay);
            if (tex_combined) glDeleteTextures(1, &tex_combined);

            // Reset texture IDs
            tex_frame = 0;
            tex_heatmap = 0;
            tex_overlay = 0;
            tex_combined = 0;
            
            // Reset resources initialized flag to force reinitialization
            resources_initialized = false;
            
            // Reset current task
            current_task.reset();
        }
        // Apply & Reload button
        ImGui::Spacing();
        if (ImGui::Button("Apply & Reload", ImVec2(-1, btnH))) {
            try {
                // Stop and clean up existing pipeline
                if (pipeline) {
                    pipeline->stop();
                    // Give some time for threads to stop properly
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                
                is_running = false;
                is_initialized = false;
                pipeline.reset();
                engine.reset();
                appConfig.reset();
                
                // Reset resources initialized flag to force reinitialization
                resources_initialized = false;
                
                // Reset CUDA resources
                if (cuda_res_heatmap) {
                    cudaGraphicsUnregisterResource(cuda_res_heatmap);
                    cuda_res_heatmap = nullptr;
                }
                if (cuda_res_overlay) {
                    cudaGraphicsUnregisterResource(cuda_res_overlay);
                    cuda_res_overlay = nullptr;
                }
                if (cuda_res_combined) {
                    cudaGraphicsUnregisterResource(cuda_res_combined);
                    cuda_res_combined = nullptr;
                }
                
                // Release OpenGL textures
                if (tex_frame) glDeleteTextures(1, &tex_frame);
                if (tex_heatmap) glDeleteTextures(1, &tex_heatmap);
                if (tex_overlay) glDeleteTextures(1, &tex_overlay);
                if (tex_combined) glDeleteTextures(1, &tex_combined);

                // Reset texture IDs
                tex_frame = 0;
                tex_heatmap = 0;
                tex_overlay = 0;
                tex_combined = 0;
                
                // Reset current task
                current_task.reset();
                
                // Update configuration
                std::string type_str = precision_items[current_precision_idx];
                trt::Precision precision = (type_str == "F16 ") ? trt::Precision::FP16 : trt::Precision::FP32;
                
                // Detect input type
                std::string input_str = video_path;
                InputType input_type = detect_input_type(input_str);
                
                std::cout << "[Dashboard] Input type detected: ";
                switch (input_type) {
                    case InputType::IMAGE:
                        std::cout << "IMAGE" << std::endl;
                        break;
                    case InputType::VIDEO:
                        std::cout << "VIDEO" << std::endl;
                        break;
                    case InputType::CAMERA:
                        std::cout << "CAMERA" << std::endl;
                        break;
                    default:
                        std::cout << "UNKNOWN" << std::endl;
                        break;
                }
                
                // Handle different input types
                if (input_type == InputType::IMAGE) {
                    // Process image input
                    cv::Mat image;
                    if (process_image_input(input_str, image)) {
                        std::cout << "[Dashboard] Image loaded successfully: " << image.cols << "x" << image.rows << std::endl;
                        
                        // Update configuration object
                        appConfig = std::unique_ptr<AppConfig>(new AppConfig(video_path, engine_path_a, "", " ", type_str, InferenceMode::SINGLE_ENGINE));
                        
                        // Recreate and load TensorRT engine
                        engine = std::make_unique<trt::TrtEngine>();
                        if (!engine->load(engine_path_a, precision)) {
                            std::cerr << "Failed to load engine: " << engine_path_a << std::endl;
                            return;
                        }
                        
                        // Initialize resources if not already initialized
                        if (!resources_initialized) {
                            InitResources();
                        }
                        
                        // For image input, use the same pipeline approach as video
                        // Create pipeline
                        pipeline = std::make_unique<pipeline::Pipeline>(video_path, static_cast<int>(this->width), static_cast<int>(this->height), engine.get(), use_anomalib_mode_);
                        
                        // Start pipeline
                        pipeline->start();
                        
                        // Set threshold
                        pipeline->setThreshold(defect_threshold);
                        std::cout << "[Dashboard] Threshold set to: " << defect_threshold << std::endl;
                        
                        // Give the pipeline some time to process the image
                        std::this_thread::sleep_for(std::chrono::milliseconds(500));
                        
                        // Try to get the processed task from the output queue
                        pipeline::FrameTaskPtr task;
                        auto& output_queue = pipeline->getOutputQueue();
                        
                        // Wait for a short time to get the processed task
                        for (int i = 0; i < 10; i++) {
                            if (output_queue.try_pop(task)) {
                                if (task->is_valid) {
                                    // Update textures with the processed image
                                    UpdateData(task);
                                    break;
                                }
                            }
                            std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        }
                        
                        // Stop the pipeline since we only need to process one image
                        pipeline->stop();
                        
                        // Set running flag
                        is_running = false; // Image processing is one-time
                        is_initialized = true;
                        std::cout << "[Dashboard] Image processed successfully" << std::endl;
                    } else {
                        std::cerr << "[Dashboard] Failed to load image: " << input_str << std::endl;
                        return;
                    }
                } else if (input_type == InputType::VIDEO || input_type == InputType::CAMERA) {
                    // For video or camera input, use the existing pipeline
                    // Update configuration object
                    appConfig = std::unique_ptr<AppConfig>(new AppConfig(video_path, engine_path_a, "", " ", type_str, InferenceMode::SINGLE_ENGINE));
                    
                    // Recreate and load TensorRT engine
                    engine = std::make_unique<trt::TrtEngine>();
                    if (!engine->load(engine_path_a, precision)) {
                        std::cerr << "Failed to load engine: " << engine_path_a << std::endl;
                        return;
                    }
                    
                    // Initialize resources if not already initialized
                    if (!resources_initialized) {
                        InitResources();
                    }
                    
                    // Recreate pipeline
                    pipeline = std::make_unique<pipeline::Pipeline>(video_path, static_cast<int>(this->width), static_cast<int>(this->height), engine.get(), use_anomalib_mode_);
                    
                    // Start pipeline
                    pipeline->start();
                    
                    // Set threshold
                    pipeline->setThreshold(defect_threshold);
                    std::cout << "[Dashboard] Threshold set to: " << defect_threshold << std::endl;
                    
                    is_running = true;
                    is_initialized = true;
                    std::cout << "[Dashboard] System initialized successfully" << std::endl;
                } else {
                    std::cerr << "[Dashboard] Unknown input type: " << input_str << std::endl;
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
    
    // Set background to black
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.08f, 0.08f, 1.0f));
    ImGui::Begin("ViewPanel", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
    // ImGui::PopStyleColor();

    // ImGui::Begin("ViewPanel", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
    // ImGui::PopStyleColor();

    // [Modified condition] Check if all three textures are ready and system is initialized
    if (tex_frame != 0 && tex_overlay != 0 && tex_heatmap != 0 && is_initialized) {
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

        // 3. Final result image (original image with semi-transparent red overlay)
        ImGui::BeginGroup();
        ImGui::Text("Defect Detection");

        // [Step 1] Get the starting absolute coordinates of the current image on the screen
        ImVec2 p_min = ImGui::GetCursorScreenPos();

        // [Step 2] Draw the combined image (original + semi-transparent red overlay)
        ImGui::Image((void*)(intptr_t)tex_combined, ImVec2(targetW, targetH));

        // [Step 3] Get the drawing pen and draw red boxes
        // TODO: Get defect rectangles from the task object
        // For now, we'll just draw a dummy rectangle
        if (appConfig) { // Ensure appConfig exists
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            
            // 【Core fix: Calculate scaling ratio】
            // Get the base resolution used during inference (backend calculates coordinates based on this resolution)
            // Use dynamic resolution values
            float baseW = static_cast<float>(width);  // Use dynamic width
            float baseH = static_cast<float>(height); // Use dynamic height

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
