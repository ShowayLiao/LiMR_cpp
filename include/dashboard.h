#pragma once
#include <memory>
#include <vector>
#include <string>
#include "imgui.h"
#include "app_config.h"
#include "pipeline/Pipeline.h"
#include "engine/TrtEngine.h"
#include <cuda_gl_interop.h>

class Dashboard {
public:
    Dashboard();
    ~Dashboard();


    void InitResources();
    
    // Data update
    void UpdateData(const pipeline::FrameTaskPtr& task);
    
    // Call this function every frame for rendering and logic processing
    void Render(int display_w, int display_h);
    
    // Get texture IDs
    void GetTextureIDs(unsigned int& tex_frame, unsigned int& tex_heatmap, unsigned int& tex_mask);

    // Helper: Handle texture updates
    void UpdateTextures();

private:
    // --- Internal drawing methods ---
    void SetupStyle();
    void DrawSidePanel(float width, float height);
    void DrawMainView(float start_x, float width, float height);
    void ReleaseResources();
    bool IsTaskOutputCompatible(const pipeline::FrameTaskPtr& task) const;
    
    // --- Core business objects (managed by smart pointers) ---
    std::unique_ptr<AppConfig> appConfig;
    std::unique_ptr<trt::TrtEngine> engine;
    std::unique_ptr<pipeline::Pipeline> pipeline;

    // --- UI state variables ---
    bool is_initialized = false;
    bool is_running = false;
    bool resources_initialized = false;
    int texture_width_ = 0;
    int texture_height_ = 0;
    
    // Texture handles
    unsigned int tex_frame = 0;   
    // [New] Pure heatmap texture
    unsigned int tex_heatmap = 0; 
    unsigned int tex_overlay = 0; 
    unsigned int tex_combined = 0; 
    
    // CUDA-OpenGL Interop resources
    cudaGraphicsResource_t cuda_res_heatmap = nullptr;
    cudaGraphicsResource_t cuda_res_overlay = nullptr;
    cudaGraphicsResource_t cuda_res_combined = nullptr;
    
    // Current task data
    pipeline::FrameTaskPtr current_task;

    // --- Input buffers (ImGui requires char*) ---
    char video_path[256] = "./input/blade.avi";
    
    // Engine path input
    char engine_path_a[256] = "./input/LiMR_merged.onnx";
    
    int current_precision_idx = 1; // Default to F16
    const char* precision_items[2] = { "F32", "F16 " };

    float defect_threshold = 0.5f;
    
    // Resolution settings
    int width = 224;  // Active model/render width for the running pipeline.
    int height = 224; // Active model/render height for the running pipeline.
    int requested_width_ = 224;
    int requested_height_ = 224;
    const char* resolution_items[2] = { "224x224", "448x448" };
    int current_resolution_idx = 0;
    bool model_input_is_dynamic_ = true;
    
    // Recent processing time (to avoid display flickering)
    double recent_processing_time = 0.0;
    
    // Anomalib mode flag
    bool use_anomalib_mode_ = false;
};
