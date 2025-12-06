#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

// --- ImGui & OpenGL Headers ---
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

// --- Your Project Headers ---
#include "video_thread.h"
#include "config.h"
#include "utils.h"

// 辅助函数：将 OpenCV Mat 转为 OpenGL 纹理
// 优化：传入引用以重用 textureID，避免内存泄漏
void update_texture(const cv::Mat& mat, GLuint& texture_id) {
    if (mat.empty()) return;

    if (texture_id == 0) {
        glGenTextures(1, &texture_id);
    }
    glBindTexture(GL_TEXTURE_2D, texture_id);
    
    // 设置纹理过滤
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    // 上传数据 (假设 mat 已经是 RGB 格式，在 video_thread 中转过了)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mat.cols, mat.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, mat.data);
}


// int main(){
    
//     // set the video and model path

//     try
//     {   
//         std::cout<<cv::getBuildInformation() << std::endl; // Print OpenCV build information
//         std::cout<<cv::cuda::getCudaEnabledDeviceCount() << " CUDA devices available." << std::endl; // Print number of CUDA devices available
//         const std::string videoSource = "../../input/blade.avi"; // Path to the video file
//         const std::string StuModelPath = "../../input/LiMR_student_16.engine"; // Path to the student model ONNX file
//         const std::string TeaModelPath = "../../input/LiMR_teacher_16.engine"; // Path to the teacher model ONNX file
//         const std::string StuOnnxPath = "../../input/LiMR_student.onnx"; // Path to the student model ONNX file
//         const std::string TeaOnnxPath = "../../input/LiMR_teacher.onnx"; // Path to the teacher model ONNX file
//         // const std::string imgSource = "../../input/IMG_9260.png "; // Path to the image file, if any
//         const std::string imgSource = " "; // Path to the image file, if any
//         const std::string type = "F32"; // Data type for the model, can be "F32" or "INT8"
//         // if you use code 'config.set_flag(trt.BuilderFlag.FP16) ' to build the model, you should also set type to "F32"
//         // because the input and output nodes of model is still FP32, only the internal computation is in FP16

//         Config NormalConfig(videoSource, 
//                     StuModelPath, 
//                     TeaModelPath,
//                     imgSource,
//                     type);

//         // check if the engine files exist, if not, build them
//         if(!std::ifstream(StuModelPath).good()) {
//             std::cout << "Building student model engine..." << std::endl;
//             build_engine(StuOnnxPath, 
//                         StuModelPath, 
//                         NormalConfig.batchSize,
//                         NormalConfig.inputWidth,
//                         NormalConfig.inputHeight,
//                         NormalConfig.outputWidth,
//                         NormalConfig.outputHeight);
            
//         } else {
//             std::cout << "Student model engine already exists." << std::endl;
//         }

//         if (!std::ifstream(TeaModelPath).good()) {
//             std::cout << "Building teacher model engine..." << std::endl;
//             build_engine(TeaOnnxPath, 
//                         TeaModelPath, 
//                         NormalConfig.batchSize,
//                         NormalConfig.inputWidth,
//                         NormalConfig.inputHeight,
//                         NormalConfig.outputWidth,
//                         NormalConfig.outputHeight);
//         } else {
//             std::cout << "Teacher model engine already exists." << std::endl;
//         }
        

//         VideoThread::VideoCaptureThread videoThread(NormalConfig);
//         videoThread.start();

//         return 0;
//     }
//     catch (const std::exception& e)
//     {
//         std::cerr << "Exception: " << e.what() << std::endl;
//         system("pause");
//         return 1;
//     }



// }

int main(int, char**) {
    // ---------------------------------------------------
    // 1. 系统初始化 (OpenCV / TensorRT 配置)
    // ---------------------------------------------------
    std::cout<<cv::getBuildInformation() << std::endl; // Print OpenCV build information
    std::cout<<cv::cuda::getCudaEnabledDeviceCount() << " CUDA devices available." << std::endl; // Print number of CUDA devices available
    const std::string videoSource = "../../input/blade.avi"; // Path to the video file
    const std::string StuModelPath = "../../input/LiMR_student_16.engine"; // Path to the student model ONNX file
    const std::string TeaModelPath = "../../input/LiMR_teacher_16.engine"; // Path to the teacher model ONNX file
    const std::string StuOnnxPath = "../../input/LiMR_student.onnx"; // Path to the student model ONNX file
    const std::string TeaOnnxPath = "../../input/LiMR_teacher.onnx"; // Path to the teacher model ONNX file
    // const std::string imgSource = "../../input/IMG_9260.png "; // Path to the image file, if any
    const std::string imgSource = " "; // Path to the image file, if any
    const std::string type = "F32"; // Data type for the model, can be "F32" or "INT8"
    // if you use code 'config.set_flag(trt.BuilderFlag.FP16) ' to build the model, you should also set type to "F32"
    // because the input and output nodes of model is still FP32, only the internal computation is in FP16

    Config NormalConfig(videoSource, 
                StuModelPath, 
                TeaModelPath,
                imgSource,
                type);

    // check if the engine files exist, if not, build them
    if(!std::ifstream(StuModelPath).good()) {
        std::cout << "Building student model engine..." << std::endl;
        build_engine(StuOnnxPath, 
                    StuModelPath, 
                    NormalConfig.batchSize,
                    NormalConfig.inputWidth,
                    NormalConfig.inputHeight,
                    NormalConfig.outputWidth,
                    NormalConfig.outputHeight);
        
    } else {
        std::cout << "Student model engine already exists." << std::endl;
    }

    if (!std::ifstream(TeaModelPath).good()) {
        std::cout << "Building teacher model engine..." << std::endl;
        build_engine(TeaOnnxPath, 
                    TeaModelPath, 
                    NormalConfig.batchSize,
                    NormalConfig.inputWidth,
                    NormalConfig.inputHeight,
                    NormalConfig.outputWidth,
                    NormalConfig.outputHeight);
    } else {
        std::cout << "Teacher model engine already exists." << std::endl;
    }
    // 假设 Config NormalConfig 已经创建好了

    // 初始化业务逻辑模块
    VideoThread::VideoCaptureThread videoProcessor(NormalConfig);

    // ---------------------------------------------------
    // 2. GUI 初始化 (GLFW + ImGui)
    // ---------------------------------------------------
    if (!glfwInit()) return 1;
    const char* glsl_version = "#version 130";
    GLFWwindow* window = glfwCreateWindow(1280, 720, "LiMR Defect Detection", NULL, NULL);
    if (window == NULL) return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // 开启垂直同步

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark(); // 深色主题
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // 状态变量
    bool is_inference_running = false; // 控制是否开始推理
    GLuint tex_frame = 0;
    GLuint tex_overlay = 0;

    // ---------------------------------------------------
    // 3. 主循环 (UI + 业务逻辑)
    // ---------------------------------------------------
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // [业务逻辑]：如果开启了推理，执行一步 update
        if (is_inference_running) {
            bool success = videoProcessor.update();
            if (success) {
                // 将更新后的 Mat 上传到 GPU
                update_texture(videoProcessor.getResultFrame(), tex_frame);
                update_texture(videoProcessor.getResultOverlay(), tex_overlay);
            }
        }

        // [UI 渲染]
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // === 绘制控制面板 ===
        {
            ImGui::Begin("Control Panel"); // 创建一个面板
            
            ImGui::Text("Status: %s", is_inference_running ? "Running" : "Paused");
            ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
            ImGui::Separator();

            // 按钮控制
            if (!is_inference_running) {
                if (ImGui::Button("Start Inference", ImVec2(150, 40))) {
                    is_inference_running = true;
                }
            } else {
                if (ImGui::Button("Stop / Pause", ImVec2(150, 40))) {
                    is_inference_running = false;
                }
            }
            
            ImGui::End();
        }

        // === 绘制视频画面 ===
        {
            ImGui::Begin("Live View");
            
            if (tex_frame != 0 && tex_overlay != 0) {
                // 并排显示两张图
                // 参数：(void*)(intptr_t)textureID, ImVec2(width, height)
                ImGui::Text("Original");
                ImGui::Image((void*)(intptr_t)tex_frame, ImVec2(448, 448));
                
                ImGui::SameLine(); // 下一个控件在同一行
                
                ImGui::Text("Anomaly Map"); 
                // 注意：SameLine 仅对紧随其后的控件有效，这里 Text 会换行，所以需要稍微调整布局逻辑
                // 简单的做法是分两个 Image 画
                ImGui::Image((void*)(intptr_t)tex_overlay, ImVec2(448, 448));
            } else {
                ImGui::Text("Press Start to visualize...");
            }

            ImGui::End();
        }

        // 渲染并交换缓冲区
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // ---------------------------------------------------
    // 4. 清理
    // ---------------------------------------------------
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}