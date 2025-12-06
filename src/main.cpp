#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include "dashboard.h" // 引入我们封装的类

int main(int, char**) {
    // 1. 系统窗口初始化
    if (!glfwInit()) return 1;
    const char* glsl_version = "#version 130";
    GLFWwindow* window = glfwCreateWindow(1600, 900, "LiMR Industrial Detector", NULL, NULL);
    if (!window) return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // 开启垂直同步

    // 2. ImGui 初始化
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // 3. ★★★ 创建 Dashboard 实例 ★★★
    Dashboard myDashboard;

    // 4. 主循环
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // 开启新一帧绘制
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // 获取窗口大小传给 Dashboard，让它自己决定怎么画
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        
        // ★★★ 调用封装好的渲染函数 ★★★
        myDashboard.Render(display_w, display_h);

        // 渲染上屏
        ImGui::Render();
        glViewport(0, 0, display_w, display_h);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // 5. 清理
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}