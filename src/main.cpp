#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include "dashboard.h" // Include our encapsulated class

int main(int, char**) {
    // 1. System window initialization
    if (!glfwInit()) return 1;
    const char* glsl_version = "#version 130";
    GLFWwindow* window = glfwCreateWindow(1600, 900, "LiMR Industrial Detector", NULL, NULL);
    if (!window) return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // Enable vertical sync

    // 2. ImGui initialization
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // 3. ★★★ Create Dashboard instance ★★★
    Dashboard myDashboard;

    // 4. Initialize Dashboard resources
    myDashboard.InitResources();
    std::cout << "[Main] Dashboard resources initialized" << std::endl;

    // 5. Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Start a new frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Get window size and pass to Dashboard, let it decide how to draw
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        
        // ★★★ Call the encapsulated render function ★★★
        myDashboard.Render(display_w, display_h);

        // Render to screen
        ImGui::Render();
        glViewport(0, 0, display_w, display_h);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // 5. Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}