cmake_minimum_required(VERSION 3.10)

project(limr LANGUAGES CUDA CXX)

# define options
option(USE_DIRECT_BLOB "Enable direct blob access" ON)

add_definitions(-DUSE_DIRECT_BLOB=${USE_DIRECT_BLOB})

# Modify to your path (原有配置)
set(OpenCV_DIR "C:/opencv_s/build/install")  
set(TRT_DIR "C:/tensorRT/TensorRT-10.12.0.36.Windows.win10.cuda-11.8/TensorRT-10.12.0.36")
set(CMAKE_CUDA_ARCHITECTURES "86")  

# -------------------------------------------------------------------
# ### [新增] 设置依赖路径 (请根据你实际解压的位置修改这里!)
# -------------------------------------------------------------------
# 假设你把它们放在了项目根目录下的 3rdparty 文件夹里
set(GLFW_DIR "${CMAKE_SOURCE_DIR}/3rdparty/glfw-3.4.bin.WIN64") 
set(IMGUI_DIR "${CMAKE_SOURCE_DIR}/3rdparty/imgui-1.92.5") 
# -------------------------------------------------------------------

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# setup CUDA
find_package(CUDA REQUIRED)
# message(STATUS "libraries: ${CUDA_LIBRARIES}") # 这行有时会刷屏，可注释
include_directories(${CUDA_INCLUDE_DIRS})

enable_language(CUDA)

include_directories(${PROJECT_SOURCE_DIR}/include)

# TensorRT
include_directories(${TRT_DIR}/include)
link_directories(${TRT_DIR}/lib)
include_directories(${TRT_DIR}/samples/common)


# OpenCV
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# torch
set(Torch_DIR "C:/libtorch/libtorch-win-shared-with-deps-debug-1.13.0+cu116/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
include_directories(${Torch_INCLUDE_DIRS})
include_directories(C:/libtorch/libtorch-win-shared-with-deps-debug-1.13.0+cu116/libtorch/include/torch/csrc/api/include)
include_directories(C:/libtorch/libtorch-win-shared-with-deps-debug-1.13.0+cu116/libtorch/include)
link_directories(C:/libtorch/libtorch-win-shared-with-deps-debug-1.13.0+cu116/libtorch/lib)

# -------------------------------------------------------------------
# ### [新增] GLFW 配置
# -------------------------------------------------------------------
include_directories(${GLFW_DIR}/include)
link_directories(${GLFW_DIR}/lib-vc2019) # 注意：这里要选对应你 VS 版本的文件夹，如 lib-vc2019

# -------------------------------------------------------------------
# ### [新增] ImGui 配置 (ImGui 是源码集成的)
# -------------------------------------------------------------------
include_directories(${IMGUI_DIR})
include_directories(${IMGUI_DIR}/backends)

# 手动指定需要的 ImGui 源文件
# 包含核心文件 + GLFW 实现 + OpenGL3 实现
file(GLOB IMGUI_SOURCES
    "${IMGUI_DIR}/*.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp"
)

# -------------------------------------------------------------------
# 源文件收集
# -------------------------------------------------------------------
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE SRCS_CU ${PROJECT_SOURCE_DIR}/src/*.cu)
list(APPEND SRCS ${SRCS_CU})

# ### [修改] 将 ImGui 的源码加入到可执行文件中
add_executable(limr ${SRCS} ${IMGUI_SOURCES})

# -------------------------------------------------------------------
# 链接库
# -------------------------------------------------------------------
target_link_libraries(limr nvinfer_10)
target_link_libraries(limr cudart)
target_link_libraries(limr nvonnxparser_10)
target_link_libraries(limr ${OpenCV_LIBS})
target_link_libraries(limr torch)

# ### [新增] 链接 GLFW 和 OpenGL
# opengl32 是 Windows 系统自带的，直接写名字即可
target_link_libraries(limr glfw3 opengl32)