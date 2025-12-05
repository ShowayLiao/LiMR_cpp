cmake_minimum_required(VERSION 3.10)

project(limr LANGUAGES CUDA CXX)

# define options
option(USE_DIRECT_BLOB "Enable direct blob access" ON)
add_definitions(-DUSE_DIRECT_BLOB=${USE_DIRECT_BLOB})

# =================================================================
# 1. 路径设置 (保持你原来的路径不变)
# =================================================================
set(OpenCV_DIR "C:/opencv_s/build/install") 
set(TRT_DIR "C:/tensorRT/TensorRT-10.12.0.36.Windows.win10.cuda-11.8/TensorRT-10.12.0.36")
set(CMAKE_CUDA_ARCHITECTURES "86") 

# LibTorch 路径
set(Torch_DIR "C:/libtorch/libtorch-win-shared-with-deps-debug-1.13.0+cu116/libtorch/share/cmake/Torch")

# =================================================================
# 2. 基础配置
# =================================================================
# 建议切换为 Release 以获得最佳性能，Debug 模式 LibTorch 会非常慢
# set(CMAKE_BUILD_TYPE Debug) 
set(CMAKE_BUILD_TYPE Release)

add_definitions(-std=c++14) # LibTorch 通常建议 C++14 或以上
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 14)

# =================================================================
# 3. 依赖包查找
# =================================================================

# CUDA
enable_language(CUDA) # 新版 CMake 推荐写法，替代 find_package(CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# TensorRT
include_directories(${TRT_DIR}/include)
link_directories(${TRT_DIR}/lib)
include_directories(${TRT_DIR}/samples/common)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Torch
find_package(Torch REQUIRED)
# 既然用了 find_package，下面这些其实通常不需要手动加，但为了保险保留你的配置
include_directories(${Torch_INCLUDE_DIRS})
include_directories(C:/libtorch/libtorch-win-shared-with-deps-debug-1.13.0+cu116/libtorch/include/torch/csrc/api/include)
include_directories(C:/libtorch/libtorch-win-shared-with-deps-debug-1.13.0+cu116/libtorch/include)
link_directories(C:/libtorch/libtorch-win-shared-with-deps-debug-1.13.0+cu116/libtorch/lib)

# =================================================================
# 4. 头文件包含 (关键修改)
# =================================================================
# 确保 src 目录也被包含，这样 #include "httplib.h" 才能找到
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src) 

# =================================================================
# 5. 源文件管理 (关键修改)
# =================================================================
# GLOB_RECURSE 会把 src 下所有的 .cpp 都找出来，包括 main.cpp
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE SRCS_CU ${PROJECT_SOURCE_DIR}/src/*.cu)

# 合并源文件列表
list(APPEND PROJECT_SOURCES ${SRCS} ${SRCS_CU})

# 生成可执行文件
# 注意：这里直接用 list，不要再手动写 main.cpp，否则会报 "multiple definition of main"
add_executable(limr ${PROJECT_SOURCES})

# =================================================================
# 6. 链接库 (关键修改)
# =================================================================

# 链接 TensorRT
target_link_libraries(limr nvinfer_10)
target_link_libraries(limr cudart)
target_link_libraries(limr nvonnxparser_10)

# 链接 OpenCV
target_link_libraries(limr ${OpenCV_LIBS})

# 链接 Torch (使用官方变量更安全)
target_link_libraries(limr ${TORCH_LIBRARIES})

# [重要] 链接 Windows 网络库 (httplib 需要)
target_link_libraries(limr ws2_32)