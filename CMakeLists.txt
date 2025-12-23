cmake_minimum_required(VERSION 3.10)

project(limr LANGUAGES CUDA CXX)

# ===================================================================
# 1. 基础配置与选项
# ===================================================================

# define options
option(USE_DIRECT_BLOB "Enable direct blob access" ON)
add_definitions(-DUSE_DIRECT_BLOB=${USE_DIRECT_BLOB})

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)

# ===================================================================
# 2. 路径设置 (请在此处修改)
# ===================================================================

set(OpenCV_DIR "C:/opencv_s/build/install") 
set(TRT_DIR "C:/tensorRT/TensorRT-10.12.0.36.Windows.win10.cuda-11.8/TensorRT-10.12.0.36")
set(CMAKE_CUDA_ARCHITECTURES "86") 

# 3rdparty 路径
set(GLFW_DIR "${CMAKE_SOURCE_DIR}/3rdparty/glfw-3.4.bin.WIN64") 
set(IMGUI_DIR "${CMAKE_SOURCE_DIR}/3rdparty/imgui-1.92.5") 

# LibTorch 路径
set(TORCH_PATH_DEBUG "C:/libtorch/libtorch-win-shared-with-deps-1.13.0+cu116")
set(TORCH_PATH_RELEASE "C:/libtorch/libtorch-win-shared-with-deps-1.13.0+cu116") 

# 根据构建类型自动选择 Torch 路径
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message(STATUS "Configuring for DEBUG build...")
    set(Torch_DIR "${TORCH_PATH_DEBUG}/libtorch/share/cmake/Torch")
else()
    message(STATUS "Configuring for RELEASE (or other) build...")
    set(Torch_DIR "${TORCH_PATH_RELEASE}/libtorch/share/cmake/Torch")
endif()

# ===================================================================
# 3. 查找包与包含目录
# ===================================================================

# setup CUDA
find_package(CUDA REQUIRED)
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
find_package(Torch REQUIRED)
include_directories(${Torch_INCLUDE_DIRS})

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    include_directories(${TORCH_PATH_DEBUG}/libtorch/include/torch/csrc/api/include)
    include_directories(${TORCH_PATH_DEBUG}/libtorch/include)
    link_directories(${TORCH_PATH_DEBUG}/libtorch/lib)
else()
    include_directories(${TORCH_PATH_RELEASE}/libtorch/include/torch/csrc/api/include)
    include_directories(${TORCH_PATH_RELEASE}/libtorch/include)
    link_directories(${TORCH_PATH_RELEASE}/libtorch/lib)
endif()


# GLFW
include_directories(${GLFW_DIR}/include)
link_directories(${GLFW_DIR}/lib-vc2019) 

# ImGui
include_directories(${IMGUI_DIR})
include_directories(${IMGUI_DIR}/backends)

# ===================================================================
# 4. 源文件收集
# ===================================================================

# ImGui 源文件
file(GLOB IMGUI_SOURCES
    "${IMGUI_DIR}/*.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp"
)

# 项目源文件
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE SRCS_CU ${PROJECT_SOURCE_DIR}/src/*.cu)
list(APPEND SRCS ${SRCS_CU})

# ===================================================================
# 5. 生成目标与链接
# ===================================================================

add_executable(limr ${SRCS} ${IMGUI_SOURCES})

target_link_libraries(limr nvinfer_10)
target_link_libraries(limr cudart)
target_link_libraries(limr nvonnxparser_10)
target_link_libraries(limr ${OpenCV_LIBS})
target_link_libraries(limr torch)
target_link_libraries(limr glfw3 opengl32)

# ===================================================================
# 6. Install 配置
# ===================================================================

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "..." FORCE)
endif()

install(TARGETS limr
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# ===================================================================
# 7. [关键修改] 自动复制依赖 DLL (带筛选逻辑)
# ===================================================================

set(ALL_DLLS "")

# 1. 收集 TensorRT DLLs
file(GLOB TRT_DLLS "${TRT_DIR}/lib/*.dll")
list(APPEND ALL_DLLS ${TRT_DLLS})

# -------------------------------------------------------------------
# 2. [修改] 收集 OpenCV DLLs 并进行过滤
# -------------------------------------------------------------------
# 先拿到所有 DLL
file(GLOB OPENCV_ALL_DLLS "${OpenCV_DIR}/x64/vc16/bin/*.dll")
set(OPENCV_FINAL_DLLS "")

foreach(tmp_dll ${OPENCV_ALL_DLLS})
    # 正则判断：文件名是否以 'd.dll' 结尾
    # 注意：Windows文件路径中可能是 d.dll 或者 d.DLL，正则忽略大小写由 CMake 处理
    if(tmp_dll MATCHES "d\\.dll$")
        # --- 这是 Debug 版 DLL ---
        if(CMAKE_BUILD_TYPE MATCHES "Debug")
            list(APPEND OPENCV_FINAL_DLLS ${tmp_dll})
        endif()
    else()
        # --- 这是 Release 版 DLL (不以 d.dll 结尾) ---
        if(NOT CMAKE_BUILD_TYPE MATCHES "Debug")
            list(APPEND OPENCV_FINAL_DLLS ${tmp_dll})
        endif()
    endif()
endforeach()

list(APPEND ALL_DLLS ${OPENCV_FINAL_DLLS})
# -------------------------------------------------------------------

# 3. 收集 LibTorch DLLs (原本逻辑已经区分了路径，这里直接GLOB即可)
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    file(GLOB TORCH_DLLS "${TORCH_PATH_DEBUG}/libtorch/lib/*.dll")
else()
    file(GLOB TORCH_DLLS "${TORCH_PATH_RELEASE}/libtorch/lib/*.dll")
endif()
list(APPEND ALL_DLLS ${TORCH_DLLS})

# 4. 收集 CUDA Runtime DLL
get_filename_component(CUDA_BIN_DIR ${CUDA_CUDART_LIBRARY} DIRECTORY)
set(CUDA_DLL_DIR "${CUDA_TOOLKIT_ROOT_DIR}/bin")
file(GLOB CUDA_DLLS "${CUDA_DLL_DIR}/cudart64_*.dll" "${CUDA_DLL_DIR}/cublas64_*.dll" "${CUDA_DLL_DIR}/curand64_*.dll")
list(APPEND ALL_DLLS ${CUDA_DLLS})

# -------------------------------------------------------------------
# 执行安装复制
# -------------------------------------------------------------------
install(FILES ${ALL_DLLS} DESTINATION bin)

# -------------------------------------------------------------------
# 编译后立即复制 (方便调试)
# -------------------------------------------------------------------
foreach(DLL_FILE ${ALL_DLLS})
    add_custom_command(TARGET limr POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${DLL_FILE}"
        "$<TARGET_FILE_DIR:limr>"
    )
endforeach()