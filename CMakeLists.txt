cmake_minimum_required(VERSION 3.21)

project(anomalib_runtime LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# All machine-specific paths are cache variables.  Set them through a CMake
# preset, the command line, or the Windows build container; do not edit this
# file for a particular developer machine.
set(LIMR_TRT_ROOT "" CACHE PATH "TensorRT SDK root (contains include and lib)")
set(LIMR_OPENCV_DIR "" CACHE PATH "Directory containing OpenCVConfig.cmake")
set(LIMR_OPENCV_BIN_DIR "" CACHE PATH "Directory containing OpenCV runtime DLLs")
set(LIMR_GLFW_DIR "${CMAKE_SOURCE_DIR}/3rdparty/glfw" CACHE PATH "GLFW source root")
set(LIMR_IMGUI_DIR "${CMAKE_SOURCE_DIR}/3rdparty/imgui" CACHE PATH "Dear ImGui source root")
set(LIMR_EXTRA_RUNTIME_DLLS "" CACHE STRING "Additional runtime DLL paths, separated by semicolons")

if(NOT WIN32)
    message(FATAL_ERROR "AnomaRT currently builds a Windows CUDA/TensorRT application. Use the Windows build container or a Windows host.")
endif()
if(NOT LIMR_TRT_ROOT)
    message(FATAL_ERROR "LIMR_TRT_ROOT is required. Point it at the TensorRT SDK root.")
endif()
if(NOT EXISTS "${LIMR_TRT_ROOT}/include/NvInfer.h")
    message(FATAL_ERROR "LIMR_TRT_ROOT does not look like a TensorRT SDK: ${LIMR_TRT_ROOT}")
endif()
if(LIMR_OPENCV_DIR)
    set(OpenCV_DIR "${LIMR_OPENCV_DIR}" CACHE PATH "" FORCE)
endif()
if(NOT OpenCV_DIR)
    message(FATAL_ERROR "LIMR_OPENCV_DIR is required and must contain OpenCVConfig.cmake.")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED CONFIG)

if(NOT LIMR_OPENCV_BIN_DIR)
    get_filename_component(_opencv_prefix "${OpenCV_DIR}/../.." ABSOLUTE)
    set(LIMR_OPENCV_BIN_DIR "${_opencv_prefix}/bin" CACHE PATH "Directory containing OpenCV runtime DLLs" FORCE)
endif()

foreach(required_dir IN ITEMS "${LIMR_GLFW_DIR}/include" "${LIMR_IMGUI_DIR}")
    if(NOT EXISTS "${required_dir}")
        message(FATAL_ERROR "Required bundled dependency is missing: ${required_dir}")
    endif()
endforeach()

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory("${LIMR_GLFW_DIR}" "${CMAKE_BINARY_DIR}/_deps/glfw")

file(GLOB IMGUI_SOURCES CONFIGURE_DEPENDS
    "${LIMR_IMGUI_DIR}/*.cpp"
    "${LIMR_IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
    "${LIMR_IMGUI_DIR}/backends/imgui_impl_opengl3.cpp")
file(GLOB_RECURSE SRCS CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE SRCS_CU CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.cu")

list(REMOVE_ITEM SRCS
    "${PROJECT_SOURCE_DIR}/src/pipeline.cpp"
    "${PROJECT_SOURCE_DIR}/src/video_thread.cpp"
    "${PROJECT_SOURCE_DIR}/src/inference.cpp"
    "${PROJECT_SOURCE_DIR}/src/inference_runner.cpp"
    "${PROJECT_SOURCE_DIR}/src/model_loader.cpp"
    "${PROJECT_SOURCE_DIR}/src/postprocess.cpp"
    "${PROJECT_SOURCE_DIR}/src/preprocess.cpp"
    "${PROJECT_SOURCE_DIR}/src/utils.cpp")
list(REMOVE_ITEM SRCS_CU "${PROJECT_SOURCE_DIR}/src/utils.cu")

add_executable(AnomaRT ${SRCS} ${SRCS_CU} ${IMGUI_SOURCES})
add_executable(test_pipeline
    "${PROJECT_SOURCE_DIR}/test/test_pipeline.cpp"
    "${PROJECT_SOURCE_DIR}/src/engine/trt_engine.cpp"
    "${PROJECT_SOURCE_DIR}/src/kernels/preprocess.cu"
    "${PROJECT_SOURCE_DIR}/src/pipeline/preprocessor.cpp"
    "${PROJECT_SOURCE_DIR}/src/pipeline/postprocessor.cpp"
    "${PROJECT_SOURCE_DIR}/src/pipeline/InputThread.cpp"
    "${PROJECT_SOURCE_DIR}/src/pipeline/InferenceThread.cpp"
    "${PROJECT_SOURCE_DIR}/src/pipeline/Pipeline.cpp"
    "${PROJECT_SOURCE_DIR}/src/kernels/postprocess_mask.cu"
    "${PROJECT_SOURCE_DIR}/src/common/kernels.cu")
add_executable(test_input_thread
    "${PROJECT_SOURCE_DIR}/test/test_input_thread.cpp"
    "${PROJECT_SOURCE_DIR}/src/pipeline/InputThread.cpp"
    "${PROJECT_SOURCE_DIR}/src/pipeline/InferenceThread.cpp"
    "${PROJECT_SOURCE_DIR}/src/pipeline/Pipeline.cpp")

foreach(target IN ITEMS AnomaRT test_pipeline test_input_thread)
    target_include_directories(${target} PRIVATE
        "${PROJECT_SOURCE_DIR}/include"
        "${LIMR_TRT_ROOT}/include"
        "${LIMR_TRT_ROOT}/samples/common"
        "${OpenCV_INCLUDE_DIRS}"
        "${LIMR_GLFW_DIR}/include"
        "${LIMR_IMGUI_DIR}"
        "${LIMR_IMGUI_DIR}/backends")
    target_link_directories(${target} PRIVATE "${LIMR_TRT_ROOT}/lib")
    target_compile_features(${target} PRIVATE cxx_std_17)
endforeach()

target_link_libraries(AnomaRT PRIVATE nvinfer_10 CUDA::cudart nvonnxparser_10 ${OpenCV_LIBS} glfw opengl32 CUDA::nppc CUDA::nppial CUDA::nppist CUDA::nppig)
target_link_libraries(test_pipeline PRIVATE nvinfer_10 CUDA::cudart nvonnxparser_10 ${OpenCV_LIBS} CUDA::nppc CUDA::nppial CUDA::nppist CUDA::nppig)
target_link_libraries(test_input_thread PRIVATE ${OpenCV_LIBS} CUDA::cudart)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/dist" CACHE PATH "Install location" FORCE)
endif()
install(TARGETS AnomaRT RUNTIME DESTINATION bin)
include(InstallRequiredSystemLibraries)

# Runtime deployment is deliberately collected by family rather than hard-coded
# DLL versions. This makes the package follow the exact CUDA/OpenCV SDK pinned
# by the build image. NVIDIA's redistribution terms still apply.
set(RUNTIME_DLLS "${LIMR_EXTRA_RUNTIME_DLLS}")
file(GLOB _opencv_dlls "${LIMR_OPENCV_BIN_DIR}/opencv_*.dll")
file(GLOB _cuda_dlls
    "${CUDAToolkit_BIN_DIR}/cudart64*.dll"
    "${CUDAToolkit_BIN_DIR}/npp*.dll"
    "${CUDAToolkit_BIN_DIR}/cublas*.dll")
file(GLOB _trt_dlls
    "${LIMR_TRT_ROOT}/lib/nvinfer*.dll"
    "${LIMR_TRT_ROOT}/lib/nvonnxparser*.dll"
    "${LIMR_TRT_ROOT}/lib/zlibwapi.dll")
list(APPEND RUNTIME_DLLS ${_opencv_dlls} ${_cuda_dlls} ${_trt_dlls} ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS})
list(REMOVE_DUPLICATES RUNTIME_DLLS)
if(RUNTIME_DLLS)
    install(FILES ${RUNTIME_DLLS} DESTINATION bin)
    foreach(dll IN LISTS RUNTIME_DLLS)
        add_custom_command(TARGET AnomaRT POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:AnomaRT>")
    endforeach()
else()
    message(WARNING "No runtime DLLs were found. The executable will build but the distributable package will be incomplete.")
endif()
