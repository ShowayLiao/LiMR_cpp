cmake_minimum_required(VERSION 3.10)

project(limr LANGUAGES CUDA CXX)

include(CTest)
enable_testing()

if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/std:c++17>
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    )
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-std=c++17>)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OpenCV_DIR "C:/opencv_s/build/install" CACHE PATH "Path containing OpenCVConfig.cmake")
set(TRT_DIR "C:/tensorRT/TensorRT-10.12.0.36.Windows.win10.cuda-11.8/TensorRT-10.12.0.36" CACHE PATH "TensorRT installation root")

set(GLFW_DIR "${CMAKE_SOURCE_DIR}/3rdparty/glfw")
set(IMGUI_DIR "${CMAKE_SOURCE_DIR}/3rdparty/imgui")

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
enable_language(CUDA)

include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories(${TRT_DIR}/include)
link_directories(${TRT_DIR}/lib)
include_directories(${TRT_DIR}/samples/common)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${GLFW_DIR}/include)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(${GLFW_DIR} ${CMAKE_BINARY_DIR}/glfw)

include_directories(${IMGUI_DIR})
include_directories(${IMGUI_DIR}/backends)

file(GLOB IMGUI_SOURCES
    "${IMGUI_DIR}/*.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp"
)

file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE SRCS_CU ${PROJECT_SOURCE_DIR}/src/*.cu)

list(REMOVE_ITEM SRCS "${PROJECT_SOURCE_DIR}/src/pipeline.cpp")
list(REMOVE_ITEM SRCS "${PROJECT_SOURCE_DIR}/src/video_thread.cpp")
list(REMOVE_ITEM SRCS "${PROJECT_SOURCE_DIR}/src/inference.cpp")
list(REMOVE_ITEM SRCS "${PROJECT_SOURCE_DIR}/src/inference_runner.cpp")
list(REMOVE_ITEM SRCS "${PROJECT_SOURCE_DIR}/src/model_loader.cpp")
list(REMOVE_ITEM SRCS "${PROJECT_SOURCE_DIR}/src/postprocess.cpp")
list(REMOVE_ITEM SRCS "${PROJECT_SOURCE_DIR}/src/preprocess.cpp")
list(REMOVE_ITEM SRCS "${PROJECT_SOURCE_DIR}/src/utils.cpp")
list(REMOVE_ITEM SRCS_CU "${PROJECT_SOURCE_DIR}/src/utils.cu")

list(APPEND SRCS ${SRCS_CU})

add_executable(limr ${SRCS} ${IMGUI_SOURCES})

add_executable(test_pipeline
    ${PROJECT_SOURCE_DIR}/test/test_pipeline.cpp
    ${PROJECT_SOURCE_DIR}/src/engine/trt_engine.cpp
    ${PROJECT_SOURCE_DIR}/src/kernels/preprocess.cu
    ${PROJECT_SOURCE_DIR}/src/pipeline/preprocessor.cpp
    ${PROJECT_SOURCE_DIR}/src/pipeline/postprocessor.cpp
    ${PROJECT_SOURCE_DIR}/src/pipeline/InputThread.cpp
    ${PROJECT_SOURCE_DIR}/src/pipeline/InferenceThread.cpp
    ${PROJECT_SOURCE_DIR}/src/pipeline/Pipeline.cpp
    ${PROJECT_SOURCE_DIR}/src/kernels/postprocess_mask.cu
    ${PROJECT_SOURCE_DIR}/src/common/kernels.cu
)

add_executable(test_input_thread
    ${PROJECT_SOURCE_DIR}/test/test_input_thread.cpp
    ${PROJECT_SOURCE_DIR}/src/engine/trt_engine.cpp
    ${PROJECT_SOURCE_DIR}/src/kernels/preprocess.cu
    ${PROJECT_SOURCE_DIR}/src/pipeline/preprocessor.cpp
    ${PROJECT_SOURCE_DIR}/src/pipeline/postprocessor.cpp
    ${PROJECT_SOURCE_DIR}/src/pipeline/InputThread.cpp
    ${PROJECT_SOURCE_DIR}/src/pipeline/InferenceThread.cpp
    ${PROJECT_SOURCE_DIR}/src/pipeline/Pipeline.cpp
    ${PROJECT_SOURCE_DIR}/src/kernels/postprocess_mask.cu
    ${PROJECT_SOURCE_DIR}/src/common/kernels.cu
)

add_executable(test_input_shape
    ${PROJECT_SOURCE_DIR}/test/test_input_shape.cpp
)

add_executable(test_safe_queue
    ${PROJECT_SOURCE_DIR}/test/test_safe_queue.cpp
)

add_executable(test_engine_contract
    ${PROJECT_SOURCE_DIR}/test/test_engine_contract.cpp
    ${PROJECT_SOURCE_DIR}/src/engine/trt_engine.cpp
)

add_executable(test_file_dialog
    ${PROJECT_SOURCE_DIR}/test/test_file_dialog.cpp
    ${PROJECT_SOURCE_DIR}/src/common/FileDialog.cpp
)

add_executable(test_dashboard_task_state
    ${PROJECT_SOURCE_DIR}/test/test_dashboard_task_state.cpp
)
target_include_directories(test_dashboard_task_state PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(test_dashboard_render_options
    ${PROJECT_SOURCE_DIR}/test/test_dashboard_render_options.cpp
)
target_include_directories(test_dashboard_render_options PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(test_anomaly_map_shape
    ${PROJECT_SOURCE_DIR}/test/test_anomaly_map_shape.cpp
)
target_include_directories(test_anomaly_map_shape PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(test_heatmap
    ${PROJECT_SOURCE_DIR}/test/test_heatmap.cu
    ${PROJECT_SOURCE_DIR}/src/common/kernels.cu
)

target_include_directories(test_input_shape PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(test_safe_queue PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(test_file_dialog PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_test(NAME input_shape_contract COMMAND test_input_shape)
add_test(NAME safe_queue_contract COMMAND test_safe_queue)
add_test(NAME engine_contract COMMAND test_engine_contract)
add_test(NAME file_dialog_contract COMMAND test_file_dialog)
add_test(NAME dashboard_task_state_contract COMMAND test_dashboard_task_state)
add_test(NAME dashboard_render_options_contract COMMAND test_dashboard_render_options)
add_test(NAME anomaly_map_shape_contract COMMAND test_anomaly_map_shape)
set_tests_properties(engine_contract PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME heatmap_contract COMMAND test_heatmap)
add_test(NAME pipeline_model_contract COMMAND test_pipeline)
set_tests_properties(pipeline_model_contract PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")


target_link_libraries(limr nvinfer_10)
target_link_libraries(limr cudart)
target_link_libraries(limr nvonnxparser_10)
target_link_libraries(limr ${OpenCV_LIBS})
target_link_libraries(limr glfw opengl32)
target_link_libraries(limr nppc)
target_link_libraries(limr nppial)
target_link_libraries(limr nppist)
target_link_libraries(limr nppig)
target_link_libraries(limr comdlg32)

target_link_libraries(test_file_dialog PRIVATE comdlg32)

target_link_libraries(test_pipeline nvinfer_10)
target_link_libraries(test_pipeline cudart)
target_link_libraries(test_pipeline nvonnxparser_10)
target_link_libraries(test_pipeline ${OpenCV_LIBS})
target_link_libraries(test_pipeline nppc)
target_link_libraries(test_pipeline nppial)
target_link_libraries(test_pipeline nppist)
target_link_libraries(test_pipeline nppig)

target_link_libraries(test_input_thread ${OpenCV_LIBS})
target_link_libraries(test_input_thread cudart)
target_link_libraries(test_input_thread nvinfer_10)
target_link_libraries(test_input_thread nvonnxparser_10)
target_link_libraries(test_input_thread nppc)
target_link_libraries(test_input_thread nppial)
target_link_libraries(test_input_thread nppist)
target_link_libraries(test_input_thread nppig)

target_link_libraries(test_engine_contract nvinfer_10)
target_link_libraries(test_engine_contract cudart)
target_link_libraries(test_engine_contract nvonnxparser_10)
target_link_libraries(test_heatmap cudart)


if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "..." FORCE)
endif()

install(TARGETS limr
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# 设置需要的动态库列表
set(REQUIRED_DLLS
    # --- OpenCV (仅保留核心模块) ---
    opencv_core455.dll
    opencv_imgcodecs455.dll
    opencv_imgproc455.dll
    opencv_videoio455.dll
    opencv_videoio_ffmpeg455_64.dll  # MP4解码必备，不做链接但必须存在

    # --- CUDA / NPP (根据你的代码依赖) ---
    cudart64_110.dll
    nppig64_11.dll      # Resize (nppiResize)
    nppidei64_11.dll    # SwapChannels (nppiSwapChannels)
    nppc64_11.dll       # NPP Core (基础依赖)

    # --- TensorRT 隐式依赖 (最容易漏的) ---
    cublas64_11.dll     # 矩阵运算核心
    cublasLt64_11.dll   # 轻量级矩阵运算 (RTX 30/40系常需)
    zlibwapi.dll        # TensorRT解压模型依赖 (经常缺失!)

    # --- TensorRT 核心 ---
    nvinfer_10.dll
    nvinfer_plugin_10.dll
    nvinfer_builder_resource_10.dll
    nvonnxparser_10.dll
)

# 构建需要拷贝的DLL列表
set(ALL_DLLS "")

# 查找并添加OpenCV DLLs
set(OPENCV_BIN_DIR "${OpenCV_DIR}/x64/vc16/bin")
foreach(dll_name ${REQUIRED_DLLS})
    if(dll_name MATCHES "^opencv_")
        set(dll_path "${OPENCV_BIN_DIR}/${dll_name}")
        if(EXISTS "${dll_path}")
            list(APPEND ALL_DLLS "${dll_path}")
        else()
            message(WARNING "OpenCV DLL not found: ${dll_path}")
        endif()
    endif()
endforeach()

# 查找并添加CUDA/NPP DLLs
set(CUDA_BIN_DIR "${CUDA_TOOLKIT_ROOT_DIR}/bin")
foreach(dll_name ${REQUIRED_DLLS})
    if(dll_name MATCHES "^(cudart|npp|cublas)")
        set(dll_path "${CUDA_BIN_DIR}/${dll_name}")
        if(EXISTS "${dll_path}")
            list(APPEND ALL_DLLS "${dll_path}")
        else()
            # 尝试查找不同版本的CUDA DLL
            file(GLOB dll_candidates "${CUDA_BIN_DIR}/${dll_name}")
            if(dll_candidates)
                list(APPEND ALL_DLLS ${dll_candidates})
            else()
                message(WARNING "CUDA/NPP DLL not found: ${dll_path}")
            endif()
        endif()
    endif()
endforeach()

# 查找并添加TensorRT DLLs
set(TRT_BIN_DIR "${TRT_DIR}/lib")
foreach(dll_name ${REQUIRED_DLLS})
    if(dll_name MATCHES "^(nvinfer|nvonnxparser|zlibwapi)")
        set(dll_path "${TRT_BIN_DIR}/${dll_name}")
        if(EXISTS "${dll_path}")
            list(APPEND ALL_DLLS "${dll_path}")
        else()
            message(WARNING "TensorRT DLL not found: ${dll_path}")
        endif()
    endif()
endforeach()

# 安装DLLs
if(ALL_DLLS)
    install(FILES ${ALL_DLLS} DESTINATION bin)

    # 构建后拷贝DLLs
    foreach(DLL_FILE ${ALL_DLLS})
        add_custom_command(TARGET limr POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DLL_FILE}"
            "$<TARGET_FILE_DIR:limr>"
        )
        add_custom_command(TARGET test_pipeline POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DLL_FILE}"
            "$<TARGET_FILE_DIR:test_pipeline>"
        )
        add_custom_command(TARGET test_input_thread POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DLL_FILE}"
            "$<TARGET_FILE_DIR:test_input_thread>"
        )
        add_custom_command(TARGET test_engine_contract POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DLL_FILE}"
            "$<TARGET_FILE_DIR:test_engine_contract>"
        )
    endforeach()
endif()
