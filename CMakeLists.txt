cmake_minimum_required(VERSION 3.10)

project(limr LANGUAGES CUDA CXX)

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/std:c++17>)
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-std=c++17>)
endif()
set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR "C:/opencv_s/build/install")
set(TRT_DIR "C:/tensorRT/TensorRT-10.12.0.36.Windows.win10.cuda-11.8/TensorRT-10.12.0.36")

set(GLFW_DIR "${CMAKE_SOURCE_DIR}/3rdparty/glfw-3.4.bin.WIN64")
set(IMGUI_DIR "${CMAKE_SOURCE_DIR}/3rdparty/imgui-1.92.5")

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
link_directories(${GLFW_DIR}/lib-vc2019)

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
    ${PROJECT_SOURCE_DIR}/src/pipeline/InputThread.cpp
    ${PROJECT_SOURCE_DIR}/src/pipeline/InferenceThread.cpp
    ${PROJECT_SOURCE_DIR}/src/pipeline/Pipeline.cpp
)

add_executable(test_inference_thread
    ${PROJECT_SOURCE_DIR}/test/test_inference_thread.cpp
    ${PROJECT_SOURCE_DIR}/src/pipeline/InputThread.cpp
    ${PROJECT_SOURCE_DIR}/src/pipeline/InferenceThread.cpp
    ${PROJECT_SOURCE_DIR}/src/pipeline/Pipeline.cpp
    ${PROJECT_SOURCE_DIR}/src/engine/trt_engine.cpp
    ${PROJECT_SOURCE_DIR}/src/kernels/preprocess.cu
    ${PROJECT_SOURCE_DIR}/src/pipeline/preprocessor.cpp
    ${PROJECT_SOURCE_DIR}/src/pipeline/postprocessor.cpp
    ${PROJECT_SOURCE_DIR}/src/kernels/postprocess_mask.cu
    ${PROJECT_SOURCE_DIR}/src/common/kernels.cu
)

target_link_libraries(limr nvinfer_10)
target_link_libraries(limr cudart)
target_link_libraries(limr nvonnxparser_10)
target_link_libraries(limr ${OpenCV_LIBS})
target_link_libraries(limr glfw3 opengl32)
target_link_libraries(limr nppc)
target_link_libraries(limr nppial)
target_link_libraries(limr nppist)
target_link_libraries(limr nppig)

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

target_link_libraries(test_inference_thread nvinfer_10)
target_link_libraries(test_inference_thread cudart)
target_link_libraries(test_inference_thread nvonnxparser_10)
target_link_libraries(test_inference_thread ${OpenCV_LIBS})
target_link_libraries(test_inference_thread nppc)
target_link_libraries(test_inference_thread nppial)
target_link_libraries(test_inference_thread nppist)
target_link_libraries(test_inference_thread nppig)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "..." FORCE)
endif()

install(TARGETS limr
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

set(ALL_DLLS "")

file(GLOB TRT_DLLS "${TRT_DIR}/lib/*.dll")
list(APPEND ALL_DLLS ${TRT_DLLS})

file(GLOB OPENCV_ALL_DLLS "${OpenCV_DIR}/x64/vc16/bin/*.dll")
set(OPENCV_FINAL_DLLS "")

foreach(tmp_dll ${OPENCV_ALL_DLLS})
    if(tmp_dll MATCHES "d\\.dll$")
        if(CMAKE_BUILD_TYPE MATCHES "Debug")
            list(APPEND OPENCV_FINAL_DLLS ${tmp_dll})
        endif()
    else()
        if(NOT CMAKE_BUILD_TYPE MATCHES "Debug")
            list(APPEND OPENCV_FINAL_DLLS ${tmp_dll})
        endif()
    endif()
endforeach()

list(APPEND ALL_DLLS ${OPENCV_FINAL_DLLS})



get_filename_component(CUDA_BIN_DIR ${CUDA_CUDART_LIBRARY} DIRECTORY)
set(CUDA_DLL_DIR "${CUDA_TOOLKIT_ROOT_DIR}/bin")
file(GLOB CUDA_DLLS "${CUDA_DLL_DIR}/cudart64_*.dll" "${CUDA_DLL_DIR}/cublas64_*.dll" "${CUDA_DLL_DIR}/curand64_*.dll")
list(APPEND ALL_DLLS ${CUDA_DLLS})

install(FILES ${ALL_DLLS} DESTINATION bin)

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
    add_custom_command(TARGET test_inference_thread POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${DLL_FILE}"
        "$<TARGET_FILE_DIR:test_inference_thread>"
    )
endforeach()
