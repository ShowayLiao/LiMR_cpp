cmake_minimum_required(VERSION 3.10)

project(limr LANGUAGES CUDA CXX)

# define options
option(USE_DIRECT_BLOB "Enable direct blob access" ON)

add_definitions(-DUSE_DIRECT_BLOB=${USE_DIRECT_BLOB})

# Modify to your path
set(OpenCV_DIR "C:/opencv_s/build/install")  
set(TRT_DIR "C:/tensorRT/TensorRT-10.12.0.36.Windows.win10.cuda-11.8/TensorRT-10.12.0.36")
set(CMAKE_CUDA_ARCHITECTURES "86")  

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# setup CUDA
find_package(CUDA REQUIRED)
message(STATUS "libraries: ${CUDA_LIBRARIES}")
message(STATUS "include path: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})

enable_language(CUDA)

include_directories(${PROJECT_SOURCE_DIR}/include)

# TensorRT
# find_package(TRT_DIR)
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




file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE SRCS_CU ${PROJECT_SOURCE_DIR}/src/*.cu)
list(APPEND SRCS ${SRCS_CU})
add_executable(limr ${PROJECT_SOURCE_DIR}/src/main.cpp ${SRCS})


target_link_libraries(limr nvinfer_10)
target_link_libraries(limr cudart)
target_link_libraries(limr nvonnxparser_10)
# target_link_libraries(limr cudaimgproc cudaarithm)
target_link_libraries(limr ${OpenCV_LIBS})
# target_link_libraries(limr ${Torch_LIBS})
target_link_libraries(limr torch)