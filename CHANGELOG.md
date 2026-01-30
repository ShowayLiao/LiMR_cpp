# Changelog

## 2026-01-28: 架构重构与性能优化

### 核心架构变更

#### 模块化设计重构
- **全新模块化架构**：将系统划分为配置管理、CUDA渲染、控制面板、推理引擎、流水线（包含预处理、后处理、推理线程）、通用工具等独立模块
- **清晰的文件组织结构**：重新组织了include/和src/目录，按功能模块分类
  - include/common/ - 通用工具和内存管理
  - include/engine/ - 推理引擎
  - include/kernels/ - CUDA内核
  - include/pipeline/ - 流水线相关组件
  - src/common/ - 通用工具实现
  - src/engine/ - 推理引擎实现
  - src/kernels/ - CUDA内核实现
  - src/pipeline/ - 流水线组件实现
- **统一的接口设计**：为各模块提供标准化的接口，降低模块间耦合度

#### 渲染系统优化
- **CUDA-OpenGL互操作**：实现了GPU-GPU直接数据传输，消除了Host-to-Device带宽瓶颈
- **自定义CUDA渲染内核**：编写了专用的CUDA Kernels替代OpenCV算子，加速颜色映射与格式转换
- **实时渲染性能提升**：渲染延迟降低至微秒级，系统吞吐量从25imgs/s提升至30imgs/s

#### 推理流水线增强
- **多线程流水线**：实现了输入线程(InputThread)和推理线程(InferenceThread)的分离
- **任务队列管理**：使用安全队列(SafeQueue)实现线程间的任务传递
- **精度优化**：同时支持FP32和FP16精度，平衡精度与性能
- **内存管理优化**：预分配和复用GPU内存缓冲区，减少内存分配开销
- **CUDA流并行**：使用CUDA流重叠计算和内存拷贝操作

#### 用户界面升级
- **交互式控制面板**：基于ImGui实现了可视化配置界面
- **实时参数调节**：支持在运行时调整视频源、模型路径、推理模式、精度等参数
- **多视图显示**：同时展示原始视频、热力图和叠加结果

### 技术细节优化

#### 预处理模块
- **GPU加速预处理**：使用cv::cuda库实现图像缩放、颜色空间转换和归一化
- **多精度支持**：同时支持FP32和FP16精度的预处理
- **内存优化**：复用GPU内存缓冲区，减少内存分配次数

#### 后处理模块
- **输出缓冲区管理**：从推理引擎获取pred_score、pred_label、anomaly_map等输出
- **GPU-CPU数据传输**：将推理结果从GPU高效复制到CPU
- **掩码生成**：基于异常图和阈值生成二值掩码
- **覆盖层生成**：从掩码生成半透明红色覆盖层
- **颜色映射**：为异常图应用颜色映射，增强可视化效果
- **图像组合**：将原始图像与覆盖层组合，生成最终结果
- **图像缩放**：使用NPP库高效调整图像大小

#### 通用工具模块
- **内存管理**：实现了CUDA内存的安全分配和释放
- **日志系统**：提供了分级日志记录功能
- **线程安全**：实现了线程安全的队列和同步机制

### 功能扩展

#### 视频处理
- **多视频源支持**：支持视频文件和摄像头输入
- **实时推理**：实现视频流的实时推理和显示
- **缺陷检测**：自动标记异常区域并计算缺陷边界框

#### 模型管理
- **TensorRT引擎**：基于TrtEngine实现高效推理
- **自动引擎生成**：当没有engine文件时，自动从onnx文件生成
- **模型路径配置**：支持通过UI界面配置模型路径

### 性能对比

| 优化阶段 | 预处理时长(ms) | 推理时长(ms) | 后处理时长(ms) | 总时长(ms) | 吞吐量(imgs/s) |
|---------|--------------|------------|--------------|----------|--------------|
| 原始C++实现 | 11 | 17 | 5 | 33 | ~30 |
| 预处理DNN加速 | 8 | 17 | 5 | 30 | ~33 |
| 预处理CUDA加速 | 4 | 17 | 5 | 26 | ~38 |
| 后处理浅拷贝优化 | 4 | 17 | 3 | 24 | ~42 |
| 推理FP16加速 | 4 | 8 | 3 | 15 | ~67 |
| 渲染优化 | 4 | 8 | 3 | <15 | >80 |

#### 渲染优化详细说明
- **流水线化处理**：实现了输入、预处理、推理、后处理的全流水线化，最大化并行度
- **单次CPU到GPU拷贝**：优化内存传输，减少CPU到GPU的数据拷贝次数，降低带宽开销
- **OpenGL显存渲染**：利用CUDA-OpenGL互操作，实现显存内直接渲染，消除Host-to-Device瓶颈
- **纯CUDA算子实现**：使用自定义CUDA Kernels替代OpenCV算子，加速颜色映射与格式转换
- **内存缓冲区复用**：预分配和复用GPU内存缓冲区，减少内存分配开销
- **CUDA流并行**：使用CUDA流重叠计算和内存拷贝操作，提升整体性能

---

## 2025-12-06: 前端UI与可视化优化

### UI系统实现
- **原生C++ UI界面**：使用第三方库实现了简单的UI页面
- **独立UI设计文件**：新增dashboard.cpp，便于后续维护
- **视频线程重构**：修改video_thread.cpp，专注于后端实时更新图片输出
- **配置参数调节**：修改main.cpp，支持在UI中手动调节配置参数

### 可视化性能优化
- **渲染管线重构**：移除CPU端的所有图像后处理逻辑
- **CUDA-OpenGL互操作**：实现显存内数据直接流转
- **自定义CUDA Kernels**：替代OpenCV算子，利用GPU并行加速颜色映射与格式转换
- **消除带宽瓶颈**：彻底解决glTexImage2D带来的Host-to-Device带宽瓶颈，渲染延迟降至微秒级
- **吞吐量提升**：系统吞吐量从25imgs/s提升至30imgs/s

---

## 2025-11-15: 核心功能实现

### 基础架构搭建
- **环境配置**：实现了OpenCV、TensorRT、LibTorch、CUDA等库的配置
- **CMake配置**：编写了完整的CMakeLists.txt，支持VSCode编译
- **双模型TensorRT加速**：实现了学生-教师模型的TensorRT加速
- **预处理实现**：提供了三种预处理方式（OpenCV、OpenCVdnn、OpenCVcuda+核函数优化）
- **后处理实现**：在C++中利用LibTorch进行后处理
- **实时视频处理**：实现了视频流的实时处理和推理

### 性能优化
- **预处理优化**：从CPU预处理优化到CUDA加速预处理
- **后处理优化**：实现了浅拷贝优化，减少数据转移开销
- **推理优化**：支持FP16精度，提升推理速度

---

# Changelog (English)

## 2026-01-28: Architecture Refactoring and Performance Optimization

### Core Architecture Changes

#### Modular Design Refactoring
- **New Modular Architecture**: Divided the system into independent modules including configuration management, CUDA rendering, dashboard, inference engine, pipeline (preprocessing, postprocessing, inference threads), and common utilities
- **Clear File Organization**: Reorganized include/ and src/ directories, classified by functional modules
  - include/common/ - Common utilities and memory management
  - include/engine/ - Inference engine
  - include/kernels/ - CUDA kernels
  - include/pipeline/ - Pipeline components
  - src/common/ - Common utilities implementation
  - src/engine/ - Inference engine implementation
  - src/kernels/ - CUDA kernels implementation
  - src/pipeline/ - Pipeline components implementation
- **Unified Interface Design**: Provided standardized interfaces for each module, reducing inter-module coupling

#### Rendering System Optimization
- **CUDA-OpenGL Interoperability**: Implemented direct GPU-GPU data transfer, eliminating Host-to-Device bandwidth bottleneck
- **Custom CUDA Rendering Kernels**: Developed specialized CUDA Kernels to replace OpenCV operators, accelerating color mapping and format conversion
- **Real-time Rendering Performance**: Reduced rendering latency to microsecond level, increased system throughput from 25imgs/s to 30imgs/s

#### Inference Pipeline Enhancement
- **Multi-threaded Pipeline**: Implemented separation of input thread (InputThread) and inference thread (InferenceThread)
- **Task Queue Management**: Used safe queue (SafeQueue) for task passing between threads

- **Precision Optimization**: Supported both FP32 and FP16 precision, balancing accuracy and performance
- **Memory Management Optimization**: Pre-allocated and reused GPU memory buffers, reducing memory allocation overhead
- **CUDA Stream Parallelism**: Used CUDA streams to overlap computation and memory copy operations

#### User Interface Upgrade
- **Interactive Control Panel**: Implemented visual configuration interface based on ImGui
- **Real-time Parameter Adjustment**: Supported runtime adjustment of video source, model paths, inference mode, precision, etc.
- **Multi-view Display**: Simultaneously displayed original video, heatmap, and overlay results

### Technical Detail Optimization

#### Preprocessing Module
- **GPU-accelerated Preprocessing**: Used cv::cuda library for image resizing, color space conversion, and normalization
- **Multi-precision Support**: Supported both FP32 and FP16 precision preprocessing
- **Memory Optimization**: Reused GPU memory buffers, reducing memory allocation times

#### Postprocessing Module
- **Output Buffer Management**: Retrieved pred_score, pred_label, anomaly_map from inference engine
- **GPU-CPU Data Transfer**: Efficiently copied inference results from GPU to CPU
- **Mask Generation**: Generated binary mask based on anomaly map and threshold
- **Overlay Generation**: Created semi-transparent red overlay from mask
- **Color Mapping**: Applied color map to anomaly map for enhanced visualization
- **Image Composition**: Combined original image with overlay to generate final result
- **Image Resizing**: Used NPP library for efficient image resizing

#### Common Utilities Module
- **Memory Management**: Implemented safe CUDA memory allocation and release
- **Logging System**: Provided hierarchical logging functionality
- **Thread Safety**: Implemented thread-safe queue and synchronization mechanisms

### Feature Extensions

#### Video Processing
- **Multi-video Source Support**: Supported video files and camera inputs
- **Real-time Inference**: Implemented real-time inference and display of video streams
- **Defect Detection**: Automatically marked abnormal regions and calculated defect bounding boxes

#### Model Management
- **TensorRT Engine**: Based on TrtEngine for efficient inference
- **Automatic Engine Generation**: Automatically generated engine files from ONNX models when no engine files exist

- **Model Path Configuration**: Supported configuring model paths through UI interface

### Performance Comparison

| Optimization Stage | Preprocessing Time(ms) | Inference Time(ms) | Postprocessing Time(ms) | Total Time(ms) | Throughput(imgs/s) |
|-------------------|----------------------|-------------------|------------------------|---------------|-------------------|
| Initial C++ Implementation | 11 | 17 | 5 | 33 | ~30 |
| Preprocessing DNN Acceleration | 8 | 17 | 5 | 30 | ~33 |
| Preprocessing CUDA Acceleration | 4 | 17 | 5 | 26 | ~38 |
| Shallow Copy Optimization | 4 | 17 | 3 | 24 | ~42 |
| FP16 Inference Acceleration | 4 | 8 | 3 | 15 | ~67 |
| Rendering Optimization | 4 | 8 | 3 | <15 | >80 |

#### Rendering Optimization Details
- **Pipelined Processing**: Implemented full pipeline for input, preprocessing, inference, and postprocessing to maximize parallelism
- **Single CPU-to-GPU Copy**: Optimized memory transfer, reduced CPU-to-GPU data copy times to lower bandwidth overhead
- **OpenGL VRAM Rendering**: Utilized CUDA-OpenGL interoperability for direct VRAM rendering, eliminating Host-to-Device bottleneck
- **Pure CUDA Operator Implementation**: Used custom CUDA Kernels instead of OpenCV operators to accelerate color mapping and format conversion
- **Memory Buffer Reuse**: Pre-allocated and reused GPU memory buffers to reduce memory allocation overhead
- **CUDA Stream Parallelism**: Used CUDA streams to overlap computation and memory copy operations for improved overall performance

---

## 2025-12-06: Frontend UI and Visualization Optimization

### UI System Implementation
- **Native C++ UI Interface**: Implemented simple UI pages using third-party libraries
- **Independent UI Design File**: Added dashboard.cpp for easier maintenance
- **Video Thread Refactoring**: Modified video_thread.cpp to focus on backend real-time image output updates
- **Configuration Parameter Adjustment**: Modified main.cpp to support manual adjustment of configuration parameters in UI

### Visualization Performance Optimization
- **Rendering Pipeline Refactoring**: Removed all CPU-side image post-processing logic
- **CUDA-OpenGL Interoperability**: Implemented direct data flow within video memory
- **Custom CUDA Kernels**: Replaced OpenCV operators with GPU-parallelized color mapping and format conversion
- **Eliminated Bandwidth Bottleneck**: Completely resolved Host-to-Device bandwidth bottleneck caused by glTexImage2D, reducing rendering latency to microsecond level
- **Throughput Improvement**: Increased system throughput from 25imgs/s to 30imgs/s

---

## 2025-11-15: Core Functionality Implementation

### Basic Architecture Setup
- **Environment Configuration**: Implemented configuration for OpenCV, TensorRT, LibTorch, CUDA, etc.
- **CMake Configuration**: Wrote complete CMakeLists.txt, supporting VSCode compilation
- **Dual-model TensorRT Acceleration**: Implemented TensorRT acceleration for student-teacher models
- **Preprocessing Implementation**: Provided three preprocessing methods (OpenCV, OpenCVdnn, OpenCVcuda+kernel function optimization)
- **Postprocessing Implementation**: Used LibTorch for postprocessing in C++
- **Real-time Video Processing**: Implemented real-time processing and inference of video streams

### Performance Optimization
- **Preprocessing Optimization**: Optimized from CPU preprocessing to CUDA-accelerated preprocessing
- **Postprocessing Optimization**: Implemented shallow copy optimization, reducing data transfer overhead
- **Inference Optimization**: Supported FP16 precision, improving inference speed
