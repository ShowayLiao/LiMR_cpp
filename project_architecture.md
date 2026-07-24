# 项目架构文档

## 1. 项目概述

### 1.1 核心功能

inference_LiMR是一个基于TensorRT的实时异常检测系统，专门用于视频流分析。系统支持单引擎和双引擎两种推理模式，能够高效地生成异常热力图，并提供直观的用户界面进行配置和监控。

### 1.2 技术栈

| 技术领域 | 核心技术 | 用途 |
|---------|---------|------|
| 编程语言 | C++ | 主要开发语言 |
| GPU计算 | CUDA | 并行计算和内存管理 |
| 深度学习推理 | TensorRT | 模型加载和高效推理 |
| 图像处理 | OpenCV | 图像预处理、后处理和可视化 |
| 图形渲染 | OpenGL | 实时视频和热力图渲染 |
| 用户界面 | ImGui | 交互式控制面板 |
| 深度学习框架 | PyTorch | 模型转换和后处理算法 |

### 1.3 目标用户

- 工业质量检测工程师
- 实时监控系统开发人员
- 深度学习推理加速研究人员

### 1.4 应用场景

- 工业产品表面缺陷检测
- 实时视频监控异常分析
- 生产流水线质量控制
- 智能设备故障检测

## 2. 目录结构

```
inference_LiMR/
├── build/                    # 构建输出目录
│   ├── Debug/                # Debug 构建输出
│   └── Release/              # Release 构建输出
├── docs/                     # 文档目录
│   ├── cmake.md              # CMake 配置文档
│   ├── install.md            # 安装说明
│   └── 安装方法.md            # 中文安装说明
├── images/                   # 图片资源
├── include/                  # 头文件目录
│   ├── app_config.h          # 应用配置定义
│   ├── cuda_render.h         # CUDA 渲染接口
│   ├── dashboard.h           # 控制面板接口
│   ├── inference.h           # 推理引擎接口
│   ├── inference_runner.h    # 推理流水线接口
│   ├── pipeline.h            # 图像处理流水线接口
│   ├── postprocess.h         # 后处理接口
│   ├── preprocess.h          # 预处理接口
│   ├── utils.cuh             # CUDA 工具函数头文件
│   ├── utils.h               # 工具函数头文件
│   └── video_thread.h        # 视频线程接口
├── src/                      # 源文件目录
│   ├── app_config.cpp        # 应用配置实现
│   ├── cuda_render.cu        # CUDA 渲染实现
│   ├── dashboard.cpp         # 控制面板实现
│   ├── inference.cpp         # 推理引擎实现
│   ├── inference_runner.cpp  # 推理流水线实现
│   ├── main.cpp              # 主程序入口
│   ├── pipeline.cpp          # 图像处理流水线实现
│   ├── postprocess.cpp       # 后处理实现
│   ├── preprocess.cpp        # 预处理实现
│   ├── utils.cpp             # 工具函数实现
│   ├── utils.cu              # CUDA 工具函数实现
│   └── video_thread.cpp      # 视频线程实现
├── .gitignore                # Git 忽略文件
├── CMakeLists.txt            # CMake 构建脚本
├── LICENSE                   # 许可证文件
└── README.md                 # 项目说明文档
```

## 3. 模块分解与职责

### 3.1 配置管理模块

**主要职责**：管理应用程序的配置参数，包括视频源、模型路径、推理模式等。

**核心组件**：
- `AppConfig`类（`include/app_config.h`）：定义应用配置结构体和接口

**关键功能**：
- 支持单引擎和双引擎两种推理模式
- 管理视频源、模型路径、精度设置等参数
- 提供配置参数的访问接口

### 3.2 CUDA渲染模块

**主要职责**：实现GPU加速的图像渲染，支持视频帧和热力图的高效显示。

**核心组件**：
- `CudaInteropTexture`类（`include/cuda_render.h`）：封装CUDA-OpenGL互操作纹理

**关键功能**：
- 管理CUDA-OpenGL互操作纹理
- 实现高效的GPU-GPU数据传输
- 支持热力图和视频帧的渲染
- 提供纹理ID访问接口

### 3.3 控制面板模块

**主要职责**：提供交互式用户界面，用于配置系统参数、监控运行状态和显示结果。

**核心组件**：
- `Dashboard`类（`include/dashboard.h`）：实现控制面板界面和逻辑

**关键功能**：
- 提供视频源和模型路径配置
- 支持推理模式和精度设置
- 显示原始视频、热力图和叠加结果
- 管理系统初始化和运行状态

### 3.4 推理流水线模块

**主要职责**：协调预处理、推理和后处理模块，实现端到端的异常检测。

**核心组件**：
- `InferenceRunner`类（`include/inference_runner.h`）：推理流水线的核心协调者

**关键功能**：
- 初始化预处理、推理引擎和后处理模块
- 协调各模块的执行顺序
- 管理GPU内存缓冲区
- 支持单引擎和双引擎推理模式
- 生成和保存热力图
- 提供性能统计信息

### 3.5 预处理模块

**主要职责**：对输入图像进行预处理，使其符合模型输入要求。

**核心组件**：
- `Preprocessor`类（`include/preprocess.h`）：实现图像预处理算法

**关键功能**：
- 图像缩放和裁剪
- 颜色空间转换
- 归一化处理
- 支持FP32和FP16精度
- GPU加速处理

### 3.6 推理引擎模块

**主要职责**：封装TensorRT推理功能，实现模型加载和高效推理。

**核心组件**：
- `InferenceEngine`类（`include/inference.h`）：封装TensorRT推理引擎

**关键功能**：
- 加载和管理TensorRT引擎
- 分配和管理推理缓冲区
- 执行推理计算
- 提供输出张量访问接口
- 支持多绑定输出

### 3.7 后处理模块

**主要职责**：对模型输出进行处理，生成异常热力图。

**核心组件**：
- `Postprocessor`类（`include/postprocess.h`）：实现后处理算法

**关键功能**：
- 计算特征图的余弦相似度
- 生成异常热力图
- 支持多尺度特征融合
- 计算异常分数和位置
- GPU加速处理

### 3.8 视频线程模块

**主要职责**：处理视频流的捕获、推理和结果输出。

**核心组件**：
- `VideoCaptureThread`类（`include/video_thread.h`）：实现视频流处理

**关键功能**：
- 视频源的打开和关闭
- 帧捕获和预处理
- 推理执行和结果处理
- 异常区域检测和标记
- 纹理更新和渲染

### 3.9 工具函数模块

**主要职责**：提供各种辅助功能，支持系统的正常运行。

**核心组件**：
- 各种工具函数（`include/utils.h`、`include/utils.cuh`）

**关键功能**：
- ONNX模型转TensorRT引擎
- 异常图计算
- 数据类型转换
- 性能优化

## 4. 文件关系与依赖

### 4.1 模块依赖关系

```
┌─────────────────┐
│   Dashboard     │
└────────┬────────┘
         │
┌────────▼────────┐
│ VideoThread     │
└────────┬────────┘
         │
┌────────▼────────┐
│ InferenceRunner │
└────────┬────────┘
         │
┌────────┼────────┬────────────────┐
│        │        │                │
▼        ▼        ▼                ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│Preprocess│ │Inference│ │Postprocess│ │CudaRender│
└─────────┘ └─────────┘ └─────────┘ └─────────┘
         │
         ▼
┌─────────┐
│  Utils  │
└─────────┘
```

### 4.2 关键文件依赖关系

| 文件 | 主要依赖 | 功能定位 |
|------|----------|----------|
| `src/app_config.cpp` | `app_config.h` | 配置管理 |
| `src/cuda_render.cu` | `cuda_render.h` | CUDA渲染 |
| `src/dashboard.cpp` | `dashboard.h`, `video_thread.h`, `app_config.h`, `utils.h` | 控制面板 |
| `src/inference_runner.cpp` | `inference_runner.h`, `preprocess.h`, `inference.h`, `postprocess.h` | 推理流水线 |
| `src/inference.cpp` | `inference.h` | 推理引擎 |
| `src/pipeline.cpp` | `pipeline.h` | 图像处理流水线 |
| `src/postprocess.cpp` | `postprocess.h` | 后处理 |
| `src/preprocess.cpp` | `preprocess.h` | 预处理 |
| `src/utils.cpp` | `utils.h` | 工具函数 |
| `src/utils.cu` | `utils.cuh` | CUDA工具函数 |
| `src/video_thread.cpp` | `video_thread.h`, `cuda_render.h`, `inference_runner.h` | 视频线程 |

## 5. 核心组件详解

### 5.1 应用配置（AppConfig）

**功能定位**：管理应用程序的配置参数，包括视频源、模型路径、推理模式等。

**接口设计**：

```cpp
enum class InferenceMode {
    SINGLE_ENGINE,  // 单引擎模式
    DUAL_ENGINE     // 双引擎模式
};

class AppConfig {
public:
    AppConfig(const std::string& videoSource = "", 
              const std::string& enginePathA = "", 
              const std::string& enginePathB = "",
              const std::string& imgSource = "",
              const std::string& type = "F32",
              InferenceMode inferenceMode = InferenceMode::SINGLE_ENGINE,
              int batchSize = 1, 
              int inputWidth = 224, 
              int inputHeight = 224, 
              int outputWidth = 224, 
              int outputHeight = 224,
              float heatmapAlpha = 0.5f,
              int colormapType = 2);
    
    // 公共成员变量，直接访问配置参数
    std::string videoSource;
    std::string enginePathA;
    std::string enginePathB;
    // ... 其他配置参数
};
```

**关键实现细节**：
- 使用枚举类型定义推理模式，支持单引擎和双引擎
- 提供默认构造函数，设置合理的默认值
- 公共成员变量设计，便于外部访问和修改

### 5.2 CUDA渲染（CudaInteropTexture）

**功能定位**：实现GPU加速的图像渲染，支持视频帧和热力图的高效显示。

**接口设计**：

```cpp
class CudaInteropTexture {
public:
    CudaInteropTexture();
    ~CudaInteropTexture();
    void init(int w, int h);
    void cleanup();
    GLuint getTextureID() const { return texture_id; }
    void upload_and_render_heatmap(const float* cpu_ptr, size_t size_bytes);
    void upload_and_render_frame(const unsigned char* cpu_ptr, size_t size_bytes);
    
private:
    GLuint texture_id;
    struct cudaGraphicsResource* cuda_res;
    void* d_buffer;      // 接收CPU数据的Buffer
    void* d_rgba_buffer; // 渲染结果Buffer
    int width, height;
};
```

**关键实现细节**：
- 使用CUDA-OpenGL互操作纹理，实现高效的GPU-GPU数据传输
- 支持两种渲染模式：热力图和视频帧
- 内部管理CUDA和OpenGL资源，确保正确释放
- 提供纹理ID访问接口，便于外部渲染

### 5.3 控制面板（Dashboard）

**功能定位**：提供交互式用户界面，用于配置系统参数、监控运行状态和显示结果。

**接口设计**：

```cpp
class Dashboard {
public:
    Dashboard();
    ~Dashboard();
    void Render(int display_w, int display_h);
    void UpdateTextures();
    
private:
    void SetupStyle();
    void DrawSidePanel(float width, float height);
    void DrawMainView(float start_x, float width, float height);
    
    // 核心业务对象
    std::unique_ptr<AppConfig> appConfig;
    std::unique_ptr<VideoThread::VideoCaptureThread> videoProcessor;
    
    // UI状态变量
    bool is_initialized = false;
    bool is_running = false;
    
    // 纹理句柄
    unsigned int tex_frame = 0;   // 原始图
    unsigned int tex_heatmap = 0; // 纯热力图
    unsigned int tex_overlay = 0; // 最终结果图
    
    // 输入缓存
    char video_path[256] = "../../input/blade.avi";
    char engine_path_a[256] = "../../input/LiMR_student_16.engine";
    char engine_path_b[256] = "../../input/LiMR_teacher_16.engine";
    
    // 精度选择
    int current_precision_idx = 0;
    const char* precision_items[2] = { "F32", "F16 " };
    
    // 推理模式选择
    int inference_mode_idx = 0;
    const char* inference_mode_items[2] = { "Single Engine", "Dual Engine" };
    
    float defect_threshold = 0.5f;
};
```

**关键实现细节**：
- 使用ImGui库实现交互式UI
- 分为侧边栏和主视图两部分
- 侧边栏用于配置参数，主视图用于显示结果
- 支持实时更新纹理和状态
- 管理视频处理线程的生命周期

### 5.4 推理流水线（InferenceRunner）

**功能定位**：协调预处理、推理和后处理模块，实现端到端的异常检测。

**接口设计**：

```cpp
class InferenceRunner {
public:
    explicit InferenceRunner(InferenceMode mode);
    ~InferenceRunner();
    
    bool init(const std::string& enginePathA, 
              const std::string& enginePathB = "", 
              int inputWidth = 256, 
              int inputHeight = 256,
              int outputWidth = 256,
              int outputHeight = 256,
              float scale = 1.0f / 255.0f,
              float mean = 0.5f);
    
    cv::Mat infer(const cv::Mat& inputImage);
    bool saveHeatmap(const cv::Mat& heatmap, const std::string& path) const;
    void printTimingStats() const;
    
private:
    InferenceMode mode_;                      // 推理模式
    Preprocessor preprocessor_;                // 预处理
    std::unique_ptr<InferenceEngine> engineA_; // 引擎A
    std::unique_ptr<InferenceEngine> engineB_; // 引擎B（仅双引擎模式使用）
    Postprocessor postprocessor_;              // 后处理
    
    // 输入输出尺寸
    int inputWidth_;
    int inputHeight_;
    int outputWidth_;
    int outputHeight_;
    
    // GPU内存缓冲区
    void* d_inputBuffer_;                     // 输入缓冲区
    void* d_anomalyMap_;                      // 异常图缓冲区
    
    // 缓冲区大小
    size_t inputBufferSize_;
    size_t anomalyMapSize_;
    
    // 计时变量
    std::chrono::high_resolution_clock::time_point preprocessStart_;
    std::chrono::high_resolution_clock::time_point preprocessEnd_;
    std::chrono::high_resolution_clock::time_point inferStart_;
    std::chrono::high_resolution_clock::time_point inferEnd_;
    std::chrono::high_resolution_clock::time_point postprocessStart_;
    std::chrono::high_resolution_clock::time_point postprocessEnd_;
    std::chrono::high_resolution_clock::time_point totalStart_;
    std::chrono::high_resolution_clock::time_point totalEnd_;
    
    // 辅助函数
    void freeGPUBuffers();
    cv::Mat generateHeatmapFromAnomalyMap(float* d_anomalyMap);
};
```

**关键实现细节**：
- 支持单引擎和双引擎两种推理模式
- 初始化时创建和配置各个模块
- 推理时协调预处理、推理和后处理的执行顺序
- 管理GPU内存缓冲区，避免重复分配
- 提供性能统计信息，便于优化
- 生成热力图并支持保存

### 5.5 预处理/后处理

**预处理（Preprocessor）**

```cpp
namespace preprocess {
    class Preprocessor {
    public:
        Preprocessor();
        ~Preprocessor();
        
        bool init(int inputWidth, int inputHeight, float scale, float mean, const std::string& type = "F32");
        void preprocess(const cv::Mat& inputFrame, void*& outputGpuPtr);
        
    private:
        int inputWidth_;
        int inputHeight_;
        std::string type_;
        cudaStream_t stream_;
        float scale_;
        float mean_[3];
        float std_[3];
        
        // GPU内存缓冲区
        cv::cuda::GpuMat gpuFrame_;
        cv::cuda::GpuMat gpuResized_;
        cv::cuda::GpuMat gpuConverted_;
        cv::cuda::GpuMat gpuNormalized_;
        cv::cuda::GpuMat gpuBlob_;
        void* outputBuffer_;
        size_t outputBufferSize_;
    };
}
```

**后处理（Postprocessor）**

```cpp
namespace postprocess {
    struct AnomalyResult {
        float* anomalyMap; // 异常图GPU指针
        float maxScore;    // 最大异常分数
        cv::Point maxLoc;  // 最大异常分数位置
    };
    
    class Postprocessor {
    public:
        Postprocessor();
        ~Postprocessor();
        
        bool init(int outputHeight = 256, int outputWidth = 256, const std::string& type = "F32");
        void postprocess(float* d_outputA, float* d_anomalyMap, int outputWidth, int outputHeight);
        void postprocess(float* d_outputA, float* d_outputB, float* d_anomalyMap, int outputWidth, int outputHeight);
        void process_and_accumulate(float* student_feature, float* teacher_feature, float* output_anomaly_map, int feature_c, int feature_h, int feature_w, int out_h, int out_w);
        
        AnomalyResult getAnomalyResult() const;
        
    private:
        int outputHeight_;
        int outputWidth_;
        int numChannels_;
        int featureHeight_;
        int featureWidth_;
        std::string type_;
        cudaStream_t stream_;
        
        AnomalyResult result_;
        
        // 辅助函数
        torch::Tensor calculateCosineSimilarity(const torch::Tensor& studentTensor, const torch::Tensor& teacherTensor);
        torch::Tensor resizeTensor(const torch::Tensor& tensor, int height, int width);
    };
}
```

**关键实现细节**：
- 预处理使用CUDA加速，支持FP32和FP16精度
- 后处理实现了余弦相似度计算和异常图生成
- 支持多尺度特征融合和累加
- 双引擎模式下比较学生模型和教师模型的输出
- 提供异常结果的访问接口

### 5.6 工具函数（Utilities）

**关键工具函数**：

```cpp
// ONNX模型转TensorRT引擎
void build_engine(const std::string& onnxModelPath,
                  const std::string& engineFilePath,
                  int batchSize = 1,
                  int inputWidth = 224,
                  int inputHeight = 224,
                  int outputWidth = 224,
                  int outputHeight = 224);

// 异常图计算
void cal_anomaly_map(std::vector<torch::Tensor>& StuOutputTensors,
                     std::vector<torch::Tensor>& TeaOutputTensors,
                     torch::Tensor& anomalyMap);

// 半精度浮点数联合体
typedef union {
    struct {
        uint16_t mantissa : 10;  // 10位尾数
        uint16_t exponent : 5;   // 5位指数
        uint16_t sign     : 1;    // 1位符号
    } bits;
    uint16_t raw_value;          // 16位原始值
} fp16;
```

**关键实现细节**：
- 提供ONNX模型转TensorRT引擎的功能
- 实现异常图计算算法
- 支持半精度浮点数操作
- 提供性能优化的辅助函数

### 5.7 视频线程（VideoThread）

```cpp
namespace VideoThread {
    class VideoCaptureThread {
    public:
        VideoCaptureThread(const AppConfig& videoConfig);
        ~VideoCaptureThread();
        void stop();
        
        bool update(float threshold);
        std::vector<cv::Rect> getDefectRects() const { return defect_rects; }
        
        // 直接返回互操作纹理ID
        unsigned int getResultFrameTexture() const { return cuda_tex_frame.getTextureID(); }
        unsigned int getResultHeatmapTexture() const { return cuda_tex_heatmap.getTextureID(); }
        
        bool isOpened() const { return cap.isOpened(); }
        
    private:
        cv::VideoCapture cap;
        std::string source;
        AppConfig config;
        std::unique_ptr<InferenceRunner> runner;
        
        // 两个互操作纹理对象
        CudaInteropTexture cuda_tex_frame;   // 原始图像
        CudaInteropTexture cuda_tex_heatmap; // 热力图
        
        std::vector<cv::Rect> defect_rects;
    };
}
```

**关键实现细节**：
- 管理视频流的捕获和处理
- 初始化推理引擎和渲染组件
- 实现帧的预处理、推理和后处理
- 支持异常区域检测和标记
- 提供纹理ID访问接口，便于外部渲染
- 支持线程安全的停止操作

## 6. 构建流程

### 6.1 编译环境要求

- Windows 10/11
- Visual Studio 2019/2022
- CUDA 11.6+
- TensorRT 8.0+
- OpenCV 4.5+
- CMake 3.15+

### 6.2 依赖项安装

1. **CUDA Toolkit**：从NVIDIA官网下载并安装
2. **TensorRT**：从NVIDIA官网下载并解压，设置环境变量
3. **OpenCV**：从OpenCV官网下载并安装，设置环境变量
4. **CMake**：从CMake官网下载并安装
5. **Visual Studio**：安装带有C++桌面开发和CUDA开发工具包的版本

### 6.3 构建脚本使用

```bash
# 创建构建目录
mkdir build
cd build

# 生成Visual Studio解决方案
cmake .. -G "Visual Studio 16 2019" -A x64

# 编译项目
cmake --build . --config Release
```

### 6.4 编译选项配置

| 编译选项 | 说明 | 默认值 |
|---------|------|--------|
| `CMAKE_BUILD_TYPE` | 构建类型 | Release |
| `CUDA_TOOLKIT_ROOT_DIR` | CUDA安装路径 | 自动检测 |
| `TENSORRT_ROOT` | TensorRT安装路径 | 自动检测 |
| `OpenCV_DIR` | OpenCV安装路径 | 自动检测 |
| `CMAKE_CUDA_ARCHITECTURES` | CUDA架构 | 自动检测 |

## 7. 使用说明

### 7.1 环境配置

1. 确保所有依赖项已正确安装
2. 设置以下环境变量：
   - `CUDA_PATH`：指向CUDA安装目录
   - `TENSORRT_DIR`：指向TensorRT安装目录
   - `OPENCV_DIR`：指向OpenCV安装目录
   - 将`%TENSORRT_DIR%/lib`和`%OPENCV_DIR%/x64/vc16/bin`添加到`PATH`环境变量

### 7.2 启动命令

```bash
# 运行可执行文件
cd build/Release
limr.exe
```

### 7.3 参数配置

1. **视频源**：在控制面板侧边栏输入视频文件路径或摄像头ID
2. **模型路径**：输入学生模型（engine_path_a）和教师模型（engine_path_b）的路径
3. **推理模式**：选择单引擎或双引擎模式
4. **精度**：选择F32或F16精度
5. **缺陷阈值**：调整异常检测的阈值

### 7.4 常见问题排查

1. **无法打开视频源**：
   - 检查视频文件路径是否正确
   - 检查摄像头是否可用
   - 确保视频格式受支持

2. **模型加载失败**：
   - 检查引擎文件路径是否正确
   - 确保引擎文件与当前CUDA版本兼容
   - 检查引擎文件是否完整

3. **性能问题**：
   - 尝试降低输入分辨率
   - 使用F16精度
   - 检查GPU利用率，确保没有其他进程占用大量GPU资源

4. **异常图全蓝或全红**：
   - 检查模型输出是否正确
   - 调整缺陷阈值
   - 检查预处理和后处理参数

## 8. 代码优化建议

### 8.1 性能优化

1. **内存管理**：
   - 减少CPU-GPU内存拷贝
   - 使用统一内存（Unified Memory）
   - 预分配和复用缓冲区

2. **并行计算**：
   - 使用CUDA流重叠计算和内存拷贝
   - 优化CUDA内核函数
   - 考虑使用TensorRT的INT8量化

3. **算法优化**：
   - 优化后处理算法，减少计算量
   - 考虑使用更高效的异常图生成方法
   - 优化多尺度特征融合策略

### 8.2 代码质量

1. **错误处理**：
   - 增强错误检查和日志记录
   - 提供更详细的错误信息
   - 考虑使用异常处理机制

2. **代码结构**：
   - 进一步模块化代码，减少耦合
   - 提取公共接口，便于扩展
   - 考虑使用设计模式提高代码可维护性

3. **文档**：
   - 增加代码注释
   - 完善API文档
   - 提供更详细的使用示例

### 8.3 功能扩展

1. **支持更多模型格式**：
   - 直接支持ONNX模型
   - 支持其他深度学习框架的模型

2. **增强可视化功能**：
   - 添加更多可视化选项
   - 支持保存结果视频
   - 提供更丰富的统计信息

3. **支持更多应用场景**：
   - 支持图像序列处理
   - 添加批量处理功能
   - 考虑支持实时摄像头流

## 9. 总结

inference_LiMR是一个功能完整、性能高效的实时异常检测系统，基于TensorRT和CUDA实现了快速推理。系统采用模块化设计，便于扩展和维护，支持单引擎和双引擎两种推理模式，提供了直观的用户界面。

该项目在工业质量检测、实时监控等领域具有广阔的应用前景。通过进一步优化性能、增强功能和提高代码质量，可以使系统更加完善，满足更多复杂场景的需求。

## 10. 参考文献

1. NVIDIA TensorRT Documentation: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html
2. OpenCV Documentation: https://docs.opencv.org/4.x/
3. CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
4. ImGui Documentation: https://github.com/ocornut/imgui/wiki
5. anomalib: https://github.com/openvinotoolkit/anomalib
