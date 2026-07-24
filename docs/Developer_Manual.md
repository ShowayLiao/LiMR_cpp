# Developer's Manual

## 1. Project Overview

### 1.1 Core Features

inference_LiMR is a TensorRT-based real-time anomaly detection system specifically designed for video stream analysis. The system supports both single-engine and dual-engine inference modes, capable of efficiently generating anomaly heatmaps and providing an intuitive user interface for configuration and monitoring.

### 1.2 Technology Stack

| Technology Area | Core Technology | Purpose |
|----------------|----------------|--------|
| Programming Language | C++ | Main development language |
| GPU Computing | CUDA | Parallel computing and memory management |
| Deep Learning Inference | TensorRT | Model loading and efficient inference |
| Image Processing | OpenCV | Image preprocessing, postprocessing, and visualization |
| Graphics Rendering | OpenGL | Real-time video and heatmap rendering |
| User Interface | ImGui | Interactive control panel |

### 1.3 Target Users

- Industrial quality inspection engineers
- Real-time monitoring system developers
- Deep learning inference acceleration researchers
- System integration engineers

### 1.4 Application Scenarios

- Industrial product surface defect detection
- Real-time video surveillance anomaly analysis
- Production line quality control
- Intelligent device fault detection

## 2. Directory Structure

```
inference_LiMR/
├── build/                    # Build output directory
│   ├── Debug/                # Debug build output
│   └── Release/              # Release build output
├── docs/                     # Documentation directory
│   ├── cmake.md              # CMake configuration documentation
│   ├── install.md            # Installation instructions
│   └── 安装方法.md            # Chinese installation instructions
├── images/                   # Image resources
├── include/                  # Header files directory
│   ├── common/               # Common components
│   │   ├── CudaMemory.hpp    # CUDA memory management
│   │   ├── Logger.h          # Logging utility
│   │   └── SafeQueue.hpp     # Thread-safe queue
│   ├── engine/               # Inference engine
│   │   └── TrtEngine.h       # TensorRT engine encapsulation
│   ├── kernels/              # CUDA kernels
│   │   └── cuda_utils.cuh    # CUDA utility functions
│   ├── pipeline/             # Pipeline components
│   │   ├── FrameTask.h       # Frame task definition
│   │   ├── InferenceThread.h # Inference thread
│   │   ├── InputThread.h     # Input thread
│   │   ├── Pipeline.h        # Pipeline management
│   │   ├── PostProcessor.h   # Post-processor
│   │   └── Preprocessor.h    # Pre-processor
│   ├── app_config.h          # Application configuration definition
│   ├── cuda_render.h         # CUDA rendering interface
│   └── dashboard.h           # Control panel interface
├── src/                      # Source files directory
│   ├── common/               # Common components implementation
│   │   └── kernels.cu        # Common CUDA kernels
│   ├── engine/               # Inference engine implementation
│   │   └── trt_engine.cpp    # TensorRT engine implementation
│   ├── kernels/              # CUDA kernels implementation
│   │   ├── postprocess.cu               # Postprocessing CUDA implementation
│   │   ├── postprocess_heatmap.cu       # Heatmap postprocessing
│   │   ├── postprocess_mask.cu          # Mask postprocessing
│   │   ├── postprocess_mask_surface.cu  # Surface mask postprocessing
│   │   └── preprocess.cu                # Preprocessing CUDA implementation
│   ├── pipeline/             # Pipeline components implementation
│   │   ├── InferenceThread.cpp  # Inference thread implementation
│   │   ├── InputThread.cpp      # Input thread implementation
│   │   ├── Pipeline.cpp         # Pipeline management implementation
│   │   ├── postprocessor.cpp    # Post-processor implementation
│   │   └── preprocessor.cpp     # Pre-processor implementation
│   ├── app_config.cpp        # Application configuration implementation
│   ├── cuda_render.cu        # CUDA rendering implementation
│   ├── dashboard.cpp         # Control panel implementation
│   └── main.cpp              # Main program entry
├── .gitignore                # Git ignore file
├── CMakeLists.txt            # CMake build script
├── LICENSE                   # License file
└── README.md                 # Project description document
```

## 3. Pipeline Architecture Details

### 3.1 Overall Architecture

inference_LiMR adopts a multi-threaded pipeline architecture, implementing a complete flow from video capture to inference and visualization. The system is mainly composed of the following core modules:

1. **Input Thread**: Responsible for reading frame data from the video source and performing initial processing
2. **Inference Thread**: Executes model inference and related processing
3. **Preprocessor**: Performs scaling, color space conversion, and normalization on frame data
4. **PostProcessor**: Generates anomaly heatmaps and overlays them on original images
5. **CUDA Rendering Module**: Implements GPU-accelerated image rendering
6. **User Interface**: Provides interactive control and result display

### 3.2 Data Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Input Thread    │────>│ Inference Thread│────>│ PostProcessor   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         ↑                       ↑                       │
         │                       │                       │
         └───────────────────────┘───────────────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ CUDA Rendering  │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ User Interface  │
                     └─────────────────┘
```

### 3.3 Key Processes

1. **Video Frame Capture**: Read frame data from video files or cameras through OpenCV
2. **Preprocessing**: Perform scaling, color space conversion, and normalization on frame data
3. **Model Inference**: Execute deep learning model inference using TensorRT
4. **Postprocessing**: Generate anomaly heatmaps and overlay them on original images
5. **Rendering Display**: Implement efficient rendering using OpenGL and CUDA interop technology
6. **User Interaction**: Provide real-time parameter adjustment and result display through ImGui

## 4. Technical Highlights

### 4.1 GPU Acceleration

- **CUDA Core Optimization**: Use CUDA kernels for efficient image processing and computation
- **GPU Memory Management**: Adopt DeviceBuffer to manage GPU memory, reducing memory leak risks
- **CUDA-OpenGL Interoperability**: Implement zero-copy GPU-GPU data transfer for improved rendering performance
- **NPP Library Acceleration**: Use NVIDIA Performance Primitives library for optimized image processing

### 4.2 Real-time Performance

- **Multi-threaded Design**: Separate video capture and processing to improve system response speed
- **TensorRT Inference**: Utilize TensorRT's optimization capabilities for efficient model inference
- **Memory Reuse**: Pre-allocate and reuse GPU memory buffers to reduce memory allocation overhead
- **Pipeline Parallelism**: Use CUDA streams to achieve parallel execution of computation and memory copy

### 4.3 Flexible Configuration

- **Dual-engine Mode**: Currently does not support student-teacher model dual-engine inference mode
- **Dynamic Resolution**: Support real-time adjustment of rendering resolution
- **Multi-precision Support**: Support both FP32 and FP16 precision model inference
- **Configurable Parameters**: Real-time adjustment of various system parameters through UI interface

### 4.4 Visualization Technology

- **Three-view Layout**: Real-time display of original image, heatmap, and overlay result in three views
- **Semi-transparent Overlay**: Use CUDA to implement efficient semi-transparent heatmap overlay
- **Dynamic Color Mapping**: Support multiple color mapping schemes to enhance visualization of anomaly regions
- **Real-time Threshold Adjustment**: Adjust anomaly detection threshold in real-time through slider

## 5. Core Component Implementation Details

### 5.1 Application Configuration (AppConfig)

**Function Positioning**: Manage application configuration parameters, including video source, model path, inference mode, etc.

**Key Implementation**:
- Support both single-engine and dual-engine inference modes
- Manage video source, model path, precision settings, and other parameters
- Provide direct access interface for configuration parameters

**Code Location**:
- `include/app_config.h`: Define application configuration structure and interface
- `src/app_config.cpp`: Implement configuration management functionality

### 5.2 CUDA Rendering (CudaInteropTexture)

**Function Positioning**: Implement GPU-accelerated image rendering, supporting efficient display of video frames and heatmaps.

**Key Implementation**:
- Use CUDA-OpenGL interop textures to implement zero-copy GPU-GPU data transfer
- Support two rendering modes: heatmap and video frame
- Internally manage CUDA and OpenGL resources to ensure proper release
- Provide texture ID access interface for external rendering

**Code Location**:
- `include/cuda_render.h`: Define CUDA rendering interface
- `src/cuda_render.cu`: Implement CUDA rendering functionality

### 5.3 Control Panel (Dashboard)

**Function Positioning**: Provide interactive user interface for configuring system parameters, monitoring operation status, and displaying results.

**Key Implementation**:
- Use ImGui library to implement interactive UI
- Divided into sidebar and main view: sidebar for parameter configuration, main view for result display
- Support real-time texture and status updates
- Manage video processing thread lifecycle
- Implement resolution adjustment functionality

**Code Location**:
- `include/dashboard.h`: Define control panel interface
- `src/dashboard.cpp`: Implement control panel functionality

### 5.4 Input Thread (InputThread)

**Function Positioning**: Responsible for reading frame data from the video source and performing initial processing.

**Key Implementation**:
- Manage video stream capture and initial processing
- Implement frame reading and buffering
- Support video files and camera input
- Provide thread-safe operation interface

**Code Location**:
- `include/pipeline/InputThread.h`: Define input thread interface
- `src/pipeline/InputThread.cpp`: Implement input thread functionality

### 5.5 Inference Thread (InferenceThread)

**Function Positioning**: Execute model inference and related processing, coordinate preprocessing and postprocessing modules.

**Key Implementation**:
- Support both single-engine and dual-engine inference modes
- Create and configure various modules during initialization
- Coordinate execution order of preprocessing, inference, and postprocessing
- Manage GPU memory buffers to avoid repeated allocation
- Provide performance statistics for optimization

**Code Location**:
- `include/pipeline/InferenceThread.h`: Define inference thread interface
- `src/pipeline/InferenceThread.cpp`: Implement inference thread functionality

### 5.6 Preprocessor

**Function Positioning**: Perform scaling, color space conversion, and normalization on frame data.

**Key Implementation**:
- Use CUDA acceleration, support both FP32 and FP16 precision
- Implement image scaling, color space conversion, and normalization
- Manage GPU memory buffers to improve processing efficiency

**Code Location**:
- `include/pipeline/Preprocessor.h`: Define preprocessing interface
- `src/pipeline/preprocessor.cpp`: Implement preprocessing functionality
- `src/kernels/preprocess.cu`: Implement preprocessing CUDA kernels

### 5.7 PostProcessor

**Function Positioning**: Generate anomaly heatmaps and overlay them on original images.

**Key Implementation**:
- Implement cosine similarity calculation and anomaly map generation
- Support multi-scale feature fusion and accumulation
- Compare student model and teacher model outputs in dual-engine mode
- Provide access interface for anomaly results
- Use CUDA to implement efficient mask generation and image overlay

**Code Location**:
- `include/pipeline/PostProcessor.h`: Define postprocessing interface
- `src/pipeline/postprocessor.cpp`: Implement postprocessor functionality
- `src/kernels/postprocess.cu`: Implement postprocessing CUDA kernels
- `src/kernels/postprocess_heatmap.cu`: Implement heatmap postprocessing
- `src/kernels/postprocess_mask.cu`: Implement mask postprocessing
- `src/kernels/postprocess_mask_surface.cu`: Implement surface mask postprocessing

### 5.8 TensorRT Engine (TrtEngine)

**Function Positioning**: Encapsulate TensorRT inference functionality, implement model loading and efficient inference.

**Key Implementation**:
- Load and manage TensorRT engine
- Allocate and manage inference buffers
- Execute inference computation
- Provide output tensor access interface
- Support multi-binding output

**Code Location**:
- `include/engine/TrtEngine.h`: Define TensorRT engine interface
- `src/engine/trt_engine.cpp`: Implement TensorRT engine functionality

### 5.9 Common Components

**Function Positioning**: Provide various auxiliary functions to support the normal operation of the system.

**Key Implementation**:
- CUDA memory management (CudaMemory.hpp)
- Logging utility (Logger.h)
- Thread-safe queue (SafeQueue.hpp)
- Common CUDA kernels implementation

**Code Location**:
- `include/common/`: Common components headers
- `src/common/`: Common components implementation


## 6. UI Development and Usage

### 6.1 Interface Layout

The system interface adopts a three-panel layout design:

1. **Left Sidebar**: Parameter configuration area, including video source, model path, inference mode, etc.
2. **Right Main View**: Result display area, containing three sub-views:
   - Original video frame
   - Anomaly heatmap
   - Overlay result map
3. **Bottom Control Bar**: Contains start/stop, pause/resume, and other control buttons

### 6.2 Core Functions

- **Video Source Configuration**: Support video files, camera input, and single image input
- **Model Configuration**: Support both single-engine and dual-engine modes
- **Precision Selection**: Support both F32 and F16 precision
- **Threshold Adjustment**: Real-time adjustment of anomaly detection threshold
- **Resolution Settings**: Support multiple preset resolutions to adapt to different display needs
- **Real-time Monitoring**: Real-time display of processing results and system status

### 6.3 Usage Flow

1. **Configure Parameters**: Input video source path, model path, and other parameters in the left sidebar
2. **Start System**: Click the "Apply & Reload" button to start the system
3. **Adjust Parameters**: Adjust threshold, resolution, and other parameters as needed
4. **Monitor Results**: Observe processing results in the right main view
5. **Control Operation**: Use buttons in the bottom control bar to control system operation status

### 6.4 Common Operations

- **Pause/Resume**: Click the "Pause" button to pause processing, click the "Resume" button to resume processing
- **Reset System**: Click the "Reset" button to reset the system to initial state
- **Adjust Resolution**: Select appropriate resolution preset in the left sidebar
- **Adjust Threshold**: Use the threshold slider to adjust anomaly detection sensitivity

## 7. Building and Deployment

### 7.1 Environment Configuration

Please refer to the detailed environment installation guide: [install.md](install.md)

### 7.2 Building Steps

```bash
# Create build directory
mkdir build
cd build

# Generate Visual Studio solution
cmake .. -G "Visual Studio 16 2019" -A x64

# Build project
cmake --build . --config Release
```

### 7.3 Running Command

```bash
# Run executable file
cd build/Release
limr.exe
```

## 8. Code Optimization and Best Practices

### 8.1 Performance Optimization

1. **Memory Management**:
   - Use DeviceBuffer to manage GPU memory, reducing memory leak risks
   - Pre-allocate and reuse buffers to reduce memory allocation overhead
   - Utilize CUDA streams to achieve parallel execution of computation and memory copy

2. **Computation Optimization**:
   - Use CUDA kernels for parallel computing
   - Utilize NPP library for optimized image processing
   - Reasonably set CUDA grid and block sizes to improve parallel efficiency

3. **Inference Optimization**:
   - Use TensorRT's INT8 quantization (if applicable)
   - Select appropriate precision mode (F32 or F16)
   - Optimize model input size to balance precision and speed

### 8.2 Code Quality

1. **Error Handling**:
   - Enhance error checking and logging
   - Provide detailed error information
   - Use exception handling mechanism (where appropriate)

2. **Code Structure**:
   - Adopt modular design to reduce code coupling
   - Extract common interfaces for easy extension
   - Use design patterns to improve code maintainability

3. **Documentation**:
   - Add code comments, especially for complex algorithms and key functions
   - Improve API documentation
   - Provide detailed usage examples

### 8.3 Debugging Techniques

1. **CUDA Debugging**:
   - Use CUDA-GDB for GPU code debugging
   - Utilize NVIDIA Nsight for performance analysis
   - Check CUDA error codes to detect problems in time

2. **Memory Debugging**:
   - Use CUDA memory checking tools to detect memory leaks
   - Monitor GPU memory usage to avoid memory overflow
   - Reasonably set buffer sizes to avoid memory waste

3. **Performance Analysis**:
   - Use NVIDIA Nsight Systems to analyze system performance
   - Monitor GPU utilization and memory bandwidth
   - Identify performance bottlenecks and optimize them targeted

## 9. Extension and Customization

### 9.1 Supporting New Models

1. **Model Conversion**:
   - Train models using PyTorch or other frameworks
   - Convert models to ONNX format
   - Use TensorRT to convert ONNX models to engine files

2. **Configuration Adjustment**:
   - Adjust configuration parameters according to model input/output size
   - Select appropriate precision mode
   - Adjust preprocessing and postprocessing parameters

### 9.2 Feature Extension

1. **Adding New Preprocessing Algorithms**:
   - Implement new preprocessing logic in `preprocess.cpp`
   - Ensure use of CUDA acceleration to improve processing efficiency

2. **Adding New Postprocessing Algorithms**:
   - Implement new postprocessing logic in `postprocess.cpp` and `postprocess_mask.cu`
   - Utilize CUDA for efficient image processing

3. **Enhancing Visualization Functions**:
   - Add new visualization options in `dashboard.cpp`
   - Implement more color mapping schemes
   - Add statistical information and analysis tools

### 9.3 System Integration

1. **Integration with Other Systems**:
   - Provide API interfaces for easy integration with other systems
   - Support custom video sources and output targets
   - Implement network communication functions to support remote monitoring

2. **Deployment to Edge Devices**:
   - Optimize models and code for edge devices
   - Reduce memory and computing requirements
   - Adapt to different hardware platforms

## 10. Troubleshooting and Debugging

### 10.1 Common Problems

1. **Cannot Open Video Source**:
   - Check if the video file path is correct
   - Check if the camera is available
   - Ensure the video format is supported

2. **Model Loading Failed**:
   - Check if the engine file path is correct
   - Ensure the engine file is compatible with the current CUDA version
   - Check if the engine file is complete

3. **Performance Issues**:
   - Try reducing input resolution
   - Use F16 precision
   - Check GPU utilization to ensure no other processes are occupying large GPU resources

4. **All Blue or All Red Anomaly Map**:
   - Check if the model output is correct
   - Adjust defect threshold
   - Check preprocessing and postprocessing parameters

5. **CUDA Errors**:
   - Check if CUDA version is compatible with TensorRT
   - Ensure GPU driver is updated
   - Check GPU memory usage to avoid memory overflow

### 10.2 Debugging Tools

1. **NVIDIA Nsight Systems**: Analyze system performance and GPU utilization
2. **NVIDIA Nsight Compute**: Analyze CUDA kernels performance
3. **CUDA-GDB**: Debug CUDA code
4. **OpenCV Debugging Tools**: Debug image processing flow
5. **Visual Studio Debugger**: Debug C++ code

### 10.3 Logging and Monitoring

- **System Logs**: Record key operations and error information
- **Performance Monitoring**: Real-time monitoring of system performance indicators
- **GPU Monitoring**: Monitor GPU utilization, memory usage, and temperature
- **Inference Monitoring**: Monitor inference time and accuracy

## 11. Summary

inference_LiMR is a feature-complete, high-performance real-time anomaly detection system that implements fast inference based on TensorRT and CUDA. The system adopts a modular design for easy extension and maintenance, and provides an intuitive user interface.

The project has broad application prospects in industrial quality inspection, real-time monitoring, and other fields. Through further performance optimization, feature enhancement, and code quality improvement, the system can be made more complete to meet the needs of more complex scenarios.

Developers can understand the system's architectural design, implementation details, and usage methods through this manual, thereby better understanding, using, and extending the system.

## 12. References

1. NVIDIA TensorRT Documentation: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html
2. OpenCV Documentation: https://docs.opencv.org/4.x/
3. CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
4. ImGui Documentation: https://github.com/ocornut/imgui/wiki
5. NVIDIA NPP Library Documentation: https://docs.nvidia.com/cuda/npp/index.html
