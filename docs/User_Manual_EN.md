# User Manual

## 1. System Overview

### 1.1 Introduction

AnomaRT is the desktop application of anomalib-runtime, a TensorRT-based native runtime for real-time anomaly detection and video stream analysis. This user manual covers dashboard operation, system startup, interfaces, and ONNX export requirements.

### 1.2 Key Features

- Real-time video anomaly detection
- Intuitive dashboard interface
- Support for both single-engine and dual-engine inference modes
- Anomaly heatmap generation
- Configurable detection thresholds
- Multiple resolution support

## 2. System Startup

### 2.1 Prerequisites

Before starting the system, ensure the following requirements are met:

- Windows 10/11 operating system
- Compatible NVIDIA GPU and driver
- The complete AnomaRT release directory, including its bundled DLL files

CUDA Toolkit, TensorRT SDK and OpenCV do not need to be installed separately
by end users.

### 2.2 Startup Procedure

1. **Prepare the executable file**
   - Navigate to the extracted AnomaRT release directory
   - Ensure all required DLL files are present (see Developer's Manual for details)

2. **Run the application**
   - Double-click `AnomaRT.exe` to start the system
   - The dashboard window will appear

3. **Initialize the system**
   - Configure the required parameters in the dashboard (see Section 3 for details)
   - Click the "INITIALIZE SYSTEM" button
   - Wait for the initialization process to complete

4. **Start processing**
   - Once initialized, the system will automatically start processing
   - If paused, click "RESUME" to start processing

## 3. Dashboard Usage

### 3.1 Dashboard Layout

The dashboard consists of the following main sections:

| Section | Description |
|---------|-------------|
| **Control Panel** | Located on the left side, contains system controls and parameters |
| **Video Display** | Located on the right side, shows the original video and anomaly heatmap |
| **Status Bar** | Located at the bottom, displays system status and performance metrics |

[Image Placeholder: Dashboard Layout]

### 3.2 Configuration Parameters

#### 3.2.1 Input Configuration

| Parameter | Description |
|-----------|-------------|
| **Video Source** | Path to video file, camera ID, or image file |
| **Resolution** | Processing and display resolution (256x256, 512x512, 640x480, 1024x768) |

#### 3.2.2 Model Configuration

| Parameter | Description |
|-----------|-------------|
| **Model Path** | Path to the model file (.onnx or .engine) |
| **Precision** | F32 or F16 |

> **Note**: .onnx format has the best compatibility; .engine format requires matching the current GPU architecture.

#### 3.2.3 Detection Configuration

| Parameter | Description |
|-----------|-------------|
| **Defect Threshold** | Anomaly detection sensitivity (0.0-1.0) |

### 3.3 Control Buttons

| Button | Function |
|--------|----------|
| **INITIALIZE SYSTEM** | Initializes the system with current configuration |
| **PAUSE** | Pauses video processing |
| **RESUME** | Resumes video processing |
| **RESET** | Resets the system to initial state |
| **Apply & Reload** | Applies new configuration and reloads the system |

[Image Placeholder: Control Panel]

### 3.4 Display Modes

The dashboard supports the following display modes:

| Mode | Description |
|------|-------------|
| **Original Video** | Displays the original input video |
| **Heatmap** | Displays the anomaly heatmap |
| **Overlay** | Displays heatmap overlay on original video |

[Image Placeholder: Display Modes]

## 4. Input/Output Interfaces

### 4.1 Input Interfaces

#### 4.1.1 Video Input

| Input Type | Description |
|------------|-------------|
| **Video File** | Supported formats: MP4, AVI, etc. |
| **Camera** | USB cameras, network cameras |
| **RTSP Stream** | Network video streams via RTSP protocol |

#### 4.1.2 Model Input

| Parameter | Description |
|-----------|-------------|
| **Input Tensor** | Shape: [1, 3, H, W], where H and W are model input dimensions |
| **Data Type** | F32 or F16 (depending on model precision) |
| **Normalization** | Image pixel values normalized to [-1, 1] range |

### 4.2 Output Interfaces

#### 4.2.1 Heatmap Output

| Output Type | Description |
|-------------|-------------|
| **Anomaly Heatmap** | Shape: [1, 1, H, W], values represent anomaly scores |
| **Visualization** | Color-coded heatmap with red indicating high anomaly areas |

#### 4.2.2 Detection Results

| Output Type | Description |
|-------------|-------------|
| **Anomaly Score** | Single value representing overall anomaly level (0.0-1.0) |
| **Defect Locations** | Coordinates of detected anomalies above threshold |
| **Status Flag** | Boolean indicating if anomalies were detected |



## 5. ONNX Export Requirements

### 5.1 Model Requirements

To ensure compatibility with AnomaRT, models must meet the following requirements when exported to ONNX format:

| Requirement | Description |
|-------------|-------------|
| **Input Shape** | Fixed input shape: [1, 3, H, W], where H and W are multiples of 32 |
| **Output Shape** | Fixed output shape: [1, 1, H, W] for heatmap generation |
| **Data Type** | Float32 (can be converted to Float16 during TensorRT optimization) |
| **Normalization** | Model should expect input normalized to [-1, 1] range |

### 5.2 Export Process

#### 5.2.1 PyTorch Model Export

```python
# Example code for exporting PyTorch model to ONNX
import torch
import torch.nn as nn

# Load your trained model
model = YourModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 256, 256)  # Adjust size as needed

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {2: 'height', 3: 'width'}, 'output': {2: 'height', 3: 'width'}}  # Optional dynamic axes
)
```

#### 5.2.2 TensorRT Conversion

After exporting the model to ONNX, it needs to be converted to TensorRT engine format:

1. **Using TensorRT ONNX Parser**
   - The system automatically converts ONNX models to TensorRT engines during initialization
   - Alternatively, you can use the TensorRT `trtexec` tool for offline conversion

2. **Offline Conversion Command**
   ```bash
   # Example command for converting ONNX to TensorRT engine
   trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
   ```

### 5.3 Model Compatibility

| Model Type | Compatibility | Notes |
|------------|---------------|-------|
| **Anomalib Models** | Fully compatible | Recommended for best results |
| **Custom Autoencoders** | Compatible | Must follow input/output requirements |
| **Other Anomaly Models** | Compatible | May require adjustment to meet input/output specifications |

## 5. Troubleshooting

### 5.1 Common Issues

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| **Cannot open video source** | Invalid video path or camera ID | Check video file path or camera connection |
| **Model loading failed** | Incompatible ONNX model or missing dependencies | Verify model format and check TensorRT installation |
| **GPU out of memory** | Resolution too high or model too large | Reduce resolution or use smaller model |
| **Low detection accuracy** | Threshold not properly adjusted | Fine-tune the defect threshold parameter |
| **Slow processing speed** | Using FP32 precision or high resolution | Switch to FP16 precision or reduce resolution |

### 5.2 Error Messages

| Error Message | Description | Solution |
|---------------|-------------|----------|
| **CUDA error: out of memory** | GPU memory insufficient | Reduce resolution or use smaller model |
| **Could not open video device** | Camera not detected or accessible | Check camera connection and permissions |
| **ONNX parser error** | Invalid ONNX model format | Ensure model was exported correctly |
| **TensorRT initialization failed** | Incompatible TensorRT version | Install recommended TensorRT version |

## 6. Best Practices

### 6.1 Performance Optimization

| Recommendation | Description |
|----------------|-------------|
| **Use FP16 Precision** | Significantly improves processing speed with minimal accuracy loss |
| **Optimize Resolution** | Balance between processing speed and detection accuracy |
| **Pre-convert Models** | Use `trtexec` to pre-convert ONNX models for faster startup |
| **Close Unnecessary Applications** | Free up GPU memory for better performance |

### 6.2 Detection Optimization

| Recommendation | Description |
|----------------|-------------|
| **Calibrate Threshold** | Adjust threshold based on specific application requirements |
| **Use Appropriate Model** | Select model size based on required accuracy and speed |
| **Consider Lighting Conditions** | Ensure consistent lighting for reliable detection |
| **Regular Model Updates** | Retrain models periodically with new data |

## 7. Summary

AnomaRT provides a native, high-performance solution for real-time anomaly detection in video streams. By following this user manual, you can configure and operate the system to meet your specific needs.

For more advanced configuration and customization options, please refer to the [Developer's Manual](开发者手册.md).
