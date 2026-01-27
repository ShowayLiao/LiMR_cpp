# ONNX Model I/O Definition

## Model Information

This document describes the input and output tensor definitions for the anomaly detection model.

## Input

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| input | float32 | `[batch_size, 3, 256, 256]` | Standard image input (NCHW) with dynamic batch size, 3 channels, 256x256 resolution |

## Output

| Index | Name | Type | Shape | Description |
|-------|------|------|-------|-------------|
| 1 | pred_score | float32 | `[dim_0, dim_1]` | Anomaly score (typically the overall anomaly score for the entire image) |
| 2 | pred_label | boolean | `[dim_0, dim_1]` | Anomaly label (True = anomaly, False = normal) |
| 3 | anomaly_map | float32 | `[dim_0, dim_1, dim_2, dim_3]` | Anomaly heatmap (pixel-level anomaly probability distribution) |
| 4 | pred_mask | boolean | `[dim_0, dim_1, dim_2, dim_3]` | Anomaly mask (pixel-level binary segmentation result) |

## Notes

- **Dynamic Batch Size**: The input tensor supports dynamic batch dimensions, allowing flexible batch processing
- **Image Format**: Input images should be in NCHW format (Batch, Channels, Height, Width)
- **Resolution**: Standard input resolution is 256x256 pixels
- **Channels**: 3-channel RGB images
- **Output Interpretation**:
  - `pred_score`: Provides a global anomaly score for the entire image
  - `pred_label`: Binary classification result indicating whether the image contains anomalies
  - `anomaly_map`: Spatial anomaly distribution across the image (useful for visualization)
  - `pred_mask`: Binary segmentation mask highlighting anomalous regions

## Usage Example

```cpp
// Input preparation
cv::Mat image = cv::imread("input.jpg");
cv::resize(image, image, cv::Size(256, 256));

// Convert to NCHW float32 format
std::vector<float> input(3 * 256 * 256);
for (int c = 0; c < 3; c++) {
    for (int h = 0; h < 256; h++) {
        for (int w = 0; w < 256; w++) {
            input[c * 256 * 256 + h * 256 + w] = image.at<cv::Vec3b>(h, w)[c] / 255.0f;
        }
    }
}

// Output processing
float* pred_score = ...;
bool* pred_label = ...;
float* anomaly_map = ...;
bool* pred_mask = ...;

// Use outputs for visualization and analysis
cv::Mat heatmap = generateHeatmap(anomaly_map);
cv::Mat mask = generateMask(pred_mask);
```
