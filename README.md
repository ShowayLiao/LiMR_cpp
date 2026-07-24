<div align="center">
  <h1>🏭 FastVision: Fast Industrial Vision Platform (C++)</h1>
  
  <p>
    <b>High-throughput, low-latency industrial vision platform based on TensorRT 10 and CUDA 11</b>
  </p>

  <p>
    <img src="https://img.shields.io/badge/C++-17-blue.svg?style=flat-square" alt="C++17">
    <img src="https://img.shields.io/badge/CUDA-11.x-green.svg?style=flat-square" alt="CUDA">
    <img src="https://img.shields.io/badge/TensorRT-10-76b900.svg?style=flat-square" alt="TensorRT">
    <img src="https://img.shields.io/badge/OpenCV-4-red.svg?style=flat-square" alt="OpenCV">
    <img src="https://img.shields.io/badge/ImGui-1.8-lightgrey.svg?style=flat-square" alt="ImGui">
    <img src="https://img.shields.io/badge/Platform-Windows-0078d7.svg?style=flat-square" alt="Windows">
  </p>

  <p>
    <b>English</b> | <a href="./README_ch.md">中文</a>
  </p>
  <p>
    <a href="https://ieeexplore.ieee.org/document/11606365">📄 Paper</a> • 
    <a href="https://github.com/ShowayLiao/LiMR">🐍 Python Prototype</a>
  </p>
</div>

---

## 📸 System Demo

<div align="center">
  <img src="docs/image-test.png" alt="System Demo" width="100%" />

  <br>
  <i>Real-time three-screen display: Original Capture (Left) | Anomaly Heatmap (Middle) | Defect Overlay (Right)</i>
</div>

<br>

## ✨ Core Highlights

<table align="center">
  <tr>
    <td align="center" width="50%">
      <h3>🚀 Extreme Performance</h3>
      <p>Pure <b>CUDA operator</b> post-processing + <b>OpenGL zero-copy</b> rendering<br>Inference latency <b>&lt; 15ms</b> (80+ FPS)</p>
    </td>
    <td align="center" width="50%">
      <h3>🎯 Anomalib Compatible</h3>
      <p>Fully compatible with <b>Anomalib</b> ecosystem<br>Seamless loading of standard models, supporting pixel-level defect segmentation and localization</p>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <h3>🖥️ Interactive Panel</h3>
      <p>Modern console built with <b>ImGui</b><br>Support for real-time threshold adjustment and model hot-switching</p>
    </td>
    <td align="center" width="50%">
      <h3>📦 Ready to Use</h3>
      <p>Provides <b>Windows one-click installer (Setup.exe)</b><br>No need to configure Python/CUDA environment, just double-click to run</p>
    </td>
  </tr>
</table>

---

## ⚡ Quick Start

### 👥 I'm an End User
> Don't want to code, just want to run the software?

1. Download the latest release: 👉 **[Download (Release)]()**
2. Need help? Check the documentation: 📖 **[User Manual](./docs/User_Manual_CN.md)**

### 👨‍💻 I'm a Developer
> Want to modify the source code or develop secondary applications?

Please refer to the compilation and build guide: 🛠️ **[Developer Environment Setup](./docs/install.md)**

---

## 📌 Performance Benchmark

We conducted comprehensive tests on an `RTX 3060` platform and achieved **more than 2x** throughput improvement compared to the original implementation.

| Optimization Stage | Preprocessing (ms) | Inference (ms) | Postprocessing (ms) | **Total Latency (ms)** | **Throughput (FPS)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Original C++ Implementation | 11 | 17 | 5 | 33 | ~30 |
| + CUDA Preprocessing | 4 | 17 | 5 | 26 | ~38 |
| + Zero-copy Postprocessing | 4 | 17 | 3 | 24 | ~42 |
| + FP16 Quantization | 4 | **8** | 3 | 15 | ~67 |
| **+ Rendering Pipeline Optimization** | **4** | **8** | **3** | **< 15** | **> 80** |

---

## 🏗️ System Architecture

The system adopts a modular design, achieving efficient interoperation between computation (CUDA) and display (OpenGL).

* **🧠 Inference Core**
    * Encapsulates `TensorRT 10`, supports FP32/FP16 dynamic precision.
    * Multi-threaded pipeline design, separating input IO from GPU computation.
* **⚡ CUDA Acceleration Layer**
    * **Preprocessor**: Color space conversion, normalization, Resize (NPP).
    * **Postprocessor**: Anomaly map generation, threshold segmentation, heatmap rendering (Custom Kernels).
* **🎨 Visualization & Interaction**
    * **Dashboard**: Control panel based on `ImGui`.
    * **Renderer**: Uses `CUDA-OpenGL Interop` to directly map VRAM textures, eliminating CPU-GPU bandwidth bottleneck.

---

## 📅 Changelog

<details>
<summary><b>v1.0.0 - 2026-01-28: Architecture Refactoring and Performance Optimization (Click to expand)</b></summary>

* **New Modular Architecture**: Refactored configuration management, rendering, inference engine, and other independent modules.
* **Rendering Optimization**: Implemented CUDA-OpenGL interoperability, eliminating Host-to-Device copy overhead.
* **Pipeline Enhancement**: Multi-threaded Pipeline, support for FP16 acceleration.
* **UI Upgrade**: Integrated ImGui for interactive parameter adjustment.

</details>

<br>

> For complete records, please refer to [CHANGELOG.md](./CHANGELOG.md)

---

## 🤝 Acknowledgements & Feedback

> If this project helps your research or work, please give it a ⭐ **Star** on GitHub!

If you find any bugs or have improvement suggestions, please submit an [Issue](https://github.com/ShowayLiao/LiMR/issues) or Pull Request.


## 📚 Reference

If you find this project useful in your research or work, please cite this project:

```tex
@article{liao2026limr,
  author  = {Shaowei Liao and Wenyong Yu and Shaolin Liao},
  title   = {Lightweight Masked Reconstruction for Real-Time Sensor-Driven
             Anomaly Detection in Industrial IoT},
  journal = {IEEE Internet of Things Journal},
  year    = {2026},
  doi     = {10.1109/JIOT.2026.3712733}
}
```

## 📜 License

This repository is licensed under the [Apache-2.0 License](LICENSE).
