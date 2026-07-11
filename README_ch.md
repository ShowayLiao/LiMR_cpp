<div align="center">
  <h1>🏭 AnomaRT：anomalib 模型原生推理运行时</h1>
  
  <p>
    <b>面向 anomalib ONNX 模型的高性能 C++/CUDA/TensorRT 部署运行时</b>
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
    <a href="./README.md">English</a> | <b>中文</b>
  </p>
  <p>
    <a href="PAPER_LINK_HERE">📄 论文地址</a> • 
    <span>anomalib 模型的社区部署配套项目</span>
  </p>
</div>

> AnomaRT 是独立的社区项目，不是 anomalib 官方项目，也不隶属于其维护团队。

---

## 📸 系统运行演示

<div align="center">
  <img src="docs/demo_placeholder.gif" alt="System Demo" width="100%" />
  <br>
  <i>实时三屏显示：原图采集 (左) | 异常热力图 (中) | 缺陷分割叠加 (右)</i>
</div>

<br>

## ✨ 核心亮点 (Highlights)

<table align="center">
  <tr>
    <td align="center" width="50%">
      <h3>🚀 极致性能</h3>
      <p>纯 <b>CUDA 算子</b>后处理 + <b>OpenGL 零拷贝</b>渲染<br>推理延迟 <b>&lt; 15ms</b> (80+ FPS)</p>
    </td>
    <td align="center" width="50%">
      <h3>🎯 Anomalib 兼容</h3>
      <p>完全兼容 <b>Anomalib</b> 生态<br>无缝加载标准模型，支持像素级缺陷分割与定位</p>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <h3>🖥️ 交互式面板</h3>
      <p>基于 <b>ImGui</b> 打造的现代化控制台<br>支持阈值实时调节、模型热切换</p>
    </td>
    <td align="center" width="50%">
      <h3>📦 开箱即用</h3>
      <p>提供 <b>Windows 一键安装包 (Setup.exe)</b><br>无需配置 Python/CUDA 环境，双击即用</p>
    </td>
  </tr>
</table>

---

## ⚡ 快速开始 (Quick Start)

### 👥 我是终端用户
> 不需要写代码，只想运行软件？

1. 下载最新发布的安装包：👉 **[点击下载 (Release)]()**
2. 遇到问题？查看文档：📖 **[用户手册 (User Manual)](./docs/User_Manual_CN.md)**

### 👨‍💻 我是开发者
> 想要修改源码或二次开发？

请查阅编译与构建指南：🛠️ **[开发者环境配置指南](./docs/安装方法.md)**

---

## 📌 性能指标 (Benchmark)

我们在 `RTX 3060` 平台上进行了完整测试，相比原始实现获得了 **2倍以上** 的吞吐量提升。

| 优化阶段 | 预处理 (ms) | 推理 (ms) | 后处理 (ms) | **总延迟 (ms)** | **吞吐量 (FPS)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 原始 C++ 实现 | 11 | 17 | 5 | 33 | ~30 |
| + 预处理 CUDA 加速 | 4 | 17 | 5 | 26 | ~38 |
| + 后处理零拷贝优化 | 4 | 17 | 3 | 24 | ~42 |
| + 推理 FP16 量化 | 4 | **8** | 3 | 15 | ~67 |
| **+ 渲染流水线优化** | **4** | **8** | **3** | **< 15** | **> 80** |

---

## 🏗️ 系统架构

本系统采用模块化设计，实现了计算（CUDA）与显示（OpenGL）的高效互操作。

* **🧠 推理核心 (Inference Engine)**
    * 封装 `TensorRT 10`，支持 FP32/FP16 动态精度。
    * 多线程流水线设计，分离输入 IO 与 GPU 计算。
* **⚡ CUDA 加速层**
    * **Preprocessor**: 颜色空间转换、归一化、Resize (NPP)。
    * **Postprocessor**: 异常图生成、阈值分割、热力图渲染 (Custom Kernels)。
* **🎨 可视化与交互**
    * **Dashboard**: 基于 `ImGui` 的控制面板。
    * **Renderer**: 使用 `CUDA-OpenGL Interop` 直接映射显存纹理，消除 CPU-GPU 带宽瓶颈。

---

## 📅 更新日志 (Changelog)

<details>
<summary><b>v1.0.0 - 2026-01-28: 架构重构与性能优化 (点击展开)</b></summary>

* **全新模块化架构**：重构配置管理、渲染、推理引擎等独立模块。
* **渲染优化**：实现 CUDA-OpenGL 互操作，消除 Host-to-Device 拷贝开销。
* **流水线增强**：多线程 Pipeline，支持 FP16 加速。
* **UI 升级**：集成 ImGui 实现交互式参数调节。

</details>

<br>

> 完整记录请参考 [CHANGELOG.md](./CHANGELOG.md)

---

## 🤝 致谢与反馈

> 如果本项目对您的研究或工作有所帮助，欢迎在 GitHub 上点一个 ⭐ **Star**！

如果您发现任何 Bug 或有改进建议，欢迎提交 [Issue](https://github.com/ShowayLiao/anomalib-runtime/issues) 或 Pull Request。

## 📜 License

This repository is licensed under the [Apache-2.0 License](LICENSE).
