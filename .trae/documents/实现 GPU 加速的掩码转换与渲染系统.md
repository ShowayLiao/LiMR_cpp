# 实现 GPU 加速的掩码转换与渲染系统

## 1. 修改 FrameTask.h 结构
- 移除基于 OpenCV 的 `heatmap_vis` 字段
- 添加 `d_overlay_gpu` (DeviceBuffer) 用于存储半透明红色 mask
- 添加 `gl_overlay_tex_id` (unsigned int) 用于 OpenGL 纹理
- 添加 `d_original_image` (DeviceBuffer) 用于存储原始图像的 GPU 内存

## 2. 实现 CUDA 算子函数
- 在 postprocessor.cpp 中添加 `launchGenerateOverlayFromMask` 函数
- 实现 0-1 图到半透明红色 mask 的转换逻辑
- 确保转换过程完全在 GPU 上执行
- 支持阈值控制以调整 mask 的透明度和覆盖范围

## 3. 修改 PostProcessor 逻辑
- 在 `process` 函数中调用新的 CUDA 算子
- 生成半透明红色 mask 并存储到 `d_overlay_gpu`
- 确保 `d_dynamic_mask` 和 `d_overlay_gpu` 之间的数据正确传递

## 4. 实现原始图像的初始 GPU 上传
- 在 Preprocessor 中添加原始图像的 GPU 上传功能
- 将上传后的 GPU 内存存储到 `d_original_image`
- 确保在处理流程开始阶段即完成上传

## 5. 修改 Dashboard 渲染逻辑
- 更新 `InitResources` 函数，初始化 `gl_overlay_tex_id`
- 更新 `UpdateData` 函数，从 `d_overlay_gpu` 更新 OpenGL 纹理
- 移除对 `heatmap_vis` 的依赖
- 实现 mask 到原图的覆盖渲染逻辑

## 6. 优化内存管理
- 确保所有 GPU 内存分配和释放操作正确执行
- 优化数据传输，减少 CPU-GPU 数据传输开销
- 确保内存管理高效，符合项目的性能要求

## 7. 测试与验证
- 验证半透明红色 mask 能够准确覆盖在原图上
- 验证整个转换过程通过 CUDA 算子在 GPU 上完成
- 验证系统性能符合要求，渲染流畅

通过以上步骤，实现全程在 GPU 上进行数据处理和渲染的架构，确保系统性能和渲染质量。