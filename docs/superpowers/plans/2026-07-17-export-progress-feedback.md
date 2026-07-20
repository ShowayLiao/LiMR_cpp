# 导出与初始化进度反馈实现计划

目标：让模型导出/加载、初始化和 Apply & Reload 期间界面持续响应，并明确显示当前阶段、成功或失败结果。

方案：将不依赖 ImGui/OpenGL 的引擎加载与管线准备放到后台线程；主线程每帧轮询结果，完成后再创建纹理、上传首帧并更新 UI。使用 `std::future` 保证任务生命周期可回收，任务运行时禁用配置按钮并绘制半透明加载遮罩。

## 任务 1：定义可测试的任务状态

- 修改：`include/dashboard.h`
- 新增：`test/test_dashboard_task_state.cpp`
- 定义空闲、运行中、成功、失败状态及忙碌判断，测试运行中/完成状态的边界。

## 任务 2：封装异步初始化结果与生命周期

- 修改：`include/dashboard.h`、`src/dashboard.cpp`
- 增加 future、线程任务结果、状态文本和错误文本。
- Dashboard 析构和 RESET 前等待已启动任务，避免后台对象悬空。
- 后台任务只操作文件、TensorRT 和 Pipeline，不触碰 ImGui/OpenGL。

## 任务 3：接入初始化与 Apply & Reload

- 修改：`src/dashboard.cpp`
- 按钮点击只提交任务并立即返回渲染循环。
- 每帧轮询任务，成功后在主线程接管 engine/pipeline、初始化纹理并显示首帧。
- 失败时保留错误信息并显示可关闭提示，不把界面留在假 READY 状态。

## 任务 4：绘制加载遮罩与结果反馈

- 修改：`src/dashboard.cpp`
- 控制面板显示“正在导出/初始化…”阶段文字和旋转圈。
- 主视图绘制半透明遮罩，阻止重复点击。
- 成功/失败消息短暂显示，并允许用户继续操作。

## 任务 5：验证

- 运行 dashboard 状态单测。
- 配置并构建可用目标；若本机缺少 TensorRT/OpenCV 构建依赖，记录具体阻塞原因。
- 检查 diff，确认后台线程不调用任何 ImGui/OpenGL API。
