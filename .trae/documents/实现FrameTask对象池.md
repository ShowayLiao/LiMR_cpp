# 实现FrameTask对象池

## 问题分析
当前每帧都会创建新的FrameTask对象，导致内存线性增长。通过实现对象池，可以循环利用旧的Task，避免重复分配内存，提高性能。

## 实现步骤

### 1. 修改FrameTask.h
- 添加`reset()`方法，仅清理状态，不释放显存
- 该方法会重置frame_id、timestamp、start_time、end_time等状态，清空peaks向量，但保留d_input、d_heatmap_gpu等DeviceBuffer

### 2. 修改Pipeline.h和Pipeline.cpp
- 添加`SafeQueue<FrameTaskPtr> task_pool_`成员变量作为空闲任务池
- 在Pipeline构造函数中预分配20个任务对象到池中
- 实现`FrameTaskPtr get_empty_task()`方法，从池中获取任务并重置
- 实现`void return_task(FrameTaskPtr task)`方法，将任务返回池中

### 3. 修改InputThread
- 为InputThread添加Pipeline指针成员
- 修改InputThread构造函数，接收Pipeline指针
- 在run方法中使用`pipeline->get_empty_task()`获取任务，替代`std::make_shared<FrameTask>()`

### 4. 修改Dashboard
- 在Render方法中，渲染完成后将current_task返回池中
- 添加检查确保pipeline存在且current_task有效
- 返回任务后重置current_task，避免重复返回

## 技术要点
- **内存复用**：通过对象池实现DeviceBuffer的复用，避免重复内存分配
- **零开销**：任务重置仅清理状态，不涉及内存操作
- **线程安全**：使用SafeQueue管理任务池，确保线程安全
- **性能优化**：当任务池为空时，可选择丢帧或创建临时对象，优先保证性能

## 预期效果
- 消除内存线性增长问题
- 减少内存分配和释放的开销
- 提高系统整体性能和稳定性
