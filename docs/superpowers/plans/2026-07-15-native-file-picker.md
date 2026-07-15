# 原生文件选择器实现计划

> **面向 AI 代理的工作者：** 必须使用 `subagent-driven-development` 或 `executing-plans` 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在 Dashboard 的输入媒体和模型路径字段旁提供 Windows 原生文件选择器，成功选择时安全地回填路径，取消时不改变当前路径。

**架构：** 将 Win32 `GetOpenFileNameA` 的调用封装在 `common/FileDialog` 中。纯路径拷贝函数接受可选的选择结果，方便单元测试；Dashboard 只负责提供文件过滤器、绘制按钮并调用封装。

**技术栈：** C++17、Win32 Common Dialog、Dear ImGui、CMake、标准库 `assert` 测试。

---

## 文件结构

- 新建 `include/common/FileDialog.h`：声明可测试的路径回填函数及 Windows 文件选择器接口。
- 新建 `src/common/FileDialog.cpp`：实现安全拷贝与 `GetOpenFileNameA` 包装。
- 修改 `src/dashboard.cpp`：为 `Video` 和 `Model Path` 增加 `Browse...` 按钮和匹配的过滤器。
- 新建 `test/test_file_dialog.cpp`：验证成功选择、安全截断与取消不覆盖路径。
- 修改 `CMakeLists.txt`：新增独立测试目标，并链接应用与该测试所需的 `comdlg32`。

### 任务 1：为路径回填行为建立失败测试

**文件：**
- 创建：`test/test_file_dialog.cpp`
- 创建：`include/common/FileDialog.h`

- [ ] **步骤 1：编写失败测试**

```cpp
#include <cassert>
#include <cstring>

#include "common/FileDialog.h"

int main() {
    char path[16] = "old-path";
    assert(common::copySelectedPath(path, sizeof(path), "C:\\input\\part.png"));
    assert(std::strcmp(path, "C:\\input\\part.") == 0);

    char unchanged[16] = "keep-this";
    assert(!common::copySelectedPath(unchanged, sizeof(unchanged), nullptr));
    assert(std::strcmp(unchanged, "keep-this") == 0);
    return 0;
}
```

- [ ] **步骤 2：运行测试，确认因缺少接口失败**

运行：`cmake --build build --target test_file_dialog`

预期：构建失败，提示 `common/FileDialog.h` 或 `copySelectedPath` 不存在。

### 任务 2：实现最小的可测试文件选择器封装

**文件：**
- 创建：`include/common/FileDialog.h`
- 创建：`src/common/FileDialog.cpp`
- 修改：`CMakeLists.txt`

- [ ] **步骤 1：声明接口**

```cpp
namespace common {
bool copySelectedPath(char* destination, std::size_t destination_size, const char* selected_path);
bool selectFile(char* destination, std::size_t destination_size, const char* filter);
}
```

- [ ] **步骤 2：实现安全回填和原生对话框**

```cpp
bool copySelectedPath(char* destination, std::size_t destination_size, const char* selected_path) {
    if (destination == nullptr || destination_size == 0 || selected_path == nullptr) return false;
    std::strncpy(destination, selected_path, destination_size - 1);
    destination[destination_size - 1] = '\0';
    return true;
}

bool selectFile(char* destination, std::size_t destination_size, const char* filter) {
    char selected_path[MAX_PATH] = {};
    OPENFILENAMEA dialog{sizeof(dialog)};
    dialog.lpstrFile = selected_path;
    dialog.nMaxFile = sizeof(selected_path);
    dialog.lpstrFilter = filter;
    dialog.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    return GetOpenFileNameA(&dialog) && copySelectedPath(destination, destination_size, selected_path);
}
```

- [ ] **步骤 3：在 CMake 中增加测试和链接库**

```cmake
add_executable(test_file_dialog
    ${PROJECT_SOURCE_DIR}/test/test_file_dialog.cpp
    ${PROJECT_SOURCE_DIR}/src/common/FileDialog.cpp
)
target_include_directories(test_file_dialog PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(limr comdlg32)
target_link_libraries(test_file_dialog comdlg32)
```

- [ ] **步骤 4：运行测试，确认通过**

运行：`cmake --build build --target test_file_dialog; .\\build\\test_file_dialog.exe`

预期：构建成功，进程退出码为 0。

### 任务 3：将文件选择器接入 Dashboard

**文件：**
- 修改：`src/dashboard.cpp` 的包含列表及 `Dashboard::DrawSidePanel` 中的配置区域。

- [ ] **步骤 1：引入封装并定义过滤器**

```cpp
#include "common/FileDialog.h"

constexpr char kInputFileFilter[] =
    "Media files\\0*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.tif;*.avi;*.mp4;*.mov;*.mkv;*.wmv;*.flv\\0"
    "All files\\0*.*\\0";
constexpr char kModelFileFilter[] =
    "Model files\\0*.onnx;*.engine;*.plan\\0All files\\0*.*\\0";
```

- [ ] **步骤 2：在每个路径输入框后绘制浏览按钮**

```cpp
ImGui::InputText("Video", video_path, sizeof(video_path));
ImGui::SameLine();
if (ImGui::Button("Browse...##input")) {
    common::selectFile(video_path, sizeof(video_path), kInputFileFilter);
}

ImGui::InputText("Model Path", engine_path_a, sizeof(engine_path_a));
ImGui::SameLine();
if (ImGui::Button("Browse...##model")) {
    common::selectFile(engine_path_a, sizeof(engine_path_a), kModelFileFilter);
}
```

- [ ] **步骤 3：构建应用**

运行：`cmake --build build --target limr`

预期：构建成功；`GetOpenFileNameA` 能解析并链接至 `comdlg32`。

- [ ] **步骤 4：手动验收**

运行 `build\\limr.exe`，确认：

1. 点击输入媒体的 `Browse...` 会打开 Windows 文件选择器，并显示图片/视频筛选项；选择文件后 `Video` 回填绝对路径。
2. 点击模型路径的 `Browse...` 会打开 Windows 文件选择器，并显示 `.onnx`、`.engine`、`.plan` 筛选项；选择文件后 `Model Path` 回填绝对路径。
3. 在任一对话框点击取消，原有路径不变。

### 任务 4：完整回归与变更审阅

**文件：**
- 修改：`docs/superpowers/plans/2026-07-15-native-file-picker.md`（仅勾选已完成步骤）。

- [ ] **步骤 1：执行已存在的轻量测试**

运行：`cmake --build build --target test_input_shape; .\\build\\test_input_shape.exe`

预期：构建成功，进程退出码为 0。

- [ ] **步骤 2：审阅变更范围**

运行：`git diff --check; git diff -- CMakeLists.txt include/common/FileDialog.h src/common/FileDialog.cpp src/dashboard.cpp test/test_file_dialog.cpp`

预期：无空白错误；变更只包含文件选择器、其测试和链接配置。
