# Windows 原生文件选择框实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在 Dashboard 的输入媒体和模型路径右侧增加浏览按钮，调用 Windows 原生文件选择框并将选中的绝对路径回填到现有配置中。

**架构：** 将 Win32 `GetOpenFileNameA` 封装在独立的 `common/FileDialog` 单元中；纯路径复制逻辑单独测试，Dashboard 只负责布局、过滤器和调用。取消选择或路径超过目标缓冲区时保持原值不变。

**技术栈：** C++17、Dear ImGui、Win32 Common Dialog、CMake、CTest/assert

---

## 文件结构

- 创建 `include/common/FileDialog.h`：声明路径回填和原生文件选择接口。
- 创建 `src/common/FileDialog.cpp`：实现安全路径回填及 `GetOpenFileNameA` 封装。
- 创建 `test/test_file_dialog.cpp`：覆盖成功、取消、超长路径三种行为。
- 修改 `src/dashboard.cpp`：加入媒体/模型过滤器及两组“路径框 + 浏览按钮 + 标签”。
- 修改 `CMakeLists.txt`：注册测试目标并为应用链接 `comdlg32`。

### 任务 1：用失败测试定义路径回填契约

**文件：**
- 创建：`include/common/FileDialog.h`
- 创建：`test/test_file_dialog.cpp`

- [ ] **步骤 1：声明可测试的路径回填接口**

```cpp
#pragma once

#include <cstddef>

namespace common {

bool copySelectedPath(char* destination,
                      std::size_t destination_size,
                      const char* selected_path);

bool selectFile(char* destination,
                std::size_t destination_size,
                const char* filter);

}  // namespace common
```

- [ ] **步骤 2：编写成功、取消和超长路径测试**

```cpp
#include <cassert>
#include <cstring>

#include "common/FileDialog.h"

int main() {
    char selected[64] = "old-path";
    assert(common::copySelectedPath(selected, sizeof(selected), "C:\\input\\part.png"));
    assert(std::strcmp(selected, "C:\\input\\part.png") == 0);

    char cancelled[64] = "keep-this";
    assert(!common::copySelectedPath(cancelled, sizeof(cancelled), nullptr));
    assert(std::strcmp(cancelled, "keep-this") == 0);

    char too_small[8] = "current";
    assert(!common::copySelectedPath(too_small, sizeof(too_small), "C:\\input\\part.png"));
    assert(std::strcmp(too_small, "current") == 0);
    return 0;
}
```

- [ ] **步骤 3：临时注册测试目标并验证链接失败**

在 `CMakeLists.txt` 的其他轻量测试旁加入：

```cmake
add_executable(test_file_dialog
    ${PROJECT_SOURCE_DIR}/test/test_file_dialog.cpp
)
target_include_directories(test_file_dialog PRIVATE ${PROJECT_SOURCE_DIR}/include)
```

运行：

```powershell
cmake -S . -B build
cmake --build build --target test_file_dialog
```

预期：链接失败，错误指出 `common::copySelectedPath` 尚未定义，证明测试确实约束待实现行为。

- [ ] **步骤 4：提交测试契约**

```powershell
git add include/common/FileDialog.h test/test_file_dialog.cpp CMakeLists.txt
git commit -m "test: define native file picker path contract"
```

### 任务 2：实现文件选择器封装

**文件：**
- 创建：`src/common/FileDialog.cpp`
- 修改：`CMakeLists.txt`
- 测试：`test/test_file_dialog.cpp`

- [ ] **步骤 1：实现保持原值的安全复制与原生对话框**

```cpp
#include "common/FileDialog.h"

#include <cstring>
#include <windows.h>
#include <commdlg.h>

namespace common {

bool copySelectedPath(char* destination,
                      std::size_t destination_size,
                      const char* selected_path) {
    if (destination == nullptr || destination_size == 0 || selected_path == nullptr) {
        return false;
    }

    const std::size_t selected_size = std::strlen(selected_path) + 1;
    if (selected_size > destination_size) {
        return false;
    }

    std::memcpy(destination, selected_path, selected_size);
    return true;
}

bool selectFile(char* destination,
                std::size_t destination_size,
                const char* filter) {
    char selected_path[1024] = {};
    OPENFILENAMEA dialog{};
    dialog.lStructSize = sizeof(dialog);
    dialog.lpstrFile = selected_path;
    dialog.nMaxFile = static_cast<DWORD>(sizeof(selected_path));
    dialog.lpstrFilter = filter;
    dialog.nFilterIndex = 1;
    dialog.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;

    if (GetOpenFileNameA(&dialog) == FALSE) {
        return false;
    }
    return copySelectedPath(destination, destination_size, selected_path);
}

}  // namespace common
```

- [ ] **步骤 2：让测试目标编译实现并链接 Win32 库**

将测试目标改为：

```cmake
add_executable(test_file_dialog
    ${PROJECT_SOURCE_DIR}/test/test_file_dialog.cpp
    ${PROJECT_SOURCE_DIR}/src/common/FileDialog.cpp
)
target_include_directories(test_file_dialog PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_file_dialog PRIVATE comdlg32)
add_test(NAME file_dialog_contract COMMAND test_file_dialog)
```

同时在 `limr` 的链接配置中加入：

```cmake
target_link_libraries(limr comdlg32)
```

- [ ] **步骤 3：构建并运行路径契约测试**

运行：

```powershell
cmake --build build --target test_file_dialog
ctest --test-dir build -R file_dialog_contract --output-on-failure
```

预期：构建成功，`file_dialog_contract` 显示 `Passed`。

- [ ] **步骤 4：提交封装实现**

```powershell
git add include/common/FileDialog.h src/common/FileDialog.cpp test/test_file_dialog.cpp CMakeLists.txt
git commit -m "feat: add Windows native file picker wrapper"
```

### 任务 3：接入 Dashboard 路径控件

**文件：**
- 修改：`src/dashboard.cpp:1-12`
- 修改：`src/dashboard.cpp:416-424`

- [ ] **步骤 1：引入文件选择器并定义与现有加载能力一致的过滤器**

在包含区加入：

```cpp
#include "common/FileDialog.h"
```

在文件内部作用域加入：

```cpp
namespace {

constexpr char kMediaFileFilter[] =
    "Image and video files\0"
    "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.tif;*.avi;*.mp4;*.mov;*.mkv;*.wmv;*.flv\0"
    "All files\0*.*\0\0";

constexpr char kModelFileFilter[] =
    "Model files (*.onnx;*.engine)\0*.onnx;*.engine\0"
    "All files\0*.*\0\0";

}  // namespace
```

模型过滤器不包含 `.plan`，因为 `src/engine/trt_engine.cpp` 当前只接受 `.onnx` 和 `.engine`。

- [ ] **步骤 2：将两个路径输入行改为“输入框 + 浏览按钮 + 标签”**

替换现有两次 `ImGui::InputText`：

```cpp
ImGui::SetNextItemWidth(190.0f);
ImGui::InputText("##InputPath", video_path, sizeof(video_path));
ImGui::SameLine();
if (ImGui::Button("...##BrowseInput")) {
    common::selectFile(video_path, sizeof(video_path), kMediaFileFilter);
}
ImGui::SameLine();
ImGui::TextUnformatted("Input");

ImGui::SetNextItemWidth(190.0f);
ImGui::InputText("##ModelPath", engine_path_a, sizeof(engine_path_a));
ImGui::SameLine();
if (ImGui::Button("...##BrowseModel")) {
    common::selectFile(engine_path_a, sizeof(engine_path_a), kModelFileFilter);
}
ImGui::SameLine();
ImGui::TextUnformatted("Model");
```

保留输入框可编辑性；`...` 按钮在窄侧栏中占用较少宽度。选择成功后写入绝对路径，取消或超长路径时不修改当前值。

- [ ] **步骤 3：构建应用目标**

运行：

```powershell
cmake --build build --target limr
```

预期：构建成功，不出现 `GetOpenFileNameA` 未解析符号或 ImGui ID 冲突。

- [ ] **步骤 4：提交 Dashboard 接入**

```powershell
git add src/dashboard.cpp
git commit -m "feat: browse for input and model files from dashboard"
```

### 任务 4：回归测试与 Windows 手工验收

**文件：**
- 验证：`CMakeLists.txt`
- 验证：`src/dashboard.cpp`
- 验证：`src/common/FileDialog.cpp`

- [ ] **步骤 1：运行轻量自动测试**

```powershell
ctest --test-dir build -R "file_dialog_contract|input_shape_contract|safe_queue_contract" --output-on-failure
```

预期：三个测试全部通过。

- [ ] **步骤 2：检查变更范围与格式**

```powershell
git diff --check
git status --short
git diff -- CMakeLists.txt include/common/FileDialog.h src/common/FileDialog.cpp src/dashboard.cpp test/test_file_dialog.cpp
```

预期：`git diff --check` 无输出；功能变更只涉及计划列出的五个代码/构建文件。

- [ ] **步骤 3：启动应用进行手工验收**

运行：

```powershell
.\build\limr.exe
```

依次确认：

1. 点击输入路径右侧 `...`，弹出 Windows 文件选择框，默认过滤图像和视频；选中文件后回填绝对路径。
2. 点击模型路径右侧 `...`，弹出 Windows 文件选择框，默认过滤 `.onnx` 和 `.engine`；选中文件后回填绝对路径。
3. 在任一对话框点击“取消”，原路径不变。
4. 两个路径框仍可手动编辑，初始化按钮仍使用当前框内的路径。

- [ ] **步骤 4：记录手工验收结果并提交必要的计划勾选更新**

```powershell
git add docs/superpowers/plans/2026-07-16-native-file-picker.md
git commit -m "docs: record native file picker verification"
```
