# Build and Installation Guide

This guide describes how to build the current Windows/C++ version of FastVision
from source. The commands and dependency versions below are based on the current
`CMakeLists.txt` and the environment used to build this repository.

> The project does **not** depend on LibTorch, cuDNN, `opencv_contrib`, or an
> OpenCV build with CUDA enabled. CUDA acceleration is implemented directly with
> CUDA, NPP, TensorRT, and CUDA-OpenGL interoperability.

## 1. Supported and verified environment

FastVision is currently a Windows x64 project.

| Component | Verified version | Notes |
| --- | --- | --- |
| Windows | Windows 10 x64 | Windows 11 should also work with a compatible toolchain |
| Visual Studio | Visual Studio 2019 16.x | MSVC v142, x64, C++17 |
| CMake | 3.15.7 | The project declares a minimum version of 3.10 |
| CUDA Toolkit | 11.6 | The current build cache uses `nvcc` 11.6 |
| TensorRT | 10.12.0.36 | Windows x64 package, TensorRT 10 libraries |
| OpenCV | 4.5.5 | Shared Release build |
| GPU | CUDA-capable NVIDIA GPU | A current NVIDIA driver is required |

The exact versions above are the known-good combination. Other versions may
work, but CUDA, Visual Studio, TensorRT, OpenCV, and the generated TensorRT
engine must be mutually compatible.

## 2. Download the required software

Download third-party software only from the official sources below.

| Dependency | Official download |
| --- | --- |
| Visual Studio 2019 | [Microsoft older Visual Studio downloads](https://visualstudio.microsoft.com/vs/older-downloads/) |
| Visual Studio 2019 Community bootstrapper | [vs_community.exe](https://aka.ms/vs/16/release/vs_community.exe) |
| CMake | [CMake downloads](https://cmake.org/download/) |
| CMake older releases | [CMake release archive](https://cmake.org/files/) |
| NVIDIA driver | [NVIDIA driver downloads](https://www.nvidia.com/Download/index.aspx) |
| CUDA Toolkit 11.6 | [CUDA 11.6 download archive](https://developer.nvidia.com/cuda-11-6-0-download-archive?target_arch=x86_64&target_os=Windows&target_version=10) |
| TensorRT | [TensorRT downloads and getting started](https://developer.nvidia.com/tensorrt-getting-started) |
| OpenCV 4.5.5 | [OpenCV 4.5.5 release](https://opencv.org/release/opencv-4-5-5/) |
| GLFW 3.4 | [Official repository](https://github.com/glfw/glfw) · [Release 3.4](https://github.com/glfw/glfw/releases/tag/3.4) · [Source ZIP](https://github.com/glfw/glfw/archive/refs/tags/3.4.zip) |
| Dear ImGui 1.92.5 | [Official repository](https://github.com/ocornut/imgui) · [Release v1.92.5](https://github.com/ocornut/imgui/releases/tag/v1.92.5) · [Source ZIP](https://github.com/ocornut/imgui/archive/refs/tags/v1.92.5.zip) |
| Git for Windows | [Git for Windows](https://git-scm.com/install/windows.html) |

The main repository currently does not track the contents of `3rdparty/glfw`
or `3rdparty/imgui` and does not contain a `.gitmodules` file. Download these
two source trees manually as described in Section 6.

## 3. Install the compiler and build tools

### 3.1 Visual Studio 2019

Install Visual Studio 2019 and select the **Desktop development with C++**
workload. Ensure that these individual components are present:

- MSVC v142 C++ x64/x86 build tools
- Windows 10 SDK
- C++ CMake tools for Windows (optional if standalone CMake is installed)

The verified CUDA 11.6 setup uses the `Visual Studio 16 2019` CMake generator.
If you switch to Visual Studio 2022, use a CUDA Toolkit and OpenCV build that
support the v143 toolchain, then change the generator accordingly.

### 3.2 CMake

Install the Windows x64 CMake package and select **Add CMake to the system
PATH** in the installer.

Verify:

```powershell
cmake --version
```

### 3.3 NVIDIA driver and CUDA Toolkit

Install Visual Studio before installing CUDA so that the CUDA installer can
register its Visual Studio integration.

Install a driver that supports the selected CUDA Toolkit, then install CUDA
11.6. A default installation is normally sufficient.

Verify:

```powershell
nvidia-smi
nvcc --version
```

If `nvcc` is not found, add the CUDA `bin` directory to `PATH`, for example:

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin
```

## 4. Install TensorRT

1. Download the Windows x64 TensorRT 10 package that matches your CUDA
   environment. The verified package is `TensorRT-10.12.0.36`.
2. Extract it to a stable path, for example:

   ```text
   C:\deps\TensorRT-10.12.0.36
   ```

3. Confirm that the extracted directory contains:

   ```text
   include\NvInfer.h
   lib\nvinfer_10.lib
   lib\nvinfer_10.dll
   lib\nvonnxparser_10.lib
   samples\common
   ```

The TensorRT root is passed to CMake as `TRT_DIR`. Do not point `TRT_DIR` at
the `include` or `lib` subdirectory.

For command-line execution outside the build directory, add TensorRT's `lib`
directory to `PATH`:

```text
C:\deps\TensorRT-10.12.0.36\lib
```

## 5. Install OpenCV

The current project uses OpenCV for image/video I/O and CPU-side image
operations. It does not require CUDA-enabled OpenCV or `opencv_contrib`.

### Option A: use an existing compatible OpenCV 4.5.5 installation

Set `OpenCV_DIR` to the directory containing `OpenCVConfig.cmake`. The verified
layout is:

```text
C:\opencv_s\build\install
├── OpenCVConfig.cmake
└── x64\vc16\bin
```

### Option B: build a minimal shared OpenCV 4.5.5 installation

After extracting the OpenCV 4.5.5 source, run:

```powershell
cmake -S C:\deps\opencv-4.5.5 `
      -B C:\deps\opencv-4.5.5-build `
      -G "Visual Studio 16 2019" -A x64 `
      -DCMAKE_INSTALL_PREFIX=C:\deps\opencv-4.5.5-install `
      -DBUILD_SHARED_LIBS=ON `
      -DBUILD_TESTS=OFF `
      -DBUILD_PERF_TESTS=OFF `
      -DBUILD_EXAMPLES=OFF

cmake --build C:\deps\opencv-4.5.5-build `
      --config Release `
      --target INSTALL
```

Use the resulting installation root as `OpenCV_DIR`:

```text
C:\deps\opencv-4.5.5-install
```

If your OpenCV DLL directory differs from `x64\vc16\bin`, add the actual
directory to `PATH`. The project's post-build copy logic currently looks for
the verified OpenCV 4.5.5/vc16 layout.

## 6. Obtain and verify the source tree

Clone the repository or extract a complete source archive:

```powershell
git clone <repository-url>
cd inference_LiMR
```

### 6.1 Download GLFW and Dear ImGui

The current source is verified with GLFW 3.4 and Dear ImGui 1.92.5. From the
repository root, clone the exact versions:

```powershell
New-Item -ItemType Directory -Force 3rdparty
git clone --branch 3.4 --depth 1 `
    https://github.com/glfw/glfw.git 3rdparty/glfw
git clone --branch v1.92.5 --depth 1 `
    https://github.com/ocornut/imgui.git 3rdparty/imgui
```

Alternatively, download the source ZIP files from the table in Section 2.
Extract and rename the directories so that the archive's extra version suffix
is removed:

```text
3rdparty\
├── glfw\
│   ├── CMakeLists.txt
│   ├── include\
│   └── src\
└── imgui\
    ├── imgui.cpp
    ├── imgui.h
    └── backends\
```

Do not place the extracted `glfw-3.4` or `imgui-1.92.5` directory one level
below these paths. `CMakeLists.txt` expects the dependency source roots to be
exactly `3rdparty/glfw` and `3rdparty/imgui`.

### 6.2 Verify the source tree

Verify that the bundled UI dependencies are present:

```text
3rdparty\glfw\CMakeLists.txt
3rdparty\imgui\imgui.cpp
3rdparty\imgui\backends\imgui_impl_glfw.cpp
```

If these files are missing, repeat the download and extraction steps above
before configuring CMake.

## 7. Configure the project

Open **Developer PowerShell for VS 2019**, change to the repository root, and
run:

```powershell
cmake -S . -B build `
      -G "Visual Studio 16 2019" -A x64 `
      -DOpenCV_DIR="C:/deps/opencv-4.5.5-install" `
      -DTRT_DIR="C:/deps/TensorRT-10.12.0.36"
```

Replace both paths with your actual installation paths. Forward slashes are
recommended in CMake command-line paths on Windows.

If more than one CUDA Toolkit is installed, select one explicitly:

```powershell
cmake -S . -B build `
      -G "Visual Studio 16 2019" -A x64 `
      -DOpenCV_DIR="C:/deps/opencv-4.5.5-install" `
      -DTRT_DIR="C:/deps/TensorRT-10.12.0.36" `
      -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6"
```

Do not reuse a build directory generated with a different Visual Studio,
CUDA, TensorRT, or OpenCV installation. Use a new build directory in that
case.

## 8. Build

Build only the application:

```powershell
cmake --build build --config Release --target limr
```

Build the application and all test executables:

```powershell
cmake --build build --config Release
```

The application is generated at:

```text
build\Release\limr.exe
```

The existing `build_test.bat` references the removed `test_single_frame`
target and should not be used for the current source tree.

## 9. Run the tests

Run the registered CTest suite:

```powershell
Push-Location build
ctest -C Release --output-on-failure
Pop-Location
```

Several contract tests are self-contained. The following tests additionally
require a CUDA-capable GPU or local test assets:

- `heatmap_contract` requires a working CUDA device.
- `pipeline_model_contract` requires
  `input/LiMR_merged.onnx`, `input/IMG_9255.png`, compatible TensorRT/CUDA
  libraries, and a working NVIDIA GPU.
- `test_input_thread` is built but is not currently registered with CTest; it
  expects `input/blade.avi` when run manually.

To run the self-contained contract tests only:

```powershell
Push-Location build
ctest -C Release --output-on-failure `
      -R "input_shape_contract|safe_queue_contract|engine_contract|file_dialog_contract|dashboard_task_state_contract|dashboard_render_options_contract|anomaly_map_shape_contract"
Pop-Location
```

## 10. Install or package the application

The CMake install prefix defaults to `build\install`. Build the install target:

```powershell
cmake --build build --config Release --target INSTALL
```

The installed executable is:

```text
build\install\bin\limr.exe
```

During build and installation, CMake attempts to copy the required OpenCV,
CUDA/NPP, and TensorRT DLLs. Review CMake warnings carefully. If a DLL is not
found automatically, add its actual directory to `PATH` or copy the DLL next
to `limr.exe`.

## 11. Run FastVision

Start the Release build:

```powershell
.\build\Release\limr.exe
```

The UI lets you select the input source and model. Model input/output
requirements are documented in [`MODEL_IO.md`](../MODEL_IO.md).

TensorRT serialized engine files are sensitive to TensorRT version, CUDA
version, GPU architecture, and build settings. If an engine cannot be loaded,
use the original ONNX model and regenerate the engine in the target
environment.

## 12. Troubleshooting

### CMake cannot find the CUDA compiler

- Run CMake from Developer PowerShell for the selected Visual Studio version.
- Confirm that `nvcc --version` works.
- If Visual Studio was installed after CUDA, repair or reinstall the CUDA
  Toolkit integration.
- Pass `CUDA_TOOLKIT_ROOT_DIR` explicitly when multiple CUDA versions exist.

### CMake cannot find OpenCV

Set `OpenCV_DIR` to the directory containing `OpenCVConfig.cmake`, not merely
the OpenCV source directory:

```powershell
-DOpenCV_DIR="C:/deps/opencv-4.5.5-install"
```

### `nvinfer_10.lib` or `nvonnxparser_10.lib` is missing

Confirm that `TRT_DIR` is the TensorRT extraction root and that its `lib`
directory contains both `.lib` and `.dll` files.

### A DLL is missing when the executable starts

Check these locations in `PATH`, or copy the required DLLs beside the
executable:

```text
<OpenCV_DIR>\x64\vc16\bin
<CUDA_ROOT>\bin
<TRT_DIR>\lib
```

Typical runtime dependencies include OpenCV 4.5.5, CUDA runtime/NPP/cuBLAS,
and TensorRT 10 DLLs. Exact filenames are listed in `CMakeLists.txt`.

### A TensorRT engine fails to deserialize

Use the same TensorRT major version that produced the engine, or regenerate it
from ONNX on the target machine. Engine files should not be treated as
portable model artifacts.

### CUDA-OpenGL interoperability fails

- Update the NVIDIA driver.
- Ensure that the application is running on the NVIDIA GPU rather than an
  integrated GPU.
- Avoid launching through Remote Desktop configurations that do not expose a
  compatible OpenGL context.

### Tests fail because files are missing

The model pipeline tests rely on files under `input/`, which are not guaranteed
to be present in every source checkout. Add the required local assets or run
the self-contained test subset shown above.
