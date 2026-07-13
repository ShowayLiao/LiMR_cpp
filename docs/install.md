# Development environment setup

AnomaRT uses OpenCV for image loading, video decoding, camera input and
`cv::Mat` storage. Its GPU preprocessing and postprocessing use CUDA/NPP and
custom CUDA kernels; **a CUDA-enabled OpenCV build is not required**.

## Required development tools

- Windows 10/11 x64;
- Visual Studio 2022 Build Tools with the **Desktop development with C++**
  workload;
- CMake 3.21 or newer and Ninja;
- CUDA Toolkit 11.8;
- TensorRT 10.12 Windows SDK;
- OpenCV 4.5.5 Windows prebuilt package.

The NVIDIA driver is also required to run or test GPU inference.

## Install OpenCV

Download and extract the standard Windows OpenCV 4.5.5 package. Do not build
OpenCV from source and do not enable `opencv_contrib` or OpenCV CUDA modules
for this project.

Assuming it is extracted to `C:\deps\OpenCV`, the required directories are:

```text
C:\deps\OpenCV\build\x64\vc16\lib  # OpenCVConfig.cmake and import libraries
C:\deps\OpenCV\build\x64\vc16\bin  # runtime DLLs
```

## Configure the project

Set the SDK roots for the current PowerShell session:

```powershell
$env:LIMR_TRT_ROOT = 'C:\deps\TensorRT'
$env:LIMR_OPENCV_DIR = 'C:\deps\OpenCV\build\x64\vc16\lib'
$env:LIMR_OPENCV_BIN_DIR = 'C:\deps\OpenCV\build\x64\vc16\bin'
```

Then configure, build and create a distributable directory:

```powershell
cmake --preset windows-release
cmake --build --preset windows-release --parallel
cmake --install out\build\windows-release
```

The package is written to `out\dist\AnomaRT`. It contains `AnomaRT.exe` and
the runtime DLLs discovered from OpenCV, CUDA and TensorRT.

## Container build

For reproducible Windows builds, use the Windows container workflow in
[`docker/windows/README.md`](../docker/windows/README.md). The toolchain image
needs the same ordinary OpenCV package at `C:\deps\OpenCV`; it does not need
a CUDA-enabled OpenCV build.
