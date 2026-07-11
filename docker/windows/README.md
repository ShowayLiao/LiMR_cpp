# Windows container build

This project produces a native Windows CUDA executable. It must therefore be
built in a **Windows container** (Docker Desktop switched to Windows
containers), not a Linux/WSL container.

Clone with the pinned source dependencies:

```powershell
git clone --recurse-submodules <repository-url>
```

For an existing clone, run `git submodule update --init --recursive` before
building. ImGui and GLFW are submodules, pinned respectively to v1.92.5 and
3.4; their compiled outputs are not committed.

`Dockerfile` consumes an internal, versioned toolchain image. Build that image
once per toolchain change and pin it by digest in CI. It must contain:

- Windows Server Core compatible with the build host;
- MSVC Build Tools 2022, CMake and Ninja;
- CUDA Toolkit 11.8;
- TensorRT 10.12 Windows SDK at `C:\deps\TensorRT`;
- OpenCV 4.5.5 Windows SDK at `C:\deps\OpenCV`.

`CMakePresets.json` expects the OpenCV CMake package at
`C:\deps\OpenCV\x64\vc16\lib`, which is the layout used by the pinned SDK
image.

TensorRT redistribution must comply with NVIDIA's applicable license. Keep the
SDK/image in a private artifact registry; do not commit vendor SDK archives to
Git.

Build a distributable package:

```powershell
docker build --platform windows/amd64 `
  --build-arg LIMR_BUILD_IMAGE=registry.example.com/anomalib-runtime/windows-cuda11.8-trt10.12@sha256:<digest> `
  -f docker/windows/Dockerfile -t anomalib-runtime-build:local .
docker create --name anomalib-runtime-export anomalib-runtime-build:local
docker cp anomalib-runtime-export:C:\src\out\dist\AnomaRT .\out\AnomaRT
docker rm anomalib-runtime-export
```

The exported `out\AnomaRT` directory is a native Windows application. Docker is
not needed on the end-user machine. The machine still needs a supported NVIDIA
driver because drivers cannot be shipped by an application package.
