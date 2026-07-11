# syntax=docker/dockerfile:1
# This image is a *build environment*, not the application runtime image.
# The base image must contain the pinned MSVC Build Tools, CMake/Ninja, CUDA
# Toolkit, TensorRT SDK and OpenCV SDK described in docker/windows/README.md.
ARG LIMR_BUILD_IMAGE=registry.example.com/anomalib-runtime/windows-cuda11.8-trt10.12:latest
FROM ${LIMR_BUILD_IMAGE}

SHELL ["powershell", "-NoProfile", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';"]
WORKDIR C:\src
COPY . C:\src

# SDK locations are image contracts, so every clone builds against the same ABI.
ENV LIMR_TRT_ROOT=C:\deps\TensorRT
ENV LIMR_OPENCV_DIR=C:\deps\OpenCV\x64\vc16\lib
ENV LIMR_OPENCV_BIN_DIR=C:\deps\OpenCV\x64\vc16\bin

RUN cmake --preset windows-release ; `
    cmake --build --preset windows-release --parallel ; `
    cmake --install C:\src\out\build\windows-release
