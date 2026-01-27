#!/bin/bash

# Build script for inference_LiMR project

echo "Building inference_LiMR project..."

# Create build directory
mkdir -p build
cd build

# Configure CMake
cmake ..

# Build
cmake --build . -j$(nproc)

echo "Build completed!"
