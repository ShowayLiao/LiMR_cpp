#!/bin/bash

# Run script for inference_LiMR project

echo "Running inference_LiMR..."

if [ ! -f "build/inference_LiMR" ]; then
    echo "Error: Executable not found. Please build the project first."
    exit 1
fi

./build/inference_LiMR "$@"
