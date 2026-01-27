@echo off
REM Build script for inference_LiMR project (Windows)

echo Building inference_LiMR project...

REM Create build directory
if not exist build mkdir build
cd build

REM Configure CMake
cmake .. -G "Visual Studio 17 2022" -A x64

REM Build
cmake --build . --config Release

echo Build completed!
cd ..
