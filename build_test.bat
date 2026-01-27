@echo off
REM Build script for test_single_frame

echo Building test_single_frame...

if not exist build mkdir build
cd build

REM Configure CMake
cmake .. -G "Visual Studio 17 2022" -A x64

if %ERRORLEVEL% NEQ 0 (
    echo CMake configuration failed!
    exit /b %ERRORLEVEL%
)

REM Build the test executable
cmake --build . --config Release --target test_single_frame

if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    exit /b %ERRORLEVEL%
)

echo Build completed successfully!
echo.
echo To run the test, execute: build\Release\test_single_frame.exe
cd ..
