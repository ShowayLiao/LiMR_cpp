#pragma once

#include <cstddef>

#include <cuda_runtime.h>

size_t getHeatmapReductionWorkspaceSize(size_t elementCount);

cudaError_t launchApplyColorMap(const float* dMap,
                                void* dOutRgba,
                                int width,
                                int height,
                                float minValue,
                                float maxValue,
                                cudaStream_t stream);

cudaError_t launchApplyColorMapAutoRange(const float* dMap,
                                         void* dOutRgba,
                                         int width,
                                         int height,
                                         void* dReductionWorkspace,
                                         size_t reductionWorkspaceBytes,
                                         float* dMinValue,
                                         float* dMaxValue,
                                         int* dInvalidValue,
                                         cudaStream_t stream);
