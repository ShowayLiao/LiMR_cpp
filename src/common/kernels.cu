#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/device/device_reduce.cuh>
#include <limits>

#include "kernels/HeatmapKernels.h"

__device__ float clamp(float v, float min, float max) { return fmaxf(min, fminf(v, max)); }

// [修改] 返回类型改为 uchar4
__device__ uchar4 floatToJet(float v) {
    uchar4 c;
    float r = clamp(1.5f - fabsf(4.0f * v - 3.0f), 0.0f, 1.0f);
    float g = clamp(1.5f - fabsf(4.0f * v - 2.0f), 0.0f, 1.0f);
    float b = clamp(1.5f - fabsf(4.0f * v - 1.0f), 0.0f, 1.0f);
    
    c.x = (unsigned char)(r * 255.0f);
    c.y = (unsigned char)(g * 255.0f);
    c.z = (unsigned char)(b * 255.0f);
    c.w = 255; // [关键] Alpha 通道设为 255 (不透明)
    return c;
}

// [修改] output 指针改为 uchar4*
__global__ void validateFiniteKernel(const float* map, int count, int* invalidValue) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && !isfinite(map[idx])) atomicExch(invalidValue, 1);
}

__global__ void applyColorMapKernel(const float* map,
                                    uchar4* output,
                                    int w,
                                    int h,
                                    float minValue,
                                    float maxValue,
                                    const int* invalidValue) {
    int idx = (blockIdx.y * blockDim.y + threadIdx.y) * w + (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < w * h) {
        if ((invalidValue && *invalidValue != 0) || !isfinite(map[idx]) ||
            !isfinite(minValue) || !isfinite(maxValue)) {
            output[idx] = make_uchar4(255, 0, 255, 255);
            return;
        }

        const float range = maxValue - minValue;
        const float normalized = range > 1e-12f ? clamp((map[idx] - minValue) / range, 0.0f, 1.0f) : 0.0f;
        output[idx] = floatToJet(normalized);
    }
}

__global__ void applyColorMapAutoRangeKernel(const float* map,
                                             uchar4* output,
                                             int w,
                                             int h,
                                             const float* minValue,
                                             const float* maxValue,
                                             const int* invalidValue) {
    const int idx = (blockIdx.y * blockDim.y + threadIdx.y) * w +
        (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= w * h) return;

    const float minVal = *minValue;
    const float maxVal = *maxValue;
    if (*invalidValue != 0 || !isfinite(map[idx]) || !isfinite(minVal) || !isfinite(maxVal)) {
        output[idx] = make_uchar4(255, 0, 255, 255);
        return;
    }

    const float range = maxVal - minVal;
    const float normalized = range > 1e-12f ? clamp((map[idx] - minVal) / range, 0.0f, 1.0f) : 0.0f;
    output[idx] = floatToJet(normalized);
}

size_t getHeatmapReductionWorkspaceSize(size_t elementCount) {
    if (elementCount == 0 || elementCount > static_cast<size_t>(std::numeric_limits<int>::max())) return 0;
    const int count = static_cast<int>(elementCount);
    size_t minBytes = 0;
    size_t maxBytes = 0;
    cub::DeviceReduce::Min(nullptr, minBytes, static_cast<const float*>(nullptr),
                           static_cast<float*>(nullptr), count);
    cub::DeviceReduce::Max(nullptr, maxBytes, static_cast<const float*>(nullptr),
                           static_cast<float*>(nullptr), count);
    return minBytes > maxBytes ? minBytes : maxBytes;
}

cudaError_t launchApplyColorMap(const float* dMap,
                                void* dOutRgba,
                                int w,
                                int h,
                                float minValue,
                                float maxValue,
                                cudaStream_t stream) {
    if (!dMap || !dOutRgba || w <= 0 || h <= 0) return cudaErrorInvalidValue;
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    applyColorMapKernel<<<grid, block, 0, stream>>>(
        dMap, static_cast<uchar4*>(dOutRgba), w, h, minValue, maxValue, nullptr);
    return cudaGetLastError();
}

cudaError_t launchApplyColorMapAutoRange(const float* dMap,
                                         void* dOutRgba,
                                         int w,
                                         int h,
                                         void* dReductionWorkspace,
                                         size_t reductionWorkspaceBytes,
                                         float* dMinValue,
                                         float* dMaxValue,
                                         int* dInvalidValue,
                                         cudaStream_t stream) {
    if (!dMap || !dOutRgba || !dReductionWorkspace || !dMinValue || !dMaxValue ||
        !dInvalidValue || w <= 0 || h <= 0) {
        return cudaErrorInvalidValue;
    }

    const size_t count = static_cast<size_t>(w) * static_cast<size_t>(h);
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) return cudaErrorInvalidValue;
    const int elementCount = static_cast<int>(count);
    const size_t requiredBytes = getHeatmapReductionWorkspaceSize(count);
    if (requiredBytes == 0 || reductionWorkspaceBytes < requiredBytes) return cudaErrorInvalidValue;

    cudaError_t status = cudaMemsetAsync(dInvalidValue, 0, sizeof(int), stream);
    if (status != cudaSuccess) return status;

    constexpr int threads = 256;
    const int blocks = (elementCount + threads - 1) / threads;
    validateFiniteKernel<<<blocks, threads, 0, stream>>>(dMap, elementCount, dInvalidValue);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;

    size_t tempBytes = reductionWorkspaceBytes;
    status = cub::DeviceReduce::Min(
        dReductionWorkspace, tempBytes, dMap, dMinValue, elementCount, stream);
    if (status != cudaSuccess) return status;

    tempBytes = reductionWorkspaceBytes;
    status = cub::DeviceReduce::Max(
        dReductionWorkspace, tempBytes, dMap, dMaxValue, elementCount, stream);
    if (status != cudaSuccess) return status;

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    applyColorMapAutoRangeKernel<<<grid, block, 0, stream>>>(
        dMap, static_cast<uchar4*>(dOutRgba), w, h, dMinValue, dMaxValue, dInvalidValue);
    return cudaGetLastError();
}
