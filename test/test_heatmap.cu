#include <cassert>
#include <cstdint>
#include <set>
#include <vector>
#include <iostream>
#include <limits>

#include <cuda_runtime.h>

#include "kernels/HeatmapKernels.h"

#undef assert
#define assert(expression) do { \
    if (!(expression)) { \
        std::cerr << "CHECK failed: " #expression << " at line " << __LINE__ << std::endl; \
        return 1; \
    } \
} while (false)

int main() {
    const std::vector<float> input{1.2F, 1.5F, 2.0F, 3.0F};
    float* dInput = nullptr;
    uchar4* dOutput = nullptr;
    void* dWorkspace = nullptr;
    float* dMin = nullptr;
    float* dMax = nullptr;
    int* dInvalid = nullptr;
    assert(cudaMalloc(&dInput, input.size() * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&dOutput, input.size() * sizeof(uchar4)) == cudaSuccess);
    const size_t workspaceBytes = getHeatmapReductionWorkspaceSize(input.size());
    assert(workspaceBytes > 0U);
    assert(cudaMalloc(&dWorkspace, workspaceBytes) == cudaSuccess);
    assert(cudaMalloc(&dMin, sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&dMax, sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&dInvalid, sizeof(int)) == cudaSuccess);
    assert(cudaMemcpy(dInput, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);

    assert(launchApplyColorMapAutoRange(
        dInput, dOutput, 4, 1, dWorkspace, workspaceBytes, dMin, dMax, dInvalid, nullptr) == cudaSuccess);
    assert(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<uchar4> output(input.size());
    std::vector<float> preserved(input.size());
    assert(cudaMemcpy(output.data(), dOutput, output.size() * sizeof(uchar4), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemcpy(preserved.data(), dInput, preserved.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);

    std::set<uint32_t> colors;
    for (const uchar4 pixel : output) {
        colors.insert(static_cast<uint32_t>(pixel.x) |
                      (static_cast<uint32_t>(pixel.y) << 8U) |
                      (static_cast<uint32_t>(pixel.z) << 16U) |
                      (static_cast<uint32_t>(pixel.w) << 24U));
    }
    assert(colors.size() >= 3U);
    assert(preserved == input);
    float minValue = 0.0F;
    float maxValue = 0.0F;
    int invalid = -1;
    assert(cudaMemcpy(&minValue, dMin, sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemcpy(&maxValue, dMax, sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemcpy(&invalid, dInvalid, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(minValue == 1.2F && maxValue == 3.0F && invalid == 0);

    const std::vector<float> constant{2.0F, 2.0F};
    assert(cudaMemcpy(dInput, constant.data(), constant.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(launchApplyColorMapAutoRange(
        dInput, dOutput, 2, 1, dWorkspace, workspaceBytes, dMin, dMax, dInvalid, nullptr) == cudaSuccess);
    assert(cudaDeviceSynchronize() == cudaSuccess);
    assert(cudaMemcpy(output.data(), dOutput, constant.size() * sizeof(uchar4), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(output[0].x == output[1].x && output[0].y == output[1].y && output[0].z == output[1].z);

    const std::vector<float> invalidInput{1.0F, std::numeric_limits<float>::quiet_NaN()};
    assert(cudaMemcpy(dInput, invalidInput.data(), invalidInput.size() * sizeof(float),
                      cudaMemcpyHostToDevice) == cudaSuccess);
    assert(launchApplyColorMapAutoRange(
        dInput, dOutput, 2, 1, dWorkspace, workspaceBytes, dMin, dMax, dInvalid, nullptr) == cudaSuccess);
    assert(cudaDeviceSynchronize() == cudaSuccess);
    assert(cudaMemcpy(&invalid, dInvalid, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemcpy(output.data(), dOutput, invalidInput.size() * sizeof(uchar4),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(invalid == 1);
    assert(output[0].x == 255 && output[0].y == 0 && output[0].z == 255);
    assert(output[1].x == 255 && output[1].y == 0 && output[1].z == 255);

    cudaFree(dInvalid);
    cudaFree(dMax);
    cudaFree(dMin);
    cudaFree(dWorkspace);
    cudaFree(dOutput);
    cudaFree(dInput);
    return 0;
}
