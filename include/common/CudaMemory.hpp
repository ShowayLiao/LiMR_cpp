#pragma once
#include <memory>
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

struct CudaDeleter {
    void operator()(void* ptr) const {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};

using DeviceBuffer = std::unique_ptr<void, CudaDeleter>;

inline DeviceBuffer make_device_buffer(size_t size) {
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, size) != cudaSuccess) {
        throw std::runtime_error("CUDA Malloc failed");
    }
    return DeviceBuffer(ptr);
}
