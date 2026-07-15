#include <cassert>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <iostream>

#include "engine/TrtEngine.h"

#undef assert
#define assert(expression) do { \
    if (!(expression)) { \
        std::cerr << "CHECK failed: " #expression << " at line " << __LINE__ << std::endl; \
        return 1; \
    } \
} while (false)

int main() {
    size_t bytes = 0;
    assert(trt::TrtEngine::calculateTensorBytes(
        nvinfer1::Dims4{1, 1, 256, 256}, nvinfer1::DataType::kFLOAT, bytes));
    assert(bytes == 1U * 1U * 256U * 256U * sizeof(float));

    assert(!trt::TrtEngine::calculateTensorBytes(
        nvinfer1::Dims4{1, 1, -1, 256}, nvinfer1::DataType::kFLOAT, bytes));
    assert(!trt::TrtEngine::calculateTensorBytes(
        nvinfer1::Dims4{1, 1, 0, 256}, nvinfer1::DataType::kFLOAT, bytes));

    nvinfer1::Dims overflowDims{};
    overflowDims.nbDims = 3;
    overflowDims.d[0] = std::numeric_limits<int32_t>::max();
    overflowDims.d[1] = std::numeric_limits<int32_t>::max();
    overflowDims.d[2] = std::numeric_limits<int32_t>::max();
    assert(!trt::TrtEngine::calculateTensorBytes(
        overflowDims, nvinfer1::DataType::kFLOAT, bytes));

    const std::filesystem::path fp16 = trt::TrtEngine::getEngineCachePath(
        "input/LiMR_merged.onnx", trt::Precision::FP16, 256, 256);
    const std::filesystem::path fp32 = trt::TrtEngine::getEngineCachePath(
        "input/LiMR_merged.onnx", trt::Precision::FP32, 256, 256);
    assert(fp16.filename().string().find("LiMR_merged.fp16.256x256.onnx-") == 0);
    assert(fp32.filename().string().find("LiMR_merged.fp32.256x256.onnx-") == 0);
    assert(fp16 != fp32);
    std::cout << "Verified cache path: " << fp16.string() << std::endl;
    std::cout << "Default cache path: "
              << trt::TrtEngine::getEngineCachePath(
                     "input/LiMR_merged.onnx", trt::Precision::FP16, 224, 224).string()
              << std::endl;

    const std::filesystem::path identityModel =
        std::filesystem::current_path() / "cache_identity_test.onnx";
    {
        std::ofstream output(identityModel, std::ios::binary | std::ios::trunc);
        output << "first model";
    }
    const auto firstIdentity = trt::TrtEngine::getEngineCachePath(
        identityModel, trt::Precision::FP16, 256, 256);
    {
        std::ofstream output(identityModel, std::ios::binary | std::ios::trunc);
        output << "second model with different content";
    }
    const auto secondIdentity = trt::TrtEngine::getEngineCachePath(
        identityModel, trt::Precision::FP16, 256, 256);
    std::filesystem::remove(identityModel);
    assert(firstIdentity != secondIdentity);

    return 0;
}
