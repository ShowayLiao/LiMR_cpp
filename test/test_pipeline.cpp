#include <algorithm>
#include <iostream>
#include <limits>
#include <cmath>
#include <set>
#include <vector>
#include <array>
#include <cstring>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "common/CudaMemory.hpp"
#include "engine/TrtEngine.h"
#include "pipeline/Preprocessor.h"
#include "kernels/HeatmapKernels.h"

namespace {

enum class DiagnosticNormalization { ZeroToOne, ImageNet, MinusOneToOne };

bool runPreprocessDiagnostic(trt::TrtEngine& engine,
                             const cv::Mat& source,
                             const nvinfer1::Dims& inputShape,
                             const nvinfer1::Dims& outputShape,
                             bool rgb,
                             DiagnosticNormalization normalization,
                             const char* label,
                             cudaStream_t stream) {
    const int width = static_cast<int>(inputShape.d[3]);
    const int height = static_cast<int>(inputShape.d[2]);
    cv::Mat resized;
    cv::resize(source, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_LINEAR);

    const size_t plane = static_cast<size_t>(width) * static_cast<size_t>(height);
    std::vector<float> hostInput(plane * 3U);
    constexpr std::array<float, 3> mean{0.485F, 0.456F, 0.406F};
    constexpr std::array<float, 3> stddev{0.229F, 0.224F, 0.225F};
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const cv::Vec3b pixel = resized.at<cv::Vec3b>(y, x);
            const size_t index = static_cast<size_t>(y) * width + x;
            for (int channel = 0; channel < 3; ++channel) {
                const int sourceChannel = rgb ? 2 - channel : channel;
                float value = static_cast<float>(pixel[sourceChannel]) / 255.0F;
                if (normalization == DiagnosticNormalization::ImageNet) {
                    value = (value - mean[channel]) / stddev[channel];
                } else if (normalization == DiagnosticNormalization::MinusOneToOne) {
                    value = value * 2.0F - 1.0F;
                }
                hostInput[static_cast<size_t>(channel) * plane + index] = value;
            }
        }
    }

    DeviceBuffer deviceInput = make_device_buffer(hostInput.size() * sizeof(float));
    if (cudaMemcpyAsync(deviceInput.get(), hostInput.data(), hostInput.size() * sizeof(float),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) return false;
    engine.infer(deviceInput.get(), 1, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) return false;

    const size_t outputPixels = static_cast<size_t>(outputShape.d[2]) * outputShape.d[3];
    std::vector<float> map(outputPixels);
    float score = 0.0F;
    uint8_t predictedLabel = 0;
    std::vector<uint8_t> predictedMask(outputPixels);
    if (cudaMemcpy(&score, engine.getBuffer("pred_score"), sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(map.data(), engine.getBuffer("anomaly_map"), map.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&predictedLabel, engine.getBuffer("pred_label"), sizeof(uint8_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(predictedMask.data(), engine.getBuffer("pred_mask"), predictedMask.size() * sizeof(uint8_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return false;

    std::vector<float> sorted = map;
    std::sort(sorted.begin(), sorted.end());
    const auto percentile = [&](double fraction) {
        return sorted[static_cast<size_t>(fraction * static_cast<double>(sorted.size() - 1U))];
    };
    const size_t above05 = static_cast<size_t>(std::count_if(map.begin(), map.end(), [](float v) { return v > 0.5F; }));
    const size_t above082 = static_cast<size_t>(std::count_if(map.begin(), map.end(), [](float v) { return v > 0.82F; }));
    const int outputWidth = static_cast<int>(outputShape.d[3]);
    const int outputHeight = static_cast<int>(outputShape.d[2]);
    double borderSum = 0.0;
    double totalSum = 0.0;
    size_t belowThree = 0;
    const size_t modelMaskPixels = static_cast<size_t>(std::count_if(
        predictedMask.begin(), predictedMask.end(), [](uint8_t v) { return v != 0; }));
    const auto minimum = std::min_element(map.begin(), map.end());
    const size_t minimumIndex = static_cast<size_t>(std::distance(map.begin(), minimum));
    size_t borderCount = 0;
    for (int y = 0; y < outputHeight; ++y) {
        for (int x = 0; x < outputWidth; ++x) {
            const float value = map[static_cast<size_t>(y) * outputWidth + x];
            totalSum += value;
            if (value < 2.999F) ++belowThree;
            if (x < 4 || y < 4 || x >= outputWidth - 4 || y >= outputHeight - 4) {
                borderSum += value;
                ++borderCount;
            }
        }
    }
    std::cout << label << ": score=" << score
              << " p01=" << percentile(0.01) << " p50=" << percentile(0.50)
              << " p95=" << percentile(0.95) << " p99=" << percentile(0.99)
              << " max=" << sorted.back()
              << " above0.5=" << (100.0 * above05 / map.size()) << "%"
              << " above0.82=" << (100.0 * above082 / map.size()) << "%"
              << " borderMean=" << (borderSum / borderCount)
              << " mean=" << (totalSum / map.size())
              << " below3=" << (100.0 * belowThree / map.size()) << "%"
              << " min=" << *minimum << "@(" << (minimumIndex % outputWidth) << ','
              << (minimumIndex / outputWidth) << ')'
              << " predLabel=" << static_cast<int>(predictedLabel)
              << " predMask=" << (100.0 * modelMaskPixels / predictedMask.size()) << "%" << std::endl;
    return true;
}

} // namespace

int main(int argc, char** argv) {
    const bool freshTrtDiagnostic =
        argc == 2 && std::strcmp(argv[1], "--fresh-trt-diagnostic") == 0;
    if (argc == 2 && std::strcmp(argv[1], "--onnx-diagnostic") == 0) {
        try {
            cv::dnn::Net net = cv::dnn::readNetFromONNX("input/LiMR_merged.onnx");
            const cv::Mat image = cv::imread("input/IMG_9255.png");
            if (net.empty() || image.empty()) return 1;
            cv::Mat blob = cv::dnn::blobFromImage(
                image, 1.0 / 255.0, cv::Size(256, 256), cv::Scalar(), true, false, CV_32F);
            net.setInput(blob, "input");
            const std::vector<cv::String> outputNames = net.getUnconnectedOutLayersNames();
            std::vector<cv::Mat> outputs;
            net.forward(outputs, outputNames);
            for (size_t i = 0; i < outputs.size(); ++i) {
                cv::Mat flat = outputs[i].reshape(1, 1);
                double minimum = 0.0;
                double maximum = 0.0;
                cv::minMaxLoc(flat, &minimum, &maximum);
                std::cout << "ONNX " << outputNames[i] << " shape=";
                for (int d = 0; d < outputs[i].dims; ++d) {
                    if (d != 0) std::cout << 'x';
                    std::cout << outputs[i].size[d];
                }
                std::cout << " range=[" << minimum << ',' << maximum << "]" << std::endl;
            }
            return 0;
        } catch (const cv::Exception& error) {
            std::cerr << "OpenCV ONNX diagnostic failed: " << error.what() << std::endl;
            return 2;
        }
    }

    trt::TrtEngine engine;
    // The fixed-shape ONNX ignores the profile dimensions, but including 256x256 in
    // the cache key deliberately bypasses the legacy 224x224 engine for diagnosis.
    const int diagnosticProfileSize = freshTrtDiagnostic ? 256 : 224;
    if (!engine.load("input/LiMR_merged.onnx", trt::Precision::FP16,
                     diagnosticProfileSize, diagnosticProfileSize)) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    cv::Mat image = cv::imread("input/IMG_9255.png");
    if (image.empty()) {
        std::cerr << "Failed to load test image" << std::endl;
        return 1;
    }

    const nvinfer1::Dims input = engine.getInputShape();
    const nvinfer1::Dims output = engine.getOutputShape();
    if (input.nbDims != 4 || output.nbDims != 4 || input.d[2] <= 0 || input.d[3] <= 0 ||
        output.d[2] <= 0 || output.d[3] <= 0 ||
        input.d[2] > std::numeric_limits<int>::max() || input.d[3] > std::numeric_limits<int>::max()) {
        std::cerr << "Engine returned unresolved shapes" << std::endl;
        return 1;
    }

    const trt::Binding* mapInfo = engine.getTensorInfo("anomaly_map");
    const trt::Binding* scoreInfo = engine.getTensorInfo("pred_score");
    const trt::Binding* labelInfo = engine.getTensorInfo("pred_label");
    if (!mapInfo || !scoreInfo || !labelInfo ||
        mapInfo->type != nvinfer1::DataType::kFLOAT ||
        scoreInfo->type != nvinfer1::DataType::kFLOAT ||
        labelInfo->type != nvinfer1::DataType::kBOOL) {
        std::cerr << "Unexpected TensorRT output types" << std::endl;
        return 1;
    }

    if ((argc == 2 && std::strcmp(argv[1], "--preprocess-diagnostic") == 0) ||
        freshTrtDiagnostic) {
        std::cout << "TensorRT bindings:" << std::endl;
        for (const auto& item : engine.getBindings()) {
            std::cout << "  " << item.first << " type=" << static_cast<int>(item.second.type)
                      << " bytes=" << item.second.size << " dims=[";
            for (int i = 0; i < item.second.dims.nbDims; ++i) {
                if (i != 0) std::cout << ',';
                std::cout << item.second.dims.d[i];
            }
            std::cout << "] input=" << item.second.isInput << std::endl;
        }
        cudaStream_t diagnosticStream = nullptr;
        if (cudaStreamCreate(&diagnosticStream) != cudaSuccess) return 1;
        const cv::Mat black(image.rows, image.cols, CV_8UC3, cv::Scalar(0, 0, 0));
        const cv::Mat white(image.rows, image.cols, CV_8UC3, cv::Scalar(255, 255, 255));
        const cv::Mat alternate = cv::imread("input/006.png");
        const bool ok =
            runPreprocessDiagnostic(engine, image, input, output, false, DiagnosticNormalization::ImageNet,
                                    "BGR ImageNet (current)", diagnosticStream) &&
            runPreprocessDiagnostic(engine, image, input, output, true, DiagnosticNormalization::ImageNet,
                                    "RGB ImageNet", diagnosticStream) &&
            runPreprocessDiagnostic(engine, image, input, output, false, DiagnosticNormalization::ZeroToOne,
                                    "BGR [0,1]", diagnosticStream) &&
            runPreprocessDiagnostic(engine, image, input, output, true, DiagnosticNormalization::ZeroToOne,
                                    "RGB [0,1]", diagnosticStream) &&
            runPreprocessDiagnostic(engine, image, input, output, false, DiagnosticNormalization::MinusOneToOne,
                                    "BGR [-1,1]", diagnosticStream) &&
            runPreprocessDiagnostic(engine, image, input, output, true, DiagnosticNormalization::MinusOneToOne,
                                    "RGB [-1,1]", diagnosticStream) &&
            runPreprocessDiagnostic(engine, black, input, output, true, DiagnosticNormalization::ZeroToOne,
                                    "BLACK RGB [0,1]", diagnosticStream) &&
            runPreprocessDiagnostic(engine, white, input, output, true, DiagnosticNormalization::ZeroToOne,
                                    "WHITE RGB [0,1]", diagnosticStream) &&
            (!alternate.empty() && runPreprocessDiagnostic(
                engine, alternate, input, output, true, DiagnosticNormalization::ZeroToOne,
                "006.png RGB [0,1]", diagnosticStream));
        cudaStreamDestroy(diagnosticStream);
        return ok ? 0 : 1;
    }

    Preprocessor preprocessor(static_cast<int>(input.d[3]), static_cast<int>(input.d[2]));
    DeviceBuffer d_input = make_device_buffer(engine.getInputSize());
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 1;

    preprocessor.process(image, d_input.get(), stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) return 1;
    engine.infer(d_input.get(), 1, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) return 1;

    float* d_score = static_cast<float*>(engine.getBuffer("pred_score"));
    float* d_map = static_cast<float*>(engine.getBuffer("anomaly_map"));
    if (!d_score || !d_map) return 1;

    const size_t pixels = static_cast<size_t>(output.d[2]) * static_cast<size_t>(output.d[3]);
    size_t expectedMapBytes = 0;
    if (!trt::TrtEngine::calculateTensorBytes(output, mapInfo->type, expectedMapBytes) ||
        mapInfo->size != expectedMapBytes || expectedMapBytes != pixels * sizeof(float)) {
        std::cerr << "Unexpected anomaly_map byte size" << std::endl;
        return 1;
    }
    float score = 0.0F;
    std::vector<float> anomalyMap(pixels);
    cudaMemcpy(&score, d_score, sizeof(score), cudaMemcpyDeviceToHost);
    cudaMemcpy(anomalyMap.data(), d_map, pixels * sizeof(float), cudaMemcpyDeviceToHost);

    DeviceBuffer dHeatmap = make_device_buffer(pixels * sizeof(uchar4));
    const size_t reductionBytes = getHeatmapReductionWorkspaceSize(pixels);
    DeviceBuffer dReduction = make_device_buffer(reductionBytes);
    DeviceBuffer dMin = make_device_buffer(sizeof(float));
    DeviceBuffer dMax = make_device_buffer(sizeof(float));
    DeviceBuffer dInvalid = make_device_buffer(sizeof(int));
    if (launchApplyColorMapAutoRange(
            d_map, dHeatmap.get(), static_cast<int>(output.d[3]), static_cast<int>(output.d[2]),
            dReduction.get(), reductionBytes, static_cast<float*>(dMin.get()),
            static_cast<float*>(dMax.get()), static_cast<int*>(dInvalid.get()), stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        std::cerr << "Heatmap rendering failed" << std::endl;
        return 1;
    }
    std::vector<uchar4> heatmap(pixels);
    int invalid = 0;
    cudaMemcpy(heatmap.data(), dHeatmap.get(), pixels * sizeof(uchar4), cudaMemcpyDeviceToHost);
    cudaMemcpy(&invalid, dInvalid.get(), sizeof(int), cudaMemcpyDeviceToHost);
    cudaStreamDestroy(stream);

    const auto range = std::minmax_element(anomalyMap.begin(), anomalyMap.end());
    if (invalid != 0 || !std::all_of(anomalyMap.begin(), anomalyMap.end(), [](float value) {
            return std::isfinite(value);
        })) {
        std::cerr << "anomaly_map contains non-finite values" << std::endl;
        return 1;
    }
    std::set<uint32_t> colors;
    for (const uchar4 pixel : heatmap) {
        colors.insert(static_cast<uint32_t>(pixel.x) |
                      (static_cast<uint32_t>(pixel.y) << 8U) |
                      (static_cast<uint32_t>(pixel.z) << 16U));
    }
    if (colors.size() < 3U) {
        std::cerr << "Heatmap is still saturated to a single color" << std::endl;
        return 1;
    }
    std::cout << "score=" << score << ", anomaly range=[" << *range.first << ", " << *range.second
              << "], heatmap colors=" << colors.size() << std::endl;
    return 0;
}
