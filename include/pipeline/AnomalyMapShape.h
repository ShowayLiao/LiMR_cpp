#pragma once

#include <NvInfer.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace pipeline {

struct AnomalyMapShape {
    int height;
    int width;
    std::size_t elements;
};

inline std::optional<AnomalyMapShape> parse_anomaly_map_shape(const nvinfer1::Dims& dims) {
    const auto to_positive_int = [](int64_t value) -> std::optional<int> {
        if (value <= 0 || value > std::numeric_limits<int>::max()) {
            return std::nullopt;
        }
        return static_cast<int>(value);
    };

    std::optional<int> height;
    std::optional<int> width;

    if (dims.nbDims == 4) {
        if (dims.d[0] != 1 || dims.d[1] != 1) return std::nullopt;
        height = to_positive_int(dims.d[2]);
        width = to_positive_int(dims.d[3]);
    } else if (dims.nbDims == 3) {
        if (dims.d[0] != 1) return std::nullopt;
        height = to_positive_int(dims.d[1]);
        width = to_positive_int(dims.d[2]);
    } else {
        return std::nullopt;
    }

    if (!height || !width) return std::nullopt;

    const std::size_t height_size = static_cast<std::size_t>(*height);
    const std::size_t width_size = static_cast<std::size_t>(*width);
    if (height_size > std::numeric_limits<std::size_t>::max() / width_size) {
        return std::nullopt;
    }

    return AnomalyMapShape{*height, *width, height_size * width_size};
}

} // namespace pipeline
