#pragma once

#include <cstddef>
#include <limits>

namespace pipeline {

inline bool tryGetFrameBufferBytes(int width, int height, size_t channels, size_t& bytes) {
    if (width <= 0 || height <= 0 || channels == 0) {
        return false;
    }

    const size_t width_size = static_cast<size_t>(width);
    const size_t height_size = static_cast<size_t>(height);
    if (width_size > std::numeric_limits<size_t>::max() / height_size) {
        return false;
    }

    const size_t pixels = width_size * height_size;
    if (pixels > std::numeric_limits<size_t>::max() / channels) {
        return false;
    }

    bytes = pixels * channels;
    return true;
}

} // namespace pipeline
