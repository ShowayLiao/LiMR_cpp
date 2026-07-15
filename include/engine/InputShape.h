#pragma once

#include <cstddef>

namespace trt {

struct InputShape {
    int batch;
    int channels;
    int height;
    int width;

    bool isConcrete() const {
        return batch > 0 && channels > 0 && height > 0 && width > 0;
    }

    bool isDynamicSpatial() const {
        return height < 0 || width < 0;
    }

    std::size_t elementCount() const {
        return static_cast<std::size_t>(batch) * channels * height * width;
    }
};

inline bool isSupportedDynamicSpatialShape(const InputShape& shape) {
    return shape.batch == 1 && shape.channels == 3 &&
           ((shape.height == 224 && shape.width == 224) ||
            (shape.height == 448 && shape.width == 448));
}

} // namespace trt
