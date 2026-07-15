#include <cassert>
#include <cstddef>
#include <limits>

#include "engine/InputShape.h"
#include "pipeline/FrameBufferLayout.h"

int main() {
    using trt::InputShape;

    const InputShape fixed{1, 3, 224, 448};
    assert(fixed.isConcrete());
    assert(fixed.elementCount() == 1U * 3U * 224U * 448U);

    const InputShape dynamic{1, 3, -1, -1};
    assert(dynamic.isDynamicSpatial());
    assert(trt::isSupportedDynamicSpatialShape(InputShape{1, 3, 224, 224}));
    assert(trt::isSupportedDynamicSpatialShape(InputShape{1, 3, 448, 448}));
    assert(!trt::isSupportedDynamicSpatialShape(InputShape{1, 3, 256, 256}));

    size_t rgba_bytes = 0;
    assert(pipeline::tryGetFrameBufferBytes(448, 448, 4, rgba_bytes));
    assert(rgba_bytes == 448U * 448U * 4U);

    size_t ignored = 0;
    assert(!pipeline::tryGetFrameBufferBytes(0, 448, 4, ignored));
    assert(!pipeline::tryGetFrameBufferBytes(
        std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), 4, ignored));

    return 0;
}
