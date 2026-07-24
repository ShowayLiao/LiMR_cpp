#include "pipeline/AnomalyMapShape.h"

#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>

namespace {

nvinfer1::Dims make_dims(std::initializer_list<int64_t> values) {
    nvinfer1::Dims dims{};
    dims.nbDims = static_cast<int>(values.size());
    int index = 0;
    for (const int64_t value : values) {
        dims.d[index++] = value;
    }
    return dims;
}

} // namespace

int main() {
    int failures = 0;
    const auto expect = [&failures](bool condition, const char* message) {
        if (!condition) {
            std::cerr << "FAILED: " << message << '\n';
            ++failures;
        }
    };

    const auto bchw = pipeline::parse_anomaly_map_shape(make_dims({1, 1, 224, 224}));
    expect(bchw.has_value(), "accept FLOAT [1,1,H,W]");
    if (bchw) {
        expect(bchw->height == 224, "BCHW height");
        expect(bchw->width == 224, "BCHW width");
        expect(bchw->elements == 224U * 224U, "BCHW element count");
    }

    const auto bhw = pipeline::parse_anomaly_map_shape(make_dims({1, 224, 224}));
    expect(bhw.has_value(), "accept FLOAT [1,H,W]");
    if (bhw) {
        expect(bhw->height == 224, "BHW height");
        expect(bhw->width == 224, "BHW width");
        expect(bhw->elements == 224U * 224U, "BHW element count");
    }
    if (bchw && bhw) {
        expect(bhw->elements == bchw->elements, "BCHW and BHW layouts have equal element counts");
    }

    expect(!pipeline::parse_anomaly_map_shape(make_dims({2, 1, 224, 224})),
           "reject BCHW batch greater than one");
    expect(!pipeline::parse_anomaly_map_shape(make_dims({1, 2, 224, 224})),
           "reject BCHW channels greater than one");
    expect(!pipeline::parse_anomaly_map_shape(make_dims({2, 224, 224})),
           "reject BHW batch greater than one");
    expect(!pipeline::parse_anomaly_map_shape(make_dims({224, 224})),
           "reject missing batch dimension");
    expect(!pipeline::parse_anomaly_map_shape(make_dims({1, 1, -1, 224})),
           "reject negative height");
    expect(!pipeline::parse_anomaly_map_shape(make_dims({1, 1, 224, -1})),
           "reject negative width");
    expect(!pipeline::parse_anomaly_map_shape(make_dims({1, 0, 224})),
           "reject zero height");
    expect(!pipeline::parse_anomaly_map_shape(make_dims({1, 224, 0})),
           "reject zero width");
    expect(!pipeline::parse_anomaly_map_shape(make_dims(
               {1, static_cast<int64_t>(std::numeric_limits<int>::max()) + 1, 224})),
           "reject height greater than INT_MAX");
    expect(!pipeline::parse_anomaly_map_shape(make_dims(
               {1, 224, static_cast<int64_t>(std::numeric_limits<int>::max()) + 1})),
           "reject width greater than INT_MAX");

    return failures == 0 ? 0 : 1;
}
