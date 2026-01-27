#pragma once

#include "pipeline/FrameTask.h"
#include "engine/TrtEngine.h"

namespace pipeline {

class PostProcessor {
public:
    PostProcessor();
    ~PostProcessor();

    void process(trt::TrtEngine* engine, FrameTaskPtr task, float threshold, cudaStream_t stream);

private:
    std::vector<float> h_score_buf_;
    std::vector<uint8_t> h_label_buf_;
};

}
