#pragma once
#include <opencv2/opencv.hpp>
#include <nppi.h>
#include "../common/CudaMemory.hpp"

class Preprocessor {
public:
    Preprocessor(int dst_width, int dst_height, bool skip_normalization = false);
    ~Preprocessor();

    void process(const cv::Mat& src_img, void* d_dst_buffer, cudaStream_t stream);

private:
    void ensureInputCapacity(size_t required_bytes, cudaStream_t stream);

    int dst_w_;
    int dst_h_;
    bool skip_normalization_;

    void* h_pinned_buffer_ = nullptr;
    size_t pinned_size_ = 0;

    DeviceBuffer d_src_img_ = nullptr;
    DeviceBuffer d_resize_temp_ = nullptr;

    NppiSize oDstSize_;
    NppiRect oDstRectROI_;
};
