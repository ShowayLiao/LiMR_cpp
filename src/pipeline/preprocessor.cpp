#include "pipeline/Preprocessor.h"
#include "common/CudaMemory.hpp"
#include <cuda_runtime.h>
#include <npp.h>
#include <stdexcept>

extern void launchLayoutConvertNorm(uint8_t* src, float* dst, int w, int h, cudaStream_t stream);

Preprocessor::Preprocessor(int w, int h) : dst_w_(w), dst_h_(h) {
    oDstSize_ = {dst_w_, dst_h_};
    oDstRectROI_ = {0, 0, dst_w_, dst_h_};

    d_resize_temp_ = make_device_buffer(dst_w_ * dst_h_ * 3 * sizeof(uint8_t));

    // 分配足够大的 pinned 内存，支持 4K 图像
    pinned_size_ = 4096 * 4096 * 3;
    if (cudaMallocHost(&h_pinned_buffer_, pinned_size_) != cudaSuccess) {
        throw std::runtime_error("Failed to allocate pinned memory");
    }
}

Preprocessor::~Preprocessor() {
    if (h_pinned_buffer_) cudaFreeHost(h_pinned_buffer_);
}

void Preprocessor::process(const cv::Mat& src_img, void* d_dst_buffer, cudaStream_t stream) {
    int src_w = src_img.cols;
    int src_h = src_img.rows;
    size_t img_bytes = src_w * src_h * 3 * sizeof(uint8_t);

    if (!d_src_img_) {
        d_src_img_ = make_device_buffer(pinned_size_);
    }

    memcpy(h_pinned_buffer_, src_img.data, img_bytes);
    cudaMemcpyAsync(d_src_img_.get(), h_pinned_buffer_, img_bytes, cudaMemcpyHostToDevice, stream);

    NppiSize oSrcSize = {src_w, src_h};
    NppiRect oSrcRectROI = {0, 0, src_w, src_h};

    nppiResize_8u_C3R(
        (const Npp8u*)d_src_img_.get(), src_w * 3, oSrcSize, oSrcRectROI,
        (Npp8u*)d_resize_temp_.get(), dst_w_ * 3, oDstSize_, oDstRectROI_,
        NPPI_INTER_LINEAR
    );

    launchLayoutConvertNorm(
        (uint8_t*)d_resize_temp_.get(),
        (float*)d_dst_buffer,
        dst_w_, dst_h_, stream
    );
}
