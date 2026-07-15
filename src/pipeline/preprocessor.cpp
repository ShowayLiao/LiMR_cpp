#include "pipeline/Preprocessor.h"
#include "common/CudaMemory.hpp"
#include <cuda_runtime.h>
#include <npp.h>
#include <cstring>
#include <stdexcept>
#include <string>

extern void launchLayoutConvertNorm(uint8_t* src, float* dst, int w, int h, cudaStream_t stream, bool skip_normalization);

Preprocessor::Preprocessor(int w, int h, bool skip_normalization)
    : dst_w_(w), dst_h_(h), skip_normalization_(skip_normalization) {
    oDstSize_ = {dst_w_, dst_h_};
    oDstRectROI_ = {0, 0, dst_w_, dst_h_};
    d_resize_temp_ = make_device_buffer(static_cast<size_t>(dst_w_) * dst_h_ * 3U);

    pinned_size_ = 1920U * 1080U * 3U;
    if (cudaMallocHost(&h_pinned_buffer_, pinned_size_) != cudaSuccess) {
        throw std::runtime_error("Failed to allocate pinned memory");
    }
}

Preprocessor::~Preprocessor() {
    if (h_pinned_buffer_) cudaFreeHost(h_pinned_buffer_);
}

void Preprocessor::ensureInputCapacity(size_t required_bytes, cudaStream_t stream) {
    if (required_bytes == 0) {
        throw std::invalid_argument("Preprocessor input size must be non-zero");
    }
    if (d_src_img_ && required_bytes <= pinned_size_) return;

    const cudaError_t sync_status = cudaStreamSynchronize(stream);
    if (sync_status != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to synchronize before resizing preprocessor buffers: ") +
                                 cudaGetErrorString(sync_status));
    }

    void* new_pinned_buffer = nullptr;
    const cudaError_t allocation_status = cudaMallocHost(&new_pinned_buffer, required_bytes);
    if (allocation_status != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to allocate pinned memory: ") + cudaGetErrorString(allocation_status));
    }

    DeviceBuffer new_device_buffer;
    try {
        new_device_buffer = make_device_buffer(required_bytes);
    } catch (...) {
        cudaFreeHost(new_pinned_buffer);
        throw;
    }

    cudaFreeHost(h_pinned_buffer_);
    h_pinned_buffer_ = new_pinned_buffer;
    d_src_img_ = std::move(new_device_buffer);
    pinned_size_ = required_bytes;
}

void Preprocessor::process(const cv::Mat& src_img, void* d_dst_buffer, cudaStream_t stream) {
    if (src_img.empty() || src_img.type() != CV_8UC3 || !d_dst_buffer) {
        throw std::invalid_argument("Preprocessor requires a non-empty CV_8UC3 image and output buffer");
    }

    const cv::Mat contiguous = src_img.isContinuous() ? src_img : src_img.clone();
    const int src_w = contiguous.cols;
    const int src_h = contiguous.rows;
    const size_t img_bytes = contiguous.total() * contiguous.elemSize();
    ensureInputCapacity(img_bytes, stream);

    std::memcpy(h_pinned_buffer_, contiguous.data, img_bytes);
    cudaMemcpyAsync(d_src_img_.get(), h_pinned_buffer_, img_bytes, cudaMemcpyHostToDevice, stream);

    NppStreamContext npp_stream_context{};
    NppStatus npp_status = nppGetStreamContext(&npp_stream_context);
    if (npp_status != NPP_SUCCESS) {
        throw std::runtime_error("Failed to create the NPP stream context");
    }
    npp_stream_context.hStream = stream;

    NppiSize oSrcSize = {src_w, src_h};
    NppiRect oSrcRectROI = {0, 0, src_w, src_h};
    npp_status = nppiResize_8u_C3R_Ctx(
        static_cast<const Npp8u*>(d_src_img_.get()), src_w * 3, oSrcSize, oSrcRectROI,
        static_cast<Npp8u*>(d_resize_temp_.get()), dst_w_ * 3, oDstSize_, oDstRectROI_,
        NPPI_INTER_LINEAR, npp_stream_context);
    if (npp_status != NPP_SUCCESS) {
        throw std::runtime_error("NPP resize failed");
    }

    launchLayoutConvertNorm(
        static_cast<uint8_t*>(d_resize_temp_.get()), static_cast<float*>(d_dst_buffer),
        dst_w_, dst_h_, stream, skip_normalization_);
}
