#pragma once

#include <cstddef>
#include <cstdint>

#include "rm_aim/backend/compute_backend.hpp"
#include "rm_aim/backend/cpu_backend.hpp"

#ifdef RM_AIM_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace rm_aim::backend {

/// 面向 Jetson Xavier NX (sm_72) / Orin NX (sm_87) 的手写 CUDA 融合计算后端。
///
/// 围绕 Jetson 统一物理内存设计了两个融合核函数：
///   1. bayer_color_thresh_kernel：uchar4 向量化读取 Bayer 原始图完成红蓝差 + 亮度阈值，
///      直接输出 1/2 降采样掩码与 Warp Ballot 活跃行位图（零全图 Demosaic）。
///   2. batch_warp_mlp_kernel：单次启动融合完成批量透视校正（28x20 写入 Shared Memory）
///      + 3 层 MLP（560 -> 64 -> 32 -> 9，[in][out] 合并访存权重）+ Softmax。
class CudaComputeBackend final : public IComputeBackend {
public:
    explicit CudaComputeBackend(MlpWeights weights);
    ~CudaComputeBackend() override;

    CudaComputeBackend(const CudaComputeBackend&) = delete;
    CudaComputeBackend& operator=(const CudaComputeBackend&) = delete;

    const char* name() const noexcept override { return "CUDA (Jetson NX Fused Kernels)"; }

    void preprocess_bayer(const FrameSlot& frame, const PreprocessParams& params,
                          HalfMaskBuffer& out_mask) override;

    void batch_warp_and_classify(const FrameSlot& frame, const WarpPatchRequest* requests,
                                 size_t count, ClassifyResult* out_results) override;

private:
#ifdef RM_AIM_WITH_CUDA
    void ensure_mask_capacity(uint32_t half_w, uint32_t half_h);
    const uint8_t* resolve_device_frame(const FrameSlot& frame);

    cudaStream_t stream_ = nullptr;

    // 常驻显存的 MLP 权重（按 [in_dim][out_dim] 行主序存放以实现 100% 合并访存）
    float* d_w1_ = nullptr;
    float* d_b1_ = nullptr;
    float* d_w2_ = nullptr;
    float* d_b2_ = nullptr;
    float* d_w3_ = nullptr;
    float* d_b3_ = nullptr;

    // 零拷贝 Mapped Pinned 掩码与活跃行缓冲
    uint8_t* h_mask_ = nullptr;
    uint8_t* d_mask_ = nullptr;
    size_t mask_cap_bytes_ = 0;

    uint8_t* h_row_nonzero_ = nullptr;
    uint8_t* d_row_nonzero_ = nullptr;
    size_t row_cap_ = 0;

    // 零拷贝 Mapped Pinned 批量透视请求与分类结果缓冲
    WarpPatchRequest* h_reqs_ = nullptr;
    WarpPatchRequest* d_reqs_ = nullptr;
    ClassifyResult* h_results_ = nullptr;
    ClassifyResult* d_results_ = nullptr;

    // 备用显存中转区（仅当传入 FrameSlot 未使用 cudaHostAllocMapped 时启用）
    uint8_t* d_stage_frame_ = nullptr;
    size_t stage_cap_bytes_ = 0;
#else
    CpuComputeBackend cpu_fallback_;
#endif
};

}  // namespace rm_aim::backend
