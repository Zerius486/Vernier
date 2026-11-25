#pragma once

#include "rm_aim/backend/compute_backend.hpp"

namespace rm_aim::backend {

/// 高效率 CPU 计算后端（Intel NUC 生产主力，同时作为 CUDA 数值对齐的 Golden Reference）。
/// 采用编译期模板特化零分支 Bayer 循环 + Eigen3 SIMD 向量化矩阵运算。
class CpuComputeBackend final : public IComputeBackend {
public:
    explicit CpuComputeBackend(MlpWeights weights);

    const char* name() const noexcept override { return "CPU (SIMD + Eigen)"; }

    void preprocess_bayer(const FrameSlot& frame, const PreprocessParams& params,
                          HalfMaskBuffer& out_mask) override;

    void batch_warp_and_classify(const FrameSlot& frame, const WarpPatchRequest* requests,
                                 size_t count, ClassifyResult* out_results) override;

private:
    MlpWeights weights_;
};

}  // namespace rm_aim::backend
