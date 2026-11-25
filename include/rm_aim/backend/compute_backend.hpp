#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "rm_aim/backend/mlp_weights.hpp"
#include "rm_aim/core/types.hpp"

namespace rm_aim::backend {

inline constexpr size_t kMaxBatchArmors = 16;

struct PreprocessParams {
    EnemyColor enemy_color = EnemyColor::Blue;
    int color_diff_thresh = 50;
    int brightness_thresh = 140;
    /// 可选动态 ROI（原图坐标系）；若 width == 0 或 height == 0 则处理整帧全图。
    RoiRect roi{};
};

/// 1/2 降采样 Bayer 颜色阈值掩码输出缓冲。
/// 掩码坐标 (hx, hy) 对应原图坐标 (roi_x + 2*hx, roi_y + 2*hy)。
struct HalfMaskBuffer {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t roi_x = 0;  // 向下对齐到偶数像素，保持 Bayer 2x2 相位
    uint32_t roi_y = 0;  // 向下对齐到偶数像素，保持 Bayer 2x2 相位
    std::vector<uint8_t> mask;         // width * height（0 或 255）
    std::vector<uint8_t> row_nonzero;  // height（0 或 1），用于 CPU 快速跳过全黑背景行

    void resize(uint32_t w, uint32_t h) {
        width = w;
        height = h;
        mask.resize(static_cast<size_t>(w) * h);
        row_nonzero.assign(h, 0);
    }
};

/// 逆透视单应矩阵请求：将归一化数字图坐标 (u ∈ [0, 20), v ∈ [0, 28)) 映射回原图坐标 (x, y)。
struct WarpPatchRequest {
    std::array<float, 9> h_inv{};  // 行主序 3x3 逆单应矩阵
};

struct ClassifyResult {
    uint8_t class_id = 0;
    float confidence = 0.0f;
    std::array<float, kMlpNumClasses> probs{};
};

/// 极简计算后端抽象接口，由以下两个后端共同实现：
/// - CpuComputeBackend（x86_64 NUC AVX2 + Eigen3 生产路径，同时作为 CUDA 精度基准）
/// - CudaComputeBackend（Jetson NX 手写融合 CUDA 核函数后端）
class IComputeBackend {
public:
    virtual ~IComputeBackend() = default;

    virtual const char* name() const noexcept = 0;

    /// 算子 1：Bayer 原始域红蓝差 + 亮度阈值 -> 输出 1/2 降采样掩码与活跃行位图。
    virtual void preprocess_bayer(const FrameSlot& frame, const PreprocessParams& params,
                                  HalfMaskBuffer& out_mask) = 0;

    /// 算子 2：批量透视校正（28x20）+ 3 层 MLP 推理 + Softmax。
    virtual void batch_warp_and_classify(const FrameSlot& frame,
                                         const WarpPatchRequest* requests, size_t count,
                                         ClassifyResult* out_results) = 0;
};

std::unique_ptr<IComputeBackend> create_compute_backend(const MlpWeights& weights,
                                                        bool prefer_cuda = true);

}  // namespace rm_aim::backend
