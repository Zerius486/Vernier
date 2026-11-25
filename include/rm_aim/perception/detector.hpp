#pragma once

#include <cstdint>
#include <vector>

#include "rm_aim/backend/compute_backend.hpp"
#include "rm_aim/core/config.hpp"
#include "rm_aim/core/types.hpp"

namespace rm_aim::perception {

struct DetectTimingUs {
    double preprocess_us = 0.0;
    double lightbar_pca_us = 0.0;
    double pair_and_classify_us = 0.0;
    double total_us = 0.0;
    bool used_roi = false;
};

/// 传统视觉 + 超轻量 MLP 装甲板检测器（无 YOLO，无全图 Demosaic）。
/// 流水线步骤：
///   1. 计算后端执行 Bayer 红蓝差阈值 -> 输出 1/2 降采样掩码与活跃行位图
///   2. 仅扫描活跃行提取连通域 -> 原图局部灰度加权 PCA 亚像素灯条角点精修
///   3. 几何约束配对装甲板 -> 构造批量逆透视单应矩阵
///   4. 计算后端一次批量执行透视校正 + MLP 分类（28x20 -> 64 -> 32 -> 9）
///   5. 根据检出结果更新下一帧动态 ROI
class ArmorDetector {
public:
    ArmorDetector(DetectorConfig cfg, EnemyColor enemy_color, backend::IComputeBackend& backend);

    void set_enemy_color(EnemyColor color) noexcept { enemy_color_ = color; }
    void reset_roi() noexcept {
        has_roi_ = false;
        roi_frame_counter_ = 0;
    }

    std::vector<Armor> detect(const FrameSlot& frame, DetectTimingUs* timing = nullptr);

    const std::vector<LightBar>& last_lightbars() const noexcept { return lightbars_; }
    const RoiRect& active_roi() const noexcept { return current_roi_; }

private:
    void extract_lightbars_pca(const FrameSlot& frame, const backend::HalfMaskBuffer& mask);
    void pair_armors_and_classify(const FrameSlot& frame, std::vector<Armor>& out_armors);
    void update_dynamic_roi(const FrameSlot& frame, const std::vector<Armor>& armors);

    DetectorConfig cfg_;
    EnemyColor enemy_color_;
    backend::IComputeBackend& backend_;

    backend::HalfMaskBuffer half_mask_;
    std::vector<uint8_t> visited_;
    std::vector<LightBar> lightbars_;

    bool has_roi_ = false;
    RoiRect current_roi_{};
    int roi_frame_counter_ = 0;
};

}  // namespace rm_aim::perception
