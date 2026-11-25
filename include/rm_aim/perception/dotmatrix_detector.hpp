#pragma once

#include <vector>

#include "rm_aim/perception/detector.hpp"

namespace rm_aim::perception {

/// 无灯条点阵装甲板检测器留空占位接口。
/// 待官方正式公布点阵样式、发光排布与几何特征后，在此补充检测与角点拟合实现。
class DotMatrixDetector final : public IArmorDetector {
public:
    DotMatrixDetector(DetectorConfig cfg, EnemyColor enemy_color,
                      backend::IComputeBackend& backend);

    const char* name() const noexcept override { return "DotMatrix (Stub)"; }

    void set_enemy_color(EnemyColor color) noexcept override { enemy_color_ = color; }
    void reset_roi() noexcept override { current_roi_ = RoiRect{}; }

    std::vector<Armor> detect(const FrameSlot& frame, DetectTimingUs* timing = nullptr) override;

    const RoiRect& active_roi() const noexcept override { return current_roi_; }

private:
    [[maybe_unused]] DetectorConfig cfg_;
    EnemyColor enemy_color_;
    [[maybe_unused]] backend::IComputeBackend& backend_;
    RoiRect current_roi_{};
};

}  // namespace rm_aim::perception
