#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "rm_aim/backend/compute_backend.hpp"
#include "rm_aim/core/config.hpp"
#include "rm_aim/core/types.hpp"

namespace rm_aim::perception {

struct DetectTimingUs {
    double preprocess_us = 0.0;
    double feature_extract_us = 0.0;
    double pair_and_classify_us = 0.0;
    double total_us = 0.0;
    bool used_roi = false;
};

/// 装甲板检测器抽象接口，编译期通过 RM_AIM_FEATURE_TYPE 绑定实现：
///   - LIGHTBAR  -> LightbarDetector
///   - DOTMATRIX -> DotMatrixDetector
class IArmorDetector {
public:
    virtual ~IArmorDetector() = default;

    virtual const char* name() const noexcept = 0;
    virtual void set_enemy_color(EnemyColor color) noexcept = 0;
    virtual void reset_roi() noexcept = 0;
    virtual std::vector<Armor> detect(const FrameSlot& frame, DetectTimingUs* timing = nullptr) = 0;
    virtual const RoiRect& active_roi() const noexcept = 0;
};

/// 编译期特征工厂函数：由 CMake 参数 RM_AIM_FEATURE_TYPE 决定链接哪一类检测器实现。
std::unique_ptr<IArmorDetector> create_armor_detector(const DetectorConfig& cfg,
                                                      EnemyColor enemy_color,
                                                      backend::IComputeBackend& backend);

}  // namespace rm_aim::perception
