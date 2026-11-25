#include "rm_aim/perception/dotmatrix_detector.hpp"

#include <utility>

namespace rm_aim::perception {

DotMatrixDetector::DotMatrixDetector(DetectorConfig cfg, EnemyColor enemy_color,
                                     backend::IComputeBackend& backend)
    : cfg_(std::move(cfg)), enemy_color_(enemy_color), backend_(backend) {}

std::vector<Armor> DotMatrixDetector::detect(const FrameSlot& frame, DetectTimingUs* timing) {
    (void)frame;
    if (timing != nullptr) {
        *timing = DetectTimingUs{};
    }

    // TODO: 待点阵装甲板官方样式与几何特征明确后实现此处，需满足下游统一接口契约：
    //   1. 返回检出的装甲板列表 std::vector<Armor>；
    //   2. Armor::corners 按顺时针 [左上, 右上, 右下, 左下] 原图像素坐标填写，
    //      与 pnp_solver.hpp 中的 3D 角点顺序保持一致；
    //   3. 填写 color、class_id、confidence、is_large；
    //   4. 如需透视裁剪接 MLP 分类或动态 ROI 追踪，可直接复用
    //      detector_utils.hpp 中的 detail::compute_patch_homography 与
    //      detail::update_dynamic_roi_state。
    return {};
}

std::unique_ptr<IArmorDetector> create_armor_detector(const DetectorConfig& cfg,
                                                      EnemyColor enemy_color,
                                                      backend::IComputeBackend& backend) {
    return std::make_unique<DotMatrixDetector>(cfg, enemy_color, backend);
}

}  // namespace rm_aim::perception
