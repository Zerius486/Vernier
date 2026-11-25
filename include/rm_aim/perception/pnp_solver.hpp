#pragma once

#include "rm_aim/core/config.hpp"
#include "rm_aim/core/types.hpp"

#ifdef RM_AIM_WITH_EIGEN
#include <Eigen/Core>
#endif

namespace rm_aim::perception {

/// RoboMaster 标准大小装甲板物理半宽与半高（单位：m）
inline constexpr double kSmallArmorHalfW = 0.135 * 0.5;
inline constexpr double kLargeArmorHalfW = 0.230 * 0.5;
inline constexpr double kArmorHalfH = 0.055 * 0.5;

/// 共面四点 IPPE PnP 求解器（含正深度与最小重投影误差消歧，以及 1D 黄金分割朝向精修）。
class PnpSolver {
public:
    explicit PnpSolver(const CameraConfig& cam_cfg);

    /// 求解相机光学坐标系（X 右、Y 下、Z 前）下的装甲板 3D 位置（m）与朝向角（rad）。
    /// 结果写入 `armor.position_cam` 与 `armor.yaw_cam`，失败返回 false。
    bool solve(Armor& armor) const;

#ifdef RM_AIM_WITH_EIGEN
    /// 基于已知重力方向与装甲板固定物理倾角（步兵/英雄 +15°，前哨站 -15°，roll=0°），
    /// 使用 1D 黄金分割搜索最小化四角点重投影误差，彻底消除 IPPE 正对翻转歧义。
    /// 参数 R_world2cam 为世界系（FLU）到相机光学系的旋转矩阵，返回优化后的世界系 yaw（rad）。
    double optimize_yaw_1d(const Armor& armor, const Eigen::Vector3d& pos_world,
                           const Eigen::Matrix3d& R_world2cam,
                           const Eigen::Vector3d& t_world2cam,
                           double armor_pitch_rad = 15.0 * 3.14159265358979323846 / 180.0) const;

    /// 去畸变 + 4-DoF（x, y, z, yaw）阻尼 Gauss-Newton 联合位姿精修：
    /// 先对四角点做镜头去畸变消除边缘畸变误差，再用 1D 黄金分割消除朝向翻转，
    /// 最后在物理固定倾角（pitch=±15°, roll=0°）约束下联合优化 3D 位置与朝向。
    void refine_pose_4dof(const Armor& armor, Eigen::Vector3d& pos_world, double& yaw_world,
                          const Eigen::Matrix3d& R_world2cam,
                          const Eigen::Vector3d& t_world2cam,
                          double armor_pitch_rad = 15.0 * 3.14159265358979323846 / 180.0) const;
#endif

private:
    CameraConfig cfg_;
};

}  // namespace rm_aim::perception
