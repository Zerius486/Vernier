#pragma once

#include "rm_aim/core/config.hpp"
#include "rm_aim/core/types.hpp"
#include "rm_aim/estimation/ekf_tracker.hpp"

namespace rm_aim::estimation {

struct BallisticSolution {
    bool valid = false;
    double yaw_rad = 0.0;        // 目标世界系 yaw（向左为正）
    double pitch_rad = 0.0;      // 目标世界系 pitch（抬头为正）
    double fly_time_s = 0.0;     // 子弹飞行时间（s）
    double horizontal_dist = 0.0;
};

/// 带空气阻力（线性阻力近似 + 竖直面迭代）的 2D 弹道解算器。
/// 坐标系约定：FLU 右手系，Yaw 向左为正，Pitch 抬头为正，水平为 0 rad。
class BallisticSolver {
public:
    explicit BallisticSolver(double air_drag_k = 0.019, double gravity = 9.7933);

    /// 给定目标在世界系下的 3D 坐标 (x, y, z) 与弹速 v0（m/s），求解发射仰角与飞行时间。
    BallisticSolution solve(double x, double y, double z, double bullet_speed) const noexcept;

private:
    double k_ = 0.019;   // 17mm 弹丸空气阻力系数（1/m）
    double g_ = 9.7933;  // 当地重力加速度（m/s^2）
};

#ifdef RM_AIM_WITH_EIGEN

/// 选板、弹道耦合预测与前馈火控解算器（完整对接 h7_framework VisionToGimbal 协议）。
class Aimer {
public:
    explicit Aimer(AimerConfig cfg);

    /// 基于当前跟踪器状态与当前云台 IMU 姿态，计算期望云台指令（含位置/速度/加速度前馈）。
    /// 切板瞬间自动对前馈做安全限幅/置零，避免阶跃跳变引起云台抽搐。
    GimbalCmd compute_command(const EkfTracker& tracker, const ImuSample& current_imu);

    int selected_armor_id() const noexcept { return last_armor_id_; }
    bool last_switched() const noexcept { return last_switched_; }
    double filtered_bullet_speed() const noexcept { return filtered_bullet_speed_; }

private:
    int select_armor_id(const VehicleState& state, double center_yaw) const;
    BallisticSolution solve_for_armor_at(const EkfTracker& tracker, int armor_id,
                                         double dt, double bullet_speed) const;
    uint8_t compute_adaptive_fire_rate(double dist_m, double yaw_err_rad, double v_yaw) const noexcept;
    void update_bullet_speed(const ImuSample& imu) noexcept;

    AimerConfig cfg_;
    BallisticSolver ballistic_;
    int last_armor_id_ = -1;
    bool last_switched_ = false;
    double filtered_bullet_speed_ = 24.0;
    uint16_t last_bullet_count_ = 0;
    bool has_bullet_count_ = false;
};

#endif  // RM_AIM_WITH_EIGEN

}  // namespace rm_aim::estimation
