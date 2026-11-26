#pragma once

#include <array>

#ifdef RM_AIM_WITH_EIGEN
#include <Eigen/Core>
#endif

namespace rm_aim::estimation {

/// 单轴轻量线性 MPC（双积分器 TinyMPC / ADMM）配置参数。
struct MpcAxisConfig {
    double dt = 0.01;        // 离散预测步长（s）
    int horizon = 25;        // 预测时域步数 N（上限 kMaxHorizon）
    double max_vel = 12.0;   // 角速度硬约束（rad/s）
    double max_acc = 60.0;   // 角加速度硬约束（rad/s^2）
    double q_pos = 200.0;    // 角度跟踪误差权重
    double q_vel = 2.0;      // 角速度跟踪误差权重
    double r_acc = 0.01;     // 角加速度控制惩罚权重
    double rho = 0.05;       // ADMM 增广拉格朗日罚系数
    int max_iters = 30;      // 最大 ADMM 迭代次数
};

/// 单轴 MPC 规划输出（可直接作为云台位置/速度/加速度前馈指令）。
struct MpcPlanOutput {
    double pos = 0.0;      // 规划目标角度（rad）
    double vel = 0.0;      // 规划目标角速度前馈（rad/s）
    double acc = 0.0;      // 规划目标角加速度前馈（rad/s^2）
    int iterations = 0;    // 实际 ADMM 迭代次数
};

#ifdef RM_AIM_WITH_EIGEN

/// 纯 Eigen 零第三方依赖的单轴轻量线性 MPC 轨迹规划器（TinyMPC / ADMM 形式）。
///
/// 动力学模型（云台单轴双积分器）：
///   x_k = [theta_k, omega_k]^T,  u_k = alpha_k
///   x_{k+1} = A * x_k + B * u_k
/// 约束：
///   |omega_k| <= max_vel,  |u_k| <= max_acc
///
/// 构造时离线预计算 2x2 离散代数 Riccati 增益矩阵，在线求解全程栈内存零堆分配，
/// 可在 CPU L1 缓存内以 ~10-30 us 完成切板平滑轨迹规划。
class MpcAxisPlanner {
public:
    static constexpr int kMaxHorizon = 64;

    explicit MpcAxisPlanner(MpcAxisConfig cfg = {});

    /// 给定当前云台状态 (pos0, vel0) 与未来 N+1 步连续展开的参考轨迹 (ref_pos, ref_vel)，
    /// 求解满足速度/加速度硬约束的最优平滑轨迹并返回控制指令。
    MpcPlanOutput solve(double pos0, double vel0, const double* ref_pos, const double* ref_vel,
                        int ref_len) const noexcept;

    const MpcAxisConfig& config() const noexcept { return cfg_; }
    int horizon() const noexcept { return N_; }
    double dt() const noexcept { return cfg_.dt; }

private:
    void precompute_riccati() noexcept;

    MpcAxisConfig cfg_{};
    int N_ = 25;

    Eigen::Matrix2d A_ = Eigen::Matrix2d::Identity();
    Eigen::Vector2d B_ = Eigen::Vector2d::Zero();
    Eigen::Matrix2d P_ = Eigen::Matrix2d::Identity();
    Eigen::RowVector2d K_ = Eigen::RowVector2d::Zero();
    Eigen::Matrix2d A_cl_T_ = Eigen::Matrix2d::Identity();
    double c1_inv_ = 1.0;  // (R_tilde + B^T P B)^(-1)
};

#endif  // RM_AIM_WITH_EIGEN

}  // namespace rm_aim::estimation
