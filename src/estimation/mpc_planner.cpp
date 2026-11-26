#include "rm_aim/estimation/mpc_planner.hpp"

#include <algorithm>
#include <cmath>

namespace rm_aim::estimation {

#ifdef RM_AIM_WITH_EIGEN

MpcAxisPlanner::MpcAxisPlanner(MpcAxisConfig cfg) : cfg_(cfg) {
    cfg_.dt = std::clamp(cfg_.dt, 0.002, 0.05);
    cfg_.horizon = std::clamp(cfg_.horizon, 4, kMaxHorizon);
    cfg_.max_vel = std::max(0.5, cfg_.max_vel);
    cfg_.max_acc = std::max(1.0, cfg_.max_acc);
    cfg_.q_pos = std::max(1e-3, cfg_.q_pos);
    cfg_.q_vel = std::max(0.0, cfg_.q_vel);
    cfg_.r_acc = std::max(1e-5, cfg_.r_acc);
    cfg_.rho = std::max(1e-3, cfg_.rho);
    cfg_.max_iters = std::clamp(cfg_.max_iters, 5, 100);
    N_ = cfg_.horizon;

    precompute_riccati();
}

void MpcAxisPlanner::precompute_riccati() noexcept {
    const double dt = cfg_.dt;
    A_ << 1.0, dt,
          0.0, 1.0;
    B_ << 0.5 * dt * dt, dt;

    // 增广二次代价：状态仅对角速度分量施加盒子约束，因此 ADMM 罚项 rho 仅叠加在 (1,1)
    Eigen::Matrix2d Q_tilde;
    Q_tilde << cfg_.q_pos, 0.0,
               0.0, cfg_.q_vel + cfg_.rho;
    const double R_tilde = cfg_.r_acc + cfg_.rho;

    // 通过定常 2x2 离散 Riccati 递推离线求解无穷时域稳态 P 与 LQR 反馈增益 K
    P_ = Q_tilde;
    for (int iter = 0; iter < 150; ++iter) {
        const double denom = R_tilde + (B_.transpose() * P_ * B_)(0, 0);
        c1_inv_ = 1.0 / std::max(1e-9, denom);
        K_ = c1_inv_ * (B_.transpose() * P_ * A_);
        const Eigen::Matrix2d P_next =
            Q_tilde + A_.transpose() * P_ * A_ - (A_.transpose() * P_ * B_) * K_;
        if ((P_next - P_).cwiseAbs().maxCoeff() < 1e-10) {
            P_ = P_next;
            break;
        }
        P_ = P_next;
    }

    const double denom = R_tilde + (B_.transpose() * P_ * B_)(0, 0);
    c1_inv_ = 1.0 / std::max(1e-9, denom);
    K_ = c1_inv_ * (B_.transpose() * P_ * A_);
    A_cl_T_ = (A_ - B_ * K_).transpose();
}

MpcPlanOutput MpcAxisPlanner::solve(double pos0, double vel0, const double* ref_pos,
                                    const double* ref_vel, int ref_len) const noexcept {
    MpcPlanOutput out{};
    out.pos = pos0;
    out.vel = 0.0;
    out.acc = 0.0;
    if (ref_pos == nullptr || ref_len <= 0 || !std::isfinite(pos0) || !std::isfinite(vel0)) {
        return out;
    }

    // 全程固定大小栈数组，零堆分配，完整常驻 CPU L1 缓存
    std::array<Eigen::Vector2d, kMaxHorizon + 1> x{};
    std::array<Eigen::Vector2d, kMaxHorizon + 1> p{};
    std::array<Eigen::Vector2d, kMaxHorizon + 1> q_ref{};
    std::array<double, kMaxHorizon> u{};
    std::array<double, kMaxHorizon> d{};

    // ADMM 约束投影辅助变量 z 与缩放对偶变量 y（分别对应角加速度 u 与角速度 omega）
    std::array<double, kMaxHorizon> zu{};
    std::array<double, kMaxHorizon> yu{};
    std::array<double, kMaxHorizon + 1> zw{};
    std::array<double, kMaxHorizon + 1> yw{};

    // 预计算参考轨迹线性项 q_k = -Q * r_k
    for (int k = 0; k <= N_; ++k) {
        const int idx = std::min(k, ref_len - 1);
        const double rp = ref_pos[idx];
        const double rv = (ref_vel != nullptr) ? ref_vel[idx] : 0.0;
        q_ref[static_cast<size_t>(k)] = Eigen::Vector2d(-cfg_.q_pos * rp, -cfg_.q_vel * rv);
    }
    const int last_idx = std::min(N_, ref_len - 1);
    const Eigen::Vector2d r_terminal(ref_pos[last_idx],
                                     (ref_vel != nullptr) ? ref_vel[last_idx] : 0.0);
    const Eigen::Vector2d p_terminal = -(P_ * r_terminal);

    x[0] = Eigen::Vector2d(pos0, std::clamp(vel0, -cfg_.max_vel, cfg_.max_vel));
    const double rho = cfg_.rho;

    int used_iters = 0;
    for (int iter = 0; iter < cfg_.max_iters; ++iter) {
        used_iters = iter + 1;

        // 1) 后向 Riccati 线性项递推（O(N) 标量/2D 向量运算）
        p[static_cast<size_t>(N_)] =
            p_terminal - Eigen::Vector2d(0.0, rho * (zw[static_cast<size_t>(N_)] -
                                                     yw[static_cast<size_t>(N_)]));
        for (int k = N_ - 1; k >= 0; --k) {
            const size_t ks = static_cast<size_t>(k);
            const double r_lin = -rho * (zu[ks] - yu[ks]);
            const Eigen::Vector2d q_lin(q_ref[ks].x(),
                                        q_ref[ks].y() - rho * (zw[ks] - yw[ks]));
            const double bt_p = B_.dot(p[ks + 1]);
            d[ks] = c1_inv_ * (bt_p + r_lin);
            p[ks] = q_lin + A_cl_T_ * p[ks + 1] - K_.transpose() * r_lin;
        }

        // 2) 前向状态滚出（从当前云台真实状态 x[0] 出发）
        for (int k = 0; k < N_; ++k) {
            const size_t ks = static_cast<size_t>(k);
            u[ks] = -(K_ * x[ks])(0, 0) - d[ks];
            x[ks + 1] = A_ * x[ks] + B_ * u[ks];
        }

        // 3) 盒子约束投影（z 更新）与对偶变量更新（y 更新），同步统计残差
        double max_primal_res = 0.0;
        double max_dual_res = 0.0;
        for (int k = 0; k < N_; ++k) {
            const size_t ks = static_cast<size_t>(k);
            const double zu_prev = zu[ks];
            zu[ks] = std::clamp(u[ks] + yu[ks], -cfg_.max_acc, cfg_.max_acc);
            const double pu = u[ks] - zu[ks];
            yu[ks] += pu;
            max_primal_res = std::max(max_primal_res, std::fabs(pu));
            max_dual_res = std::max(max_dual_res, std::fabs(zu[ks] - zu_prev));

            const double zw_prev = zw[ks];
            zw[ks] = std::clamp(x[ks].y() + yw[ks], -cfg_.max_vel, cfg_.max_vel);
            const double pw = x[ks].y() - zw[ks];
            yw[ks] += pw;
            max_primal_res = std::max(max_primal_res, std::fabs(pw));
            max_dual_res = std::max(max_dual_res, std::fabs(zw[ks] - zw_prev));
        }
        {
            const size_t ns = static_cast<size_t>(N_);
            const double zw_prev = zw[ns];
            zw[ns] = std::clamp(x[ns].y() + yw[ns], -cfg_.max_vel, cfg_.max_vel);
            const double pw = x[ns].y() - zw[ns];
            yw[ns] += pw;
            max_primal_res = std::max(max_primal_res, std::fabs(pw));
            max_dual_res = std::max(max_dual_res, std::fabs(zw[ns] - zw_prev));
        }

        if (max_primal_res < 1e-3 && max_dual_res < 1e-3) {
            break;
        }
    }

    // 输出严格满足物理硬约束的第 1 步控制轨迹
    const double acc0 = std::clamp(zu[0], -cfg_.max_acc, cfg_.max_acc);
    const Eigen::Vector2d x1 = A_ * x[0] + B_ * acc0;
    const double vel1 = std::clamp(x1.y(), -cfg_.max_vel, cfg_.max_vel);

    if (std::isfinite(x1.x()) && std::isfinite(vel1) && std::isfinite(acc0)) {
        out.pos = x1.x();
        out.vel = vel1;
        out.acc = acc0;
        out.iterations = used_iters;
    }
    return out;
}

#endif  // RM_AIM_WITH_EIGEN

}  // namespace rm_aim::estimation
