#include "rm_aim/estimation/ballistic.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace rm_aim::estimation {

namespace {

constexpr double kDegToRad = std::numbers::pi_v<double> / 180.0;

}  // namespace

BallisticSolver::BallisticSolver(double air_drag_k, double gravity)
    : k_(air_drag_k), g_(gravity) {}

BallisticSolution BallisticSolver::solve(double x, double y, double z,
                                         double bullet_speed) const noexcept {
    BallisticSolution sol{};
    const double d = std::hypot(x, y);
    if (d < 0.05 || !std::isfinite(d) || !std::isfinite(z)) {
        return sol;
    }

    // 保护弹速下限，避免串口初始帧弹速为 0 导致除零
    const double v0 = (bullet_speed > 8.0 && bullet_speed < 40.0) ? bullet_speed : 24.0;
    sol.yaw_rad = std::atan2(y, x);
    sol.horizontal_dist = d;

    // 完整耦合二次空气阻力动力学方程（四阶龙格-库塔 RK4 数值积分）：
    //   dx/dt = vx,  dz/dt = vz
    //   dvx/dt = -k * |v| * vx,  dvz/dt = -g - k * |v| * vz
    // 相比水平/竖直解耦近似，在大仰角（打高地/前哨站）与 5~7m 远距下精度可达 < 0.5 mm
    auto simulate_rk4 = [&](double pitch_rad, double& z_at_d, double& t_at_d) -> bool {
        double px = 0.0;
        double pz = 0.0;
        double vx = v0 * std::cos(pitch_rad);
        double vz = v0 * std::sin(pitch_rad);
        if (vx < 1.0) return false;

        // 粗估飞行时间并划分为 24 步（约 5~10 ms/步，RK4 截断误差 < 0.1 mm）
        const double t_est = (k_ > 1e-6) ? ((std::exp(k_ * d) - 1.0) / (k_ * vx)) : (d / vx);
        const double dt = std::clamp(t_est / 24.0, 0.001, 0.025);
        double t = 0.0;

        auto deriv = [&](double ivx, double ivz, double& dvx, double& dvz) {
            const double speed = std::hypot(ivx, ivz);
            dvx = -k_ * speed * ivx;
            dvz = -g_ - k_ * speed * ivz;
        };

        for (int step = 0; step < 64; ++step) {
            double dvx1, dvz1, dvx2, dvz2, dvx3, dvz3, dvx4, dvz4;
            deriv(vx, vz, dvx1, dvz1);
            const double vx2 = vx + 0.5 * dt * dvx1;
            const double vz2 = vz + 0.5 * dt * dvz1;
            deriv(vx2, vz2, dvx2, dvz2);
            const double vx3 = vx + 0.5 * dt * dvx2;
            const double vz3 = vz + 0.5 * dt * dvz2;
            deriv(vx3, vz3, dvx3, dvz3);
            const double vx4 = vx + dt * dvx3;
            const double vz4 = vz + dt * dvz3;
            deriv(vx4, vz4, dvx4, dvz4);

            const double next_px = px + (dt / 6.0) * (vx + 2.0 * vx2 + 2.0 * vx3 + vx4);
            const double next_pz = pz + (dt / 6.0) * (vz + 2.0 * vz2 + 2.0 * vz3 + vz4);
            const double next_vx = vx + (dt / 6.0) * (dvx1 + 2.0 * dvx2 + 2.0 * dvx3 + dvx4);
            const double next_vz = vz + (dt / 6.0) * (dvz1 + 2.0 * dvz2 + 2.0 * dvz3 + dvz4);

            if (next_px >= d) {
                const double frac = (next_px > px) ? ((d - px) / (next_px - px)) : 0.0;
                z_at_d = pz + frac * (next_pz - pz);
                t_at_d = t + frac * dt;
                return true;
            }

            px = next_px;
            pz = next_pz;
            vx = next_vx;
            vz = next_vz;
            t += dt;
            if (vx < 0.5) return false;
        }
        return false;
    };

    // 外层打靶法迭代调整虚拟瞄准高度 z_aim，6~8 次即可收敛到 < 0.2 mm
    double z_aim = z;
    double pitch = std::atan2(z_aim, d);
    double t_fly = d / v0;
    bool converged = false;

    for (int iter = 0; iter < 8; ++iter) {
        pitch = std::atan2(z_aim, d);
        double z_hit = 0.0;
        double t_hit = 0.0;
        if (!simulate_rk4(pitch, z_hit, t_hit)) {
            break;
        }
        t_fly = t_hit;
        const double err_z = z - z_hit;
        z_aim += err_z;
        converged = true;
        if (std::fabs(err_z) < 2e-4) {
            break;
        }
    }

    sol.pitch_rad = std::atan2(z_aim, d);  // 抬头为正
    sol.fly_time_s = t_fly;
    sol.valid = converged && std::isfinite(sol.pitch_rad) && std::isfinite(sol.fly_time_s);
    return sol;
}

#ifdef RM_AIM_WITH_EIGEN

Aimer::Aimer(AimerConfig cfg)
    : cfg_(std::move(cfg)),
      use_mpc_(cfg_.planner == "mpc"),
      yaw_mpc_(MpcAxisConfig{
          .dt = cfg_.mpc_dt,
          .horizon = cfg_.mpc_horizon,
          .max_vel = cfg_.mpc_max_yaw_vel,
          .max_acc = cfg_.mpc_max_yaw_acc,
          .q_pos = cfg_.mpc_q_pos,
          .q_vel = cfg_.mpc_q_vel,
          .r_acc = cfg_.mpc_r_acc,
      }),
      pitch_mpc_(MpcAxisConfig{
          .dt = cfg_.mpc_dt,
          .horizon = cfg_.mpc_horizon,
          .max_vel = cfg_.mpc_max_pitch_vel,
          .max_acc = cfg_.mpc_max_pitch_acc,
          .q_pos = cfg_.mpc_q_pos,
          .q_vel = cfg_.mpc_q_vel,
          .r_acc = cfg_.mpc_r_acc,
      }) {}

int Aimer::select_armor_id(const VehicleState& state, double center_yaw, int prev_id) const {
    const auto armors = EkfTracker::compute_armors(state);
    if (armors.empty()) return 0;

    const double coming_rad = cfg_.coming_angle_deg * kDegToRad;
    const double leaving_rad = cfg_.leaving_angle_deg * kDegToRad;

    // 计算每块装甲板相对“正对云台方向（center_yaw + pi）”的偏角
    const double ideal_armor_yaw = wrap_to_pi(center_yaw + std::numbers::pi_v<double>);

    // 滞回策略：若上一帧选中的板仍未转出离开角，且转速不是极端情况，优先保持避免抖动切板
    if (prev_id >= 0 && prev_id < static_cast<int>(armors.size())) {
        const double rel = shortest_angular_distance(ideal_armor_yaw,
                                                     armors[static_cast<size_t>(prev_id)].yaw);
        const bool moving_out = (state.v_yaw >= 0.0 && rel > leaving_rad) ||
                                (state.v_yaw < 0.0 && rel < -leaving_rad);
        if (!moving_out && std::fabs(rel) < coming_rad) {
            return prev_id;
        }
    }

    // 否则选择最接近正对云台的装甲板（高速旋转时略微偏向即将转入的那块板）
    int best_id = 0;
    double best_score = 1e18;
    const double spin_bias = (std::fabs(state.v_yaw) > 1.5)
                                 ? ((state.v_yaw > 0.0) ? -0.15 : 0.15)
                                 : 0.0;
    for (const auto& a : armors) {
        const double rel = shortest_angular_distance(ideal_armor_yaw, a.yaw);
        const double score = std::fabs(rel - spin_bias);
        if (score < best_score) {
            best_score = score;
            best_id = a.id;
        }
    }
    return best_id;
}

BallisticSolution Aimer::solve_for_armor_at(const EkfTracker& tracker, int armor_id,
                                            double dt, double bullet_speed) const {
    const VehicleState s = tracker.predict_state(dt);
    const auto armors = EkfTracker::compute_armors(s);
    const auto& a = armors[static_cast<size_t>(armor_id % static_cast<int>(armors.size()))];
    return ballistic_.solve(a.x, a.y, a.z, bullet_speed);
}

void Aimer::update_bullet_speed(const ImuSample& imu) noexcept {
    // 仅在合理物理区间内更新实时弹速（既兼容摩擦轮转速映射弹速，也兼容裁判系统测速）
    if (imu.bullet_speed > 10.0f && imu.bullet_speed < 35.0f) {
        if (!has_bullet_count_) {
            filtered_bullet_speed_ = static_cast<double>(imu.bullet_speed);
            last_bullet_count_ = imu.bullet_count;
            has_bullet_count_ = true;
        } else if (imu.bullet_count != last_bullet_count_) {
            // 检测到实际出弹递增时用更高权重刷新滤波弹速，及时跟随连发掉速
            const double a = std::clamp(cfg_.bullet_speed_alpha, 0.05, 1.0);
            filtered_bullet_speed_ =
                (1.0 - a) * filtered_bullet_speed_ + a * static_cast<double>(imu.bullet_speed);
            last_bullet_count_ = imu.bullet_count;
        } else {
            // 无新发弹计数时缓慢跟踪下位机实时摩擦轮映射弹速
            filtered_bullet_speed_ =
                0.95 * filtered_bullet_speed_ + 0.05 * static_cast<double>(imu.bullet_speed);
        }
    }
}

uint8_t Aimer::compute_adaptive_fire_rate(double dist_m, double yaw_err_rad,
                                          double v_yaw) const noexcept {
    const double max_hz = static_cast<double>(std::max(cfg_.min_fire_rate_hz, cfg_.max_fire_rate_hz));
    const double min_hz = static_cast<double>(std::min(cfg_.min_fire_rate_hz, cfg_.max_fire_rate_hz));

    // 距离因子：2m 内满频，2m~6m 线性降频以稳定摩擦轮转速与远距弹道
    const double dist_factor = std::clamp(1.0 - (dist_m - 2.0) / 5.0, 0.25, 1.0);
    // 转速因子：高速小陀螺时适度降频以减少切板空档浪费热量
    const double spin_factor = std::clamp(1.0 - (std::fabs(v_yaw) - 3.0) / 12.0, 0.40, 1.0);
    // 对准精度因子：越贴近中心射频越高
    const double align_factor = std::clamp(1.0 - std::fabs(yaw_err_rad) / (3.0 * kDegToRad), 0.50, 1.0);

    const double factor = dist_factor * spin_factor * align_factor;
    const double hz = min_hz + factor * (max_hz - min_hz);
    return static_cast<uint8_t>(std::clamp(std::lround(hz), static_cast<long>(min_hz), 40L));
}

GimbalCmd Aimer::compute_command(const EkfTracker& tracker, const ImuSample& current_imu) {
    GimbalCmd cmd{};
    last_switched_ = false;
    update_bullet_speed(current_imu);

    if (!tracker.is_tracking()) {
        last_armor_id_ = -1;
        return cmd;
    }

    const VehicleState cur = tracker.vehicle_state();
    const double bullet_speed = filtered_bullet_speed_;
    const double sys_delay = (std::fabs(cur.v_yaw) >= cfg_.decision_speed_rad_s)
                                 ? cfg_.high_speed_delay_s
                                 : cfg_.low_speed_delay_s;

    // 第 1 步：用当前车体中心粗算飞行时间，预测未来车体状态并选定目标装甲板
    const BallisticSolution center_sol = ballistic_.solve(cur.xc, cur.yc, cur.zc, bullet_speed);
    if (!center_sol.valid) return cmd;

    const double t_coarse = center_sol.fly_time_s + sys_delay;
    const VehicleState pred_coarse = tracker.predict_state(t_coarse);
    const double center_yaw = std::atan2(pred_coarse.yc, pred_coarse.xc);
    const int chosen_id = select_armor_id(pred_coarse, center_yaw, last_armor_id_);

    last_switched_ = (last_armor_id_ != -1) && (chosen_id != last_armor_id_);
    last_armor_id_ = chosen_id;

    // 第 2 步：对选定装甲板做 2 次飞行时间耦合迭代，得到精确打击时刻 t_hit
    double t_hit = t_coarse;
    BallisticSolution sol{};
    for (int iter = 0; iter < 2; ++iter) {
        sol = solve_for_armor_at(tracker, chosen_id, t_hit, bullet_speed);
        if (!sol.valid) return cmd;
        t_hit = sol.fly_time_s + sys_delay;
    }

    // 第 3 步：叠加机械零位偏置，并对下位机当前 yaw 做最短角展开（避免跨 ±pi 跳变）
    const double yaw_offset = cfg_.yaw_offset_deg * kDegToRad;
    const double pitch_offset = cfg_.pitch_offset_deg * kDegToRad;
    const double target_yaw_world = sol.yaw_rad + yaw_offset;
    const double target_pitch_world = sol.pitch_rad + pitch_offset;
    const double yaw_err = shortest_angular_distance(current_imu.yaw, target_yaw_world);
    const double pitch_err = target_pitch_world - static_cast<double>(current_imu.pitch);

    cmd.yaw = static_cast<float>(static_cast<double>(current_imu.yaw) + yaw_err);
    cmd.pitch = static_cast<float>(target_pitch_world);

    // 第 4 步：Planning 轨迹规划与前馈生成（支持 "mpc" 轻量线性轨迹规划 与 "direct" 中心差分）
    if (use_mpc_) {
        const int N = yaw_mpc_.horizon();
        const double dt = yaw_mpc_.dt();
        std::array<double, MpcAxisPlanner::kMaxHorizon + 1> ref_yaw{};
        std::array<double, MpcAxisPlanner::kMaxHorizon + 1> ref_yaw_vel{};
        std::array<double, MpcAxisPlanner::kMaxHorizon + 1> ref_pitch{};
        std::array<double, MpcAxisPlanner::kMaxHorizon + 1> ref_pitch_vel{};
        std::array<int, MpcAxisPlanner::kMaxHorizon + 1> ref_ids{};

        ref_yaw[0] = static_cast<double>(current_imu.yaw) + yaw_err;
        ref_pitch[0] = target_pitch_world;
        ref_ids[0] = chosen_id;

        int sim_id = chosen_id;
        for (int k = 1; k <= N; ++k) {
            const double tk = t_hit + static_cast<double>(k) * dt;
            const VehicleState pk = tracker.predict_state(tk);
            const double c_yaw_k = std::atan2(pk.yc, pk.xc);
            sim_id = select_armor_id(pk, c_yaw_k, sim_id);
            ref_ids[static_cast<size_t>(k)] = sim_id;

            const auto armors_k = EkfTracker::compute_armors(pk);
            const auto& ak = armors_k[static_cast<size_t>(sim_id % static_cast<int>(armors_k.size()))];
            const BallisticSolution sol_k = ballistic_.solve(ak.x, ak.y, ak.z, bullet_speed);
            if (sol_k.valid) {
                const double y_raw = sol_k.yaw_rad + yaw_offset;
                const double p_raw = sol_k.pitch_rad + pitch_offset;
                const double prev_y = ref_yaw[static_cast<size_t>(k - 1)];
                ref_yaw[static_cast<size_t>(k)] =
                    prev_y + shortest_angular_distance(prev_y, y_raw);
                ref_pitch[static_cast<size_t>(k)] = p_raw;
            } else {
                ref_yaw[static_cast<size_t>(k)] = ref_yaw[static_cast<size_t>(k - 1)];
                ref_pitch[static_cast<size_t>(k)] = ref_pitch[static_cast<size_t>(k - 1)];
            }
        }

        // 计算参考角速度序列；在切板跃变步处将参考角速度置零，避免引入虚假冲激速度
        for (int k = 0; k < N; ++k) {
            const size_t ks = static_cast<size_t>(k);
            if (ref_ids[ks + 1] == ref_ids[ks]) {
                ref_yaw_vel[ks] = std::clamp((ref_yaw[ks + 1] - ref_yaw[ks]) / dt,
                                             -cfg_.mpc_max_yaw_vel, cfg_.mpc_max_yaw_vel);
                ref_pitch_vel[ks] = std::clamp((ref_pitch[ks + 1] - ref_pitch[ks]) / dt,
                                               -cfg_.mpc_max_pitch_vel, cfg_.mpc_max_pitch_vel);
            } else {
                ref_yaw_vel[ks] = 0.0;
                ref_pitch_vel[ks] = 0.0;
            }
        }
        ref_yaw_vel[static_cast<size_t>(N)] = ref_yaw_vel[static_cast<size_t>(N - 1)];
        ref_pitch_vel[static_cast<size_t>(N)] = ref_pitch_vel[static_cast<size_t>(N - 1)];

        const MpcPlanOutput py =
            yaw_mpc_.solve(static_cast<double>(current_imu.yaw),
                           static_cast<double>(current_imu.yaw_vel), ref_yaw.data(),
                           ref_yaw_vel.data(), N + 1);
        const MpcPlanOutput pp =
            pitch_mpc_.solve(static_cast<double>(current_imu.pitch),
                             static_cast<double>(current_imu.pitch_vel), ref_pitch.data(),
                             ref_pitch_vel.data(), N + 1);

        cmd.yaw = static_cast<float>(py.pos);
        cmd.yaw_vel = static_cast<float>(py.vel);
        cmd.yaw_acc = static_cast<float>(py.acc);
        cmd.pitch = static_cast<float>(pp.pos);
        cmd.pitch_vel = static_cast<float>(pp.vel);
        cmd.pitch_acc = static_cast<float>(pp.acc);
    } else if (!last_switched_) {
        constexpr double h = 0.002;  // 2 ms 中心差分步长
        const BallisticSolution sol_m = solve_for_armor_at(tracker, chosen_id, t_hit - h, bullet_speed);
        const BallisticSolution sol_p = solve_for_armor_at(tracker, chosen_id, t_hit + h, bullet_speed);
        if (sol_m.valid && sol_p.valid) {
            const double dy_p = shortest_angular_distance(sol.yaw_rad, sol_p.yaw_rad);
            const double dy_m = shortest_angular_distance(sol_m.yaw_rad, sol.yaw_rad);
            const double dp_p = sol_p.pitch_rad - sol.pitch_rad;
            const double dp_m = sol.pitch_rad - sol_m.pitch_rad;

            const double yaw_vel = (dy_p + dy_m) / (2.0 * h);
            const double yaw_acc = (dy_p - dy_m) / (h * h);
            const double pitch_vel = (dp_p + dp_m) / (2.0 * h);
            const double pitch_acc = (dp_p - dp_m) / (h * h);

            // 限幅保护，防止极端噪声放大
            cmd.yaw_vel = static_cast<float>(std::clamp(yaw_vel, -12.0, 12.0));
            cmd.yaw_acc = static_cast<float>(std::clamp(yaw_acc, -60.0, 60.0));
            cmd.pitch_vel = static_cast<float>(std::clamp(pitch_vel, -8.0, 8.0));
            cmd.pitch_acc = static_cast<float>(std::clamp(pitch_acc, -40.0, 40.0));
        }
    }

    // 第 5 步：根据距离与角度容差决定是否开火及自适应建议射频（mode: 1=仅瞄准, 2=瞄准并开火）
    const double tol_rad = ((sol.horizontal_dist < cfg_.judge_distance_m)
                                ? cfg_.first_tolerance_deg
                                : cfg_.second_tolerance_deg) *
                           kDegToRad;
    const bool in_window = (std::fabs(yaw_err) <= tol_rad) && (std::fabs(pitch_err) <= tol_rad);
    if (cfg_.auto_fire && in_window) {
        cmd.mode = 2U;
        cmd.fire_rate_hz = compute_adaptive_fire_rate(sol.horizontal_dist, yaw_err, cur.v_yaw);
    } else {
        cmd.mode = 1U;
        cmd.fire_rate_hz = 0U;
    }
    return cmd;
}

#endif  // RM_AIM_WITH_EIGEN

}  // namespace rm_aim::estimation
