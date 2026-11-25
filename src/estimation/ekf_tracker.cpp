#include "rm_aim/estimation/ekf_tracker.hpp"

#ifdef RM_AIM_WITH_EIGEN

#include <algorithm>
#include <cmath>
#include <numbers>

namespace rm_aim::estimation {

namespace {

constexpr double kPi = std::numbers::pi_v<double>;
constexpr double kTwoPi = 2.0 * kPi;

/// 由整车状态向量计算第 armor_id 块装甲板的预期观测 [xa, ya, za, yaw_a]。
EkfTracker::MeasVec armor_measurement_model(const EkfTracker::StateVec& x, int armor_id,
                                            int armor_num) {
    const double step = kTwoPi / static_cast<double>(armor_num);
    const double yaw_i = x(6) + static_cast<double>(armor_id) * step;
    const bool is_odd = (armor_num == 4) && ((armor_id & 1) != 0);
    const double r = is_odd ? x(10) : x(9);
    const double z = is_odd ? (x(4) + x(11)) : x(4);

    EkfTracker::MeasVec z_pred;
    z_pred(0) = x(0) - r * std::cos(yaw_i);
    z_pred(1) = x(2) - r * std::sin(yaw_i);
    z_pred(2) = z;
    z_pred(3) = wrap_to_pi(yaw_i);
    return z_pred;
}

/// 观测模型对 12 维状态的解析雅可比矩阵 H (4x12)。
EkfTracker::HMat armor_measurement_jacobian(const EkfTracker::StateVec& x, int armor_id,
                                            int armor_num) {
    const double step = kTwoPi / static_cast<double>(armor_num);
    const double yaw_i = x(6) + static_cast<double>(armor_id) * step;
    const bool is_odd = (armor_num == 4) && ((armor_id & 1) != 0);
    const double r = is_odd ? x(10) : x(9);
    const double c = std::cos(yaw_i);
    const double s = std::sin(yaw_i);

    EkfTracker::HMat H = EkfTracker::HMat::Zero();
    H(0, 0) = 1.0;
    H(0, 6) = r * s;
    H(0, is_odd ? 10 : 9) = -c;

    H(1, 2) = 1.0;
    H(1, 6) = -r * c;
    H(1, is_odd ? 10 : 9) = -s;

    H(2, 4) = 1.0;
    if (is_odd) {
        H(2, 11) = 1.0;
    }

    H(3, 6) = 1.0;
    return H;
}

}  // namespace

EkfTracker::EkfTracker(TrackerConfig cfg, double singer_tau)
    : cfg_(std::move(cfg)),
      singer_tau_(singer_tau),
      predictor_(std::make_shared<SingerMotionPredictor>()) {}

void EkfTracker::set_motion_predictor(std::shared_ptr<IMotionPredictor> predictor) {
    if (predictor) {
        predictor_ = std::move(predictor);
    } else {
        predictor_ = std::make_shared<SingerMotionPredictor>();
    }
}

void EkfTracker::reset() {
    state_ = TrackState::Lost;
    last_ts_ns_ = 0;
    detect_count_ = 0;
    lost_count_ = 0;
    last_q_scale_ = 1.0;
    x_.setZero();
    P_.setIdentity();
    if (predictor_) {
        predictor_->reset();
    }
}

void EkfTracker::init_from_obs(const WorldArmorObs& obs) {
    tracked_class_id_ = obs.class_id;
    tracked_is_large_ = obs.is_large;
    // class_id == 6 约定为前哨站（3 块装甲板），其余地面车辆为 4 块
    armor_num_ = (obs.class_id == 6) ? 3 : 4;
    const double init_r = (obs.class_id == 6) ? 0.2765 : 0.26;

    x_.setZero();
    x_(0) = obs.pos_world.x() + init_r * std::cos(obs.yaw_world);
    x_(2) = obs.pos_world.y() + init_r * std::sin(obs.yaw_world);
    x_(4) = obs.pos_world.z();
    x_(6) = wrap_to_pi(obs.yaw_world);
    x_(9) = init_r;
    x_(10) = init_r;
    x_(11) = 0.0;

    P_.setZero();
    P_.diagonal() << 0.05, 4.0,   // xc, vx
                     0.05, 4.0,   // yc, vy
                     0.05, 2.0,   // zc, vz
                     0.10, 25.0, 100.0,  // yaw, v_yaw, a_yaw
                     0.01, 0.01, 0.01;   // r1, r2, dz

    detect_count_ = 1;
    lost_count_ = 0;
    last_q_scale_ = 1.0;
    state_ = (cfg_.min_detect_count <= 1) ? TrackState::Tracking : TrackState::Detecting;
    if (predictor_) {
        predictor_->reset();
    }
}

void EkfTracker::ekf_predict(double dt, double q_scale) {
    if (dt <= 0.0) return;

    const double tau = singer_tau_;
    const double decay = std::exp(-dt / tau);
    const double one_minus = 1.0 - decay;
    const double c_pos = tau * (dt - tau * one_minus);
    const double c_vel = tau * one_minus;

    // 12x12 状态转移矩阵 F（Singer 阻尼角加速解析转移）
    StateMat F = StateMat::Identity();
    F(0, 1) = dt;
    F(2, 3) = dt;
    F(4, 5) = dt;
    F(6, 7) = dt;
    F(6, 8) = c_pos;
    F(7, 8) = c_vel;
    F(8, 8) = decay;

    x_ = F * x_;
    x_(6) = wrap_to_pi(x_(6));

    // 分段白噪声过程协方差 Q（新息自适应膨胀系数 q_scale 仅作用于角速度/角加速度通道）
    StateMat Q = StateMat::Zero();
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt3 * dt;

    const double q_xy = 20.0;
    const double q_z = 8.0;
    const double q_alpha = 220.0 * q_scale;  // Singer 角加速度驱动方差

    auto fill_cv_block = [&](int idx, double q) {
        Q(idx, idx) = 0.25 * dt4 * q;
        Q(idx, idx + 1) = 0.5 * dt3 * q;
        Q(idx + 1, idx) = 0.5 * dt3 * q;
        Q(idx + 1, idx + 1) = dt2 * q;
    };
    fill_cv_block(0, q_xy);
    fill_cv_block(2, q_xy);
    fill_cv_block(4, q_z);

    // 旋转通道 (yaw, v_yaw, a_yaw) 过程噪声
    Q(6, 6) = 0.25 * dt4 * q_alpha + 1e-5 * dt;
    Q(7, 7) = dt2 * q_alpha + 0.5 * q_scale * dt;
    Q(8, 8) = (1.0 - decay * decay) * q_alpha;
    Q(6, 7) = 0.5 * dt3 * q_alpha;
    Q(7, 6) = Q(6, 7);
    Q(7, 8) = dt * q_alpha * (1.0 - decay);
    Q(8, 7) = Q(7, 8);

    // 几何结构参数 (r1, r2, dz) 极缓慢漂移
    Q(9, 9) = 1e-4 * dt;
    Q(10, 10) = 1e-4 * dt;
    Q(11, 11) = 1e-4 * dt;

    P_ = F * P_ * F.transpose() + Q;
}

int EkfTracker::match_best_armor_id(const WorldArmorObs& obs, double& out_cost) const {
    int best_id = 0;
    double best_cost = 1e18;
    for (int i = 0; i < armor_num_; ++i) {
        const MeasVec z_pred = armor_measurement_model(x_, i, armor_num_);
        const double dp = (obs.pos_world - z_pred.head<3>()).norm();
        const double dyaw = std::fabs(shortest_angular_distance(z_pred(3), obs.yaw_world));
        const double cost = dp + 0.35 * dyaw;
        if (cost < best_cost) {
            best_cost = cost;
            best_id = i;
        }
    }
    out_cost = best_cost;
    return best_id;
}

void EkfTracker::ekf_update_armor(int armor_id, const WorldArmorObs& obs) {
    const MeasVec z_pred = armor_measurement_model(x_, armor_id, armor_num_);
    const HMat H = armor_measurement_jacobian(x_, armor_id, armor_num_);

    MeasVec y;
    y(0) = obs.pos_world.x() - z_pred(0);
    y(1) = obs.pos_world.y() - z_pred(1);
    y(2) = obs.pos_world.z() - z_pred(2);
    y(3) = shortest_angular_distance(z_pred(3), obs.yaw_world);

    // 随距离自适应的测量噪声 R（距离越远，PnP 位置与朝向噪声越大）
    const double dist = std::max(0.5, obs.pos_world.norm());
    MeasMat R = MeasMat::Zero();
    const double r_pos = 0.0008 * dist * dist;
    R(0, 0) = r_pos;
    R(1, 1) = r_pos;
    R(2, 2) = r_pos;
    R(3, 3) = 0.008 + 0.0015 * dist;

    MeasMat S = H * P_ * H.transpose() + R;

    // 鲁棒位置野点门控（抗半遮挡/反光测距跳变）+ Huber 平滑降权
    const Eigen::Vector3d y_pos = y.head<3>();
    const Eigen::Matrix3d S_pos = S.topLeftCorner<3, 3>();
    const double nis_pos = y_pos.transpose() * S_pos.ldlt().solve(y_pos);
    if (state_ == TrackState::Tracking && nis_pos > 36.0) {
        // 超过 ~6 sigma 的极端测距野点直接拒收，避免污染整车中心与速度估计
        return;
    }
    if (nis_pos > 6.25) {
        // 中度异常观测按 Huber 权重膨胀位置测量方差 R，平滑抑制单帧抖动
        const double huber_scale = std::sqrt(nis_pos) / 2.5;
        R.topLeftCorner<3, 3>() *= (huber_scale * huber_scale);
        S = H * P_ * H.transpose() + R;
    }

    // 新息自适应检测：计算角度残差归一化卡方值，检测脉冲转速突变
    const double nis_yaw = (y(3) * y(3)) / std::max(1e-6, S(3, 3));
    if (nis_yaw > 4.0) {
        // 检测到剧烈变加速，临时膨胀旋转协方差以迅速提升卡尔曼增益
        const double inflate = std::clamp(nis_yaw / 4.0, 1.0, 12.0);
        last_q_scale_ = inflate;
        P_(7, 7) += 0.8 * inflate;
        P_(8, 8) += 12.0 * inflate;
        S = H * P_ * H.transpose() + R;
    } else {
        // 平稳后平滑回落到基准过程噪声，保持低抖动
        last_q_scale_ = 0.85 * last_q_scale_ + 0.15 * 1.0;
    }

    const Eigen::Matrix<double, kStateDim, kMeasDim> K = P_ * H.transpose() * S.inverse();
    x_ = x_ + K * y;
    x_(6) = wrap_to_pi(x_(6));

    // 物理合理性钳位：限制装甲板半径与高度差，防止极端噪点导致发散
    x_(9) = std::clamp(x_(9), 0.14, 0.42);
    x_(10) = std::clamp(x_(10), 0.14, 0.42);
    x_(11) = std::clamp(x_(11), -0.15, 0.15);

    // Joseph 稳定形式协方差更新，保证半正定性
    const StateMat I_KH = StateMat::Identity() - K * H;
    P_ = I_KH * P_ * I_KH.transpose() + K * R * K.transpose();
}

void EkfTracker::update(uint64_t timestamp_ns, const std::vector<WorldArmorObs>& observations) {
    if (state_ == TrackState::Lost) {
        if (observations.empty()) return;
        // 优先锁定距光轴/原点最近的装甲板
        const auto best_it = std::min_element(
            observations.begin(), observations.end(),
            [](const WorldArmorObs& a, const WorldArmorObs& b) {
                return a.pos_world.squaredNorm() < b.pos_world.squaredNorm();
            });
        init_from_obs(*best_it);
        last_ts_ns_ = timestamp_ns;
        return;
    }

    const double dt = (timestamp_ns > last_ts_ns_)
                          ? static_cast<double>(timestamp_ns - last_ts_ns_) * 1e-9
                          : 0.0;
    last_ts_ns_ = timestamp_ns;

    // 超过 0.5 秒无有效帧直接复位
    if (dt > 0.5) {
        reset();
        return;
    }

    ekf_predict(dt, last_q_scale_);

    // 筛选与当前追踪类别一致且代价最小的观测（支持同一帧观测到同车 2 块板依次更新）
    bool matched_any = false;
    for (const auto& obs : observations) {
        if (obs.class_id != tracked_class_id_) continue;
        double cost = 0.0;
        const int armor_id = match_best_armor_id(obs, cost);
        if (cost < 0.85) {
            ekf_update_armor(armor_id, obs);
            matched_any = true;
        }
    }

    const int max_lost =
        (tracked_class_id_ == 6) ? cfg_.outpost_max_temp_lost_count : cfg_.max_temp_lost_count;

    if (matched_any) {
        lost_count_ = 0;
        if (state_ == TrackState::Detecting) {
            ++detect_count_;
            if (detect_count_ >= cfg_.min_detect_count) {
                state_ = TrackState::Tracking;
            }
        } else if (state_ == TrackState::TempLost) {
            state_ = TrackState::Tracking;
        }
        if (predictor_) {
            predictor_->on_state_updated(timestamp_ns, vehicle_state());
        }
    } else {
        if (state_ == TrackState::Detecting) {
            reset();
        } else {
            ++lost_count_;
            if (lost_count_ > max_lost) {
                reset();
            } else {
                state_ = TrackState::TempLost;
            }
        }
    }
}

VehicleState EkfTracker::vehicle_state() const noexcept {
    VehicleState s{};
    s.xc = x_(0);
    s.vx = x_(1);
    s.yc = x_(2);
    s.vy = x_(3);
    s.zc = x_(4);
    s.vz = x_(5);
    s.yaw = x_(6);
    s.v_yaw = x_(7);
    s.a_yaw = x_(8);
    s.r1 = x_(9);
    s.r2 = x_(10);
    s.dz = x_(11);
    s.singer_tau = singer_tau_;
    s.class_id = tracked_class_id_;
    s.is_large = tracked_is_large_;
    s.armor_num = armor_num_;
    return s;
}

VehicleState EkfTracker::predict_state(double dt) const {
    const VehicleState cur = vehicle_state();
    if (predictor_) {
        return predictor_->predict(cur, dt);
    }
    SingerMotionPredictor fallback;
    return fallback.predict(cur, dt);
}

std::vector<PredictedArmor> EkfTracker::compute_armors(const VehicleState& s) {
    std::vector<PredictedArmor> armors;
    armors.reserve(static_cast<size_t>(s.armor_num));
    const double step = kTwoPi / static_cast<double>(s.armor_num);

    for (int i = 0; i < s.armor_num; ++i) {
        const double yaw_i = wrap_to_pi(s.yaw + static_cast<double>(i) * step);
        const bool is_odd = (s.armor_num == 4) && ((i & 1) != 0);
        const double r = is_odd ? s.r2 : s.r1;
        const double z = is_odd ? (s.zc + s.dz) : s.zc;

        PredictedArmor a{};
        a.id = i;
        a.x = s.xc - r * std::cos(yaw_i);
        a.y = s.yc - r * std::sin(yaw_i);
        a.z = z;
        a.yaw = yaw_i;
        a.radius = r;
        armors.push_back(a);
    }
    return armors;
}

}  // namespace rm_aim::estimation

#endif  // RM_AIM_WITH_EIGEN
