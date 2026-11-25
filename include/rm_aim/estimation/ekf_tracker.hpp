#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "rm_aim/core/config.hpp"
#include "rm_aim/estimation/imu_buffer.hpp"
#include "rm_aim/estimation/motion_model.hpp"

#ifdef RM_AIM_WITH_EIGEN
#include <Eigen/Dense>
#endif

namespace rm_aim::estimation {

enum class TrackState : uint8_t {
    Lost = 0,
    Detecting = 1,
    Tracking = 2,
    TempLost = 3,
};

struct PredictedArmor {
    int id = 0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double yaw = 0.0;
    double radius = 0.26;
};

#ifdef RM_AIM_WITH_EIGEN

/// 12 维阻尼角加速（Singer）整车扩展卡尔曼滤波器 + 新息自适应过程噪声 Q。
///
/// 状态向量 x (12 维)：
///   [0] xc, [1] vx, [2] yc, [3] vy, [4] zc, [5] vz,
///   [6] yaw, [7] v_yaw, [8] a_yaw, [9] r1, [10] r2, [11] dz
///
/// 反制非匀速/脉冲小陀螺核心机制：
///   1. 引入阻尼角加速度 a_yaw（时间常数 tau），消除脉冲加速沿的角速度相位滞后且外推不过冲；
///   2. 实时监测角度新息的卡方马氏距离，检测脉冲突变时自动膨胀 Q_omega / Q_alpha，
///      在 3~5 帧内快速咬住新转速并在平稳后自动回落。
class EkfTracker {
public:
    static constexpr int kStateDim = 12;
    static constexpr int kMeasDim = 4;

    using StateVec = Eigen::Matrix<double, kStateDim, 1>;
    using StateMat = Eigen::Matrix<double, kStateDim, kStateDim>;
    using MeasVec = Eigen::Matrix<double, kMeasDim, 1>;
    using MeasMat = Eigen::Matrix<double, kMeasDim, kMeasDim>;
    using HMat = Eigen::Matrix<double, kMeasDim, kStateDim>;

    explicit EkfTracker(TrackerConfig cfg, double singer_tau = 0.20);

    /// 注入自定义外挂运动预测器（传入 nullptr 则恢复默认 SingerMotionPredictor）。
    void set_motion_predictor(std::shared_ptr<IMotionPredictor> predictor);

    /// 用当前帧的世界系装甲板观测列表更新整车跟踪器。
    void update(uint64_t timestamp_ns, const std::vector<WorldArmorObs>& observations);

    void reset();

    TrackState track_state() const noexcept { return state_; }
    bool is_tracking() const noexcept {
        return state_ == TrackState::Tracking || state_ == TrackState::TempLost;
    }

    VehicleState vehicle_state() const noexcept;

    /// 通过当前绑定的运动预测器（默认 Singer）预测未来 dt 秒的整车状态。
    VehicleState predict_state(double dt) const;

    /// 计算给定整车状态对应的所有装甲板 3D 位置与朝向。
    static std::vector<PredictedArmor> compute_armors(const VehicleState& s);

    /// 最近一次角度新息自适应膨胀倍率（用于调试/遥测观察脉冲响应）。
    double last_q_scale() const noexcept { return last_q_scale_; }

private:
    void init_from_obs(const WorldArmorObs& obs);
    void ekf_predict(double dt, double q_scale);
    void ekf_update_armor(int armor_id, const WorldArmorObs& obs);
    int match_best_armor_id(const WorldArmorObs& obs, double& out_cost) const;

    TrackerConfig cfg_;
    double singer_tau_ = 0.20;

    TrackState state_ = TrackState::Lost;
    uint64_t last_ts_ns_ = 0;
    int detect_count_ = 0;
    int lost_count_ = 0;
    uint8_t tracked_class_id_ = 0;
    bool tracked_is_large_ = false;
    int armor_num_ = 4;

    StateVec x_ = StateVec::Zero();
    StateMat P_ = StateMat::Identity();
    double last_q_scale_ = 1.0;

    std::shared_ptr<IMotionPredictor> predictor_;
};

#endif  // RM_AIM_WITH_EIGEN

}  // namespace rm_aim::estimation
