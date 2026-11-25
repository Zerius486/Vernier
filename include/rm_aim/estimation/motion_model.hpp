#pragma once

#include <cmath>
#include <cstdint>

namespace rm_aim::estimation {

/// 整车状态快照（对应 12 维 Singer EKF 估计值），用于运动预测与外挂预测器接口。
struct VehicleState {
    double xc = 0.0, vx = 0.0;      // 车体中心 X 位置与速度（m, m/s）
    double yc = 0.0, vy = 0.0;      // 车体中心 Y 位置与速度（m, m/s）
    double zc = 0.0, vz = 0.0;      // 车体中心 Z 位置与速度（m, m/s）
    double yaw = 0.0;               // 0 号装甲板世界系朝向角（rad）
    double v_yaw = 0.0;             // 自转角速度（rad/s）
    double a_yaw = 0.0;             // 自转角加速度（rad/s^2）
    double r1 = 0.26;               // 偶号装甲板（0/2）旋转半径（m）
    double r2 = 0.26;               // 奇号装甲板（1/3）旋转半径（m）
    double dz = 0.0;                // 奇号板相对偶号板的高度差（m）
    double singer_tau = 0.20;       // Singer 阻尼时间常数（s）
    uint8_t class_id = 0;
    bool is_large = false;
    int armor_num = 4;              // 装甲板数量（常规车辆为 4，前哨站为 3）
};

/// 外挂运动预测器抽象接口（预留扩展点）。
/// 后续训练赛如需外挂周期脉冲频谱预测、神经网络轨迹预测等非匀速模型，
/// 只需继承该接口并注入 EkfTracker / Aimer 即可，无需改动核心滤波框架。
class IMotionPredictor {
public:
    virtual ~IMotionPredictor() = default;

    /// 每帧滤波更新后回调，供外挂模型收集历史转速/位姿序列（默认空实现）。
    virtual void on_state_updated(uint64_t timestamp_ns, const VehicleState& state) {
        (void)timestamp_ns;
        (void)state;
    }

    /// 切换目标或丢失目标时重置外挂内部历史缓存。
    virtual void reset() {}

    /// 预测 dt 秒后的整车状态。
    virtual VehicleState predict(const VehicleState& current, double dt) const = 0;
};

/// 默认内置预测器：基于 12 维阻尼角加速（Singer）模型的精确解析外推。
/// 角加速度按 exp(-t / tau) 衰减：近程快速响应脉冲加减速，远程自动收敛回匀速不过冲。
class SingerMotionPredictor final : public IMotionPredictor {
public:
    VehicleState predict(const VehicleState& cur, double dt) const override {
        VehicleState next = cur;
        next.xc = cur.xc + cur.vx * dt;
        next.yc = cur.yc + cur.vy * dt;
        next.zc = cur.zc + cur.vz * dt;

        const double tau = cur.singer_tau;
        if (tau > 1e-4 && std::fabs(dt) > 1e-7) {
            const double decay = std::exp(-dt / tau);
            const double one_minus = 1.0 - decay;
            // 对 a(t) = a0 * exp(-t/tau) 做解析一次与二次积分
            next.yaw = cur.yaw + cur.v_yaw * dt + cur.a_yaw * tau * (dt - tau * one_minus);
            next.v_yaw = cur.v_yaw + cur.a_yaw * tau * one_minus;
            next.a_yaw = cur.a_yaw * decay;
        } else {
            next.yaw = cur.yaw + cur.v_yaw * dt;
            next.v_yaw = cur.v_yaw;
            next.a_yaw = 0.0;
        }
        return next;
    }
};

}  // namespace rm_aim::estimation
