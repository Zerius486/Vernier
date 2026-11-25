#pragma once

#include <array>
#include <cstdint>
#include <mutex>

#include "rm_aim/core/config.hpp"
#include "rm_aim/core/types.hpp"

#ifdef RM_AIM_WITH_EIGEN
#include <Eigen/Geometry>
#endif

namespace rm_aim::estimation {

/// 高频 IMU 姿态时间同步缓冲（支持 500~1000 Hz 写入，按曝光中点时间戳做四元数 Slerp 软同步）。
class ImuBuffer {
public:
    static constexpr size_t kCapacity = 256;

    ImuBuffer() = default;

    /// 串口接收线程调用：写入带 steady_clock 纳秒时间戳的 IMU 采样。
    void push(const ImuSample& sample);

    /// 算法线程调用：查询指定时间戳（如图像曝光中点）的插值 IMU 姿态。
    /// 若缓冲区为空返回 false；若时间戳略超出范围则钳位到最近边界帧。
    bool query(uint64_t timestamp_ns, ImuSample& out) const;

    bool latest(ImuSample& out) const;
    size_t size() const;
    void clear();

private:
    mutable std::mutex mutex_;
    std::array<ImuSample, kCapacity> ring_{};
    size_t head_ = 0;   // 下一个写入位置
    size_t count_ = 0;  // 当前有效元素数量
};

/// 将角度归一化到 (-pi, pi]。
double wrap_to_pi(double angle) noexcept;

/// 计算从 from 到 to 的最短有符号角位移（范围 (-pi, pi]）。
double shortest_angular_distance(double from, double to) noexcept;

#ifdef RM_AIM_WITH_EIGEN

/// 世界系下（FLU 右手系：X 前、Y 左、Z 上）的装甲板观测结果。
struct WorldArmorObs {
    Eigen::Vector3d pos_world = Eigen::Vector3d::Zero();
    double yaw_world = 0.0;  // 装甲板外法线在世界系水平面上的朝向角（向左为正，rad）
    uint8_t class_id = 0;
    bool is_large = false;
    float confidence = 0.0f;
};

/// 计算世界系（FLU）到相机光学系的刚体变换 (R_w2c, t_w2c)，
/// 满足 p_cam = R_w2c * p_world + t_w2c，用于 1D 黄金分割重投影优化。
void compute_world_to_cam_transform(const ImuSample& imu, const ExtrinsicsConfig& ext,
                                    Eigen::Matrix3d& R_w2c, Eigen::Vector3d& t_w2c);

/// 将相机光学系下的装甲板观测转换到世界系（FLU 右手系）。
/// 变换链：p_world = R_gimbal2world(q) * (R_camera2gimbal * p_cam + t_camera2gimbal)
WorldArmorObs transform_armor_to_world(const Armor& armor_cam, const ImuSample& imu,
                                       const ExtrinsicsConfig& ext);

#endif  // RM_AIM_WITH_EIGEN

}  // namespace rm_aim::estimation
