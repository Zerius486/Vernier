#include "rm_aim/estimation/imu_buffer.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace rm_aim::estimation {

namespace {

constexpr double kPi = std::numbers::pi_v<double>;
constexpr double kTwoPi = 2.0 * kPi;

/// 两个单位四元数之间的最短路径球面线性插值（Slerp）。
Quatf slerp_quat(const Quatf& a, const Quatf& b, float t) noexcept {
    float dot = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
    Quatf b_adj = b;
    if (dot < 0.0f) {
        dot = -dot;
        b_adj = {-b.w, -b.x, -b.y, -b.z};
    }

    Quatf res{};
    if (dot > 0.9995f) {
        // 夹角极小时退化为线性插值并归一化，避免除零
        res.w = a.w + t * (b_adj.w - a.w);
        res.x = a.x + t * (b_adj.x - a.x);
        res.y = a.y + t * (b_adj.y - a.y);
        res.z = a.z + t * (b_adj.z - a.z);
    } else {
        const float theta = std::acos(std::clamp(dot, -1.0f, 1.0f));
        const float sin_theta = std::sin(theta);
        const float w0 = std::sin((1.0f - t) * theta) / sin_theta;
        const float w1 = std::sin(t * theta) / sin_theta;
        res.w = w0 * a.w + w1 * b_adj.w;
        res.x = w0 * a.x + w1 * b_adj.x;
        res.y = w0 * a.y + w1 * b_adj.y;
        res.z = w0 * a.z + w1 * b_adj.z;
    }

    const float norm = std::sqrt(res.w * res.w + res.x * res.x + res.y * res.y + res.z * res.z);
    if (norm > 1e-6f) {
        const float inv = 1.0f / norm;
        res.w *= inv;
        res.x *= inv;
        res.y *= inv;
        res.z *= inv;
    } else {
        res = {1.0f, 0.0f, 0.0f, 0.0f};
    }
    return res;
}

}  // namespace

double wrap_to_pi(double angle) noexcept {
    double a = std::fmod(angle + kPi, kTwoPi);
    if (a <= 0.0) a += kTwoPi;
    return a - kPi;
}

double shortest_angular_distance(double from, double to) noexcept {
    return wrap_to_pi(to - from);
}

void ImuBuffer::push(const ImuSample& sample) {
    std::lock_guard<std::mutex> lock(mutex_);
    ring_[head_] = sample;
    head_ = (head_ + 1U) % kCapacity;
    if (count_ < kCapacity) {
        ++count_;
    }
}

bool ImuBuffer::query(uint64_t timestamp_ns, ImuSample& out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (count_ == 0) return false;

    auto at = [&](size_t logical_idx) -> const ImuSample& {
        const size_t oldest = (head_ + kCapacity - count_) % kCapacity;
        return ring_[(oldest + logical_idx) % kCapacity];
    };

    if (count_ == 1 || timestamp_ns <= at(0).timestamp_ns) {
        out = at(0);
        return true;
    }
    if (timestamp_ns >= at(count_ - 1U).timestamp_ns) {
        out = at(count_ - 1U);
        return true;
    }

    // 二分查找满足 at(lo).ts <= timestamp_ns < at(lo + 1).ts 的相邻区间
    size_t lo = 0;
    size_t hi = count_ - 1U;
    while (lo + 1U < hi) {
        const size_t mid = lo + (hi - lo) / 2U;
        if (at(mid).timestamp_ns <= timestamp_ns) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    const ImuSample& s0 = at(lo);
    const ImuSample& s1 = at(hi);
    const uint64_t dt = s1.timestamp_ns - s0.timestamp_ns;
    const float alpha = (dt > 0) ? static_cast<float>(static_cast<double>(timestamp_ns - s0.timestamp_ns) /
                                                      static_cast<double>(dt))
                                 : 0.0f;

    out = s1;
    out.timestamp_ns = timestamp_ns;
    out.q = slerp_quat(s0.q, s1.q, alpha);
    out.yaw = static_cast<float>(
        wrap_to_pi(static_cast<double>(s0.yaw) +
                   static_cast<double>(alpha) * shortest_angular_distance(s0.yaw, s1.yaw)));
    out.pitch = s0.pitch + alpha * (s1.pitch - s0.pitch);
    out.yaw_vel = s0.yaw_vel + alpha * (s1.yaw_vel - s0.yaw_vel);
    out.pitch_vel = s0.pitch_vel + alpha * (s1.pitch_vel - s0.pitch_vel);
    return true;
}

bool ImuBuffer::latest(ImuSample& out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (count_ == 0) return false;
    const size_t idx = (head_ + kCapacity - 1U) % kCapacity;
    out = ring_[idx];
    return true;
}

size_t ImuBuffer::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return count_;
}

void ImuBuffer::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    head_ = 0;
    count_ = 0;
}

#ifdef RM_AIM_WITH_EIGEN

void compute_world_to_cam_transform(const ImuSample& imu, const ExtrinsicsConfig& ext,
                                    Eigen::Matrix3d& R_w2c, Eigen::Vector3d& t_w2c) {
    Eigen::Matrix3d R_c2g;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            R_c2g(r, c) = ext.r_camera2gimbal[static_cast<size_t>(r * 3 + c)];
        }
    }
    const Eigen::Vector3d t_c2g(ext.t_camera2gimbal[0], ext.t_camera2gimbal[1],
                                ext.t_camera2gimbal[2]);

    Eigen::Matrix3d R_g2i;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            R_g2i(r, c) = ext.r_gimbal2imubody[static_cast<size_t>(r * 3 + c)];
        }
    }

    Eigen::Quaterniond q_i2w(imu.q.w, imu.q.x, imu.q.y, imu.q.z);
    if (q_i2w.squaredNorm() > 1e-9) {
        q_i2w.normalize();
    } else {
        q_i2w = Eigen::Quaterniond::Identity();
    }
    const Eigen::Matrix3d R_g2w = q_i2w.toRotationMatrix() * R_g2i;

    // p_world = R_g2w * (R_c2g * p_cam + t_c2g) = R_c2w * p_cam + t_c2w
    // => p_cam = R_c2w^T * (p_world - t_c2w)
    const Eigen::Matrix3d R_c2w = R_g2w * R_c2g;
    const Eigen::Vector3d t_c2w = R_g2w * t_c2g;
    R_w2c = R_c2w.transpose();
    t_w2c = -R_w2c * t_c2w;
}

WorldArmorObs transform_armor_to_world(const Armor& armor_cam, const ImuSample& imu,
                                       const ExtrinsicsConfig& ext) {
    // 相机到云台的外参旋转矩阵与平移向量
    Eigen::Matrix3d R_c2g;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            R_c2g(r, c) = ext.r_camera2gimbal[static_cast<size_t>(r * 3 + c)];
        }
    }
    const Eigen::Vector3d t_c2g(ext.t_camera2gimbal[0], ext.t_camera2gimbal[1],
                                ext.t_camera2gimbal[2]);

    // 云台到 IMU 机体系外参
    Eigen::Matrix3d R_g2i;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            R_g2i(r, c) = ext.r_gimbal2imubody[static_cast<size_t>(r * 3 + c)];
        }
    }

    // 四元数（w, x, y, z）给出的 IMU 机体到世界系（FLU）旋转矩阵
    Eigen::Quaterniond q_i2w(imu.q.w, imu.q.x, imu.q.y, imu.q.z);
    if (q_i2w.squaredNorm() > 1e-9) {
        q_i2w.normalize();
    } else {
        q_i2w = Eigen::Quaterniond::Identity();
    }
    const Eigen::Matrix3d R_g2w = q_i2w.toRotationMatrix() * R_g2i;

    const Eigen::Vector3d p_cam(armor_cam.position_cam.x, armor_cam.position_cam.y,
                                armor_cam.position_cam.z);
    const Eigen::Vector3d p_gimbal = R_c2g * p_cam + t_c2g;
    const Eigen::Vector3d p_world = R_g2w * p_gimbal;

    // 将相机系下装甲板水平朝向法向量（正对相机时为 [0, 0, -1]）旋转到世界系
    const Eigen::Vector3d n_cam(std::sin(armor_cam.yaw_cam), 0.0, -std::cos(armor_cam.yaw_cam));
    const Eigen::Vector3d n_world = R_g2w * (R_c2g * n_cam);

    WorldArmorObs obs{};
    obs.pos_world = p_world;
    obs.yaw_world = std::atan2(n_world.y(), n_world.x());
    obs.class_id = armor_cam.class_id;
    obs.is_large = armor_cam.is_large;
    obs.confidence = armor_cam.confidence;
    return obs;
}

#endif  // RM_AIM_WITH_EIGEN

}  // namespace rm_aim::estimation
