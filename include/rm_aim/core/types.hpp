#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace rm_aim {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

inline uint64_t to_steady_ns(TimePoint tp) noexcept {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count());
}

inline TimePoint from_steady_ns(uint64_t ns) noexcept {
    return TimePoint(std::chrono::nanoseconds(ns));
}

enum class EnemyColor : uint8_t {
    Red = 0,
    Blue = 1,
};

enum class PixelFormat : uint8_t {
    Unknown = 0,
    BayerRG8 = 1,
    BayerBG8 = 2,
    BayerGR8 = 3,
    BayerGB8 = 4,
    Mono8 = 5,
    BGR8 = 6,
};

inline uint32_t bytes_per_pixel(PixelFormat fmt) noexcept {
    switch (fmt) {
        case PixelFormat::BayerRG8:
        case PixelFormat::BayerBG8:
        case PixelFormat::BayerGR8:
        case PixelFormat::BayerGB8:
        case PixelFormat::Mono8:
            return 1;
        case PixelFormat::BGR8:
            return 3;
        default:
            return 0;
    }
}

/// 预分配图像帧槽。
/// 在 Jetson NX (RM_AIM_WITH_CUDA=1) 上，`host_ptr` 通过 `cudaHostAllocMapped` 分配，
/// `device_ptr` 指向同一块物理内存，实现零拷贝；
/// 在 x86_64 NUC 上，`host_ptr` 为 64 字节对齐的常规主机内存，`device_ptr` 为 nullptr。
struct FrameSlot {
    uint8_t* host_ptr = nullptr;
    uint8_t* device_ptr = nullptr;
    size_t capacity = 0;

    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t stride = 0;
    uint32_t byte_size = 0;
    PixelFormat format = PixelFormat::Unknown;

    uint64_t frame_id = 0;
    uint64_t timestamp_ns = 0;  // steady_clock 纳秒时间戳（曝光中点）
};

struct Point2f {
    float x = 0.0f;
    float y = 0.0f;
};

struct Vec3f {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct Quatf {
    float w = 1.0f;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct RoiRect {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t width = 0;
    uint32_t height = 0;
};

struct ImuSample {
    uint64_t timestamp_ns = 0;
    uint8_t mode = 0;          // 0: 空闲, 1: 自瞄, 2: 小符, 3: 大符
    Quatf q{};                 // 四元数（w, x, y, z 顺序）
    float yaw = 0.0f;          // 云台 yaw 角（rad）
    float yaw_vel = 0.0f;      // 云台 yaw 角速度（rad/s）
    float pitch = 0.0f;        // 云台 pitch 角（rad）
    float pitch_vel = 0.0f;    // 云台 pitch 角速度（rad/s）
    float bullet_speed = 0.0f; // 弹速（m/s）
    uint16_t bullet_count = 0; // 累计发弹数
};

struct LightBar {
    Point2f top{};
    Point2f bottom{};
    Point2f center{};
    float length = 0.0f;
    float width = 0.0f;
    float angle_rad = 0.0f;
    EnemyColor color = EnemyColor::Red;
};

struct Armor {
    std::array<Point2f, 4> corners{};  // 装甲板四角点：左上、右上、右下、左下
    EnemyColor color = EnemyColor::Red;
    uint8_t class_id = 0;
    float confidence = 0.0f;
    bool is_large = false;
    Vec3f position_cam{};              // 相机坐标系下的三维位置（m）
    float yaw_cam = 0.0f;              // 相机坐标系下的装甲板朝向（rad）
};

struct GimbalCmd {
    uint8_t mode = 0;            // 0: 不控制, 1: 控制云台, 2: 控制云台并开火
    float yaw = 0.0f;            // 期望 yaw 角（rad）
    float yaw_vel = 0.0f;        // 期望 yaw 角速度前馈（rad/s）
    float yaw_acc = 0.0f;        // 期望 yaw 角加速度前馈（rad/s^2）
    float pitch = 0.0f;          // 期望 pitch 角（rad）
    float pitch_vel = 0.0f;      // 期望 pitch 角速度前馈（rad/s）
    float pitch_acc = 0.0f;      // 期望 pitch 角加速度前馈（rad/s^2）
    uint8_t fire_rate_hz = 0;    // 视觉建议射频（Hz，0 表示停火/默认；下位机保留热量裁决权）
};

}  // namespace rm_aim
