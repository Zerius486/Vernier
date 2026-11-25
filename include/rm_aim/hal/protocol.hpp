#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "rm_aim/core/types.hpp"
#include "rm_aim/hal/crc.hpp"

namespace rm_aim::hal {

#pragma pack(push, 1)

// 与 h7_framework/Skywalker/Application/Inc/app_protocol.h 字节级严格对齐的通信帧结构
struct GimbalToVisionPacket {
    uint8_t head[2];       // 帧头，固定为 {'A', 'B'}
    uint8_t mode;          // 0: 空闲, 1: 自瞄, 2: 小符, 3: 大符
    float q[4];            // 姿态四元数，w, x, y, z 顺序
    float yaw;             // 云台 yaw 角度（rad）
    float yaw_velocity;    // 云台 yaw 角速度（rad/s）
    float pitch;           // 云台 pitch 角度（rad）
    float pitch_velocity;  // 云台 pitch 角速度（rad/s）
    float bullet_speed;    // 弹速（m/s）
    uint16_t bullet_count; // 累计发弹数
    uint16_t crc16;        // 小端序 CRC16 校验
};

struct VisionToGimbalPacket {
    uint8_t head[2];          // 帧头，固定为 {'A', 'B'}
    uint8_t mode;             // 0: 不控制, 1: 控制云台, 2: 控制云台并开火
    float yaw;                // 视觉期望 yaw 角度（rad）
    float yaw_velocity;       // 视觉期望 yaw 角速度前馈（rad/s）
    float yaw_acceleration;   // 视觉期望 yaw 角加速度前馈（rad/s^2）
    float pitch;              // 视觉期望 pitch 角度（rad）
    float pitch_velocity;     // 视觉期望 pitch 角速度前馈（rad/s）
    float pitch_acceleration; // 视觉期望 pitch 角加速度前馈（rad/s^2）
    uint8_t fire_rate_hz;     // 视觉建议射频（Hz，0 表示停火/默认；下位机保留热量裁决权）
    uint16_t crc16;           // 小端序 CRC16 校验
};

#pragma pack(pop)

static_assert(sizeof(GimbalToVisionPacket) == 43, "GimbalToVisionPacket 必须严格为 43 字节");
static_assert(sizeof(VisionToGimbalPacket) == 30, "VisionToGimbalPacket 必须严格为 30 字节");

inline bool is_valid_packet(const GimbalToVisionPacket& p) noexcept {
    return p.head[0] == 'A' && p.head[1] == 'B' && p.mode <= 3U &&
           std::isfinite(p.q[0]) && std::isfinite(p.q[1]) &&
           std::isfinite(p.q[2]) && std::isfinite(p.q[3]) &&
           std::isfinite(p.yaw) && std::isfinite(p.yaw_velocity) &&
           std::isfinite(p.pitch) && std::isfinite(p.pitch_velocity) &&
           std::isfinite(p.bullet_speed);
}

inline bool is_valid_packet(const VisionToGimbalPacket& p) noexcept {
    return p.head[0] == 'A' && p.head[1] == 'B' && p.mode <= 2U && p.fire_rate_hz <= 40U &&
           std::isfinite(p.yaw) && std::isfinite(p.yaw_velocity) &&
           std::isfinite(p.yaw_acceleration) && std::isfinite(p.pitch) &&
           std::isfinite(p.pitch_velocity) && std::isfinite(p.pitch_acceleration);
}

inline VisionToGimbalPacket pack_vision_cmd(const GimbalCmd& cmd) noexcept {
    VisionToGimbalPacket pkt{};
    pkt.head[0] = 'A';
    pkt.head[1] = 'B';
    const bool ok = cmd.mode <= 2U && cmd.fire_rate_hz <= 40U &&
                    std::isfinite(cmd.yaw) && std::isfinite(cmd.yaw_vel) &&
                    std::isfinite(cmd.yaw_acc) && std::isfinite(cmd.pitch) &&
                    std::isfinite(cmd.pitch_vel) && std::isfinite(cmd.pitch_acc);
    if (ok) {
        pkt.mode = cmd.mode;
        pkt.yaw = cmd.yaw;
        pkt.yaw_velocity = cmd.yaw_vel;
        pkt.yaw_acceleration = cmd.yaw_acc;
        pkt.pitch = cmd.pitch;
        pkt.pitch_velocity = cmd.pitch_vel;
        pkt.pitch_acceleration = cmd.pitch_acc;
        pkt.fire_rate_hz = (cmd.mode == 2U) ? cmd.fire_rate_hz : 0U;
    } else {
        // 遇到 NaN/Inf 时回退为安全零指令，与下位机 VisionProtocolFrameIsValid 保持一致
        pkt.mode = 0U;
        pkt.fire_rate_hz = 0U;
    }
    const uint16_t c = crc16(reinterpret_cast<const uint8_t*>(&pkt),
                             sizeof(VisionToGimbalPacket) - sizeof(uint16_t));
    pkt.crc16 = c;  // x86_64 与 ARM64 Linux 均为小端序
    return pkt;
}

inline GimbalToVisionPacket pack_gimbal_state(const ImuSample& s) noexcept {
    GimbalToVisionPacket pkt{};
    pkt.head[0] = 'A';
    pkt.head[1] = 'B';
    pkt.mode = s.mode;
    pkt.q[0] = s.q.w;
    pkt.q[1] = s.q.x;
    pkt.q[2] = s.q.y;
    pkt.q[3] = s.q.z;
    pkt.yaw = s.yaw;
    pkt.yaw_velocity = s.yaw_vel;
    pkt.pitch = s.pitch;
    pkt.pitch_velocity = s.pitch_vel;
    pkt.bullet_speed = s.bullet_speed;
    pkt.bullet_count = s.bullet_count;
    pkt.crc16 = crc16(reinterpret_cast<const uint8_t*>(&pkt),
                      sizeof(GimbalToVisionPacket) - sizeof(uint16_t));
    return pkt;
}

inline ImuSample unpack_gimbal_packet(const GimbalToVisionPacket& p, uint64_t ts_ns) noexcept {
    ImuSample s{};
    s.timestamp_ns = ts_ns;
    s.mode = p.mode;
    s.q = {p.q[0], p.q[1], p.q[2], p.q[3]};
    s.yaw = p.yaw;
    s.yaw_vel = p.yaw_velocity;
    s.pitch = p.pitch;
    s.pitch_vel = p.pitch_velocity;
    s.bullet_speed = p.bullet_speed;
    s.bullet_count = p.bullet_count;
    return s;
}

/// 滑动窗口字节流解包器，用于解析定长 'AB' 帧头 + CRC16 协议帧（抗粘包、半包与脏字节）。
template <typename PacketT>
class StreamPacketParser {
public:
    explicit StreamPacketParser(size_t reserve_bytes = sizeof(PacketT) * 8U) {
        buf_.reserve(reserve_bytes);
    }

    template <typename Callback>
    size_t feed(const uint8_t* data, size_t len, Callback&& on_packet) {
        if (data == nullptr || len == 0) return 0;

        // 防止持续无效垃圾数据导致缓冲区无限增长
        if (buf_.size() + len > sizeof(PacketT) * 16U) {
            buf_.clear();
            ++overflow_resets_;
        }
        buf_.insert(buf_.end(), data, data + len);

        size_t parsed = 0;
        size_t offset = 0;
        while (buf_.size() - offset >= sizeof(PacketT)) {
            const uint8_t* ptr = buf_.data() + offset;
            if (ptr[0] != 'A' || ptr[1] != 'B') {
                ++offset;
                continue;
            }
            if (!check_crc16(ptr, sizeof(PacketT))) {
                ++crc_errors_;
                ++offset;
                continue;
            }

            PacketT pkt{};
            std::memcpy(&pkt, ptr, sizeof(PacketT));
            offset += sizeof(PacketT);

            if (!is_valid_packet(pkt)) {
                ++invalid_frames_;
                continue;
            }

            ++valid_frames_;
            ++parsed;
            on_packet(pkt);
        }

        if (offset > 0) {
            buf_.erase(buf_.begin(), buf_.begin() + static_cast<std::ptrdiff_t>(offset));
        }
        return parsed;
    }

    uint64_t valid_frames() const noexcept { return valid_frames_; }
    uint64_t crc_errors() const noexcept { return crc_errors_; }
    uint64_t invalid_frames() const noexcept { return invalid_frames_; }
    uint64_t overflow_resets() const noexcept { return overflow_resets_; }

private:
    std::vector<uint8_t> buf_;
    uint64_t valid_frames_ = 0;
    uint64_t crc_errors_ = 0;
    uint64_t invalid_frames_ = 0;
    uint64_t overflow_resets_ = 0;
};

}  // namespace rm_aim::hal
