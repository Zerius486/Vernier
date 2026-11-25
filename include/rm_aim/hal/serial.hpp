#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "rm_aim/core/config.hpp"
#include "rm_aim/core/types.hpp"
#include "rm_aim/hal/protocol.hpp"

namespace rm_aim::hal {

struct SerialStats {
    uint64_t rx_packets = 0;
    uint64_t tx_packets = 0;
    uint64_t crc_errors = 0;
    uint64_t tx_errors = 0;
    uint64_t reconnects = 0;
    bool connected = false;
};

class ISerialLink {
public:
    using ImuCallback = std::function<void(const ImuSample&)>;

    virtual ~ISerialLink() = default;
    virtual bool start(ImuCallback on_imu) = 0;
    virtual void stop() = 0;
    virtual bool send_command(const GimbalCmd& cmd) = 0;
    virtual SerialStats stats() const = 0;
};

/// 统一 POSIX UART（/dev/ttyTHS*、/dev/ttyUSB*）与 USB-CDC（/dev/ttyACM*）串口驱动。
/// 使用 Linux termios2 支持任意非标准波特率，epoll 异步非阻塞收包，并支持 USB-CDC 断线重枚举自动重连。
class PosixSerialLink final : public ISerialLink {
public:
    explicit PosixSerialLink(SerialConfig cfg);
    ~PosixSerialLink() override;

    bool start(ImuCallback on_imu) override;
    void stop() override;
    bool send_command(const GimbalCmd& cmd) override;
    SerialStats stats() const override;

private:
    bool open_port(const std::string& dev_path);
    bool try_connect();
    void close_port();
    void rx_loop();

    SerialConfig cfg_;
    ImuCallback on_imu_;
    StreamPacketParser<GimbalToVisionPacket> parser_;

    mutable std::mutex fd_mutex_;
    int fd_ = -1;
    std::string active_device_;

    std::atomic<bool> running_{false};
    std::atomic<bool> connected_{false};
    std::atomic<uint64_t> tx_packets_{0};
    std::atomic<uint64_t> tx_errors_{0};
    std::atomic<uint64_t> reconnects_{0};
    std::thread rx_thread_;
};

/// 纯内存虚拟回环串口，用于无硬件单元测试与离线仿真。
class LoopbackSerialLink final : public ISerialLink {
public:
    using CmdCallback = std::function<void(const VisionToGimbalPacket&)>;

    explicit LoopbackSerialLink(CmdCallback on_cmd_sent = nullptr);
    ~LoopbackSerialLink() override = default;

    bool start(ImuCallback on_imu) override;
    void stop() override;
    bool send_command(const GimbalCmd& cmd) override;
    SerialStats stats() const override;

    /// 注入模拟下位机 STM32H7 发来的原始字节流。
    size_t inject_rx_bytes(const uint8_t* data, size_t len);

private:
    ImuCallback on_imu_;
    CmdCallback on_cmd_sent_;
    StreamPacketParser<GimbalToVisionPacket> rx_parser_;
    StreamPacketParser<VisionToGimbalPacket> tx_parser_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> tx_packets_{0};
};

std::unique_ptr<ISerialLink> create_serial_link(const SerialConfig& cfg);

}  // namespace rm_aim::hal
