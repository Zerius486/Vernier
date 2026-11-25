#pragma once

#include <cstdint>
#include <string>

namespace rm_aim::debug {

/// 极简 PlotJuggler UDP JSON 遥测发送器（零外部依赖，关闭时零开销）。
class UdpTelemetry {
public:
    UdpTelemetry() = default;
    ~UdpTelemetry();

    UdpTelemetry(const UdpTelemetry&) = delete;
    UdpTelemetry& operator=(const UdpTelemetry&) = delete;

    bool open(const std::string& host, uint16_t port);
    void close();

    /// 发送一行 JSON 遥测数据到 PlotJuggler（UDP Server 监听对应端口即可直接出图）。
    void send_json(const std::string& json_payload) noexcept;

    bool is_open() const noexcept { return fd_ >= 0; }

private:
    int fd_ = -1;
    std::string host_;
    uint16_t port_ = 0;
};

}  // namespace rm_aim::debug
