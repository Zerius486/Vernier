#include "rm_aim/debug/udp_telemetry.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace rm_aim::debug {

UdpTelemetry::~UdpTelemetry() { close(); }

bool UdpTelemetry::open(const std::string& host, uint16_t port) {
    close();
    fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd_ < 0) return false;

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        close();
        return false;
    }
    // 使用 connected UDP 简化 send 调用并避免每次重复构造地址结构
    if (::connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        close();
        return false;
    }
    host_ = host;
    port_ = port;
    return true;
}

void UdpTelemetry::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

void UdpTelemetry::send_json(const std::string& json_payload) noexcept {
    if (fd_ < 0 || json_payload.empty()) return;
    (void)::send(fd_, json_payload.data(), json_payload.size(), MSG_DONTWAIT | MSG_NOSIGNAL);
}

}  // namespace rm_aim::debug
