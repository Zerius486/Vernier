#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace rm_aim::debug {

/// 零外部依赖的极简 POSIX Socket MJPEG HTTP 推流服务器（面向 SSH 无头调试）。
/// 浏览器直接访问 `http://<设备IP>:<port>/` 即可查看实时标注画面；
/// 当没有任何浏览器连接时 `has_clients()` 返回 false，主循环完全跳过画图与 JPEG 编码（0 CPU 开销）。
class MjpegServer {
public:
    explicit MjpegServer(uint16_t port = 8080);
    ~MjpegServer();

    MjpegServer(const MjpegServer&) = delete;
    MjpegServer& operator=(const MjpegServer&) = delete;

    bool start();
    void stop();

    /// 返回当前是否有在线浏览器客户端（用于主循环零开销门控）。
    bool has_clients() const noexcept {
        return client_count_.load(std::memory_order_acquire) > 0;
    }

    /// 推送一帧已编码好的 JPEG 字节流。
    void publish_jpeg(const std::vector<uint8_t>& jpeg_bytes);

private:
    void accept_loop();
    void serve_client(int client_fd);

    uint16_t port_;
    int listen_fd_ = -1;
    std::atomic<bool> running_{false};
    std::atomic<int> client_count_{0};
    std::thread accept_thread_;

    std::mutex frame_mutex_;
    std::vector<uint8_t> latest_jpeg_;
    uint64_t frame_seq_ = 0;
};

}  // namespace rm_aim::debug
