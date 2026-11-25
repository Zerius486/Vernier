#include "rm_aim/debug/mjpeg_server.hpp"

#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

namespace rm_aim::debug {

namespace {

bool send_all(int fd, const void* data, size_t len) noexcept {
    const auto* ptr = static_cast<const uint8_t*>(data);
    size_t sent = 0;
    while (sent < len) {
        const ssize_t n = ::send(fd, ptr + sent, len - sent, MSG_NOSIGNAL);
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

}  // namespace

MjpegServer::MjpegServer(uint16_t port) : port_(port) {}

MjpegServer::~MjpegServer() { stop(); }

bool MjpegServer::start() {
    if (running_.exchange(true)) return true;

    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        running_.store(false, std::memory_order_release);
        return false;
    }

    int opt = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // 设置 accept 超时，便于 stop() 时优雅退出
    timeval tv{};
    tv.tv_sec = 0;
    tv.tv_usec = 200'000;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port_);

    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0 ||
        ::listen(listen_fd_, 4) < 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
        running_.store(false, std::memory_order_release);
        return false;
    }

    accept_thread_ = std::thread(&MjpegServer::accept_loop, this);
    return true;
}

void MjpegServer::stop() {
    if (!running_.exchange(false)) return;
    if (listen_fd_ >= 0) {
        ::shutdown(listen_fd_, SHUT_RDWR);
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
    if (accept_thread_.joinable()) {
        accept_thread_.join();
    }
}

void MjpegServer::publish_jpeg(const std::vector<uint8_t>& jpeg_bytes) {
    if (jpeg_bytes.empty()) return;
    std::lock_guard<std::mutex> lock(frame_mutex_);
    latest_jpeg_ = jpeg_bytes;
    ++frame_seq_;
}

void MjpegServer::accept_loop() {
    while (running_.load(std::memory_order_acquire)) {
        sockaddr_in client_addr{};
        socklen_t len = sizeof(client_addr);
        const int cfd = ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&client_addr), &len);
        if (cfd < 0) {
            continue;
        }
        // 限制最大并发观看数，避免多浏览器拖垮网络带宽
        if (client_count_.load(std::memory_order_acquire) >= 3) {
            ::close(cfd);
            continue;
        }
        std::thread(&MjpegServer::serve_client, this, cfd).detach();
    }
}

void MjpegServer::serve_client(int client_fd) {
    client_count_.fetch_add(1, std::memory_order_acq_rel);

    // 读掉 HTTP 请求头
    char req_buf[512];
    timeval tv{};
    tv.tv_sec = 1;
    ::setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    (void)::recv(client_fd, req_buf, sizeof(req_buf), 0);

    static constexpr const char* kHttpHeader =
        "HTTP/1.1 200 OK\r\n"
        "Cache-Control: no-cache\r\n"
        "Pragma: no-cache\r\n"
        "Connection: close\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=vernierframe\r\n\r\n";

    if (send_all(client_fd, kHttpHeader, std::strlen(kHttpHeader))) {
        uint64_t last_sent = 0;
        std::vector<uint8_t> local_buf;
        while (running_.load(std::memory_order_acquire)) {
            uint64_t seq = 0;
            {
                std::lock_guard<std::mutex> lock(frame_mutex_);
                seq = frame_seq_;
                if (seq != last_sent && !latest_jpeg_.empty()) {
                    local_buf = latest_jpeg_;
                }
            }
            if (seq == last_sent || local_buf.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(15));
                continue;
            }

            const std::string part_hdr =
                "--vernierframe\r\n"
                "Content-Type: image/jpeg\r\n"
                "Content-Length: " +
                std::to_string(local_buf.size()) + "\r\n\r\n";
            if (!send_all(client_fd, part_hdr.data(), part_hdr.size()) ||
                !send_all(client_fd, local_buf.data(), local_buf.size()) ||
                !send_all(client_fd, "\r\n", 2)) {
                break;
            }
            last_sent = seq;
            // 调试画面限流至 ~30 FPS，节省无线/网口带宽
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
    }

    ::close(client_fd);
    client_count_.fetch_sub(1, std::memory_order_acq_rel);
}

}  // namespace rm_aim::debug
