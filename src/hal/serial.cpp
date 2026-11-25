#include "rm_aim/hal/serial.hpp"

#include <asm/termbits.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <sys/epoll.h>
#include <sys/ioctl.h>
#include <unistd.h>

namespace rm_aim::hal {

namespace {

bool configure_termios2(int fd, uint32_t baudrate) {
    struct termios2 tio {};
    if (ioctl(fd, TCGETS2, &tio) != 0) {
        return false;
    }

    // 原始 8N1 透传模式，关闭流控、回显与行缓冲处理
    tio.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL | IXON | IXOFF | IXANY);
    tio.c_oflag &= ~OPOST;
    tio.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
    tio.c_cflag &= ~(CSIZE | PARENB | PARODD | CSTOPB | CRTSCTS);
    tio.c_cflag |= (CS8 | CREAD | CLOCAL);

    tio.c_cflag &= ~CBAUD;
    tio.c_cflag |= BOTHER;
    tio.c_ispeed = baudrate;
    tio.c_ospeed = baudrate;

    tio.c_cc[VMIN] = 0;
    tio.c_cc[VTIME] = 0;

    return ioctl(fd, TCSETS2, &tio) == 0;
}

}  // namespace

PosixSerialLink::PosixSerialLink(SerialConfig cfg) : cfg_(std::move(cfg)) {}

PosixSerialLink::~PosixSerialLink() { stop(); }

bool PosixSerialLink::open_port(const std::string& dev_path) {
    const int fd = ::open(dev_path.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) {
        return false;
    }

    if (!configure_termios2(fd, cfg_.baudrate)) {
        ::close(fd);
        return false;
    }

    // 清空残留收发缓存
    ioctl(fd, TCFLSH, TCIOFLUSH);

    std::lock_guard<std::mutex> lock(fd_mutex_);
    if (fd_ >= 0) {
        ::close(fd_);
    }
    fd_ = fd;
    active_device_ = dev_path;
    connected_.store(true, std::memory_order_release);
    return true;
}

bool PosixSerialLink::try_connect() {
    if (open_port(cfg_.device)) {
        return true;
    }

    // 针对 USB-CDC / USB-UART 断线后重新枚举（如 /dev/ttyACM0 -> /dev/ttyACM1）的备选扫描
    if (!cfg_.fallback_prefix.empty()) {
        for (int i = 0; i < 8; ++i) {
            const std::string candidate = cfg_.fallback_prefix + std::to_string(i);
            if (candidate == cfg_.device) continue;
            if (std::filesystem::exists(candidate) && open_port(candidate)) {
                return true;
            }
        }
    }
    return false;
}

void PosixSerialLink::close_port() {
    std::lock_guard<std::mutex> lock(fd_mutex_);
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    connected_.store(false, std::memory_order_release);
}

bool PosixSerialLink::start(ImuCallback on_imu) {
    if (running_.exchange(true)) {
        return true;
    }
    on_imu_ = std::move(on_imu);
    try_connect();
    rx_thread_ = std::thread(&PosixSerialLink::rx_loop, this);
    return true;
}

void PosixSerialLink::stop() {
    if (!running_.exchange(false)) {
        return;
    }
    if (rx_thread_.joinable()) {
        rx_thread_.join();
    }
    close_port();
}

bool PosixSerialLink::send_command(const GimbalCmd& cmd) {
    const VisionToGimbalPacket pkt = pack_vision_cmd(cmd);
    const auto* bytes = reinterpret_cast<const uint8_t*>(&pkt);

    std::lock_guard<std::mutex> lock(fd_mutex_);
    if (fd_ < 0) {
        tx_errors_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    size_t total_written = 0;
    while (total_written < sizeof(pkt)) {
        const ssize_t n = ::write(fd_, bytes + total_written, sizeof(pkt) - total_written);
        if (n > 0) {
            total_written += static_cast<size_t>(n);
        } else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            continue;
        } else {
            tx_errors_.fetch_add(1, std::memory_order_relaxed);
            ::close(fd_);
            fd_ = -1;
            connected_.store(false, std::memory_order_release);
            return false;
        }
    }

    tx_packets_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void PosixSerialLink::rx_loop() {
    const int epfd = epoll_create1(EPOLL_CLOEXEC);
    uint8_t rx_buf[256];
    int registered_fd = -1;

    while (running_.load(std::memory_order_acquire)) {
        int current_fd = -1;
        {
            std::lock_guard<std::mutex> lock(fd_mutex_);
            current_fd = fd_;
        }

        if (current_fd < 0) {
            registered_fd = -1;
            if (!cfg_.auto_reconnect) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(cfg_.reconnect_interval_ms));
            if (try_connect()) {
                reconnects_.fetch_add(1, std::memory_order_relaxed);
            }
            continue;
        }

        if (epfd >= 0 && registered_fd != current_fd) {
            epoll_event ev{};
            ev.events = EPOLLIN | EPOLLERR | EPOLLHUP;
            ev.data.fd = current_fd;
            if (epoll_ctl(epfd, EPOLL_CTL_ADD, current_fd, &ev) == 0) {
                registered_fd = current_fd;
            }
        }

        epoll_event events[1];
        const int nfds = (epfd >= 0) ? epoll_wait(epfd, events, 1, 50) : 1;
        if (nfds <= 0) {
            continue;
        }

        if (epfd >= 0 && (events[0].events & (EPOLLERR | EPOLLHUP))) {
            close_port();
            registered_fd = -1;
            continue;
        }

        const ssize_t n = ::read(current_fd, rx_buf, sizeof(rx_buf));
        if (n > 0) {
            const uint64_t ts_ns = to_steady_ns(Clock::now());
            parser_.feed(rx_buf, static_cast<size_t>(n), [&](const GimbalToVisionPacket& pkt) {
                if (on_imu_) {
                    on_imu_(unpack_gimbal_packet(pkt, ts_ns));
                }
            });
        } else if (n == 0 || (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)) {
            close_port();
            registered_fd = -1;
        }
    }

    if (epfd >= 0) {
        ::close(epfd);
    }
}

SerialStats PosixSerialLink::stats() const {
    return SerialStats{
        .rx_packets = parser_.valid_frames(),
        .tx_packets = tx_packets_.load(std::memory_order_relaxed),
        .crc_errors = parser_.crc_errors(),
        .tx_errors = tx_errors_.load(std::memory_order_relaxed),
        .reconnects = reconnects_.load(std::memory_order_relaxed),
        .connected = connected_.load(std::memory_order_acquire),
    };
}

// ---------------------------------------------------------------------------
// LoopbackSerialLink
// ---------------------------------------------------------------------------

LoopbackSerialLink::LoopbackSerialLink(CmdCallback on_cmd_sent)
    : on_cmd_sent_(std::move(on_cmd_sent)) {}

bool LoopbackSerialLink::start(ImuCallback on_imu) {
    on_imu_ = std::move(on_imu);
    running_.store(true, std::memory_order_release);
    return true;
}

void LoopbackSerialLink::stop() { running_.store(false, std::memory_order_release); }

bool LoopbackSerialLink::send_command(const GimbalCmd& cmd) {
    if (!running_.load(std::memory_order_acquire)) return false;
    const VisionToGimbalPacket pkt = pack_vision_cmd(cmd);
    tx_packets_.fetch_add(1, std::memory_order_relaxed);
    if (on_cmd_sent_) {
        tx_parser_.feed(reinterpret_cast<const uint8_t*>(&pkt), sizeof(pkt), on_cmd_sent_);
    }
    return true;
}

size_t LoopbackSerialLink::inject_rx_bytes(const uint8_t* data, size_t len) {
    if (!running_.load(std::memory_order_acquire)) return 0;
    const uint64_t ts_ns = to_steady_ns(Clock::now());
    return rx_parser_.feed(data, len, [&](const GimbalToVisionPacket& pkt) {
        if (on_imu_) {
            on_imu_(unpack_gimbal_packet(pkt, ts_ns));
        }
    });
}

SerialStats LoopbackSerialLink::stats() const {
    return SerialStats{
        .rx_packets = rx_parser_.valid_frames(),
        .tx_packets = tx_packets_.load(std::memory_order_relaxed),
        .crc_errors = rx_parser_.crc_errors(),
        .tx_errors = 0,
        .reconnects = 0,
        .connected = running_.load(std::memory_order_acquire),
    };
}

std::unique_ptr<ISerialLink> create_serial_link(const SerialConfig& cfg) {
    if (cfg.device == "loopback") {
        return std::make_unique<LoopbackSerialLink>();
    }
    return std::make_unique<PosixSerialLink>(cfg);
}

}  // namespace rm_aim::hal
