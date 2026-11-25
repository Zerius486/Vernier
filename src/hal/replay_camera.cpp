#include "rm_aim/hal/camera.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>

namespace rm_aim::hal {

// ---------------------------------------------------------------------------
// RawFrameWriter
// ---------------------------------------------------------------------------
RawFrameWriter::~RawFrameWriter() { close(); }

bool RawFrameWriter::open(const std::string& path) {
    close();
    const std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
    }

    file_.open(path, std::ios::binary | std::ios::trunc);
    if (!file_.is_open()) return false;

    RawFileHeader hdr{};
    std::memcpy(hdr.magic, "RMRAW001", 8);
    hdr.version = 1;
    hdr.reserved = 0;
    file_.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    return file_.good();
}

bool RawFrameWriter::write(const FrameSlot& slot) {
    if (!file_.is_open() || slot.host_ptr == nullptr || slot.byte_size == 0) {
        return false;
    }

    RawFrameHeader fhdr{};
    fhdr.frame_id = slot.frame_id;
    fhdr.timestamp_ns = slot.timestamp_ns;
    fhdr.width = slot.width;
    fhdr.height = slot.height;
    fhdr.stride = slot.stride;
    fhdr.byte_size = slot.byte_size;
    fhdr.format = static_cast<uint8_t>(slot.format);

    file_.write(reinterpret_cast<const char*>(&fhdr), sizeof(fhdr));
    file_.write(reinterpret_cast<const char*>(slot.host_ptr),
                static_cast<std::streamsize>(slot.byte_size));
    return file_.good();
}

void RawFrameWriter::close() {
    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }
}

// ---------------------------------------------------------------------------
// ReplayCameraSource
// ---------------------------------------------------------------------------
ReplayCameraSource::ReplayCameraSource(std::string path, double target_fps, bool loop)
    : path_(std::move(path)), target_fps_(target_fps), loop_(loop) {}

ReplayCameraSource::~ReplayCameraSource() { stop(); }

bool ReplayCameraSource::start(LatestFramePool& pool) {
    if (running_.exchange(true)) return true;
    thread_ = std::thread(&ReplayCameraSource::worker, this, &pool);
    return true;
}

void ReplayCameraSource::stop() {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) thread_.join();
    streaming_.store(false, std::memory_order_release);
}

void ReplayCameraSource::worker(LatestFramePool* pool) {
    std::ifstream file(path_, std::ios::binary);
    if (!file.is_open()) {
        errors_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    RawFileHeader hdr{};
    file.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!file.good() || std::memcmp(hdr.magic, "RMRAW001", 8) != 0) {
        errors_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    streaming_.store(true, std::memory_order_release);
    const auto frame_period =
        (target_fps_ > 0.0)
            ? std::chrono::nanoseconds(static_cast<int64_t>(1e9 / target_fps_))
            : std::chrono::nanoseconds(0);

    while (running_.load(std::memory_order_acquire)) {
        const auto t0 = Clock::now();
        RawFrameHeader fhdr{};
        file.read(reinterpret_cast<char*>(&fhdr), sizeof(fhdr));
        if (!file.good()) {
            if (loop_) {
                file.clear();
                file.seekg(sizeof(RawFileHeader), std::ios::beg);
                continue;
            }
            break;
        }

        FrameSlot& slot = pool->acquire_write_slot();
        if (fhdr.byte_size > slot.capacity) {
            errors_.fetch_add(1, std::memory_order_relaxed);
            file.seekg(fhdr.byte_size, std::ios::cur);
            continue;
        }

        file.read(reinterpret_cast<char*>(slot.host_ptr),
                  static_cast<std::streamsize>(fhdr.byte_size));
        if (!file.good()) {
            break;
        }

        slot.frame_id = fhdr.frame_id;
        slot.timestamp_ns = fhdr.timestamp_ns;
        slot.width = fhdr.width;
        slot.height = fhdr.height;
        slot.stride = fhdr.stride;
        slot.byte_size = fhdr.byte_size;
        slot.format = static_cast<PixelFormat>(fhdr.format);

        pool->publish_write_slot();
        captured_.fetch_add(1, std::memory_order_relaxed);

        if (frame_period.count() > 0) {
            std::this_thread::sleep_until(t0 + frame_period);
        }
    }

    streaming_.store(false, std::memory_order_release);
}

CameraStats ReplayCameraSource::stats() const {
    return CameraStats{
        .captured_frames = captured_.load(std::memory_order_relaxed),
        .dropped_frames = 0,
        .errors = errors_.load(std::memory_order_relaxed),
        .reconnects = 0,
        .streaming = streaming_.load(std::memory_order_acquire),
    };
}

// ---------------------------------------------------------------------------
// SyntheticCameraSource（生成 BayerRG8 合成装甲板灯条，用于离线压测）
// ---------------------------------------------------------------------------
SyntheticCameraSource::SyntheticCameraSource(uint32_t width, uint32_t height,
                                             double target_fps, EnemyColor color)
    : width_(width), height_(height), target_fps_(target_fps), color_(color) {}

SyntheticCameraSource::~SyntheticCameraSource() { stop(); }

bool SyntheticCameraSource::start(LatestFramePool& pool) {
    if (running_.exchange(true)) return true;
    thread_ = std::thread(&SyntheticCameraSource::worker, this, &pool);
    return true;
}

void SyntheticCameraSource::stop() {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) thread_.join();
    streaming_.store(false, std::memory_order_release);
}

void SyntheticCameraSource::worker(LatestFramePool* pool) {
    streaming_.store(true, std::memory_order_release);
    const auto period =
        (target_fps_ > 0.0)
            ? std::chrono::nanoseconds(static_cast<int64_t>(1e9 / target_fps_))
            : std::chrono::nanoseconds(0);

    uint64_t seq = 0;
    while (running_.load(std::memory_order_acquire)) {
        const auto t0 = Clock::now();
        FrameSlot& slot = pool->acquire_write_slot();
        const uint32_t bytes = width_ * height_;
        if (bytes > slot.capacity) {
            break;
        }

        // 模拟低曝光暗背景
        std::memset(slot.host_ptr, 8, bytes);

        // 按 BayerRG8 排布绘制两条移动的竖直灯条：
        // (偶行, 偶列) = R，(奇行, 奇列) = B，其余 = G
        const uint32_t cx = static_cast<uint32_t>(
            width_ / 2 + static_cast<int32_t>(120.0 * std::sin(static_cast<double>(seq) * 0.05)));
        const uint32_t cy = height_ / 2;
        const uint32_t half_w = 50;
        const uint32_t half_h = 26;

        auto draw_bar = [&](uint32_t bx) {
            const uint32_t x0 = (bx > 4) ? bx - 4 : 0;
            const uint32_t x1 = (bx + 4 < width_) ? bx + 4 : width_ - 1;
            const uint32_t y0 = (cy > half_h) ? cy - half_h : 0;
            const uint32_t y1 = (cy + half_h < height_) ? cy + half_h : height_ - 1;

            for (uint32_t y = y0; y <= y1; ++y) {
                uint8_t* row = slot.host_ptr + static_cast<size_t>(y) * width_;
                for (uint32_t x = x0; x <= x1; ++x) {
                    const bool is_r = ((y & 1U) == 0U) && ((x & 1U) == 0U);
                    const bool is_b = ((y & 1U) == 1U) && ((x & 1U) == 1U);
                    if (color_ == EnemyColor::Red) {
                        row[x] = is_r ? 250 : (is_b ? 20 : 120);
                    } else {
                        row[x] = is_b ? 250 : (is_r ? 20 : 120);
                    }
                }
            }
        };

        draw_bar(cx - half_w);
        draw_bar(cx + half_w);

        slot.frame_id = ++seq;
        slot.timestamp_ns = to_steady_ns(t0);
        slot.width = width_;
        slot.height = height_;
        slot.stride = width_;
        slot.byte_size = bytes;
        slot.format = PixelFormat::BayerRG8;

        pool->publish_write_slot();
        captured_.fetch_add(1, std::memory_order_relaxed);

        if (period.count() > 0) {
            std::this_thread::sleep_until(t0 + period);
        }
    }

    streaming_.store(false, std::memory_order_release);
}

CameraStats SyntheticCameraSource::stats() const {
    return CameraStats{
        .captured_frames = captured_.load(std::memory_order_relaxed),
        .dropped_frames = 0,
        .errors = 0,
        .reconnects = 0,
        .streaming = streaming_.load(std::memory_order_acquire),
    };
}

std::unique_ptr<ICameraSource> create_camera_source(const CameraConfig& cfg, EnemyColor color) {
    if (cfg.source == "replay") {
        return std::make_unique<ReplayCameraSource>(cfg.replay_path, cfg.frame_rate, true);
    }
    if (cfg.source == "synthetic") {
        return std::make_unique<SyntheticCameraSource>(1280, 1024, cfg.frame_rate, color);
    }
#ifdef RM_AIM_WITH_MVS
    return std::make_unique<HikCameraSource>(cfg);
#else
    return std::make_unique<SyntheticCameraSource>(1280, 1024, cfg.frame_rate, color);
#endif
}

}  // namespace rm_aim::hal
