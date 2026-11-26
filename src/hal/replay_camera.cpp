#include "rm_aim/hal/camera.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>

namespace rm_aim::hal {

// ---------------------------------------------------------------------------
// RawFrameWriter（RMRAW002 异步后台写盘：图像帧 + IMU 同步交错写入）
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
    std::memcpy(hdr.magic, "RMRAW002", 8);
    hdr.version = 2;
    hdr.reserved = 0;
    file_.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    if (!file_.good()) {
        file_.close();
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.clear();
        queued_frames_ = 0;
        stop_ = false;
        open_ = true;
        dropped_frames_.store(0, std::memory_order_relaxed);
    }
    writer_thread_ = std::thread(&RawFrameWriter::writer_loop, this);
    return true;
}

bool RawFrameWriter::write(const FrameSlot& slot) {
    if (slot.host_ptr == nullptr || slot.byte_size == 0) {
        return false;
    }

    constexpr size_t kMaxQueuedFrames = 32;
    PendingChunk chunk{};
    chunk.type = RawChunkType::Frame;
    chunk.fhdr.frame_id = slot.frame_id;
    chunk.fhdr.timestamp_ns = slot.timestamp_ns;
    chunk.fhdr.width = slot.width;
    chunk.fhdr.height = slot.height;
    chunk.fhdr.stride = slot.stride;
    chunk.fhdr.byte_size = slot.byte_size;
    chunk.fhdr.format = static_cast<uint8_t>(slot.format);
    chunk.pixels.assign(slot.host_ptr, slot.host_ptr + slot.byte_size);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!open_) return false;
        // 磁盘写入跟不上时丢弃新帧，防止内存暴涨或阻塞实时视觉主循环
        if (queued_frames_ >= kMaxQueuedFrames) {
            dropped_frames_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        queue_.push_back(std::move(chunk));
        ++queued_frames_;
    }
    cv_.notify_one();
    return true;
}

bool RawFrameWriter::write_imu(const ImuSample& sample) {
    constexpr size_t kMaxTotalChunks = 4096;
    PendingChunk chunk{};
    chunk.type = RawChunkType::Imu;
    chunk.imu = sample;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!open_ || queue_.size() >= kMaxTotalChunks) return false;
        queue_.push_back(std::move(chunk));
    }
    cv_.notify_one();
    return true;
}

void RawFrameWriter::writer_loop() {
    while (true) {
        PendingChunk chunk{};
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) {
                if (stop_) break;
                continue;
            }
            chunk = std::move(queue_.front());
            queue_.pop_front();
            if (chunk.type == RawChunkType::Frame && queued_frames_ > 0) {
                --queued_frames_;
            }
        }

        if (!file_.is_open()) continue;
        if (chunk.type == RawChunkType::Frame) {
            RawChunkHeader chdr{};
            chdr.type = static_cast<uint8_t>(RawChunkType::Frame);
            chdr.payload_size =
                static_cast<uint32_t>(sizeof(RawFrameHeader) + chunk.pixels.size());
            file_.write(reinterpret_cast<const char*>(&chdr), sizeof(chdr));
            file_.write(reinterpret_cast<const char*>(&chunk.fhdr), sizeof(chunk.fhdr));
            file_.write(reinterpret_cast<const char*>(chunk.pixels.data()),
                        static_cast<std::streamsize>(chunk.pixels.size()));
        } else if (chunk.type == RawChunkType::Imu) {
            RawChunkHeader chdr{};
            chdr.type = static_cast<uint8_t>(RawChunkType::Imu);
            chdr.payload_size = static_cast<uint32_t>(sizeof(ImuSample));
            file_.write(reinterpret_cast<const char*>(&chdr), sizeof(chdr));
            file_.write(reinterpret_cast<const char*>(&chunk.imu), sizeof(chunk.imu));
        }
    }
}

void RawFrameWriter::close() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!open_ && !writer_thread_.joinable()) {
            if (file_.is_open()) {
                file_.flush();
                file_.close();
            }
            return;
        }
        open_ = false;
        stop_ = true;
    }
    cv_.notify_all();
    if (writer_thread_.joinable()) {
        writer_thread_.join();
    }
    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }
}

// ---------------------------------------------------------------------------
// RawLogReader（顺序读取 RMRAW001 / RMRAW002）
// ---------------------------------------------------------------------------
bool RawLogReader::open(const std::string& path) {
    close();
    file_.open(path, std::ios::binary);
    if (!file_.is_open()) return false;

    RawFileHeader hdr{};
    file_.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!file_.good()) {
        close();
        return false;
    }

    if (std::memcmp(hdr.magic, "RMRAW001", 8) == 0) {
        version_ = 1;
        return true;
    }
    if (std::memcmp(hdr.magic, "RMRAW002", 8) == 0) {
        version_ = 2;
        return true;
    }

    close();
    return false;
}

bool RawLogReader::read_next(RawChunkType& out_type, RawFrameHeader& out_fhdr,
                             std::vector<uint8_t>& out_pixels, ImuSample& out_imu) {
    if (!file_.is_open()) return false;

    if (version_ == 1) {
        file_.read(reinterpret_cast<char*>(&out_fhdr), sizeof(out_fhdr));
        if (!file_.good()) return false;
        out_pixels.resize(out_fhdr.byte_size);
        file_.read(reinterpret_cast<char*>(out_pixels.data()),
                   static_cast<std::streamsize>(out_fhdr.byte_size));
        if (!file_.good()) return false;
        out_type = RawChunkType::Frame;
        return true;
    }

    while (true) {
        RawChunkHeader chdr{};
        file_.read(reinterpret_cast<char*>(&chdr), sizeof(chdr));
        if (!file_.good()) return false;

        if (chdr.type == static_cast<uint8_t>(RawChunkType::Frame)) {
            if (chdr.payload_size < sizeof(RawFrameHeader)) return false;
            file_.read(reinterpret_cast<char*>(&out_fhdr), sizeof(out_fhdr));
            if (!file_.good()) return false;
            const uint32_t expected_bytes = chdr.payload_size - static_cast<uint32_t>(sizeof(RawFrameHeader));
            if (out_fhdr.byte_size != expected_bytes) return false;
            out_pixels.resize(out_fhdr.byte_size);
            file_.read(reinterpret_cast<char*>(out_pixels.data()),
                       static_cast<std::streamsize>(out_fhdr.byte_size));
            if (!file_.good()) return false;
            out_type = RawChunkType::Frame;
            return true;
        }

        if (chdr.type == static_cast<uint8_t>(RawChunkType::Imu)) {
            if (chdr.payload_size != sizeof(ImuSample)) return false;
            file_.read(reinterpret_cast<char*>(&out_imu), sizeof(out_imu));
            if (!file_.good()) return false;
            out_type = RawChunkType::Imu;
            return true;
        }

        // 跳过未来版本新增的未知扩展块，保证前向兼容
        file_.seekg(static_cast<std::streamoff>(chdr.payload_size), std::ios::cur);
        if (!file_.good()) return false;
    }
}

bool RawLogReader::rewind() {
    if (!file_.is_open()) return false;
    file_.clear();
    file_.seekg(sizeof(RawFileHeader), std::ios::beg);
    return file_.good();
}

void RawLogReader::close() {
    if (file_.is_open()) {
        file_.close();
    }
    version_ = 0;
}

// ---------------------------------------------------------------------------
// ReplayCameraSource
// ---------------------------------------------------------------------------
ReplayCameraSource::ReplayCameraSource(std::string path, double target_fps, bool loop)
    : path_(std::move(path)), target_fps_(target_fps), loop_(loop) {}

ReplayCameraSource::~ReplayCameraSource() { stop(); }

void ReplayCameraSource::set_imu_callback(std::function<void(const ImuSample&)> cb) {
    imu_cb_ = std::move(cb);
}

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
    RawLogReader reader;
    if (!reader.open(path_)) {
        errors_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    streaming_.store(true, std::memory_order_release);
    const auto frame_period =
        (target_fps_ > 0.0)
            ? std::chrono::nanoseconds(static_cast<int64_t>(1e9 / target_fps_))
            : std::chrono::nanoseconds(0);

    std::vector<uint8_t> pixels;
    RawFrameHeader fhdr{};
    ImuSample imu{};
    RawChunkType type = RawChunkType::Frame;

    while (running_.load(std::memory_order_acquire)) {
        const auto t0 = Clock::now();
        if (!reader.read_next(type, fhdr, pixels, imu)) {
            if (loop_ && reader.rewind()) {
                continue;
            }
            break;
        }

        if (type == RawChunkType::Imu) {
            if (imu_cb_) {
                imu_cb_(imu);
            }
            continue;
        }

        FrameSlot& slot = pool->acquire_write_slot();
        if (fhdr.byte_size > slot.capacity) {
            errors_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        std::memcpy(slot.host_ptr, pixels.data(), fhdr.byte_size);
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
