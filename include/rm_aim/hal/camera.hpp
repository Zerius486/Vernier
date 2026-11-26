#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "rm_aim/core/config.hpp"
#include "rm_aim/core/ring_buffer.hpp"
#include "rm_aim/core/types.hpp"

namespace rm_aim::hal {

struct CameraStats {
    uint64_t captured_frames = 0;
    uint64_t dropped_frames = 0;
    uint64_t errors = 0;
    uint64_t reconnects = 0;
    bool streaming = false;
};

class ICameraSource {
public:
    virtual ~ICameraSource() = default;

    /// 启动采图线程，将图像持续写入传入的零拷贝 LatestFramePool。
    virtual bool start(LatestFramePool& pool) = 0;
    virtual void stop() = 0;
    virtual CameraStats stats() const = 0;
};

// ---------------------------------------------------------------------------
// 紧凑无损原始 Bayer/Mono + IMU 同步录包格式（.rmraw），用于离线精确复现与调参
//   - RMRAW001 (v1)：仅包含连续 RawFrameHeader + 像素体（向下兼容）
//   - RMRAW002 (v2)：流式分块（RawChunkHeader），交错记录原始图像帧与高频 ImuSample
// ---------------------------------------------------------------------------
enum class RawChunkType : uint8_t {
    Frame = 1,
    Imu = 2,
};

#pragma pack(push, 1)
struct RawFileHeader {
    char magic[8];  // 魔数："RMRAW001" 或 "RMRAW002"
    uint32_t version;
    uint32_t reserved;
};

struct RawChunkHeader {
    uint8_t type;         // 对应 RawChunkType
    uint8_t reserved[3];
    uint32_t payload_size;
};

struct RawFrameHeader {
    uint64_t frame_id;
    uint64_t timestamp_ns;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t byte_size;
    uint8_t format;  // 对应 PixelFormat 枚举值
    uint8_t padding[3];
};
#pragma pack(pop)

/// 异步无阻塞录包写入器，输出 RMRAW002 帧与 IMU 同步流。
class RawFrameWriter {
public:
    RawFrameWriter() = default;
    ~RawFrameWriter();

    bool open(const std::string& path);
    bool write(const FrameSlot& slot);
    bool write_imu(const ImuSample& sample);
    void close();

    uint64_t dropped_frames() const noexcept {
        return dropped_frames_.load(std::memory_order_relaxed);
    }

private:
    struct PendingChunk {
        RawChunkType type = RawChunkType::Frame;
        RawFrameHeader fhdr{};
        std::vector<uint8_t> pixels;
        ImuSample imu{};
    };

    void writer_loop();

    std::ofstream file_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<PendingChunk> queue_;
    size_t queued_frames_ = 0;
    bool open_ = false;
    bool stop_ = false;
    std::atomic<uint64_t> dropped_frames_{0};
    std::thread writer_thread_;
};

/// 顺序录包读取器，兼容 RMRAW001 与 RMRAW002。
class RawLogReader {
public:
    bool open(const std::string& path);
    bool read_next(RawChunkType& out_type, RawFrameHeader& out_fhdr,
                   std::vector<uint8_t>& out_pixels, ImuSample& out_imu);
    bool rewind();
    void close();

    uint32_t version() const noexcept { return version_; }
    bool is_open() const noexcept { return file_.is_open(); }

private:
    std::ifstream file_;
    uint32_t version_ = 0;
};

/// 离线回放源：将 .rmraw（v1/v2）录包中的图像帧与同步 IMU 回放到系统主链路。
class ReplayCameraSource final : public ICameraSource {
public:
    ReplayCameraSource(std::string path, double target_fps = 0.0, bool loop = false);
    ~ReplayCameraSource() override;

    /// 注册同步 IMU 回放回调（应在 start() 前调用，用于将 RMRAW002 内嵌 IMU 注入 ImuBuffer）。
    void set_imu_callback(std::function<void(const ImuSample&)> cb);

    bool start(LatestFramePool& pool) override;
    void stop() override;
    CameraStats stats() const override;

private:
    void worker(LatestFramePool* pool);

    std::string path_;
    double target_fps_;
    bool loop_;
    std::function<void(const ImuSample&)> imu_cb_;
    std::atomic<bool> running_{false};
    std::atomic<bool> streaming_{false};
    std::atomic<uint64_t> captured_{0};
    std::atomic<uint64_t> errors_{0};
    std::thread thread_;
};

/// 合成图像源：生成带两条移动竖直灯条的 BayerRG8 画面，用于无相机压测与单测。
class SyntheticCameraSource final : public ICameraSource {
public:
    SyntheticCameraSource(uint32_t width = 1280, uint32_t height = 1024,
                          double target_fps = 200.0, EnemyColor color = EnemyColor::Blue);
    ~SyntheticCameraSource() override;

    bool start(LatestFramePool& pool) override;
    void stop() override;
    CameraStats stats() const override;

private:
    void worker(LatestFramePool* pool);

    uint32_t width_;
    uint32_t height_;
    double target_fps_;
    EnemyColor color_;
    std::atomic<bool> running_{false};
    std::atomic<bool> streaming_{false};
    std::atomic<uint64_t> captured_{0};
    std::thread thread_;
};

#ifdef RM_AIM_WITH_MVS
/// 海康 MVS 工业相机零拷贝直连驱动。
class HikCameraSource final : public ICameraSource {
public:
    explicit HikCameraSource(CameraConfig cfg);
    ~HikCameraSource() override;

    bool start(LatestFramePool& pool) override;
    void stop() override;
    CameraStats stats() const override;

private:
    bool open_and_start_grabbing();
    void stop_and_close();
    void daemon_loop();
    void capture_loop();
    bool reset_usb_device() const;

    CameraConfig cfg_;
    LatestFramePool* pool_ = nullptr;
    void* handle_ = nullptr;

    std::atomic<bool> running_{false};
    std::atomic<bool> capturing_{false};
    std::atomic<uint64_t> captured_{0};
    std::atomic<uint64_t> errors_{0};
    std::atomic<uint64_t> reconnects_{0};

    std::thread daemon_thread_;
    std::thread capture_thread_;
};
#endif

std::unique_ptr<ICameraSource> create_camera_source(const CameraConfig& cfg, EnemyColor color);

}  // namespace rm_aim::hal
