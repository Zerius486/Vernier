#pragma once

#include <atomic>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <thread>

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
// 紧凑无损原始 Bayer/Mono 录包格式（.rmraw），用于离线复现与调参
// ---------------------------------------------------------------------------
#pragma pack(push, 1)
struct RawFileHeader {
    char magic[8];  // 魔数："RMRAW001"
    uint32_t version;
    uint32_t reserved;
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

class RawFrameWriter {
public:
    RawFrameWriter() = default;
    ~RawFrameWriter();

    bool open(const std::string& path);
    bool write(const FrameSlot& slot);
    void close();

private:
    std::ofstream file_;
};

/// 离线回放源：按指定帧率（或全速）将 .rmraw 录包回放到 LatestFramePool。
class ReplayCameraSource final : public ICameraSource {
public:
    ReplayCameraSource(std::string path, double target_fps = 0.0, bool loop = false);
    ~ReplayCameraSource() override;

    bool start(LatestFramePool& pool) override;
    void stop() override;
    CameraStats stats() const override;

private:
    void worker(LatestFramePool* pool);

    std::string path_;
    double target_fps_;
    bool loop_;
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
/// 海康 MVS 工业相机零拷贝直连驱动（强制 BayerRG8、手动曝光/关闭自动白平衡、原生 USB 复位）。
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
