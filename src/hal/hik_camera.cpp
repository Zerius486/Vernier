#include "rm_aim/hal/camera.hpp"

#ifdef RM_AIM_WITH_MVS

#include <MvCameraControl.h>

#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <linux/usbdevice_fs.h>
#include <sstream>
#include <sys/ioctl.h>
#include <unistd.h>

namespace rm_aim::hal {

namespace {

PixelFormat map_mvs_pixel_format(MvGvspPixelType type) noexcept {
    switch (type) {
        case PixelType_Gvsp_BayerRG8: return PixelFormat::BayerRG8;
        case PixelType_Gvsp_BayerBG8: return PixelFormat::BayerBG8;
        case PixelType_Gvsp_BayerGR8: return PixelFormat::BayerGR8;
        case PixelType_Gvsp_BayerGB8: return PixelFormat::BayerGB8;
        case PixelType_Gvsp_Mono8:    return PixelFormat::Mono8;
        case PixelType_Gvsp_BGR8_Packed: return PixelFormat::BGR8;
        default: return PixelFormat::Unknown;
    }
}

std::string read_sysfs_trimmed(const std::filesystem::path& p) {
    std::ifstream f(p);
    std::string s;
    if (std::getline(f, s)) {
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) {
            s.pop_back();
        }
    }
    return s;
}

}  // namespace

HikCameraSource::HikCameraSource(CameraConfig cfg) : cfg_(std::move(cfg)) {}

HikCameraSource::~HikCameraSource() { stop(); }

bool HikCameraSource::start(LatestFramePool& pool) {
    if (running_.exchange(true)) return true;
    pool_ = &pool;
    daemon_thread_ = std::thread(&HikCameraSource::daemon_loop, this);
    return true;
}

void HikCameraSource::stop() {
    if (!running_.exchange(false)) return;
    capturing_.store(false, std::memory_order_release);
    if (daemon_thread_.joinable()) {
        daemon_thread_.join();
    }
    stop_and_close();
}

bool HikCameraSource::reset_usb_device() const {
    const auto sep = cfg_.vid_pid.find(':');
    if (sep == std::string::npos) return false;
    const std::string target_vid = cfg_.vid_pid.substr(0, sep);
    const std::string target_pid = cfg_.vid_pid.substr(sep + 1);

    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator("/sys/bus/usb/devices", ec)) {
        const auto dir = entry.path();
        if (read_sysfs_trimmed(dir / "idVendor") != target_vid) continue;
        if (read_sysfs_trimmed(dir / "idProduct") != target_pid) continue;

        const std::string busnum = read_sysfs_trimmed(dir / "busnum");
        const std::string devnum = read_sysfs_trimmed(dir / "devnum");
        if (busnum.empty() || devnum.empty()) continue;

        std::ostringstream dev_path;
        dev_path << "/dev/bus/usb/"
                 << std::setw(3) << std::setfill('0') << std::stoi(busnum) << "/"
                 << std::setw(3) << std::setfill('0') << std::stoi(devnum);

        const int fd = ::open(dev_path.str().c_str(), O_WRONLY);
        if (fd < 0) return false;
        const int rc = ioctl(fd, USBDEVFS_RESET, 0);
        ::close(fd);
        return rc == 0;
    }
    return false;
}

bool HikCameraSource::open_and_start_grabbing() {
    MV_CC_DEVICE_INFO_LIST dev_list{};
    if (MV_CC_EnumDevices(MV_USB_DEVICE | MV_GIGE_DEVICE, &dev_list) != MV_OK ||
        dev_list.nDeviceNum == 0) {
        return false;
    }

    if (MV_CC_CreateHandle(&handle_, dev_list.pDeviceInfo[0]) != MV_OK) {
        handle_ = nullptr;
        return false;
    }

    if (MV_CC_OpenDevice(handle_) != MV_OK) {
        MV_CC_DestroyHandle(handle_);
        handle_ = nullptr;
        return false;
    }

    // 面向低延迟传统 + CUDA Bayer 流水线的最优硬件配置：
    // - 强制输出 BayerRG8（相对 RGB8 带宽降至 1/3，免 CPU 全图 Demosaic）
    // - 关闭自动曝光、自动增益与自动白平衡（避免红蓝通道漂移）
    MV_CC_SetEnumValue(handle_, "PixelFormat", PixelType_Gvsp_BayerRG8);
    MV_CC_SetEnumValue(handle_, "ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
    MV_CC_SetEnumValue(handle_, "GainAuto", MV_GAIN_MODE_OFF);
    MV_CC_SetEnumValue(handle_, "BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_OFF);
    MV_CC_SetFloatValue(handle_, "ExposureTime", static_cast<float>(cfg_.exposure_us));
    MV_CC_SetFloatValue(handle_, "Gain", static_cast<float>(cfg_.gain));
    MV_CC_SetBoolValue(handle_, "AcquisitionFrameRateEnable", true);
    MV_CC_SetFloatValue(handle_, "AcquisitionFrameRate", static_cast<float>(cfg_.frame_rate));

    // 压缩 SDK 内部缓存队列深度，避免旧帧堆积拉高延迟
    MV_CC_SetImageNodeNum(handle_, 2);

    if (MV_CC_StartGrabbing(handle_) != MV_OK) {
        MV_CC_CloseDevice(handle_);
        MV_CC_DestroyHandle(handle_);
        handle_ = nullptr;
        return false;
    }

    capturing_.store(true, std::memory_order_release);
    capture_thread_ = std::thread(&HikCameraSource::capture_loop, this);
    return true;
}

void HikCameraSource::stop_and_close() {
    capturing_.store(false, std::memory_order_release);
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    if (handle_ != nullptr) {
        MV_CC_StopGrabbing(handle_);
        MV_CC_CloseDevice(handle_);
        MV_CC_DestroyHandle(handle_);
        handle_ = nullptr;
    }
}

void HikCameraSource::capture_loop() {
    MV_FRAME_OUT raw{};
    const int64_t offset_ns =
        static_cast<int64_t>((cfg_.time_offset_ms * 1e6) - (cfg_.exposure_us * 500.0));

    while (capturing_.load(std::memory_order_acquire) && running_.load(std::memory_order_acquire)) {
        // 直接在 MVS SDK 内阻塞等待新帧（无任何人为 sleep 延迟）
        const unsigned int ret = MV_CC_GetImageBuffer(handle_, &raw, 100);
        if (ret != MV_OK) {
            errors_.fetch_add(1, std::memory_order_relaxed);
            break;
        }

        const int64_t now_ns = static_cast<int64_t>(to_steady_ns(Clock::now()));
        const uint64_t midpoint_ns = (now_ns + offset_ns > 0)
                                         ? static_cast<uint64_t>(now_ns + offset_ns)
                                         : static_cast<uint64_t>(now_ns);

        const auto& info = raw.stFrameInfo;
        const PixelFormat fmt = map_mvs_pixel_format(info.enPixelType);
        const uint32_t bytes = info.nFrameLen;

        FrameSlot& slot = pool_->acquire_write_slot();
        if (fmt != PixelFormat::Unknown && bytes <= slot.capacity) {
            std::memcpy(slot.host_ptr, raw.pBufAddr, bytes);
            slot.width = info.nWidth;
            slot.height = info.nHeight;
            slot.stride = info.nWidth * bytes_per_pixel(fmt);
            slot.byte_size = bytes;
            slot.format = fmt;
            slot.frame_id = info.nFrameNum;
            slot.timestamp_ns = midpoint_ns;
            pool_->publish_write_slot();
            captured_.fetch_add(1, std::memory_order_relaxed);
        } else {
            errors_.fetch_add(1, std::memory_order_relaxed);
        }

        if (MV_CC_FreeImageBuffer(handle_, &raw) != MV_OK) {
            errors_.fetch_add(1, std::memory_order_relaxed);
            break;
        }
    }

    capturing_.store(false, std::memory_order_release);
}

void HikCameraSource::daemon_loop() {
    open_and_start_grabbing();

    while (running_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        if (capturing_.load(std::memory_order_acquire)) {
            continue;
        }

        stop_and_close();
        reset_usb_device();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if (open_and_start_grabbing()) {
            reconnects_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    stop_and_close();
}

CameraStats HikCameraSource::stats() const {
    return CameraStats{
        .captured_frames = captured_.load(std::memory_order_relaxed),
        .dropped_frames = pool_ ? pool_->dropped_frames() : 0,
        .errors = errors_.load(std::memory_order_relaxed),
        .reconnects = reconnects_.load(std::memory_order_relaxed),
        .streaming = capturing_.load(std::memory_order_acquire),
    };
}

}  // namespace rm_aim::hal

#endif  // RM_AIM_WITH_MVS
