#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "rm_aim/backend/compute_backend.hpp"
#include "rm_aim/backend/mlp_weights.hpp"
#include "rm_aim/core/config.hpp"
#include "rm_aim/core/ring_buffer.hpp"
#include "rm_aim/debug/mjpeg_server.hpp"
#include "rm_aim/debug/udp_telemetry.hpp"
#include "rm_aim/estimation/ballistic.hpp"
#include "rm_aim/estimation/ekf_tracker.hpp"
#include "rm_aim/estimation/imu_buffer.hpp"
#include "rm_aim/hal/camera.hpp"
#include "rm_aim/hal/serial.hpp"
#include "rm_aim/perception/detector.hpp"
#include "rm_aim/perception/pnp_solver.hpp"

#ifdef RM_AIM_WITH_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace {

std::atomic<bool> g_stop{false};

void handle_signal(int) { g_stop.store(true, std::memory_order_release); }

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    const std::string cfg_path = (argc > 1) ? argv[1] : "config/default.yaml";
    rm_aim::AppConfig cfg;
    try {
        cfg = rm_aim::AppConfig::load(cfg_path);
    } catch (const std::exception& e) {
        std::cerr << "[Vernier] 加载配置失败: " << e.what() << "\n";
        return 1;
    }

    // 加载超轻量 MLP 权重；若尚未训练真实权重文件则回退到占位权重并提示
    rm_aim::backend::MlpWeights weights;
    if (std::filesystem::exists(cfg.detector.mlp_weights_path)) {
        weights = rm_aim::backend::MlpWeights::load(cfg.detector.mlp_weights_path);
    } else {
        std::cerr << "[Vernier] 提示：未找到权重文件 '" << cfg.detector.mlp_weights_path
                  << "'，当前使用测试占位权重（实车前请用 scripts/train_mlp.py 导出）。\n";
        weights = rm_aim::backend::MlpWeights::make_dummy(3);
    }

    auto backend = rm_aim::backend::create_compute_backend(weights, true);
    rm_aim::perception::ArmorDetector detector(cfg.detector, cfg.enemy_color, *backend);
    rm_aim::perception::PnpSolver pnp_solver(cfg.camera);

    rm_aim::estimation::ImuBuffer imu_buffer;
#ifdef RM_AIM_WITH_EIGEN
    rm_aim::estimation::EkfTracker tracker(cfg.tracker, 0.20);
    rm_aim::estimation::Aimer aimer(cfg.aimer);
#endif

    // 预分配 3 槽零拷贝帧池（最大覆盖 1440x1080 单通道 Bayer 约 1.6MB，预留 4MB）
    rm_aim::LatestFramePool frame_pool(3, 4U * 1024U * 1024U);
    auto camera = rm_aim::hal::create_camera_source(cfg.camera, cfg.enemy_color);
    auto serial = rm_aim::hal::create_serial_link(cfg.serial);

    // 可选无头调试组件（未开启时零开销）
    rm_aim::debug::MjpegServer mjpeg(cfg.debug.mjpeg_port);
    if (cfg.debug.enable_mjpeg) {
        if (mjpeg.start()) {
            std::cout << "[Vernier] MJPEG 调试推流已启动: http://<ip>:" << cfg.debug.mjpeg_port
                      << "/\n";
        }
    }

    rm_aim::debug::UdpTelemetry telemetry;
    if (cfg.debug.enable_udp_telemetry) {
        telemetry.open(cfg.debug.udp_host, cfg.debug.udp_port);
    }

    rm_aim::hal::RawFrameWriter recorder;
    if (cfg.debug.enable_record) {
        recorder.open(cfg.debug.record_path);
    }

    std::cout << "[Vernier] 启动完成 | 后端: " << backend->name()
              << " | 零拷贝: " << (frame_pool.is_cuda_zero_copy() ? "ON" : "OFF")
              << " | 相机源: " << cfg.camera.source
              << " | 串口: " << cfg.serial.device << "\n";

    serial->start([&](const rm_aim::ImuSample& sample) { imu_buffer.push(sample); });
    camera->start(frame_pool);

    uint64_t proc_frames = 0;
    auto last_stat_tp = rm_aim::Clock::now();
    uint64_t stat_frames = 0;
    double fps = 0.0;

    while (!g_stop.load(std::memory_order_acquire)) {
        const rm_aim::FrameSlot* slot = frame_pool.try_acquire_latest();
        if (slot == nullptr) {
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            continue;
        }

        if (cfg.debug.enable_record) {
            recorder.write(*slot);
        }

        rm_aim::perception::DetectTimingUs timing{};
        auto armors = detector.detect(*slot, &timing);

        // 查询与曝光中点对齐的 IMU 姿态（无串口数据时回退为单位四元数）
        rm_aim::ImuSample synced_imu{};
        if (!imu_buffer.query(slot->timestamp_ns, synced_imu)) {
            synced_imu.timestamp_ns = slot->timestamp_ns;
            synced_imu.q = {1.0f, 0.0f, 0.0f, 0.0f};
            synced_imu.bullet_speed = 24.0f;
        }

        rm_aim::GimbalCmd cmd{};
#ifdef RM_AIM_WITH_EIGEN
        Eigen::Matrix3d R_w2c;
        Eigen::Vector3d t_w2c;
        rm_aim::estimation::compute_world_to_cam_transform(synced_imu, cfg.extrinsics, R_w2c, t_w2c);

        std::vector<rm_aim::estimation::WorldArmorObs> world_obs;
        world_obs.reserve(armors.size());
        for (auto& a : armors) {
            if (pnp_solver.solve(a)) {
                auto obs =
                    rm_aim::estimation::transform_armor_to_world(a, synced_imu, cfg.extrinsics);
                // 去畸变 + 物理倾角约束下的 4-DoF 联合位姿精修（前哨站 -15°，其余 +15°）
                constexpr double kDeg15 = 15.0 * 3.14159265358979323846 / 180.0;
                const double armor_pitch = (a.class_id == 6) ? -kDeg15 : kDeg15;
                pnp_solver.refine_pose_4dof(a, obs.pos_world, obs.yaw_world, R_w2c, t_w2c,
                                            armor_pitch);
                world_obs.push_back(obs);
            }
        }
        tracker.update(slot->timestamp_ns, world_obs);
        cmd = aimer.compute_command(tracker, synced_imu);
#else
        (void)pnp_solver;
#endif
        serial->send_command(cmd);

#ifdef RM_AIM_WITH_OPENCV
        // 仅当有浏览器连接 MJPEG 服务器时才解码画框并编码 JPEG，无人看时 0 CPU 开销
        if (cfg.debug.enable_mjpeg && mjpeg.has_clients() && slot->width > 0 && slot->height > 0) {
            cv::Mat raw(static_cast<int>(slot->height), static_cast<int>(slot->width),
                        CV_8UC1, slot->host_ptr, slot->stride);
            cv::Mat bgr;
            cv::cvtColor(raw, bgr, cv::COLOR_BayerRG2BGR);
            for (const auto& a : armors) {
                for (int i = 0; i < 4; ++i) {
                    const auto& p0 = a.corners[static_cast<size_t>(i)];
                    const auto& p1 = a.corners[static_cast<size_t>((i + 1) % 4)];
                    cv::line(bgr, {static_cast<int>(p0.x), static_cast<int>(p0.y)},
                             {static_cast<int>(p1.x), static_cast<int>(p1.y)},
                             cv::Scalar(0, 255, 0), 2);
                }
            }
            std::vector<uint8_t> jpeg;
            cv::imencode(".jpg", bgr, jpeg, {cv::IMWRITE_JPEG_QUALITY, 75});
            mjpeg.publish_jpeg(jpeg);
        }
#endif

        frame_pool.release_read_slot();
        ++proc_frames;
        ++stat_frames;

        const auto now = rm_aim::Clock::now();
        const double elapsed_s = std::chrono::duration<double>(now - last_stat_tp).count();
        if (elapsed_s >= 0.25) {
            fps = static_cast<double>(stat_frames) / elapsed_s;
            stat_frames = 0;
            last_stat_tp = now;

            const auto s_stats = serial->stats();
            if (cfg.debug.enable_tui) {
                std::printf(
                    "\r\033[K[Vernier] FPS:%6.1f | Det:%6.1fus(%s) | Armors:%zu | "
                    "Cmd(m=%u,y=%.2f,p=%.2f,hz=%u) | Ser(rx=%llu,tx=%llu,%s)",
                    fps, timing.total_us, timing.used_roi ? "ROI" : "FULL", armors.size(),
                    cmd.mode, cmd.yaw, cmd.pitch, cmd.fire_rate_hz,
                    static_cast<unsigned long long>(s_stats.rx_packets),
                    static_cast<unsigned long long>(s_stats.tx_packets),
                    s_stats.connected ? "OK" : "OFF");
                std::fflush(stdout);
            }

            if (telemetry.is_open()) {
                char buf[256];
                std::snprintf(
                    buf, sizeof(buf),
                    "{\"fps\":%.1f,\"det_us\":%.1f,\"armors\":%zu,\"mode\":%u,"
                    "\"yaw\":%.4f,\"pitch\":%.4f,\"fire_hz\":%u,\"v0\":%.2f}",
                    fps, timing.total_us, armors.size(), cmd.mode, cmd.yaw, cmd.pitch,
                    cmd.fire_rate_hz, synced_imu.bullet_speed);
                telemetry.send_json(buf);
            }
        }
    }

    if (cfg.debug.enable_tui) {
        std::cout << "\n";
    }
    camera->stop();
    serial->stop();
    mjpeg.stop();
    recorder.close();
    std::cout << "[Vernier] 已优雅退出，共处理 " << proc_frames << " 帧。\n";
    return 0;
}
