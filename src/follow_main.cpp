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

void print_usage(const char* prog) {
    std::cout << "用法: " << prog << " [config.yaml] [--no-ff] [--mjpeg]\n"
              << "  说明   : 纯瞄准跟随测试程序，代码级强制禁火（mode 最高为 1，fire_rate_hz 恒为 0）\n"
              << "  --no-ff: 关闭速度/加速度前馈，仅输出目标位置角度（便于单独排查外参与位置环）\n"
              << "  --mjpeg: 临时强制开启 MJPEG 网页调试推流\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    std::string cfg_path = "config/default.yaml";
    bool disable_feedforward = false;
    bool force_mjpeg = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--no-ff") {
            disable_feedforward = true;
        } else if (arg == "--mjpeg") {
            force_mjpeg = true;
        } else {
            cfg_path = arg;
        }
    }

    rm_aim::AppConfig cfg;
    try {
        cfg = rm_aim::AppConfig::load(cfg_path);
    } catch (const std::exception& e) {
        std::cerr << "[Vernier-FOLLOW] 加载配置失败: " << e.what() << "\n";
        return 1;
    }

    // 代码级安全锁：无论配置文件如何设置，纯跟随测试程序严禁触发开火
    cfg.aimer.auto_fire = false;
    if (force_mjpeg) {
        cfg.debug.enable_mjpeg = true;
    }

    // 加载超轻量 MLP 权重；若尚未训练真实权重则回退占位权重
    rm_aim::backend::MlpWeights weights;
    if (std::filesystem::exists(cfg.detector.mlp_weights_path)) {
        weights = rm_aim::backend::MlpWeights::load(cfg.detector.mlp_weights_path);
    } else {
        std::cerr << "[Vernier-FOLLOW] 提示：未找到权重文件 '" << cfg.detector.mlp_weights_path
                  << "'，当前使用测试占位权重。\n";
        weights = rm_aim::backend::MlpWeights::make_dummy(3);
    }

    auto backend = rm_aim::backend::create_compute_backend(weights, true);
    auto detector = rm_aim::perception::create_armor_detector(cfg.detector, cfg.enemy_color, *backend);
    rm_aim::perception::PnpSolver pnp_solver(cfg.camera);

    rm_aim::estimation::ImuBuffer imu_buffer;
#ifdef RM_AIM_WITH_EIGEN
    rm_aim::estimation::EkfTracker tracker(cfg.tracker, 0.20);
    rm_aim::estimation::Aimer aimer(cfg.aimer);
#endif

    rm_aim::LatestFramePool frame_pool(3, 4U * 1024U * 1024U);
    auto camera = rm_aim::hal::create_camera_source(cfg.camera, cfg.enemy_color);
    auto serial = rm_aim::hal::create_serial_link(cfg.serial);

    rm_aim::debug::MjpegServer mjpeg(cfg.debug.mjpeg_port);
    if (cfg.debug.enable_mjpeg && mjpeg.start()) {
        std::cout << "[Vernier-FOLLOW] MJPEG 调试推流已启动: http://<ip>:"
                  << cfg.debug.mjpeg_port << "/\n";
    }

    rm_aim::debug::UdpTelemetry telemetry;
    if (cfg.debug.enable_udp_telemetry) {
        telemetry.open(cfg.debug.udp_host, cfg.debug.udp_port);
    }

    std::cout << "[Vernier-FOLLOW | NO-FIRE] 启动完成 | 后端: " << backend->name()
              << " | 特征: " << detector->name()
              << " | 前馈: " << (disable_feedforward ? "OFF" : "ON")
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

        rm_aim::perception::DetectTimingUs timing{};
        auto armors = detector->detect(*slot, &timing);

        rm_aim::ImuSample synced_imu{};
        const bool has_imu = imu_buffer.query(slot->timestamp_ns, synced_imu);
        if (!has_imu) {
            synced_imu.timestamp_ns = slot->timestamp_ns;
            synced_imu.mode = 1U;
            synced_imu.q = {1.0f, 0.0f, 0.0f, 0.0f};
            synced_imu.bullet_speed = 24.0f;
        }

        rm_aim::GimbalCmd cmd{};
#ifdef RM_AIM_WITH_EIGEN
        if (has_imu && synced_imu.mode == 0U) {
            tracker.reset();
            detector->reset_roi();
        } else {
            Eigen::Matrix3d R_w2c;
            Eigen::Vector3d t_w2c;
            rm_aim::estimation::compute_world_to_cam_transform(synced_imu, cfg.extrinsics, R_w2c,
                                                               t_w2c);

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
        }
#else
        (void)pnp_solver;
#endif

        // 发送前二次硬钳位：绝对禁止开火，仅允许不控制(0)或纯瞄准跟随(1)
        if (cmd.mode >= 1U) {
            cmd.mode = 1U;
        }
        cmd.fire_rate_hz = 0U;

        if (disable_feedforward) {
            cmd.yaw_vel = 0.0f;
            cmd.yaw_acc = 0.0f;
            cmd.pitch_vel = 0.0f;
            cmd.pitch_acc = 0.0f;
        }

        serial->send_command(cmd);

#ifdef RM_AIM_WITH_OPENCV
        // 仅当有浏览器连接时才按实际 Bayer 排布解码标注并推送，无人观看时 0 CPU 开销
        if (cfg.debug.enable_mjpeg && mjpeg.has_clients() && slot->width > 0 && slot->height > 0) {
            cv::Mat raw(static_cast<int>(slot->height), static_cast<int>(slot->width),
                        CV_8UC1, slot->host_ptr, slot->stride);
            cv::Mat bgr;
            switch (slot->format) {
                case rm_aim::PixelFormat::BayerRG8: cv::cvtColor(raw, bgr, cv::COLOR_BayerRG2BGR); break;
                case rm_aim::PixelFormat::BayerBG8: cv::cvtColor(raw, bgr, cv::COLOR_BayerBG2BGR); break;
                case rm_aim::PixelFormat::BayerGR8: cv::cvtColor(raw, bgr, cv::COLOR_BayerGR2BGR); break;
                case rm_aim::PixelFormat::BayerGB8: cv::cvtColor(raw, bgr, cv::COLOR_BayerGB2BGR); break;
                default: cv::cvtColor(raw, bgr, cv::COLOR_GRAY2BGR); break;
            }
            for (const auto& a : armors) {
                for (int i = 0; i < 4; ++i) {
                    const auto& p0 = a.corners[static_cast<size_t>(i)];
                    const auto& p1 = a.corners[static_cast<size_t>((i + 1) % 4)];
                    cv::line(bgr, {static_cast<int>(p0.x), static_cast<int>(p0.y)},
                             {static_cast<int>(p1.x), static_cast<int>(p1.y)},
                             cv::Scalar(0, 255, 255), 2);
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
                    "\r\033[K[FOLLOW|NO-FIRE] FPS:%6.1f | Det:%6.1fus(%s) | Armors:%zu | "
                    "Cmd(m=%u,y=%.2f,p=%.2f,yv=%.2f) | Ser(rx=%llu,tx=%llu,%s)",
                    fps, timing.total_us, timing.used_roi ? "ROI" : "FULL", armors.size(),
                    cmd.mode, cmd.yaw, cmd.pitch, cmd.yaw_vel,
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
                    "\"yaw\":%.4f,\"pitch\":%.4f,\"yaw_vel\":%.4f,\"pitch_vel\":%.4f}",
                    fps, timing.total_us, armors.size(), cmd.mode, cmd.yaw, cmd.pitch,
                    cmd.yaw_vel, cmd.pitch_vel);
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
    std::cout << "[Vernier-FOLLOW] 已优雅退出，共处理 " << proc_frames << " 帧。\n";
    return 0;
}
