#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <poll.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "rm_aim/backend/compute_backend.hpp"
#include "rm_aim/backend/mlp_weights.hpp"
#include "rm_aim/core/config.hpp"
#include "rm_aim/core/ring_buffer.hpp"
#include "rm_aim/debug/mjpeg_server.hpp"
#include "rm_aim/estimation/imu_buffer.hpp"
#include "rm_aim/hal/camera.hpp"
#include "rm_aim/hal/serial.hpp"
#include "rm_aim/perception/detector.hpp"
#include "rm_aim/perception/pnp_solver.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

std::atomic<bool> g_stop{false};

void handle_signal(int) { g_stop.store(true, std::memory_order_release); }

void print_usage(const char* prog) {
    std::cout << "Vernier SSH 无头标定与校准工具\n"
              << "用法:\n"
              << "  1) 相机内参标定（棋盘格 + MJPEG 浏览器看图）:\n"
              << "     " << prog
              << " intrinsic [config.yaml] [--cols 9] [--rows 6] [--square 0.025] [--port 8080]\n"
              << "     终端按键: [Enter] 抓拍当前有效棋盘格帧 | [c] 开始解算 | [q] 退出\n\n"
              << "  2) 软同步时间差 + 外参旋转联合校准（静止装甲板 + 甩云台）:\n"
              << "     " << prog << " sync [config.yaml] [--duration 6] [--port 8080]\n\n"
              << "  3) 从 .rmraw 录包导出 28x20 轻量 MLP 数字训练样本:\n"
              << "     " << prog << " dump_digits [config.yaml] --input capture.rmraw --out dataset/raw [--stride 3]\n";
}

/// 非阻塞读取一行标准输入命令（用于 SSH 终端交互）
bool poll_stdin_line(std::string& out_line) {
    pollfd pfd{};
    pfd.fd = STDIN_FILENO;
    pfd.events = POLLIN;
    if (::poll(&pfd, 1, 0) > 0 && (pfd.revents & POLLIN)) {
        return static_cast<bool>(std::getline(std::cin, out_line));
    }
    return false;
}

/// 将单通道 Bayer 或 Mono 帧转为灰度图与 BGR 图（供棋盘格检测与 MJPEG 显示）
void decode_frame_to_bgr_gray(const rm_aim::FrameSlot& slot, cv::Mat& bgr, cv::Mat& gray) {
    cv::Mat raw(static_cast<int>(slot.height), static_cast<int>(slot.width), CV_8UC1,
                slot.host_ptr, slot.stride);
    switch (slot.format) {
        case rm_aim::PixelFormat::BayerRG8:
            cv::cvtColor(raw, bgr, cv::COLOR_BayerRG2BGR);
            cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
            break;
        case rm_aim::PixelFormat::BayerBG8:
            cv::cvtColor(raw, bgr, cv::COLOR_BayerBG2BGR);
            cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
            break;
        case rm_aim::PixelFormat::BayerGR8:
            cv::cvtColor(raw, bgr, cv::COLOR_BayerGR2BGR);
            cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
            break;
        case rm_aim::PixelFormat::BayerGB8:
            cv::cvtColor(raw, bgr, cv::COLOR_BayerGB2BGR);
            cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
            break;
        default:
            gray = raw.clone();
            cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
            break;
    }
}

// ============================================================================
// 子命令 1：相机内参标定（intrinsic）
// ============================================================================
int run_intrinsic_calib(const rm_aim::AppConfig& cfg, int board_cols, int board_rows,
                        float square_m, uint16_t mjpeg_port) {
    const cv::Size board_size(board_cols, board_rows);
    std::vector<cv::Point3f> obj_template;
    obj_template.reserve(static_cast<size_t>(board_cols * board_rows));
    for (int r = 0; r < board_rows; ++r) {
        for (int c = 0; c < board_cols; ++c) {
            obj_template.emplace_back(static_cast<float>(c) * square_m,
                                      static_cast<float>(r) * square_m, 0.0f);
        }
    }

    rm_aim::LatestFramePool pool(3, 4U * 1024U * 1024U);
    auto camera = rm_aim::hal::create_camera_source(cfg.camera, cfg.enemy_color);
    rm_aim::debug::MjpegServer mjpeg(mjpeg_port);
    if (mjpeg.start()) {
        std::cout << "[Calib-Intrinsic] MJPEG 预览已启动: http://<设备IP>:" << mjpeg_port << "/\n";
    }
    std::cout << "[Calib-Intrinsic] 棋盘格内角点: " << board_cols << "x" << board_rows
              << "，方格边长: " << square_m << " m\n"
              << "[Calib-Intrinsic] 按 [Enter] 抓拍有效帧，采满 15~25 帧后输入 'c' 解算，输入 'q' 退出。\n";

    camera->start(pool);

    std::vector<std::vector<cv::Point3f>> obj_points;
    std::vector<std::vector<cv::Point2f>> img_points;
    cv::Size img_size;
    std::vector<cv::Point2f> latest_corners;
    bool latest_found = false;

    while (!g_stop.load(std::memory_order_acquire)) {
        const rm_aim::FrameSlot* slot = pool.try_acquire_latest();
        if (slot != nullptr) {
            cv::Mat bgr, gray;
            decode_frame_to_bgr_gray(*slot, bgr, gray);
            img_size = gray.size();
            pool.release_read_slot();

            std::vector<cv::Point2f> corners;
            const bool found = cv::findChessboardCorners(
                gray, board_size, corners,
                cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE |
                    cv::CALIB_CB_FAST_CHECK);
            if (found) {
                cv::cornerSubPix(
                    gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                    cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
                latest_corners = corners;
                latest_found = true;
            } else {
                latest_found = false;
            }

            if (mjpeg.has_clients()) {
                if (found) {
                    cv::drawChessboardCorners(bgr, board_size, corners, found);
                }
                const std::string status = "Captured: " + std::to_string(img_points.size()) +
                                           (found ? " [BOARD OK]" : " [NO BOARD]");
                cv::putText(bgr, status, {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 1.0,
                            found ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
                std::vector<uint8_t> jpeg;
                cv::imencode(".jpg", bgr, jpeg, {cv::IMWRITE_JPEG_QUALITY, 75});
                mjpeg.publish_jpeg(jpeg);
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        std::string cmd;
        if (poll_stdin_line(cmd)) {
            if (cmd == "q" || cmd == "Q") {
                break;
            } else if (cmd == "c" || cmd == "C") {
                if (img_points.size() < 6) {
                    std::cout << "[Calib-Intrinsic] 有效帧数不足（当前 " << img_points.size()
                              << "，至少需要 6 帧，建议 15~25 帧）。\n";
                    continue;
                }
                cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
                cv::Mat D = cv::Mat::zeros(1, 5, CV_64F);
                std::vector<cv::Mat> rvecs, tvecs;
                const double rms = cv::calibrateCamera(obj_points, img_points, img_size, K, D,
                                                       rvecs, tvecs);

                std::cout << "\n========== 内参标定结果 ==========\n"
                          << std::fixed << std::setprecision(12)
                          << "# 重投影误差: " << rms << " px (样本数: " << img_points.size() << ")\n"
                          << "camera_matrix: ["
                          << K.at<double>(0, 0) << ", " << K.at<double>(0, 1) << ", " << K.at<double>(0, 2) << ", "
                          << K.at<double>(1, 0) << ", " << K.at<double>(1, 1) << ", " << K.at<double>(1, 2) << ", "
                          << K.at<double>(2, 0) << ", " << K.at<double>(2, 1) << ", " << K.at<double>(2, 2) << "]\n"
                          << "distort_coeffs: ["
                          << D.at<double>(0, 0) << ", " << D.at<double>(0, 1) << ", "
                          << D.at<double>(0, 2) << ", " << D.at<double>(0, 3) << ", "
                          << D.at<double>(0, 4) << "]\n"
                          << "==================================\n";
                break;
            } else {
                // 直接按回车抓拍当前帧
                if (latest_found) {
                    obj_points.push_back(obj_template);
                    img_points.push_back(latest_corners);
                    std::cout << "[Calib-Intrinsic] 已抓拍第 " << img_points.size() << " 帧。\n";
                } else {
                    std::cout << "[Calib-Intrinsic] 当前画面未检测到完整棋盘格，未抓拍。\n";
                }
            }
        }
    }

    camera->stop();
    mjpeg.stop();
    return 0;
}

// ============================================================================
// 子命令 2：软同步时间偏移 + 外参旋转联合校准（sync）
// ============================================================================
struct RecordedObs {
    uint64_t ts_ns = 0;
    Eigen::Vector3d p_cam = Eigen::Vector3d::Zero();
};

/// 在有序 IMU 轨迹中按指定时间戳做四元数 Slerp 查询
bool query_imu_vector(const std::vector<rm_aim::ImuSample>& samples, uint64_t ts_ns,
                      Eigen::Quaterniond& q_out) {
    if (samples.size() < 2 || ts_ns < samples.front().timestamp_ns ||
        ts_ns > samples.back().timestamp_ns) {
        return false;
    }

    size_t lo = 0;
    size_t hi = samples.size() - 1U;
    while (lo + 1U < hi) {
        const size_t mid = lo + (hi - lo) / 2U;
        if (samples[mid].timestamp_ns <= ts_ns) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    const auto& s0 = samples[lo];
    const auto& s1 = samples[hi];
    const double dt = static_cast<double>(s1.timestamp_ns - s0.timestamp_ns);
    const double alpha = (dt > 0.0) ? static_cast<double>(ts_ns - s0.timestamp_ns) / dt : 0.0;

    Eigen::Quaterniond q0(s0.q.w, s0.q.x, s0.q.y, s0.q.z);
    Eigen::Quaterniond q1(s1.q.w, s1.q.x, s1.q.y, s1.q.z);
    q0.normalize();
    q1.normalize();
    q_out = q0.slerp(alpha, q1);
    return true;
}

/// 计算给定时间偏移与外参旋转矩阵下，所有观测转换到世界系后的 3D 位置标准差（m）
double eval_world_pos_std(const std::vector<RecordedObs>& obs_list,
                          const std::vector<rm_aim::ImuSample>& imu_list,
                          const Eigen::Matrix3d& R_c2g, const Eigen::Vector3d& t_c2g,
                          const Eigen::Matrix3d& R_g2i, double time_offset_ms) {
    std::vector<Eigen::Vector3d> pts;
    pts.reserve(obs_list.size());
    const int64_t offset_ns = static_cast<int64_t>(time_offset_ms * 1e6);

    for (const auto& o : obs_list) {
        const int64_t q_ts = static_cast<int64_t>(o.ts_ns) + offset_ns;
        if (q_ts < 0) continue;
        Eigen::Quaterniond q_i2w;
        if (!query_imu_vector(imu_list, static_cast<uint64_t>(q_ts), q_i2w)) continue;
        const Eigen::Matrix3d R_g2w = q_i2w.toRotationMatrix() * R_g2i;
        pts.push_back(R_g2w * (R_c2g * o.p_cam + t_c2g));
    }

    if (pts.size() < 10) return 1e9;

    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& p : pts) mean += p;
    mean /= static_cast<double>(pts.size());

    double var_sum = 0.0;
    for (const auto& p : pts) var_sum += (p - mean).squaredNorm();
    return std::sqrt(var_sum / static_cast<double>(pts.size()));
}

int run_sync_extrinsic_calib(const rm_aim::AppConfig& cfg, double duration_s,
                             uint16_t mjpeg_port) {
    rm_aim::backend::MlpWeights weights =
        std::filesystem::exists(cfg.detector.mlp_weights_path)
            ? rm_aim::backend::MlpWeights::load(cfg.detector.mlp_weights_path)
            : rm_aim::backend::MlpWeights::make_dummy(3);

    auto backend = rm_aim::backend::create_compute_backend(weights, true);
    auto detector = rm_aim::perception::create_armor_detector(cfg.detector, cfg.enemy_color, *backend);
    rm_aim::perception::PnpSolver pnp_solver(cfg.camera);

    rm_aim::LatestFramePool pool(3, 4U * 1024U * 1024U);
    auto camera = rm_aim::hal::create_camera_source(cfg.camera, cfg.enemy_color);
    auto serial = rm_aim::hal::create_serial_link(cfg.serial);

    rm_aim::debug::MjpegServer mjpeg(mjpeg_port);
    if (mjpeg.start()) {
        std::cout << "[Calib-Sync] MJPEG 预览已启动: http://<设备IP>:" << mjpeg_port << "/\n";
    }

    std::mutex imu_mtx;
    std::vector<rm_aim::ImuSample> imu_history;
    imu_history.reserve(static_cast<size_t>(duration_s * 1200.0));

    serial->start([&](const rm_aim::ImuSample& s) {
        std::lock_guard<std::mutex> lock(imu_mtx);
        imu_history.push_back(s);
    });
    camera->start(pool);

    std::cout << "[Calib-Sync] 请将一块装甲板静止放在车前 2~3m 处，并在接下来 "
              << duration_s << " 秒内左右摆动 ±20°、上下俯仰 ±10° 云台...\n";

    std::vector<RecordedObs> obs_list;
    obs_list.reserve(static_cast<size_t>(duration_s * 250.0));
    const auto t_start = rm_aim::Clock::now();

    while (!g_stop.load(std::memory_order_acquire)) {
        const double elapsed = std::chrono::duration<double>(rm_aim::Clock::now() - t_start).count();
        if (elapsed >= duration_s) break;

        const rm_aim::FrameSlot* slot = pool.try_acquire_latest();
        if (slot == nullptr) {
            std::this_thread::sleep_for(std::chrono::microseconds(300));
            continue;
        }

        auto armors = detector->detect(*slot);
        if (!armors.empty() && pnp_solver.solve(armors[0])) {
            obs_list.push_back(RecordedObs{
                .ts_ns = slot->timestamp_ns,
                .p_cam = {armors[0].position_cam.x, armors[0].position_cam.y,
                          armors[0].position_cam.z},
            });
        }

        if (mjpeg.has_clients()) {
            cv::Mat bgr, gray;
            decode_frame_to_bgr_gray(*slot, bgr, gray);
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

        pool.release_read_slot();
    }

    camera->stop();
    serial->stop();
    mjpeg.stop();

    std::cout << "[Calib-Sync] 采集完成：有效装甲板帧 " << obs_list.size()
              << " 帧，IMU 采样 " << imu_history.size() << " 帧。\n";
    if (obs_list.size() < 30 || imu_history.size() < 50) {
        std::cerr << "[Calib-Sync] 样本数不足（请确认海康相机识别到装甲板且串口持续收到 IMU）。\n";
        return 1;
    }

    Eigen::Matrix3d R_c2g_init, R_g2i;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            R_c2g_init(r, c) = cfg.extrinsics.r_camera2gimbal[static_cast<size_t>(r * 3 + c)];
            R_g2i(r, c) = cfg.extrinsics.r_gimbal2imubody[static_cast<size_t>(r * 3 + c)];
        }
    }
    const Eigen::Vector3d t_c2g(cfg.extrinsics.t_camera2gimbal[0],
                                cfg.extrinsics.t_camera2gimbal[1],
                                cfg.extrinsics.t_camera2gimbal[2]);

    const double init_std = eval_world_pos_std(obs_list, imu_history, R_c2g_init, t_c2g, R_g2i, 0.0);

    // Step A: 粗扫 + 精扫软同步时间偏移 time_offset_ms ∈ [-20, +20] ms
    double best_dt_ms = 0.0;
    double best_std = init_std;
    for (double dt = -20.0; dt <= 20.0; dt += 0.5) {
        const double s = eval_world_pos_std(obs_list, imu_history, R_c2g_init, t_c2g, R_g2i, dt);
        if (s < best_std) {
            best_std = s;
            best_dt_ms = dt;
        }
    }
    for (double dt = best_dt_ms - 1.0; dt <= best_dt_ms + 1.0; dt += 0.1) {
        const double s = eval_world_pos_std(obs_list, imu_history, R_c2g_init, t_c2g, R_g2i, dt);
        if (s < best_std) {
            best_std = s;
            best_dt_ms = dt;
        }
    }

    // Step B: 多尺度坐标下降精修安装小角度误差 (roll, pitch, yaw) ∈ [-5°, +5°]
    constexpr double kDegToRad = std::numbers::pi_v<double> / 180.0;
    double euler_deg[3] = {0.0, 0.0, 0.0};
    auto build_R = [&](const double e[3]) -> Eigen::Matrix3d {
        const Eigen::AngleAxisd rx(e[0] * kDegToRad, Eigen::Vector3d::UnitX());
        const Eigen::AngleAxisd ry(e[1] * kDegToRad, Eigen::Vector3d::UnitY());
        const Eigen::AngleAxisd rz(e[2] * kDegToRad, Eigen::Vector3d::UnitZ());
        return (rz * ry * rx).toRotationMatrix() * R_c2g_init;
    };

    for (double step : {0.5, 0.1, 0.02}) {
        for (int iter = 0; iter < 12; ++iter) {
            bool improved = false;
            for (int axis = 0; axis < 3; ++axis) {
                for (double dir : {-1.0, 1.0}) {
                    double cand[3] = {euler_deg[0], euler_deg[1], euler_deg[2]};
                    cand[axis] = std::clamp(cand[axis] + dir * step, -5.0, 5.0);
                    const double s = eval_world_pos_std(obs_list, imu_history, build_R(cand),
                                                        t_c2g, R_g2i, best_dt_ms);
                    if (s + 1e-6 < best_std) {
                        best_std = s;
                        euler_deg[axis] = cand[axis];
                        improved = true;
                    }
                }
            }
            if (!improved) break;
        }
    }

    const Eigen::Matrix3d R_best = build_R(euler_deg);
    std::cout << "\n========== 时间同步与外参校准结果 ==========\n"
              << std::fixed << std::setprecision(2)
              << "# 世界系静止目标位置抖动(标准差): "
              << (init_std * 1000.0) << " mm -> " << (best_std * 1000.0) << " mm\n"
              << "# 外参旋转微调量 (deg): roll=" << euler_deg[0]
              << ", pitch=" << euler_deg[1] << ", yaw=" << euler_deg[2] << "\n"
              << "time_offset_ms: " << (cfg.camera.time_offset_ms + best_dt_ms) << "\n"
              << std::setprecision(6)
              << "r_camera2gimbal: ["
              << R_best(0, 0) << ", " << R_best(0, 1) << ", " << R_best(0, 2) << ", "
              << R_best(1, 0) << ", " << R_best(1, 1) << ", " << R_best(1, 2) << ", "
              << R_best(2, 0) << ", " << R_best(2, 1) << ", " << R_best(2, 2) << "]\n"
              << "============================================\n";
    return 0;
}

// ============================================================================
// 子命令 3：从 .rmraw 录包导出与运行期完全一致的 28x20 数字训练样本（dump_digits）
// ============================================================================
int run_dump_digits(const rm_aim::AppConfig& cfg, const std::string& input_path,
                    const std::string& out_dir, int frame_stride) {
    if (input_path.empty()) {
        std::cerr << "[Calib-Dump] 请通过 --input 指定 .rmraw 录包路径。\n";
        return 1;
    }

    std::ifstream file(input_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Calib-Dump] 无法打开录包文件: " << input_path << "\n";
        return 1;
    }

    rm_aim::hal::RawFileHeader hdr{};
    file.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!file.good() || std::memcmp(hdr.magic, "RMRAW001", 8) != 0) {
        std::cerr << "[Calib-Dump] 无效的 .rmraw 文件头。\n";
        return 1;
    }

    std::filesystem::create_directories(out_dir);

    // 使用占位权重放行所有几何配对成功的候选装甲板，以便完整收集正/负样本
    rm_aim::backend::MlpWeights dummy = rm_aim::backend::MlpWeights::make_dummy(3);
    auto backend = rm_aim::backend::create_compute_backend(dummy, false);
    rm_aim::DetectorConfig det_cfg = cfg.detector;
    det_cfg.use_dynamic_roi = false;
    det_cfg.min_confidence = 0.0f;
    auto detector = rm_aim::perception::create_armor_detector(det_cfg, cfg.enemy_color, *backend);

    std::vector<uint8_t> raw_buf;
    uint64_t frame_idx = 0;
    uint64_t saved_crops = 0;
    const int stride_step = std::max(1, frame_stride);

    while (!g_stop.load(std::memory_order_acquire)) {
        rm_aim::hal::RawFrameHeader fhdr{};
        file.read(reinterpret_cast<char*>(&fhdr), sizeof(fhdr));
        if (!file.good()) break;

        raw_buf.resize(fhdr.byte_size);
        file.read(reinterpret_cast<char*>(raw_buf.data()),
                  static_cast<std::streamsize>(fhdr.byte_size));
        if (!file.good()) break;

        if ((frame_idx++ % static_cast<uint64_t>(stride_step)) != 0) continue;

        rm_aim::FrameSlot slot{};
        slot.host_ptr = raw_buf.data();
        slot.capacity = raw_buf.size();
        slot.width = fhdr.width;
        slot.height = fhdr.height;
        slot.stride = fhdr.stride;
        slot.byte_size = fhdr.byte_size;
        slot.format = static_cast<rm_aim::PixelFormat>(fhdr.format);
        slot.frame_id = fhdr.frame_id;
        slot.timestamp_ns = fhdr.timestamp_ns;

        const auto armors = detector->detect(slot);
        cv::Mat raw_gray(static_cast<int>(slot.height), static_cast<int>(slot.width), CV_8UC1,
                         slot.host_ptr, slot.stride);

        for (size_t i = 0; i < armors.size(); ++i) {
            const auto& c = armors[i].corners;
#ifdef RM_AIM_FEATURE_DOTMATRIX
            // TODO: 待点阵样式确定后，按实际角点定义调整样本裁剪区域；当前暂直接映射四角点
            const std::vector<cv::Point2f> src_pts = {
                {c[0].x, c[0].y}, {c[1].x, c[1].y}, {c[2].x, c[2].y}, {c[3].x, c[3].y}};
#else
            // 灯条模式下与运行期 lightbar_detector.cpp 完全一致的上下 2x 外扩数字区域裁剪
            const auto expand_pair = [](rm_aim::Point2f top, rm_aim::Point2f bot) {
                const rm_aim::Point2f mid{0.5f * (top.x + bot.x), 0.5f * (top.y + bot.y)};
                return std::pair<cv::Point2f, cv::Point2f>{
                    {mid.x + (top.x - mid.x) * 2.0f, mid.y + (top.y - mid.y) * 2.0f},
                    {mid.x + (bot.x - mid.x) * 2.0f, mid.y + (bot.y - mid.y) * 2.0f}};
            };
            const auto [lt, lb] = expand_pair(c[0], c[3]);
            const auto [rt, rb] = expand_pair(c[1], c[2]);
            const std::vector<cv::Point2f> src_pts = {lt, rt, rb, lb};
#endif
            const std::vector<cv::Point2f> dst_pts = {
                {0.0f, 0.0f},
                {static_cast<float>(rm_aim::backend::kMlpInputW - 1U), 0.0f},
                {static_cast<float>(rm_aim::backend::kMlpInputW - 1U),
                 static_cast<float>(rm_aim::backend::kMlpInputH - 1U)},
                {0.0f, static_cast<float>(rm_aim::backend::kMlpInputH - 1U)},
            };

            const cv::Mat M = cv::getPerspectiveTransform(src_pts, dst_pts);
            cv::Mat patch;
            cv::warpPerspective(raw_gray, patch, M,
                                cv::Size(static_cast<int>(rm_aim::backend::kMlpInputW),
                                         static_cast<int>(rm_aim::backend::kMlpInputH)),
                                cv::INTER_LINEAR);

            const std::string out_file =
                (std::filesystem::path(out_dir) /
                 ("f" + std::to_string(fhdr.frame_id) + "_a" + std::to_string(i) + ".png"))
                    .string();
            if (cv::imwrite(out_file, patch)) {
                ++saved_crops;
            }
        }
    }

    std::cout << "[Calib-Dump] 导出完成：扫描 " << frame_idx << " 帧，输出 "
              << saved_crops << " 张 28x20 数字样本至 " << out_dir << "\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        print_usage(argv[0]);
        return 0;
    }

    const std::string mode = argv[1];
    std::string cfg_path = "config/default.yaml";
    int cols = 9;
    int rows = 6;
    float square = 0.025f;
    double duration = 6.0;
    uint16_t port = 8080;
    std::string input_path;
    std::string out_dir = "dataset/raw";
    int stride = 3;

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--cols" && i + 1 < argc) {
            cols = std::stoi(argv[++i]);
        } else if (a == "--rows" && i + 1 < argc) {
            rows = std::stoi(argv[++i]);
        } else if (a == "--square" && i + 1 < argc) {
            square = std::stof(argv[++i]);
        } else if (a == "--duration" && i + 1 < argc) {
            duration = std::stod(argv[++i]);
        } else if (a == "--port" && i + 1 < argc) {
            port = static_cast<uint16_t>(std::stoul(argv[++i]));
        } else if (a == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (a == "--out" && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (a == "--stride" && i + 1 < argc) {
            stride = std::stoi(argv[++i]);
        } else if (!a.empty() && a[0] != '-') {
            cfg_path = a;
        }
    }

    rm_aim::AppConfig cfg;
    try {
        cfg = rm_aim::AppConfig::load(cfg_path);
    } catch (const std::exception& e) {
        std::cerr << "[Vernier-Calib] 加载配置失败: " << e.what() << "\n";
        return 1;
    }

    if (mode == "intrinsic") {
        return run_intrinsic_calib(cfg, cols, rows, square, port);
    }
    if (mode == "sync") {
        return run_sync_extrinsic_calib(cfg, duration, port);
    }
    if (mode == "dump_digits") {
        return run_dump_digits(cfg, input_path, out_dir, stride);
    }

    std::cerr << "[Vernier-Calib] 未知子命令: " << mode << "\n";
    print_usage(argv[0]);
    return 1;
}
