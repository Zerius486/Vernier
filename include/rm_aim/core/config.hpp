#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "rm_aim/core/types.hpp"

namespace rm_aim {

struct CameraConfig {
    std::string source = "hik";  // 数据源："hik"（海康）、"replay"（录包回放）、"synthetic"（合成流）
    std::string replay_path;
    double exposure_us = 2000.0;
    double gain = 16.0;
    double frame_rate = 200.0;
    std::string vid_pid = "2bdf:0001";
    double time_offset_ms = 0.0;
    std::array<double, 9> camera_matrix{1776.9477, 0.0, 756.3124,
                                        0.0, 1776.0591, 566.1654,
                                        0.0, 0.0, 1.0};
    std::array<double, 5> distort_coeffs{0.0, 0.0, 0.0, 0.0, 0.0};
};

struct ExtrinsicsConfig {
    std::array<double, 9> r_gimbal2imubody{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    std::array<double, 9> r_camera2gimbal{0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0};
    std::array<double, 3> t_camera2gimbal{0.1, 0.0, 0.05};
};

struct SerialConfig {
    std::string device = "/dev/ttyACM0";
    std::string fallback_prefix = "/dev/ttyACM";
    uint32_t baudrate = 921600;
    bool auto_reconnect = true;
    uint32_t reconnect_interval_ms = 200;
};

struct DetectorConfig {
    int color_diff_thresh = 50;
    int brightness_thresh = 140;
    float max_angle_error_deg = 45.0f;
    float min_lightbar_ratio = 1.5f;
    float max_lightbar_ratio = 20.0f;
    float min_lightbar_length = 8.0f;
    float min_armor_ratio = 1.0f;
    float max_armor_ratio = 5.0f;
    float max_side_ratio = 1.5f;
    float max_rectangular_error_deg = 25.0f;
    std::string mlp_weights_path = "models/armor_mlp.bin";
    float min_confidence = 0.80f;
    bool use_dynamic_roi = true;
    float roi_scale = 2.5f;
    int full_scan_interval = 15;
    // TODO: 待点阵装甲板样式公布后，在此补充点阵专属配置项
};

struct TrackerConfig {
    int min_detect_count = 5;
    int max_temp_lost_count = 15;
    int outpost_max_temp_lost_count = 75;
};

struct AimerConfig {
    double yaw_offset_deg = -2.5;
    double pitch_offset_deg = 0.0;
    double coming_angle_deg = 70.0;
    double leaving_angle_deg = 30.0;
    double decision_speed_rad_s = 7.0;
    double high_speed_delay_s = 0.066;
    double low_speed_delay_s = 0.015;
    double first_tolerance_deg = 3.0;
    double second_tolerance_deg = 2.0;
    double judge_distance_m = 2.0;
    bool auto_fire = true;
    uint8_t max_fire_rate_hz = 20;    // 近距离正对最大建议射频（Hz）
    uint8_t min_fire_rate_hz = 6;     // 远距离/高速切板保精度最小射频（Hz）
    double bullet_speed_alpha = 0.3;  // 发弹后实时弹速一阶滤波系数
};

struct DebugConfig {
    bool enable_tui = true;
    bool enable_mjpeg = false;
    uint16_t mjpeg_port = 8080;
    bool enable_udp_telemetry = false;
    std::string udp_host = "127.0.0.1";
    uint16_t udp_port = 9870;
    bool enable_record = false;
    std::string record_path = "records/capture.rmraw";
};

struct AppConfig {
    EnemyColor enemy_color = EnemyColor::Blue;
    CameraConfig camera;
    ExtrinsicsConfig extrinsics;
    SerialConfig serial;
    DetectorConfig detector;
    TrackerConfig tracker;
    AimerConfig aimer;
    DebugConfig debug;

    /// 零外部依赖的轻量 YAML 子集解析器（支持两级分组、标量、字符串与内联数组）。
    static AppConfig load(const std::string& path);
};

}  // namespace rm_aim
