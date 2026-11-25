#include "rm_aim/core/config.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace rm_aim {
namespace {

std::string trim(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(start, end - start);
}

std::string strip_comment(const std::string& line) {
    bool in_single = false;
    bool in_double = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '\'' && !in_double) {
            in_single = !in_single;
        } else if (c == '"' && !in_single) {
            in_double = !in_double;
        } else if (c == '#' && !in_single && !in_double) {
            return line.substr(0, i);
        }
    }
    return line;
}

std::string unquote(const std::string& s) {
    if (s.size() >= 2 &&
        ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\''))) {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

bool parse_bool(const std::string& v) {
    std::string lower = v;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lower == "true" || lower == "1" || lower == "yes" || lower == "on";
}

template <size_t N>
void parse_double_array(const std::string& raw, std::array<double, N>& out) {
    std::string s = trim(raw);
    if (s.size() >= 2 && s.front() == '[' && s.back() == ']') {
        s = s.substr(1, s.size() - 2);
    }
    std::stringstream ss(s);
    std::string item;
    size_t idx = 0;
    while (std::getline(ss, item, ',') && idx < N) {
        std::string t = trim(item);
        if (!t.empty()) {
            out[idx++] = std::stod(t);
        }
    }
    if (idx != N) {
        throw std::runtime_error("Expected array of length " + std::to_string(N) + ", got: " + raw);
    }
}

}  // namespace

AppConfig AppConfig::load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open config file: " + path);
    }

    std::unordered_map<std::string, std::string> kv;
    std::string current_section;
    std::string line;

    while (std::getline(file, line)) {
        std::string clean = strip_comment(line);
        if (trim(clean).empty()) {
            continue;
        }

        size_t indent = 0;
        while (indent < clean.size() && clean[indent] == ' ') {
            ++indent;
        }

        const size_t colon = clean.find(':');
        if (colon == std::string::npos) {
            continue;
        }

        const std::string key = trim(clean.substr(0, colon));
        const std::string val = trim(clean.substr(colon + 1));

        if (indent == 0 && val.empty()) {
            current_section = key;
        } else if (indent == 0) {
            kv[key] = val;
        } else {
            kv[current_section + "." + key] = val;
        }
    }

    auto get_str = [&](const std::string& k, std::string& dst) {
        if (auto it = kv.find(k); it != kv.end()) dst = unquote(it->second);
    };
    auto get_int = [&](const std::string& k, int& dst) {
        if (auto it = kv.find(k); it != kv.end()) dst = std::stoi(it->second);
    };
    auto get_u32 = [&](const std::string& k, uint32_t& dst) {
        if (auto it = kv.find(k); it != kv.end()) dst = static_cast<uint32_t>(std::stoul(it->second));
    };
    auto get_u16 = [&](const std::string& k, uint16_t& dst) {
        if (auto it = kv.find(k); it != kv.end()) dst = static_cast<uint16_t>(std::stoul(it->second));
    };
    auto get_flt = [&](const std::string& k, float& dst) {
        if (auto it = kv.find(k); it != kv.end()) dst = std::stof(it->second);
    };
    auto get_dbl = [&](const std::string& k, double& dst) {
        if (auto it = kv.find(k); it != kv.end()) dst = std::stod(it->second);
    };
    auto get_bool = [&](const std::string& k, bool& dst) {
        if (auto it = kv.find(k); it != kv.end()) dst = parse_bool(it->second);
    };

    AppConfig cfg;

    if (auto it = kv.find("target.enemy_color"); it != kv.end()) {
        std::string color = unquote(it->second);
        std::transform(color.begin(), color.end(), color.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        cfg.enemy_color = (color == "red") ? EnemyColor::Red : EnemyColor::Blue;
    }

    // 相机参数
    get_str("camera.source", cfg.camera.source);
    get_str("camera.replay_path", cfg.camera.replay_path);
    get_dbl("camera.exposure_us", cfg.camera.exposure_us);
    get_dbl("camera.gain", cfg.camera.gain);
    get_dbl("camera.frame_rate", cfg.camera.frame_rate);
    get_str("camera.vid_pid", cfg.camera.vid_pid);
    get_dbl("camera.time_offset_ms", cfg.camera.time_offset_ms);
    if (auto it = kv.find("camera.camera_matrix"); it != kv.end()) {
        parse_double_array(it->second, cfg.camera.camera_matrix);
    }
    if (auto it = kv.find("camera.distort_coeffs"); it != kv.end()) {
        parse_double_array(it->second, cfg.camera.distort_coeffs);
    }

    // 手眼外参
    if (auto it = kv.find("extrinsics.r_gimbal2imubody"); it != kv.end()) {
        parse_double_array(it->second, cfg.extrinsics.r_gimbal2imubody);
    }
    if (auto it = kv.find("extrinsics.r_camera2gimbal"); it != kv.end()) {
        parse_double_array(it->second, cfg.extrinsics.r_camera2gimbal);
    }
    if (auto it = kv.find("extrinsics.t_camera2gimbal"); it != kv.end()) {
        parse_double_array(it->second, cfg.extrinsics.t_camera2gimbal);
    }

    // 串口通信参数
    get_str("serial.device", cfg.serial.device);
    get_str("serial.fallback_prefix", cfg.serial.fallback_prefix);
    get_u32("serial.baudrate", cfg.serial.baudrate);
    get_bool("serial.auto_reconnect", cfg.serial.auto_reconnect);
    get_u32("serial.reconnect_interval_ms", cfg.serial.reconnect_interval_ms);

    // 装甲板检测参数
    get_int("detector.color_diff_thresh", cfg.detector.color_diff_thresh);
    get_int("detector.brightness_thresh", cfg.detector.brightness_thresh);
    get_flt("detector.max_angle_error_deg", cfg.detector.max_angle_error_deg);
    get_flt("detector.min_lightbar_ratio", cfg.detector.min_lightbar_ratio);
    get_flt("detector.max_lightbar_ratio", cfg.detector.max_lightbar_ratio);
    get_flt("detector.min_lightbar_length", cfg.detector.min_lightbar_length);
    get_flt("detector.min_armor_ratio", cfg.detector.min_armor_ratio);
    get_flt("detector.max_armor_ratio", cfg.detector.max_armor_ratio);
    get_flt("detector.max_side_ratio", cfg.detector.max_side_ratio);
    get_flt("detector.max_rectangular_error_deg", cfg.detector.max_rectangular_error_deg);
    get_str("detector.mlp_weights_path", cfg.detector.mlp_weights_path);
    get_flt("detector.min_confidence", cfg.detector.min_confidence);
    get_bool("detector.use_dynamic_roi", cfg.detector.use_dynamic_roi);
    get_flt("detector.roi_scale", cfg.detector.roi_scale);
    get_int("detector.full_scan_interval", cfg.detector.full_scan_interval);

    // 跟踪器参数
    get_int("tracker.min_detect_count", cfg.tracker.min_detect_count);
    get_int("tracker.max_temp_lost_count", cfg.tracker.max_temp_lost_count);
    get_int("tracker.outpost_max_temp_lost_count", cfg.tracker.outpost_max_temp_lost_count);

    // 弹道与火控参数
    get_dbl("aimer.yaw_offset_deg", cfg.aimer.yaw_offset_deg);
    get_dbl("aimer.pitch_offset_deg", cfg.aimer.pitch_offset_deg);
    get_dbl("aimer.coming_angle_deg", cfg.aimer.coming_angle_deg);
    get_dbl("aimer.leaving_angle_deg", cfg.aimer.leaving_angle_deg);
    get_dbl("aimer.decision_speed_rad_s", cfg.aimer.decision_speed_rad_s);
    get_dbl("aimer.high_speed_delay_s", cfg.aimer.high_speed_delay_s);
    get_dbl("aimer.low_speed_delay_s", cfg.aimer.low_speed_delay_s);
    get_dbl("aimer.first_tolerance_deg", cfg.aimer.first_tolerance_deg);
    get_dbl("aimer.second_tolerance_deg", cfg.aimer.second_tolerance_deg);
    get_dbl("aimer.judge_distance_m", cfg.aimer.judge_distance_m);
    get_bool("aimer.auto_fire", cfg.aimer.auto_fire);
    if (auto it = kv.find("aimer.max_fire_rate_hz"); it != kv.end()) {
        cfg.aimer.max_fire_rate_hz = static_cast<uint8_t>(std::stoul(it->second));
    }
    if (auto it = kv.find("aimer.min_fire_rate_hz"); it != kv.end()) {
        cfg.aimer.min_fire_rate_hz = static_cast<uint8_t>(std::stoul(it->second));
    }
    get_dbl("aimer.bullet_speed_alpha", cfg.aimer.bullet_speed_alpha);

    // 调试与可视化参数
    get_bool("debug.enable_tui", cfg.debug.enable_tui);
    get_bool("debug.enable_mjpeg", cfg.debug.enable_mjpeg);
    get_u16("debug.mjpeg_port", cfg.debug.mjpeg_port);
    get_bool("debug.enable_udp_telemetry", cfg.debug.enable_udp_telemetry);
    get_str("debug.udp_host", cfg.debug.udp_host);
    get_u16("debug.udp_port", cfg.debug.udp_port);
    get_bool("debug.enable_record", cfg.debug.enable_record);
    get_str("debug.record_path", cfg.debug.record_path);

    return cfg;
}

}  // namespace rm_aim
