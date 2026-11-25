#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <vector>

#include "rm_aim/core/config.hpp"
#include "rm_aim/estimation/ballistic.hpp"
#include "rm_aim/estimation/ekf_tracker.hpp"
#include "rm_aim/estimation/imu_buffer.hpp"
#include "rm_aim/estimation/motion_model.hpp"
#include "rm_aim/perception/pnp_solver.hpp"

using namespace rm_aim;
using namespace rm_aim::estimation;
using namespace rm_aim::perception;

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::cerr << "\nCHECK FAILED at " << __FILE__ << ":" << __LINE__ << ": "     \
                      << #cond << "\n";                                                  \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)

namespace {

constexpr double kPi = std::numbers::pi_v<double>;
constexpr double kDegToRad = kPi / 180.0;

void test_imu_buffer_slerp() {
    std::cout << "[TEST] ImuBuffer 四元数 Slerp 与跨 ±pi 插值... ";

    ImuBuffer buf;
    // 绕 Z 轴从 +170° 转到 -170°（最短路径应经过 180°）
    auto make_yaw_quat = [](double yaw) -> Quatf {
        return {static_cast<float>(std::cos(0.5 * yaw)), 0.0f, 0.0f,
                static_cast<float>(std::sin(0.5 * yaw))};
    };

    ImuSample s0{};
    s0.timestamp_ns = 1'000'000;
    s0.yaw = static_cast<float>(170.0 * kDegToRad);
    s0.pitch = 0.10f;
    s0.q = make_yaw_quat(s0.yaw);

    ImuSample s1{};
    s1.timestamp_ns = 3'000'000;
    s1.yaw = static_cast<float>(-170.0 * kDegToRad);
    s1.pitch = 0.20f;
    s1.q = make_yaw_quat(s1.yaw);

    buf.push(s0);
    buf.push(s1);

    ImuSample mid{};
    CHECK(buf.query(2'000'000, mid));
    // 中点 yaw 应为 ±pi，pitch 应为 0.15
    CHECK(std::fabs(std::fabs(mid.yaw) - kPi) < 1e-3);
    CHECK(std::fabs(mid.pitch - 0.15f) < 1e-4f);
    // 对应四元数 |w| 应接近 0，|z| 应接近 1
    CHECK(std::fabs(mid.q.w) < 1e-3f && std::fabs(std::fabs(mid.q.z) - 1.0f) < 1e-3f);

    std::cout << "OK\n";
}

void test_pnp_1d_yaw_optimization(const AppConfig& cfg) {
    std::cout << "[TEST] PnpSolver 1D 黄金分割朝向消歧... ";

#ifdef RM_AIM_WITH_EIGEN
    CameraConfig cam = cfg.camera;
    cam.distort_coeffs = {0.0, 0.0, 0.0, 0.0, 0.0};
    PnpSolver solver(cam);

    // 相机水平正视前方：世界系 FLU (X前,Y左,Z上) -> 相机系 (X右=-Y, Y下=-Z, Z前=X)
    Eigen::Matrix3d R_w2c;
    R_w2c << 0.0, -1.0,  0.0,
             0.0,  0.0, -1.0,
             1.0,  0.0,  0.0;
    const Eigen::Vector3d t_w2c = Eigen::Vector3d::Zero();

    const Eigen::Vector3d pos_w(3.2, 0.15, 0.05);
    const double true_yaw = wrap_to_pi(kPi + 22.0 * kDegToRad);  // 相对正对偏转 22°
    const double pitch_rad = 15.0 * kDegToRad;

    const double hw = kSmallArmorHalfW;
    const double hh = kArmorHalfH;
    const double cp = std::cos(pitch_rad);
    const double sp = std::sin(pitch_rad);
    const std::array<Eigen::Vector3d, 4> local_pts = {
        Eigen::Vector3d{-hh * sp,  hw,  hh * cp},
        Eigen::Vector3d{-hh * sp, -hw,  hh * cp},
        Eigen::Vector3d{ hh * sp, -hw, -hh * cp},
        Eigen::Vector3d{ hh * sp,  hw, -hh * cp},
    };

    Eigen::Matrix3d R_a2w;
    R_a2w << std::cos(true_yaw), -std::sin(true_yaw), 0.0,
             std::sin(true_yaw),  std::cos(true_yaw), 0.0,
                            0.0,                 0.0, 1.0;

    Armor armor{};
    armor.is_large = false;
    for (size_t i = 0; i < 4; ++i) {
        const Eigen::Vector3d pc = R_w2c * (pos_w + R_a2w * local_pts[i]);
        armor.corners[i] = {
            static_cast<float>(cam.camera_matrix[0] * (pc.x() / pc.z()) + cam.camera_matrix[2]),
            static_cast<float>(cam.camera_matrix[4] * (pc.y() / pc.z()) + cam.camera_matrix[5]),
        };
    }

    const double est_yaw = solver.optimize_yaw_1d(armor, pos_w, R_w2c, t_w2c, pitch_rad);
    const double err_deg = std::fabs(shortest_angular_distance(true_yaw, est_yaw)) * 180.0 / kPi;
    CHECK(err_deg < 0.5);

    std::cout << std::fixed << std::setprecision(3) << "OK (err=" << err_deg << " deg)\n";
#else
    (void)cfg;
    std::cout << "SKIPPED\n";
#endif
}

void test_ekf_pulse_spin_and_extensible_predictor(const AppConfig& cfg) {
    std::cout << "[TEST] 12 维 Singer EKF + 新息自适应 Q 追踪脉冲变速小陀螺... ";

#ifdef RM_AIM_WITH_EIGEN
    EkfTracker tracker(cfg.tracker, 0.20);

    // 仿真 200 FPS 脉冲变速小陀螺：
    //   0.0s ~ 0.6s: 匀速 3.5 rad/s
    //   0.6s ~ 1.2s: 脉冲跃升至 9.0 rad/s（带物理平滑上升）
    const double dt = 0.005;  // 200 FPS
    double xc = 3.0, yc = 0.2, zc = 0.1;
    const double vx = 0.4, vy = -0.3;
    double theta = 0.0;
    double omega = 3.5;
    const double r1 = 0.26, r2 = 0.24, dz = 0.02;

    double max_q_scale_seen = 1.0;

    for (int step = 0; step < 260; ++step) {
        const double t = step * dt;
        const double target_omega = (t < 0.60) ? 3.5 : 9.0;
        // 模拟底盘转动惯量一阶响应（时间常数 0.05s）
        omega += (target_omega - omega) * (dt / 0.05);
        theta = wrap_to_pi(theta + omega * dt);
        xc += vx * dt;
        yc += vy * dt;

        // 选出最朝向原点的可见装甲板生成观测
        std::vector<WorldArmorObs> obs_list;
        const double view_dir = wrap_to_pi(std::atan2(yc, xc) + kPi);
        for (int i = 0; i < 4; ++i) {
            const double yaw_i = wrap_to_pi(theta + i * (0.5 * kPi));
            if (std::fabs(shortest_angular_distance(view_dir, yaw_i)) < 65.0 * kDegToRad) {
                const bool odd = (i & 1) != 0;
                const double r = odd ? r2 : r1;
                WorldArmorObs obs{};
                obs.pos_world = {xc - r * std::cos(yaw_i), yc - r * std::sin(yaw_i),
                                 odd ? (zc + dz) : zc};
                obs.yaw_world = yaw_i;
                obs.class_id = 3;
                obs_list.push_back(obs);
            }
        }

        const uint64_t ts_ns = static_cast<uint64_t>((step + 1) * 5'000'000ULL);
        tracker.update(ts_ns, obs_list);
        max_q_scale_seen = std::max(max_q_scale_seen, tracker.last_q_scale());
    }

    CHECK(tracker.is_tracking());
    const VehicleState est = tracker.vehicle_state();
    // 校验 1.3s 脉冲稳定后中心位置、平移速度与新转速（9.0 rad/s）已精确咬住
    CHECK(std::hypot(est.xc - xc, est.yc - yc) < 0.06);
    CHECK(std::fabs(est.vx - vx) < 0.25);
    CHECK(std::fabs(est.v_yaw - omega) < 0.8);

    // 验证外挂 IMotionPredictor 接口可正常插拔
    struct CustomStubPredictor final : public IMotionPredictor {
        mutable int calls = 0;
        VehicleState predict(const VehicleState& cur, double dt_s) const override {
            ++calls;
            return SingerMotionPredictor{}.predict(cur, dt_s);
        }
    };
    auto custom = std::make_shared<CustomStubPredictor>();
    tracker.set_motion_predictor(custom);
    (void)tracker.predict_state(0.12);
    CHECK(custom->calls == 1);

    std::cout << std::fixed << std::setprecision(2)
              << "OK (v_yaw_est=" << est.v_yaw << " vs true=" << omega
              << " rad/s, max_Q_scale=" << max_q_scale_seen << ")\n";
#else
    (void)cfg;
    std::cout << "SKIPPED\n";
#endif
}

void test_ballistic_and_aimer(const AppConfig& cfg) {
    std::cout << "[TEST] 空气阻力弹道解算 + Aimer 前馈与切板保护... ";

    BallisticSolver solver(0.019, 9.7933);
    // 水平距离 5.0m，目标与枪口等高 (z=0)，弹速 24m/s，需抬头补偿重力下坠（pitch > 0）
    const BallisticSolution sol = solver.solve(5.0, 0.0, 0.0, 24.0);
    CHECK(sol.valid);
    CHECK(sol.pitch_rad > 0.02 && sol.pitch_rad < 0.12);  // 约 2.5°~3.5° 抬头补偿
    CHECK(sol.fly_time_s > 0.20 && sol.fly_time_s < 0.25);

    // 回代验证物理落点精度 (< 1 mm)
    const double t = sol.fly_time_s;
    const double z_hit = 24.0 * std::sin(sol.pitch_rad) * t - 0.5 * 9.7933 * t * t;
    CHECK(std::fabs(z_hit - 0.0) < 0.001);

#ifdef RM_AIM_WITH_EIGEN
    EkfTracker tracker(cfg.tracker, 0.20);
    // 喂入若干帧横向平移目标，验证 Aimer 产出合理 yaw_vel 前馈
    for (int i = 0; i < 15; ++i) {
        WorldArmorObs obs{};
        obs.pos_world = {3.0, -0.5 + i * 0.01, 0.2};
        obs.yaw_world = kPi;
        obs.class_id = 3;
        tracker.update(static_cast<uint64_t>((i + 1) * 5'000'000ULL), {obs});
    }

    Aimer aimer(cfg.aimer);
    ImuSample imu{};
    imu.yaw = 0.0f;
    imu.pitch = 0.0f;
    imu.bullet_speed = 24.0f;

    const GimbalCmd cmd = aimer.compute_command(tracker, imu);
    CHECK(cmd.mode == 1 || cmd.mode == 2);
    CHECK(std::isfinite(cmd.yaw) && std::isfinite(cmd.pitch));
    CHECK(cmd.yaw_vel > 0.1f);  // 目标向 +Y（左）移动，期望左转角速度前馈为正
    CHECK(cmd.pitch > 0.0f);    // 目标高于枪口且有重力下坠，抬头为正
#endif

    std::cout << std::fixed << std::setprecision(2)
              << "OK (5m pitch=+" << (sol.pitch_rad * 180.0 / kPi)
              << " deg, t_fly=" << (sol.fly_time_s * 1000.0) << " ms)\n";
}

}  // namespace

int main(int argc, char** argv) {
    const std::string cfg_path = (argc > 1) ? argv[1] : "config/default.yaml";
    const AppConfig cfg = AppConfig::load(cfg_path);

    test_imu_buffer_slerp();
    test_pnp_1d_yaw_optimization(cfg);
    test_ekf_pulse_spin_and_extensible_predictor(cfg);
    test_ballistic_and_aimer(cfg);

    std::cout << "All Phase 3 tests passed!\n";
    return 0;
}
