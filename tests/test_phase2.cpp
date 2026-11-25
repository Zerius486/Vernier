#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

#include "rm_aim/backend/compute_backend.hpp"
#include "rm_aim/backend/mlp_weights.hpp"
#include "rm_aim/core/config.hpp"
#include "rm_aim/core/ring_buffer.hpp"
#include "rm_aim/perception/detector.hpp"
#include "rm_aim/perception/pnp_solver.hpp"

using namespace rm_aim;
using namespace rm_aim::backend;
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

#if !defined(RM_AIM_FEATURE_DOTMATRIX) || defined(RM_AIM_WITH_CUDA)
/// 绘制干净的 1280x1024 合成 BayerRG8 装甲板图像（两条竖直红/蓝灯条）。
void fill_synthetic_bayer_armor(FrameSlot& slot, uint32_t width, uint32_t height,
                                float cx, float cy, float half_w, float half_h,
                                EnemyColor color) {
    slot.width = width;
    slot.height = height;
    slot.stride = width;
    slot.byte_size = width * height;
    slot.format = PixelFormat::BayerRG8;
    std::memset(slot.host_ptr, 8, slot.byte_size);

    auto draw_bar = [&](float bx) {
        const uint32_t x0 = static_cast<uint32_t>(std::max(0.0f, bx - 4.0f));
        const uint32_t x1 = static_cast<uint32_t>(std::min(static_cast<float>(width - 1U), bx + 4.0f));
        const uint32_t y0 = static_cast<uint32_t>(std::max(0.0f, cy - half_h));
        const uint32_t y1 = static_cast<uint32_t>(std::min(static_cast<float>(height - 1U), cy + half_h));

        for (uint32_t y = y0; y <= y1; ++y) {
            uint8_t* row = slot.host_ptr + static_cast<size_t>(y) * width;
            for (uint32_t x = x0; x <= x1; ++x) {
                const bool is_r = ((y & 1U) == 0U) && ((x & 1U) == 0U);
                const bool is_b = ((y & 1U) == 1U) && ((x & 1U) == 1U);
                if (color == EnemyColor::Red) {
                    row[x] = is_r ? 250 : (is_b ? 15 : 130);
                } else {
                    row[x] = is_b ? 250 : (is_r ? 15 : 130);
                }
            }
        }
    };

    draw_bar(cx - half_w);
    draw_bar(cx + half_w);
}
#endif

void test_mlp_weights_io() {
    std::cout << "[TEST] MlpWeights binary save/load (VRMLP001)... ";

    const auto tmp = std::filesystem::temp_directory_path() / "vernier_test_mlp.bin";
    const MlpWeights orig = MlpWeights::make_dummy(4);
    CHECK(orig.valid());
    CHECK(orig.save(tmp.string()));

    const MlpWeights loaded = MlpWeights::load(tmp.string());
    CHECK(loaded.valid());
    CHECK(std::fabs(loaded.b3[4] - 4.0f) < 1e-6f);
    CHECK(std::fabs(loaded.b3[0] - (-2.0f)) < 1e-6f);

    std::filesystem::remove(tmp);
    std::cout << "OK\n";
}

void test_detector_and_benchmark(const AppConfig& cfg) {
    MlpWeights weights = MlpWeights::make_dummy(3);  // 测试权重：固定预测 3 号步兵
    auto backend = create_compute_backend(weights, true);
    auto detector = create_armor_detector(cfg.detector, EnemyColor::Blue, *backend);

#ifdef RM_AIM_FEATURE_DOTMATRIX
    std::cout << "[TEST] " << detector->name() << " interface contract... ";

    LatestFramePool pool(3, 1280 * 1024);
    FrameSlot& slot = pool.acquire_write_slot();
    slot.width = 1280;
    slot.height = 1024;
    slot.stride = 1280;
    slot.byte_size = 1280 * 1024;
    slot.format = PixelFormat::BayerRG8;
    std::memset(slot.host_ptr, 0, slot.byte_size);

    detector->set_enemy_color(EnemyColor::Red);
    detector->set_enemy_color(EnemyColor::Blue);
    detector->reset_roi();

    DetectTimingUs t_full{};
    const auto armors_full = detector->detect(slot, &t_full);
    CHECK(detector->name() != nullptr);
    CHECK(!t_full.used_roi);
    CHECK(armors_full.empty());
    CHECK(detector->active_roi().width == 0 && detector->active_roi().height == 0);

    std::cout << "OK (placeholder)\n";
#else
    std::cout << "[TEST] " << detector->name() << " + Dynamic ROI... ";

    LatestFramePool pool(3, 1280 * 1024);
    FrameSlot& slot = pool.acquire_write_slot();
    constexpr float kCx = 640.0f;
    constexpr float kCy = 512.0f;
    constexpr float kHalfW = 58.0f;
    constexpr float kHalfH = 26.0f;
    fill_synthetic_bayer_armor(slot, 1280, 1024, kCx, kCy, kHalfW, kHalfH, EnemyColor::Blue);

    // 第 1 帧：全图扫描
    DetectTimingUs t_full{};
    const auto armors_full = detector->detect(slot, &t_full);
    CHECK(!t_full.used_roi);
    CHECK(armors_full.size() == 1);
    CHECK(armors_full[0].class_id == 3);
    CHECK(armors_full[0].confidence > 0.95f);

    // 校验角点中心精度（< 2.5 像素）
    const float det_cx = 0.25f * (armors_full[0].corners[0].x + armors_full[0].corners[1].x +
                                  armors_full[0].corners[2].x + armors_full[0].corners[3].x);
    const float det_cy = 0.25f * (armors_full[0].corners[0].y + armors_full[0].corners[1].y +
                                  armors_full[0].corners[2].y + armors_full[0].corners[3].y);
    CHECK(std::fabs(det_cx - kCx) < 2.5f);
    CHECK(std::fabs(det_cy - kCy) < 2.5f);

    // 第 2 帧：自动切换为动态 ROI 追踪
    DetectTimingUs t_roi{};
    const auto armors_roi = detector->detect(slot, &t_roi);
    CHECK(t_roi.used_roi);
    CHECK(armors_roi.size() == 1);

    // 基准压测：100 帧全图扫描 vs 100 帧动态 ROI 追踪
    constexpr int kIters = 100;
    double sum_full_us = 0.0;
    for (int i = 0; i < kIters; ++i) {
        detector->reset_roi();
        DetectTimingUs t{};
        auto a = detector->detect(slot, &t);
        CHECK(a.size() == 1);
        sum_full_us += t.total_us;
    }

    double sum_roi_us = 0.0;
    for (int i = 0; i < kIters; ++i) {
        DetectTimingUs t{};
        auto a = detector->detect(slot, &t);
        CHECK(a.size() == 1);
        sum_roi_us += t.total_us;
    }

    const double avg_full = sum_full_us / kIters;
    const double avg_roi = sum_roi_us / kIters;
    std::cout << "OK\n";
    std::cout << std::fixed << std::setprecision(1)
              << "       Backend: " << backend->name() << "\n"
              << "       1280x1024 Full-Frame: " << avg_full << " us ("
              << (1e6 / avg_full) << " FPS)\n"
              << "       1280x1024 Dynamic ROI: " << avg_roi << " us ("
              << (1e6 / avg_roi) << " FPS)\n";
#endif
}

void test_pnp_solver(const AppConfig& cfg) {
    std::cout << "[TEST] IPPE PnP Solver (synthetic 3D projection -> 3D recovery)... ";

#ifdef RM_AIM_WITH_OPENCV
    // 使用无畸变针孔模型 + 实车内参，在 Z = 3.0 m 处生成理论投影并反解验证
    CameraConfig cam = cfg.camera;
    cam.distort_coeffs = {0.0, 0.0, 0.0, 0.0, 0.0};
    PnpSolver solver(cam);

    const double fx = cam.camera_matrix[0];
    const double cx = cam.camera_matrix[2];
    const double fy = cam.camera_matrix[4];
    const double cy = cam.camera_matrix[5];

    const double tx = 0.25;
    const double ty = -0.10;
    const double tz = 3.00;
    const double hw = kSmallArmorHalfW;
    const double hh = kArmorHalfH;

    auto project = [&](double x, double y, double z) -> Point2f {
        return {static_cast<float>(fx * (x / z) + cx),
                static_cast<float>(fy * (y / z) + cy)};
    };

    Armor armor{};
    armor.is_large = false;
    armor.corners = {
        project(tx - hw, ty - hh, tz),
        project(tx + hw, ty - hh, tz),
        project(tx + hw, ty + hh, tz),
        project(tx - hw, ty + hh, tz),
    };

    CHECK(solver.solve(armor));
    CHECK(std::fabs(armor.position_cam.x - tx) < 0.005);
    CHECK(std::fabs(armor.position_cam.y - ty) < 0.005);
    CHECK(std::fabs(armor.position_cam.z - tz) < 0.005);
    std::cout << "OK (pos=[" << armor.position_cam.x << ", "
              << armor.position_cam.y << ", " << armor.position_cam.z << "] m)\n";
#else
    (void)cfg;
    std::cout << "SKIPPED (OpenCV not available)\n";
#endif
}

void test_cpu_vs_cuda_equivalence(const AppConfig& cfg) {
#ifdef RM_AIM_WITH_CUDA
    std::cout << "[TEST] CPU vs CUDA exact equivalence (mask + fused MLP)... ";

    MlpWeights w = MlpWeights::make_dummy(5);
    // 填充非零确定性权重，完整检验三层矩阵乘与 Softmax 的数值一致性
    for (size_t i = 0; i < w.w1.size(); ++i) w.w1[i] = 0.01f * std::sin(static_cast<float>(i));
    for (size_t i = 0; i < w.w2.size(); ++i) w.w2[i] = 0.02f * std::cos(static_cast<float>(i));
    for (size_t i = 0; i < w.w3.size(); ++i) w.w3[i] = 0.03f * std::sin(static_cast<float>(i));

    auto cpu_be = create_compute_backend(w, false);
    auto cuda_be = create_compute_backend(w, true);

    LatestFramePool pool(3, 1280 * 1024);
    FrameSlot& slot = pool.acquire_write_slot();
    fill_synthetic_bayer_armor(slot, 1280, 1024, 640.0f, 512.0f, 58.0f, 26.0f, EnemyColor::Blue);

    PreprocessParams pp{};
    pp.enemy_color = EnemyColor::Blue;
    pp.color_diff_thresh = cfg.detector.color_diff_thresh;
    pp.brightness_thresh = cfg.detector.brightness_thresh;

    HalfMaskBuffer m_cpu, m_cuda;
    cpu_be->preprocess_bayer(slot, pp, m_cpu);
    cuda_be->preprocess_bayer(slot, pp, m_cuda);
    CHECK(m_cpu.mask == m_cuda.mask);
    CHECK(m_cpu.row_nonzero == m_cuda.row_nonzero);

    WarpPatchRequest req{};
    req.h_inv = {5.0f, 0.0f, 580.0f, 0.0f, 2.0f, 480.0f, 0.0f, 0.0f, 1.0f};
    ClassifyResult r_cpu{}, r_cuda{};
    cpu_be->batch_warp_and_classify(slot, &req, 1, &r_cpu);
    cuda_be->batch_warp_and_classify(slot, &req, 1, &r_cuda);

    CHECK(r_cpu.class_id == r_cuda.class_id);
    for (uint32_t c = 0; c < kMlpNumClasses; ++c) {
        CHECK(std::fabs(r_cpu.probs[c] - r_cuda.probs[c]) < 1e-4f);
    }

    // 校验算子 3（batch_fit_lightbars）CPU vs CUDA 亚像素端点等价性 (< 0.05 px)
    LightbarBboxRequest lb_req{574, 480, 590, 544};
    LightbarFitResult lb_cpu{}, lb_cuda{};
    cpu_be->batch_fit_lightbars(slot, &lb_req, 1,
                                static_cast<float>(cfg.detector.brightness_thresh), &lb_cpu);
    cuda_be->batch_fit_lightbars(slot, &lb_req, 1,
                                 static_cast<float>(cfg.detector.brightness_thresh), &lb_cuda);
    CHECK(lb_cpu.valid == 1 && lb_cuda.valid == 1);
    CHECK(std::fabs(lb_cpu.top.x - lb_cuda.top.x) < 0.05f);
    CHECK(std::fabs(lb_cpu.top.y - lb_cuda.top.y) < 0.05f);
    CHECK(std::fabs(lb_cpu.bottom.x - lb_cuda.bottom.x) < 0.05f);
    CHECK(std::fabs(lb_cpu.bottom.y - lb_cuda.bottom.y) < 0.05f);

    std::cout << "OK (zero-copy=" << (pool.is_cuda_zero_copy() ? "ON" : "OFF") << ")\n";
#else
    (void)cfg;
#endif
}

}  // namespace

int main(int argc, char** argv) {
    const std::string cfg_path = (argc > 1) ? argv[1] : "config/default.yaml";
    const AppConfig cfg = AppConfig::load(cfg_path);

    test_mlp_weights_io();
    test_detector_and_benchmark(cfg);
    test_pnp_solver(cfg);
    test_cpu_vs_cuda_equivalence(cfg);

    std::cout << "All Phase 2 tests passed!\n";
    return 0;
}
