#include "rm_aim/perception/detector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

#ifdef RM_AIM_WITH_EIGEN
#include <Eigen/Dense>
#endif

namespace rm_aim::perception {
namespace {

constexpr float kRadToDeg = 180.0f / std::numbers::pi_v<float>;

/// 计算 3x3 逆透视单应矩阵，将归一化数字图四角
/// (0,0), (W-1,0), (W-1,H-1), (0,H-1) 映射回原图四边形（左上、右上、右下、左下）。
bool compute_patch_homography(const std::array<Point2f, 4>& dst, std::array<float, 9>& h_out) {
    constexpr float W = static_cast<float>(backend::kMlpInputW - 1U);
    constexpr float H = static_cast<float>(backend::kMlpInputH - 1U);
    const std::array<Point2f, 4> src = {
        Point2f{0.0f, 0.0f},
        Point2f{W, 0.0f},
        Point2f{W, H},
        Point2f{0.0f, H},
    };

#ifdef RM_AIM_WITH_EIGEN
    Eigen::Matrix<double, 8, 8> A;
    Eigen::Matrix<double, 8, 1> b;
    for (int i = 0; i < 4; ++i) {
        const double u = src[i].x;
        const double v = src[i].y;
        const double x = dst[i].x;
        const double y = dst[i].y;
        A.row(2 * i) << u, v, 1.0, 0.0, 0.0, 0.0, -x * u, -x * v;
        b(2 * i) = x;
        A.row(2 * i + 1) << 0.0, 0.0, 0.0, u, v, 1.0, -y * u, -y * v;
        b(2 * i + 1) = y;
    }

    const Eigen::Matrix<double, 8, 1> sol = A.partialPivLu().solve(b);
    if (!sol.allFinite()) return false;

    for (int i = 0; i < 8; ++i) {
        h_out[static_cast<size_t>(i)] = static_cast<float>(sol(i));
    }
    h_out[8] = 1.0f;
    return true;
#else
    (void)dst;
    h_out = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    return false;
#endif
}

inline float dist2d(Point2f a, Point2f b) noexcept {
    return std::hypot(a.x - b.x, a.y - b.y);
}

}  // namespace

ArmorDetector::ArmorDetector(DetectorConfig cfg, EnemyColor enemy_color,
                             backend::IComputeBackend& backend)
    : cfg_(std::move(cfg)), enemy_color_(enemy_color), backend_(backend) {}

std::vector<Armor> ArmorDetector::detect(const FrameSlot& frame, DetectTimingUs* timing) {
    const auto t0 = Clock::now();
    std::vector<Armor> armors;

    const bool use_roi =
        cfg_.use_dynamic_roi && has_roi_ && (roi_frame_counter_ < cfg_.full_scan_interval);

    backend::PreprocessParams pp{};
    pp.enemy_color = enemy_color_;
    pp.color_diff_thresh = cfg_.color_diff_thresh;
    pp.brightness_thresh = cfg_.brightness_thresh;
    pp.roi = use_roi ? current_roi_ : RoiRect{};

    backend_.preprocess_bayer(frame, pp, half_mask_);
    const auto t1 = Clock::now();

    extract_lightbars_pca(frame, half_mask_);
    const auto t2 = Clock::now();

    pair_armors_and_classify(frame, armors);
    const auto t3 = Clock::now();

    // 若动态 ROI 本帧未检出目标，立即在同一帧内回退到全图扫描
    if (use_roi && armors.empty()) {
        has_roi_ = false;
        roi_frame_counter_ = 0;
        pp.roi = RoiRect{};
        backend_.preprocess_bayer(frame, pp, half_mask_);
        extract_lightbars_pca(frame, half_mask_);
        pair_armors_and_classify(frame, armors);
    }

    update_dynamic_roi(frame, armors);

    if (timing != nullptr) {
        timing->preprocess_us =
            std::chrono::duration<double, std::micro>(t1 - t0).count();
        timing->lightbar_pca_us =
            std::chrono::duration<double, std::micro>(t2 - t1).count();
        timing->pair_and_classify_us =
            std::chrono::duration<double, std::micro>(t3 - t2).count();
        timing->total_us =
            std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
        timing->used_roi = use_roi;
    }
    return armors;
}

void ArmorDetector::extract_lightbars_pca(const FrameSlot& frame,
                                          const backend::HalfMaskBuffer& mask) {
    lightbars_.clear();
    if (mask.width == 0 || mask.height == 0) return;

    const uint32_t mw = mask.width;
    const uint32_t mh = mask.height;
    visited_.assign(static_cast<size_t>(mw) * mh, 0);

    std::vector<uint32_t> stack;
    stack.reserve(256);

    std::array<backend::LightbarBboxRequest, backend::kMaxBatchLightbars> bbox_reqs{};
    size_t bbox_count = 0;

    for (uint32_t hy = 0; hy < mh && bbox_count < backend::kMaxBatchLightbars; ++hy) {
        // 利用活跃行位图直接跳过全黑背景行
        if (mask.row_nonzero[hy] == 0) continue;

        const size_t row_off = static_cast<size_t>(hy) * mw;
        for (uint32_t hx = 0; hx < mw && bbox_count < backend::kMaxBatchLightbars; ++hx) {
            const size_t idx = row_off + hx;
            if (mask.mask[idx] == 0 || visited_[idx] != 0) continue;

            // 在 1/2 降采样掩码上做 8 邻域泛洪提取连通域外接框
            uint32_t min_x = hx, max_x = hx;
            uint32_t min_y = hy, max_y = hy;
            uint32_t pixel_count = 0;

            visited_[idx] = 1;
            stack.clear();
            stack.push_back(static_cast<uint32_t>(idx));

            while (!stack.empty()) {
                const uint32_t cur = stack.back();
                stack.pop_back();
                const uint32_t cx = cur % mw;
                const uint32_t cy = cur / mw;

                ++pixel_count;
                min_x = std::min(min_x, cx);
                max_x = std::max(max_x, cx);
                min_y = std::min(min_y, cy);
                max_y = std::max(max_y, cy);

                const uint32_t y_lo = (cy > 0) ? cy - 1U : 0U;
                const uint32_t y_hi = (cy + 1U < mh) ? cy + 1U : mh - 1U;
                const uint32_t x_lo = (cx > 0) ? cx - 1U : 0U;
                const uint32_t x_hi = (cx + 1U < mw) ? cx + 1U : mw - 1U;

                for (uint32_t ny = y_lo; ny <= y_hi; ++ny) {
                    const size_t nrow = static_cast<size_t>(ny) * mw;
                    for (uint32_t nx = x_lo; nx <= x_hi; ++nx) {
                        const size_t nidx = nrow + nx;
                        if (mask.mask[nidx] != 0 && visited_[nidx] == 0) {
                            visited_[nidx] = 1;
                            stack.push_back(static_cast<uint32_t>(nidx));
                        }
                    }
                }
            }

            if (pixel_count < 3U) continue;

            // 将 1/2 掩码外接框映射回原图局部窗口（外扩 2 像素），交给后端批量执行 PCA + 梯度亚像素精修
            bbox_reqs[bbox_count++] = backend::LightbarBboxRequest{
                .fx0 = (mask.roi_x + min_x * 2U >= 2U) ? (mask.roi_x + min_x * 2U - 2U) : 0U,
                .fy0 = (mask.roi_y + min_y * 2U >= 2U) ? (mask.roi_y + min_y * 2U - 2U) : 0U,
                .fx1 = std::min(frame.width - 1U, mask.roi_x + (max_x + 1U) * 2U + 2U),
                .fy1 = std::min(frame.height - 1U, mask.roi_y + (max_y + 1U) * 2U + 2U),
            };
        }
    }

    if (bbox_count == 0) return;

    // 一次批量调用计算后端（CPU 或 CUDA Kernel 3）完成所有候选灯条的 PCA + 梯度边缘亚像素精修
    std::array<backend::LightbarFitResult, backend::kMaxBatchLightbars> fit_results{};
    backend_.batch_fit_lightbars(frame, bbox_reqs.data(), bbox_count,
                                 static_cast<float>(cfg_.brightness_thresh), fit_results.data());

    for (size_t k = 0; k < bbox_count; ++k) {
        const auto& fit = fit_results[k];
        if (!fit.valid) continue;

        const float ratio = fit.length / std::max(1e-3f, fit.width);
        const float tilt_deg = std::fabs(fit.angle_rad) * kRadToDeg;
        if (fit.length < cfg_.min_lightbar_length) continue;
        if (ratio < cfg_.min_lightbar_ratio || ratio > cfg_.max_lightbar_ratio) continue;
        if (tilt_deg > cfg_.max_angle_error_deg) continue;

        LightBar lb{};
        lb.top = fit.top;
        lb.bottom = fit.bottom;
        lb.center = fit.center;
        lb.length = fit.length;
        lb.width = fit.width;
        lb.angle_rad = fit.angle_rad;
        lb.color = enemy_color_;
        lightbars_.push_back(lb);
    }

    // 按 x 坐标从左到右排序，保证配对结果确定性
    std::sort(lightbars_.begin(), lightbars_.end(),
              [](const LightBar& a, const LightBar& b) { return a.center.x < b.center.x; });
}

void ArmorDetector::pair_armors_and_classify(const FrameSlot& frame,
                                             std::vector<Armor>& out_armors) {
    struct Candidate {
        Armor armor;
        backend::WarpPatchRequest req;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(backend::kMaxBatchArmors);

    const size_t n = lightbars_.size();
    for (size_t i = 0; i < n && candidates.size() < backend::kMaxBatchArmors; ++i) {
        for (size_t j = i + 1; j < n && candidates.size() < backend::kMaxBatchArmors; ++j) {
            const LightBar& left = lightbars_[i];
            const LightBar& right = lightbars_[j];

            // 左右灯条长度比约束
            const float side_ratio = std::max(left.length, right.length) /
                                     std::max(1e-3f, std::min(left.length, right.length));
            if (side_ratio > cfg_.max_side_ratio) continue;

            // 灯条平行度约束
            const float angle_diff_deg = std::fabs(left.angle_rad - right.angle_rad) * kRadToDeg;
            if (angle_diff_deg > cfg_.max_rectangular_error_deg) continue;

            // 装甲板宽高比约束
            const float avg_len = 0.5f * (left.length + right.length);
            const float center_dist = dist2d(left.center, right.center);
            const float armor_ratio = center_dist / std::max(1e-3f, avg_len);
            if (armor_ratio < cfg_.min_armor_ratio || armor_ratio > cfg_.max_armor_ratio) {
                continue;
            }

            // 矩形度约束：两灯条中心连线与灯条法线的夹角
            const float dx = right.center.x - left.center.x;
            const float dy = right.center.y - left.center.y;
            const float conn_angle = std::atan2(dy, dx);
            const float avg_tilt = 0.5f * (left.angle_rad + right.angle_rad);
            const float rect_err_deg = std::fabs(conn_angle + avg_tilt) * kRadToDeg;
            if (rect_err_deg > cfg_.max_rectangular_error_deg) continue;

            // 将灯条上下方向外扩 2 倍，覆盖装甲板中央数字区域用于 MLP 裁剪分类
            const auto expand_end = [](Point2f p, Point2f center, float scale) -> Point2f {
                return {center.x + (p.x - center.x) * scale,
                        center.y + (p.y - center.y) * scale};
            };
            const std::array<Point2f, 4> crop_quad = {
                expand_end(left.top, left.center, 2.0f),
                expand_end(right.top, right.center, 2.0f),
                expand_end(right.bottom, right.center, 2.0f),
                expand_end(left.bottom, left.center, 2.0f),
            };

            Candidate cand{};
            if (!compute_patch_homography(crop_quad, cand.req.h_inv)) {
                continue;
            }

            cand.armor.corners = {left.top, right.top, right.bottom, left.bottom};
            cand.armor.color = enemy_color_;
            cand.armor.is_large = (armor_ratio > 3.2f);
            candidates.push_back(cand);
        }
    }

    if (candidates.empty()) return;

    // 一次批量调用 CPU / CUDA 融合 Warp + MLP 后端
    std::array<backend::WarpPatchRequest, backend::kMaxBatchArmors> reqs{};
    std::array<backend::ClassifyResult, backend::kMaxBatchArmors> results{};
    const size_t count = candidates.size();
    for (size_t k = 0; k < count; ++k) {
        reqs[k] = candidates[k].req;
    }

    backend_.batch_warp_and_classify(frame, reqs.data(), count, results.data());

    for (size_t k = 0; k < count; ++k) {
        const auto& res = results[k];
        // class_id == 0 约定为负样本（非装甲板）
        if (res.class_id == 0 || res.confidence < cfg_.min_confidence) {
            continue;
        }
        Armor armor = candidates[k].armor;
        armor.class_id = res.class_id;
        armor.confidence = res.confidence;
        out_armors.push_back(armor);
    }
}

void ArmorDetector::update_dynamic_roi(const FrameSlot& frame, const std::vector<Armor>& armors) {
    if (!cfg_.use_dynamic_roi || armors.empty()) {
        has_roi_ = false;
        roi_frame_counter_ = 0;
        return;
    }

    float min_x = static_cast<float>(frame.width);
    float min_y = static_cast<float>(frame.height);
    float max_x = 0.0f;
    float max_y = 0.0f;
    for (const auto& a : armors) {
        for (const auto& c : a.corners) {
            min_x = std::min(min_x, c.x);
            min_y = std::min(min_y, c.y);
            max_x = std::max(max_x, c.x);
            max_y = std::max(max_y, c.y);
        }
    }

    const float cx = 0.5f * (min_x + max_x);
    const float cy = 0.5f * (min_y + max_y);
    const float half_w = std::max(64.0f, 0.5f * (max_x - min_x) * cfg_.roi_scale);
    const float half_h = std::max(64.0f, 0.5f * (max_y - min_y) * cfg_.roi_scale);

    const uint32_t x0 = static_cast<uint32_t>(std::max(0.0f, cx - half_w)) & ~1U;
    const uint32_t y0 = static_cast<uint32_t>(std::max(0.0f, cy - half_h)) & ~1U;
    const uint32_t x1 = std::min(frame.width, static_cast<uint32_t>(cx + half_w)) & ~1U;
    const uint32_t y1 = std::min(frame.height, static_cast<uint32_t>(cy + half_h)) & ~1U;

    if (x1 > x0 + 16U && y1 > y0 + 16U) {
        current_roi_ = RoiRect{x0, y0, x1 - x0, y1 - y0};
        has_roi_ = true;
        ++roi_frame_counter_;
    } else {
        has_roi_ = false;
        roi_frame_counter_ = 0;
    }
}

}  // namespace rm_aim::perception
