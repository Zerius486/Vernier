#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "rm_aim/backend/compute_backend.hpp"
#include "rm_aim/core/config.hpp"
#include "rm_aim/core/types.hpp"

#ifdef RM_AIM_WITH_EIGEN
#include <Eigen/Dense>
#endif

namespace rm_aim::perception::detail {

/// 计算 3x3 逆透视单应矩阵，将归一化数字图四角
/// (0,0), (W-1,0), (W-1,H-1), (0,H-1) 映射回原图四边形（左上、右上、右下、左下）。
inline bool compute_patch_homography(const std::array<Point2f, 4>& dst,
                                     std::array<float, 9>& h_out) {
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

/// 根据本帧检出的装甲板外接矩形更新下一帧动态 ROI 状态。
inline void update_dynamic_roi_state(const DetectorConfig& cfg, const FrameSlot& frame,
                                     const std::vector<Armor>& armors, bool& has_roi,
                                     RoiRect& current_roi, int& roi_frame_counter) {
    if (!cfg.use_dynamic_roi || armors.empty()) {
        has_roi = false;
        roi_frame_counter = 0;
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
    const float half_w = std::max(64.0f, 0.5f * (max_x - min_x) * cfg.roi_scale);
    const float half_h = std::max(64.0f, 0.5f * (max_y - min_y) * cfg.roi_scale);

    const uint32_t x0 = static_cast<uint32_t>(std::max(0.0f, cx - half_w)) & ~1U;
    const uint32_t y0 = static_cast<uint32_t>(std::max(0.0f, cy - half_h)) & ~1U;
    const uint32_t x1 = std::min(frame.width, static_cast<uint32_t>(cx + half_w)) & ~1U;
    const uint32_t y1 = std::min(frame.height, static_cast<uint32_t>(cy + half_h)) & ~1U;

    if (x1 > x0 + 16U && y1 > y0 + 16U) {
        current_roi = RoiRect{x0, y0, x1 - x0, y1 - y0};
        has_roi = true;
        ++roi_frame_counter;
    } else {
        has_roi = false;
        roi_frame_counter = 0;
    }
}

}  // namespace rm_aim::perception::detail
