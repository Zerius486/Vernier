#include "rm_aim/backend/cpu_backend.hpp"
#include "rm_aim/backend/cuda_backend.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef RM_AIM_WITH_EIGEN
#include <Eigen/Core>
#endif

namespace rm_aim::backend {
namespace {

template <PixelFormat Fmt, EnemyColor Color>
void run_bayer_thresh(const uint8_t* src, uint32_t stride, uint32_t roi_x, uint32_t roi_y,
                      uint32_t half_w, uint32_t half_h, int diff_thresh, int bright_thresh,
                      uint8_t* dst_mask, uint8_t* row_nonzero) noexcept {
    // 2x2 Bayer 单元内的通道排布：
    // 偶行：[e0, e1]，奇行：[o0, o1]
    // BayerRG8: e0=R, e1=G, o0=G, o1=B
    // BayerBG8: e0=B, e1=G, o0=G, o1=R
    // BayerGR8: e0=G, e1=R, o0=B, o1=G
    // BayerGB8: e0=G, e1=B, o0=R, o1=G
#if defined(__AVX2__)
    const bool use_avx2 = (diff_thresh >= 0 && diff_thresh <= 255 &&
                           bright_thresh >= 0 && bright_thresh <= 255);
    const __m128i shuf_even = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14,
                                            -1, -1, -1, -1, -1, -1, -1, -1);
    const __m128i shuf_odd = _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15,
                                           -1, -1, -1, -1, -1, -1, -1, -1);
    const __m128i v_bright = _mm_set1_epi8(static_cast<char>(bright_thresh));
    const __m128i v_diff = _mm_set1_epi8(static_cast<char>(diff_thresh));
    const __m128i v_zero = _mm_setzero_si128();
    const __m128i v_all_ff = _mm_set1_epi8(-1);

    // 从 32 字节（16 对 Bayer 像素）中快速抽取 16 字节偶索引或奇索引通道
    auto extract_16 = [&](const uint8_t* ptr, bool odd_col) -> __m128i {
        const __m256i v32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
        const __m128i lo = _mm256_castsi256_si128(v32);
        const __m128i hi = _mm256_extracti128_si256(v32, 1);
        const __m128i& shuf = odd_col ? shuf_odd : shuf_even;
        return _mm_unpacklo_epi64(_mm_shuffle_epi8(lo, shuf), _mm_shuffle_epi8(hi, shuf));
    };
#endif

    for (uint32_t hy = 0; hy < half_h; ++hy) {
        const uint8_t* even = src + static_cast<size_t>(roi_y + hy * 2U) * stride + roi_x;
        const uint8_t* odd = even + stride;
        uint8_t* out_row = dst_mask + static_cast<size_t>(hy) * half_w;
        uint8_t any_hit = 0;
        uint32_t hx = 0;

#if defined(__AVX2__)
        if (use_avx2) {
            // 每次迭代并行处理 16 个 2x2 Bayer 单元（32x2 原图像素）
            for (; hx + 16U <= half_w; hx += 16U) {
                const uint32_t x2 = hx * 2U;
                __m128i vr, vb;
                if constexpr (Fmt == PixelFormat::BayerRG8) {
                    vr = extract_16(even + x2, false);
                    vb = extract_16(odd + x2, true);
                } else if constexpr (Fmt == PixelFormat::BayerBG8) {
                    vb = extract_16(even + x2, false);
                    vr = extract_16(odd + x2, true);
                } else if constexpr (Fmt == PixelFormat::BayerGR8) {
                    vr = extract_16(even + x2, true);
                    vb = extract_16(odd + x2, false);
                } else {  // BayerGB8
                    vb = extract_16(even + x2, true);
                    vr = extract_16(odd + x2, false);
                }

                const __m128i v_pri = (Color == EnemyColor::Red) ? vr : vb;
                const __m128i v_sec = (Color == EnemyColor::Red) ? vb : vr;

                // 无符号饱和减法：a > t 等价于 subs_epu8(a, t) != 0
                const __m128i eq_b = _mm_cmpeq_epi8(_mm_subs_epu8(v_pri, v_bright), v_zero);
                const __m128i eq_d =
                    _mm_cmpeq_epi8(_mm_subs_epu8(_mm_subs_epu8(v_pri, v_sec), v_diff), v_zero);
                const __m128i mask16 = _mm_andnot_si128(_mm_or_si128(eq_b, eq_d), v_all_ff);

                _mm_storeu_si128(reinterpret_cast<__m128i*>(out_row + hx), mask16);
                if (_mm_movemask_epi8(mask16) != 0) {
                    any_hit = 1U;
                }
            }
        }
#endif

        // 标量尾部路径（处理不足 16 块的边界像素，及无 AVX2 平台）
        for (; hx < half_w; ++hx) {
            const uint32_t x2 = hx * 2U;
            int r = 0;
            int b = 0;
            if constexpr (Fmt == PixelFormat::BayerRG8) {
                r = even[x2];
                b = odd[x2 + 1U];
            } else if constexpr (Fmt == PixelFormat::BayerBG8) {
                b = even[x2];
                r = odd[x2 + 1U];
            } else if constexpr (Fmt == PixelFormat::BayerGR8) {
                r = even[x2 + 1U];
                b = odd[x2];
            } else {  // BayerGB8
                b = even[x2 + 1U];
                r = odd[x2];
            }

            const int primary = (Color == EnemyColor::Red) ? r : b;
            const int secondary = (Color == EnemyColor::Red) ? b : r;
            const bool hit = (primary > bright_thresh) && ((primary - secondary) > diff_thresh);
            out_row[hx] = hit ? 255U : 0U;
            any_hit |= static_cast<uint8_t>(hit);
        }
        row_nonzero[hy] = any_hit;
    }
}

/// 直接在单通道 Bayer/Mono 原图上做快速双线性亮度采样（省去全图 Demosaic 开销）。
inline float sample_gray_bilinear(const FrameSlot& frame, float fx, float fy) noexcept {
    if (fx < 0.0f || fy < 0.0f ||
        fx >= static_cast<float>(frame.width - 1U) ||
        fy >= static_cast<float>(frame.height - 1U)) {
        return 0.0f;
    }

    const uint32_t x0 = static_cast<uint32_t>(fx);
    const uint32_t y0 = static_cast<uint32_t>(fy);
    const float dx = fx - static_cast<float>(x0);
    const float dy = fy - static_cast<float>(y0);

    const uint8_t* r0 = frame.host_ptr + static_cast<size_t>(y0) * frame.stride;
    const uint8_t* r1 = r0 + frame.stride;

    const float v00 = static_cast<float>(r0[x0]);
    const float v10 = static_cast<float>(r0[x0 + 1U]);
    const float v01 = static_cast<float>(r1[x0]);
    const float v11 = static_cast<float>(r1[x0 + 1U]);

    const float top = v00 + dx * (v10 - v00);
    const float bot = v01 + dx * (v11 - v01);
    return (top + dy * (bot - top)) * (1.0f / 255.0f);
}

void mlp_forward_single(const float* in560, const MlpWeights& w, ClassifyResult& out) noexcept {
#ifdef RM_AIM_WITH_EIGEN
    using RowMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using RowVec = Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>;

    Eigen::Map<const RowVec> x(in560, kMlpInputDim);
    Eigen::Map<const RowMat> w1(w.w1.data(), kMlpInputDim, kMlpHidden1);
    Eigen::Map<const RowVec> b1(w.b1.data(), kMlpHidden1);
    Eigen::Map<const RowMat> w2(w.w2.data(), kMlpHidden1, kMlpHidden2);
    Eigen::Map<const RowVec> b2(w.b2.data(), kMlpHidden2);
    Eigen::Map<const RowMat> w3(w.w3.data(), kMlpHidden2, kMlpNumClasses);
    Eigen::Map<const RowVec> b3(w.b3.data(), kMlpNumClasses);

    const RowVec h1 = (x * w1 + b1).cwiseMax(0.0f);
    const RowVec h2 = (h1 * w2 + b2).cwiseMax(0.0f);
    const RowVec logits = h2 * w3 + b3;

    const float max_l = logits.maxCoeff();
    float sum = 0.0f;
    for (uint32_t c = 0; c < kMlpNumClasses; ++c) {
        out.probs[c] = std::exp(logits[c] - max_l);
        sum += out.probs[c];
    }
#else
    float h1[kMlpHidden1];
    for (uint32_t j = 0; j < kMlpHidden1; ++j) h1[j] = w.b1[j];
    for (uint32_t i = 0; i < kMlpInputDim; ++i) {
        const float xi = in560[i];
        const float* row = &w.w1[i * kMlpHidden1];
        for (uint32_t j = 0; j < kMlpHidden1; ++j) h1[j] += xi * row[j];
    }
    for (uint32_t j = 0; j < kMlpHidden1; ++j) h1[j] = std::max(0.0f, h1[j]);

    float h2[kMlpHidden2];
    for (uint32_t j = 0; j < kMlpHidden2; ++j) h2[j] = w.b2[j];
    for (uint32_t i = 0; i < kMlpHidden1; ++i) {
        const float xi = h1[i];
        const float* row = &w.w2[i * kMlpHidden2];
        for (uint32_t j = 0; j < kMlpHidden2; ++j) h2[j] += xi * row[j];
    }
    for (uint32_t j = 0; j < kMlpHidden2; ++j) h2[j] = std::max(0.0f, h2[j]);

    float logits[kMlpNumClasses];
    for (uint32_t j = 0; j < kMlpNumClasses; ++j) logits[j] = w.b3[j];
    for (uint32_t i = 0; i < kMlpHidden2; ++i) {
        const float xi = h2[i];
        const float* row = &w.w3[i * kMlpNumClasses];
        for (uint32_t j = 0; j < kMlpNumClasses; ++j) logits[j] += xi * row[j];
    }

    float max_l = logits[0];
    for (uint32_t c = 1; c < kMlpNumClasses; ++c) max_l = std::max(max_l, logits[c]);
    float sum = 0.0f;
    for (uint32_t c = 0; c < kMlpNumClasses; ++c) {
        out.probs[c] = std::exp(logits[c] - max_l);
        sum += out.probs[c];
    }
#endif

    const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    uint8_t best_cls = 0;
    float best_p = -1.0f;
    for (uint32_t c = 0; c < kMlpNumClasses; ++c) {
        out.probs[c] *= inv_sum;
        if (out.probs[c] > best_p) {
            best_p = out.probs[c];
            best_cls = static_cast<uint8_t>(c);
        }
    }
    out.class_id = best_cls;
    out.confidence = best_p;
}

}  // namespace

CpuComputeBackend::CpuComputeBackend(MlpWeights weights) : weights_(std::move(weights)) {}

void CpuComputeBackend::preprocess_bayer(const FrameSlot& frame, const PreprocessParams& params,
                                         HalfMaskBuffer& out_mask) {
    if (frame.host_ptr == nullptr || frame.width < 2 || frame.height < 2) {
        out_mask.resize(0, 0);
        return;
    }

    // 裁剪并将 ROI 起始坐标对齐到偶数像素，保持 Bayer 2x2 相位不错乱
    uint32_t rx = params.roi.x & ~1U;
    uint32_t ry = params.roi.y & ~1U;
    uint32_t rw = (params.roi.width > 0) ? params.roi.width : frame.width;
    uint32_t rh = (params.roi.height > 0) ? params.roi.height : frame.height;
    if (rx >= frame.width) rx = 0;
    if (ry >= frame.height) ry = 0;
    if (rx + rw > frame.width) rw = frame.width - rx;
    if (ry + rh > frame.height) rh = frame.height - ry;

    const uint32_t half_w = rw / 2U;
    const uint32_t half_h = rh / 2U;
    out_mask.roi_x = rx;
    out_mask.roi_y = ry;
    out_mask.resize(half_w, half_h);

    auto dispatch_color = [&]<PixelFormat Fmt>() {
        if (params.enemy_color == EnemyColor::Red) {
            run_bayer_thresh<Fmt, EnemyColor::Red>(frame.host_ptr, frame.stride, rx, ry,
                                                   half_w, half_h, params.color_diff_thresh,
                                                   params.brightness_thresh, out_mask.mask.data(),
                                                   out_mask.row_nonzero.data());
        } else {
            run_bayer_thresh<Fmt, EnemyColor::Blue>(frame.host_ptr, frame.stride, rx, ry,
                                                    half_w, half_h, params.color_diff_thresh,
                                                    params.brightness_thresh, out_mask.mask.data(),
                                                    out_mask.row_nonzero.data());
        }
    };

    switch (frame.format) {
        case PixelFormat::BayerRG8: dispatch_color.template operator()<PixelFormat::BayerRG8>(); break;
        case PixelFormat::BayerBG8: dispatch_color.template operator()<PixelFormat::BayerBG8>(); break;
        case PixelFormat::BayerGR8: dispatch_color.template operator()<PixelFormat::BayerGR8>(); break;
        case PixelFormat::BayerGB8: dispatch_color.template operator()<PixelFormat::BayerGB8>(); break;
        default:
            std::memset(out_mask.mask.data(), 0, out_mask.mask.size());
            break;
    }
}

void CpuComputeBackend::batch_warp_and_classify(const FrameSlot& frame,
                                                const WarpPatchRequest* requests, size_t count,
                                                ClassifyResult* out_results) {
    if (frame.host_ptr == nullptr || requests == nullptr || out_results == nullptr || count == 0) {
        return;
    }

    const size_t n = std::min(count, kMaxBatchArmors);
    float patch[kMlpInputDim];

    for (size_t k = 0; k < n; ++k) {
        const auto& H = requests[k].h_inv;
        for (uint32_t v = 0; v < kMlpInputH; ++v) {
            const float fv = static_cast<float>(v);
            for (uint32_t u = 0; u < kMlpInputW; ++u) {
                const float fu = static_cast<float>(u);
                const float w_hom = H[6] * fu + H[7] * fv + H[8];
                const float inv_w = (std::fabs(w_hom) > 1e-6f) ? (1.0f / w_hom) : 0.0f;
                const float sx = (H[0] * fu + H[1] * fv + H[2]) * inv_w;
                const float sy = (H[3] * fu + H[4] * fv + H[5]) * inv_w;
                patch[v * kMlpInputW + u] = sample_gray_bilinear(frame, sx, sy);
            }
        }
        mlp_forward_single(patch, weights_, out_results[k]);
    }
}

void CpuComputeBackend::batch_fit_lightbars(const FrameSlot& frame,
                                            const LightbarBboxRequest* requests, size_t count,
                                            float brightness_thresh,
                                            LightbarFitResult* out_results) {
    if (frame.host_ptr == nullptr || requests == nullptr || out_results == nullptr || count == 0) {
        return;
    }

    const size_t n = std::min(count, kMaxBatchLightbars);
    const float thresh_f = brightness_thresh * 0.5f;

    auto sample_bayer_cell_avg = [&](double px, double py) -> double {
        if (px < 0.5 || py < 0.5 ||
            px >= static_cast<double>(frame.width - 2U) ||
            py >= static_cast<double>(frame.height - 2U)) {
            return 0.0;
        }
        auto bilin = [&](double x, double y) -> double {
            const uint32_t x0 = static_cast<uint32_t>(x);
            const uint32_t y0 = static_cast<uint32_t>(y);
            const double dx = x - static_cast<double>(x0);
            const double dy = y - static_cast<double>(y0);
            const uint8_t* r0 = frame.host_ptr + static_cast<size_t>(y0) * frame.stride;
            const uint8_t* r1 = r0 + frame.stride;
            const double top = r0[x0] + dx * (r0[x0 + 1U] - r0[x0]);
            const double bot = r1[x0] + dx * (r1[x0 + 1U] - r1[x0]);
            return top + dy * (bot - top);
        };
        return 0.25 * (bilin(px - 0.5, py - 0.5) + bilin(px + 0.5, py - 0.5) +
                       bilin(px - 0.5, py + 0.5) + bilin(px + 0.5, py + 0.5));
    };

    for (size_t k = 0; k < n; ++k) {
        LightbarFitResult& res = out_results[k];
        res = LightbarFitResult{};

        const uint32_t fx0 = std::min(requests[k].fx0, frame.width - 1U);
        const uint32_t fy0 = std::min(requests[k].fy0, frame.height - 1U);
        const uint32_t fx1 = std::min(requests[k].fx1, frame.width - 1U);
        const uint32_t fy1 = std::min(requests[k].fy1, frame.height - 1U);
        if (fx1 <= fx0 || fy1 <= fy0) continue;

        // 阶段 1：亮度加权质心
        double sum_w = 0.0, sum_x = 0.0, sum_y = 0.0;
        for (uint32_t y = fy0; y <= fy1; ++y) {
            const uint8_t* row = frame.host_ptr + static_cast<size_t>(y) * frame.stride;
            for (uint32_t x = fx0; x <= fx1; ++x) {
                const float val = static_cast<float>(row[x]);
                if (val <= thresh_f) continue;
                const double w = static_cast<double>(val - thresh_f);
                sum_w += w;
                sum_x += w * static_cast<double>(x);
                sum_y += w * static_cast<double>(y);
            }
        }
        if (sum_w <= 1e-6) continue;

        const double cx = sum_x / sum_w;
        const double cy = sum_y / sum_w;

        // 阶段 2：2x2 加权协方差矩阵与 PCA 主轴解析求解
        double cxx = 0.0, cxy = 0.0, cyy = 0.0;
        for (uint32_t y = fy0; y <= fy1; ++y) {
            const uint8_t* row = frame.host_ptr + static_cast<size_t>(y) * frame.stride;
            for (uint32_t x = fx0; x <= fx1; ++x) {
                const float val = static_cast<float>(row[x]);
                if (val <= thresh_f) continue;
                const double w = static_cast<double>(val - thresh_f);
                const double dx = static_cast<double>(x) - cx;
                const double dy = static_cast<double>(y) - cy;
                cxx += w * dx * dx;
                cxy += w * dx * dy;
                cyy += w * dy * dy;
            }
        }
        cxx /= sum_w;
        cxy /= sum_w;
        cyy /= sum_w;

        const double trace = cxx + cyy;
        const double diff = cxx - cyy;
        const double disc = std::hypot(diff, 2.0 * cxy);
        const double lambda1 = 0.5 * (trace + disc);
        const double lambda2 = std::max(1e-6, 0.5 * (trace - disc));
        if (lambda1 <= lambda2) continue;

        double ux = 0.0, uy = 1.0;
        if (std::fabs(cxy) > 1e-9 || std::fabs(diff) > 1e-9) {
            const double theta = 0.5 * std::atan2(2.0 * cxy, diff);
            ux = std::cos(theta);
            uy = std::sin(theta);
        }
        if (uy < 0.0) {
            ux = -ux;
            uy = -uy;
        }

        // 阶段 3：高亮像素粗略端点投影范围
        double min_proj = 1e9, max_proj = -1e9;
        for (uint32_t y = fy0; y <= fy1; ++y) {
            const uint8_t* row = frame.host_ptr + static_cast<size_t>(y) * frame.stride;
            for (uint32_t x = fx0; x <= fx1; ++x) {
                if (static_cast<float>(row[x]) <= thresh_f) continue;
                const double proj =
                    (static_cast<double>(x) - cx) * ux + (static_cast<double>(y) - cy) * uy;
                min_proj = std::min(min_proj, proj);
                max_proj = std::max(max_proj, proj);
            }
        }

        // 阶段 4：2x2 Bayer 单元均值 1D 亮度剖面 + 梯度峰值抛物线亚像素端点精修
        if (max_proj - min_proj >= 4.0) {
            constexpr double kStep = 0.5;
            const double s_start = min_proj - 3.0;
            const double s_end = max_proj + 3.0;
            const int n_samples = static_cast<int>(std::floor((s_end - s_start) / kStep)) + 1;
            if (n_samples >= 10 && n_samples <= 512) {
                double profile[512];
                for (int i = 0; i < n_samples; ++i) {
                    const double s = s_start + i * kStep;
                    profile[i] = sample_bayer_cell_avg(cx + s * ux, cy + s * uy);
                }

                const int edge_band = std::min(n_samples / 2, static_cast<int>(6.0 / kStep));
                int best_top = -1;
                double max_g_top = 8.0;
                for (int i = 1; i < edge_band; ++i) {
                    const double g = profile[i + 1] - profile[i - 1];
                    if (g > max_g_top) {
                        max_g_top = g;
                        best_top = i;
                    }
                }

                int best_bot = -1;
                double max_g_bot = 8.0;
                for (int i = n_samples - edge_band; i < n_samples - 1; ++i) {
                    const double g = profile[i - 1] - profile[i + 1];
                    if (g > max_g_bot) {
                        max_g_bot = g;
                        best_bot = i;
                    }
                }

                auto refine_peak = [&](int idx, bool is_rising) -> double {
                    double sub = 0.0;
                    if (idx >= 2 && idx <= n_samples - 3) {
                        const double g_m = is_rising ? (profile[idx] - profile[idx - 2])
                                                     : (profile[idx - 2] - profile[idx]);
                        const double g_0 = is_rising ? (profile[idx + 1] - profile[idx - 1])
                                                     : (profile[idx - 1] - profile[idx + 1]);
                        const double g_p = is_rising ? (profile[idx + 2] - profile[idx])
                                                     : (profile[idx] - profile[idx + 2]);
                        const double denom = g_m - 2.0 * g_0 + g_p;
                        if (std::fabs(denom) > 1e-6) {
                            sub = std::clamp(0.5 * (g_m - g_p) / denom, -0.5, 0.5);
                        }
                    }
                    return s_start + (static_cast<double>(idx) + sub) * kStep;
                };

                if (best_top >= 0 && best_bot > best_top) {
                    min_proj = refine_peak(best_top, true);
                    max_proj = refine_peak(best_bot, false);
                }
            }
        }

        res.top = {static_cast<float>(cx + min_proj * ux),
                   static_cast<float>(cy + min_proj * uy)};
        res.bottom = {static_cast<float>(cx + max_proj * ux),
                      static_cast<float>(cy + max_proj * uy)};
        res.center = {static_cast<float>(cx), static_cast<float>(cy)};
        res.length = static_cast<float>(max_proj - min_proj);
        res.width = std::max(1.0f, static_cast<float>(std::sqrt(12.0 * lambda2)));
        res.angle_rad = static_cast<float>(std::atan2(ux, uy));
        res.valid = 1U;
    }
}

std::unique_ptr<IComputeBackend> create_compute_backend(const MlpWeights& weights,
                                                        bool prefer_cuda) {
#ifdef RM_AIM_WITH_CUDA
    if (prefer_cuda) {
        return std::make_unique<CudaComputeBackend>(weights);
    }
#else
    (void)prefer_cuda;
#endif
    return std::make_unique<CpuComputeBackend>(weights);
}

}  // namespace rm_aim::backend
