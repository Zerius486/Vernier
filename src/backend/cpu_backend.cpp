#include "rm_aim/backend/cpu_backend.hpp"
#include "rm_aim/backend/cuda_backend.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>

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
    for (uint32_t hy = 0; hy < half_h; ++hy) {
        const uint8_t* even = src + static_cast<size_t>(roi_y + hy * 2U) * stride + roi_x;
        const uint8_t* odd = even + stride;
        uint8_t* out_row = dst_mask + static_cast<size_t>(hy) * half_w;
        uint8_t any_hit = 0;

        for (uint32_t hx = 0; hx < half_w; ++hx) {
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
