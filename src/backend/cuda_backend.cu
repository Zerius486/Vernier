#include "rm_aim/backend/cuda_backend.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace rm_aim::backend {

#ifdef RM_AIM_WITH_CUDA

namespace {

inline void cuda_check(cudaError_t err, const char* expr) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error at ") + expr + ": " +
                                 cudaGetErrorString(err));
    }
}

#define RM_CUDA_CHECK(call) cuda_check((call), #call)

// ============================================================================
// 核函数 1：bayer_color_thresh_kernel（Bayer 域红蓝差 + 亮度阈值 + 活跃行融合算子）
// ============================================================================
// - 每个线程一次处理横向相邻的 2 个 2x2 Bayer 单元（共 4x2 原图像素）。
// - 地址满足 4 字节对齐时使用 uchar4 向量化读取，32 线程 Warp 合并为 128B 缓存行事务。
// - 利用 __ballot_sync 在 Warp 内归约活跃行，幂等写入 row_nonzero[hy]，零原子争用。
template <PixelFormat Fmt, EnemyColor Color>
__device__ __forceinline__ uint8_t eval_bayer_cell(uint8_t e0, uint8_t e1,
                                                   uint8_t o0, uint8_t o1,
                                                   int diff_thresh, int bright_thresh) {
    int r = 0;
    int b = 0;
    if constexpr (Fmt == PixelFormat::BayerRG8) {
        r = static_cast<int>(e0);
        b = static_cast<int>(o1);
    } else if constexpr (Fmt == PixelFormat::BayerBG8) {
        b = static_cast<int>(e0);
        r = static_cast<int>(o1);
    } else if constexpr (Fmt == PixelFormat::BayerGR8) {
        r = static_cast<int>(e1);
        b = static_cast<int>(o0);
    } else {  // BayerGB8
        b = static_cast<int>(e1);
        r = static_cast<int>(o0);
    }

    const int primary = (Color == EnemyColor::Red) ? r : b;
    const int secondary = (Color == EnemyColor::Red) ? b : r;
    return ((primary > bright_thresh) && ((primary - secondary) > diff_thresh)) ? 255U : 0U;
}

template <PixelFormat Fmt, EnemyColor Color>
__global__ void bayer_color_thresh_kernel(
    const uint8_t* __restrict__ bayer_src, uint32_t stride,
    uint32_t roi_x, uint32_t roi_y, uint32_t half_w, uint32_t half_h,
    int diff_thresh, int bright_thresh,
    uint8_t* __restrict__ mask_half, uint8_t* __restrict__ row_nonzero) {
    const uint32_t hy = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t hx = (blockIdx.x * blockDim.x + threadIdx.x) * 2U;

    bool hit_any = false;
    if (hy < half_h && hx < half_w) {
        const uint32_t sy = roi_y + hy * 2U;
        const uint32_t sx = roi_x + hx * 2U;
        const uint8_t* even = bayer_src + static_cast<size_t>(sy) * stride + sx;
        const uint8_t* odd = even + stride;
        uint8_t* out_row = mask_half + static_cast<size_t>(hy) * half_w;

        // 快速路径：4 字节对齐时用 uchar4 一次读取 2 个 Bayer 单元
        if ((hx + 1U < half_w) && ((reinterpret_cast<uintptr_t>(even) & 3U) == 0U) &&
            ((stride & 3U) == 0U)) {
            const uchar4 ev4 = *reinterpret_cast<const uchar4*>(even);
            const uchar4 od4 = *reinterpret_cast<const uchar4*>(odd);

            const uint8_t m0 = eval_bayer_cell<Fmt, Color>(
                ev4.x, ev4.y, od4.x, od4.y, diff_thresh, bright_thresh);
            const uint8_t m1 = eval_bayer_cell<Fmt, Color>(
                ev4.z, ev4.w, od4.z, od4.w, diff_thresh, bright_thresh);

            out_row[hx] = m0;
            out_row[hx + 1U] = m1;
            hit_any = (m0 | m1) != 0U;
        } else {
            const uint8_t m0 = eval_bayer_cell<Fmt, Color>(
                even[0], even[1], odd[0], odd[1], diff_thresh, bright_thresh);
            out_row[hx] = m0;
            hit_any = (m0 != 0U);

            if (hx + 1U < half_w) {
                const uint8_t m1 = eval_bayer_cell<Fmt, Color>(
                    even[2], even[3], odd[2], odd[3], diff_thresh, bright_thresh);
                out_row[hx + 1U] = m1;
                hit_any = hit_any || (m1 != 0U);
            }
        }
    }

    // Warp 级活跃行归约（blockDim.x == 32，每行块恰好对应 1 个 Warp）
    const unsigned ballot = __ballot_sync(0xFFFFFFFFU, hit_any);
    if (hy < half_h && threadIdx.x == 0 && ballot != 0U) {
        row_nonzero[hy] = 1U;
    }
}

// ============================================================================
// 核函数 2：batch_warp_mlp_kernel（批量透视校正 + 3 层 MLP + Softmax 全融合算子）
// ============================================================================
// - Grid：gridDim.x = count（最多 16 个候选装甲板）
// - Block：blockDim.x = 64（2 个 Warp，恰好等于第一隐藏层宽度 kMlpHidden1）
// - 零中间全局显存写回；Shared Memory 硬件广播消除 Bank Conflict；
//   [in_dim][out_dim] 权重排布保证 100% 合并访存。
__device__ __forceinline__ float sample_gray_bilinear_dev(
    const uint8_t* __restrict__ src, uint32_t width, uint32_t height, uint32_t stride,
    float fx, float fy) {
    if (fx < 0.0f || fy < 0.0f ||
        fx >= static_cast<float>(width - 1U) ||
        fy >= static_cast<float>(height - 1U)) {
        return 0.0f;
    }

    const uint32_t x0 = static_cast<uint32_t>(fx);
    const uint32_t y0 = static_cast<uint32_t>(fy);
    const float dx = fx - static_cast<float>(x0);
    const float dy = fy - static_cast<float>(y0);

    const uint8_t* r0 = src + static_cast<size_t>(y0) * stride;
    const uint8_t* r1 = r0 + stride;

    const float v00 = static_cast<float>(r0[x0]);
    const float v10 = static_cast<float>(r0[x0 + 1U]);
    const float v01 = static_cast<float>(r1[x0]);
    const float v11 = static_cast<float>(r1[x0 + 1U]);

    const float top = fmaf(dx, v10 - v00, v00);
    const float bot = fmaf(dx, v11 - v01, v01);
    return fmaf(dy, bot - top, top) * (1.0f / 255.0f);
}

__global__ void batch_warp_mlp_kernel(
    const uint8_t* __restrict__ bayer_src, uint32_t width, uint32_t height, uint32_t stride,
    const WarpPatchRequest* __restrict__ reqs,
    const float* __restrict__ w1, const float* __restrict__ b1,
    const float* __restrict__ w2, const float* __restrict__ b2,
    const float* __restrict__ w3, const float* __restrict__ b3,
    ClassifyResult* __restrict__ out_results) {
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;  // [0, 63]

    __shared__ float s_H[9];
    __shared__ float s_in[kMlpInputDim];   // 560 floats = 2.24 KB
    __shared__ float s_h1[kMlpHidden1];    // 64 floats
    __shared__ float s_h2[kMlpHidden2];    // 32 floats
    __shared__ float s_logits[kMlpNumClasses];

    if (tid < 9U) {
        s_H[tid] = reqs[bid].h_inv[tid];
    }
    __syncthreads();

    // 阶段 A：64 线程并行透视校正（28x20 = 560 像素）直接写入 Shared Memory
    for (uint32_t idx = tid; idx < kMlpInputDim; idx += blockDim.x) {
        const float fu = static_cast<float>(idx % kMlpInputW);
        const float fv = static_cast<float>(idx / kMlpInputW);
        const float w_hom = fmaf(s_H[6], fu, fmaf(s_H[7], fv, s_H[8]));
        const float inv_w = (fabsf(w_hom) > 1e-6f) ? (1.0f / w_hom) : 0.0f;
        const float sx = fmaf(s_H[0], fu, fmaf(s_H[1], fv, s_H[2])) * inv_w;
        const float sy = fmaf(s_H[3], fu, fmaf(s_H[4], fv, s_H[5])) * inv_w;
        s_in[idx] = sample_gray_bilinear_dev(bayer_src, width, height, stride, sx, sy);
    }
    __syncthreads();

    // 阶段 B：FC1 (560 -> 64) + ReLU
    // 每线程在寄存器中累加 1 个输出神经元；s_in[i] 向整个 Warp 广播读取（零 Bank Conflict）
    {
        float acc = b1[tid];
        #pragma unroll 4
        for (uint32_t i = 0; i < kMlpInputDim; ++i) {
            acc = fmaf(s_in[i], w1[i * kMlpHidden1 + tid], acc);
        }
        s_h1[tid] = fmaxf(acc, 0.0f);
    }
    __syncthreads();

    // 阶段 C：FC2 (64 -> 32) + ReLU（单个 Warp）
    if (tid < kMlpHidden2) {
        float acc = b2[tid];
        #pragma unroll 4
        for (uint32_t i = 0; i < kMlpHidden1; ++i) {
            acc = fmaf(s_h1[i], w2[i * kMlpHidden2 + tid], acc);
        }
        s_h2[tid] = fmaxf(acc, 0.0f);
    }
    __syncthreads();

    // 阶段 D：FC3 (32 -> 9) + Softmax（Warp 0 内完成）
    if (tid < kMlpNumClasses) {
        float acc = b3[tid];
        #pragma unroll 4
        for (uint32_t i = 0; i < kMlpHidden2; ++i) {
            acc = fmaf(s_h2[i], w3[i * kMlpNumClasses + tid], acc);
        }
        s_logits[tid] = acc;
    }
    __syncwarp();

    if (tid == 0U) {
        float max_l = s_logits[0];
        #pragma unroll
        for (uint32_t c = 1; c < kMlpNumClasses; ++c) {
            max_l = fmaxf(max_l, s_logits[c]);
        }

        float probs[kMlpNumClasses];
        float sum = 0.0f;
        #pragma unroll
        for (uint32_t c = 0; c < kMlpNumClasses; ++c) {
            probs[c] = expf(s_logits[c] - max_l);
            sum += probs[c];
        }

        const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        uint8_t best_cls = 0;
        float best_p = -1.0f;
        ClassifyResult& res = out_results[bid];
        #pragma unroll
        for (uint32_t c = 0; c < kMlpNumClasses; ++c) {
            const float p = probs[c] * inv_sum;
            res.probs[c] = p;
            if (p > best_p) {
                best_p = p;
                best_cls = static_cast<uint8_t>(c);
            }
        }
        res.class_id = best_cls;
        res.confidence = best_p;
    }
}

template <PixelFormat Fmt>
void launch_bayer_kernel(EnemyColor color, const uint8_t* d_src, uint32_t stride,
                         uint32_t rx, uint32_t ry, uint32_t half_w, uint32_t half_h,
                         int diff_thresh, int bright_thresh,
                         uint8_t* d_mask, uint8_t* d_row_nonzero, cudaStream_t stream) {
    // 每个线程横向处理 2 个 Bayer 单元
    const dim3 block(32, 8);
    const dim3 grid(((half_w + 1U) / 2U + block.x - 1U) / block.x,
                    (half_h + block.y - 1U) / block.y);
    if (color == EnemyColor::Red) {
        bayer_color_thresh_kernel<Fmt, EnemyColor::Red><<<grid, block, 0, stream>>>(
            d_src, stride, rx, ry, half_w, half_h, diff_thresh, bright_thresh, d_mask, d_row_nonzero);
    } else {
        bayer_color_thresh_kernel<Fmt, EnemyColor::Blue><<<grid, block, 0, stream>>>(
            d_src, stride, rx, ry, half_w, half_h, diff_thresh, bright_thresh, d_mask, d_row_nonzero);
    }
}

}  // namespace

CudaComputeBackend::CudaComputeBackend(MlpWeights weights) {
    RM_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

    auto upload_vec = [](const std::vector<float>& v, float** d_ptr) {
        const size_t bytes = v.size() * sizeof(float);
        RM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(d_ptr), bytes));
        RM_CUDA_CHECK(cudaMemcpy(*d_ptr, v.data(), bytes, cudaMemcpyHostToDevice));
    };

    upload_vec(weights.w1, &d_w1_);
    upload_vec(weights.b1, &d_b1_);
    upload_vec(weights.w2, &d_w2_);
    upload_vec(weights.b2, &d_b2_);
    upload_vec(weights.w3, &d_w3_);
    upload_vec(weights.b3, &d_b3_);

    // 预分配零拷贝 Mapped Pinned 批量 MLP 输入输出缓冲
    RM_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_reqs_),
                                kMaxBatchArmors * sizeof(WarpPatchRequest),
                                cudaHostAllocMapped));
    RM_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_reqs_), h_reqs_, 0));

    RM_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_results_),
                                kMaxBatchArmors * sizeof(ClassifyResult),
                                cudaHostAllocMapped));
    RM_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_results_), h_results_, 0));

    // 默认预分配 1280x1024 的 1/2 降采样掩码（640x512）
    ensure_mask_capacity(640, 512);
}

CudaComputeBackend::~CudaComputeBackend() {
    if (stream_ != nullptr) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }
    if (d_w1_) cudaFree(d_w1_);
    if (d_b1_) cudaFree(d_b1_);
    if (d_w2_) cudaFree(d_w2_);
    if (d_b2_) cudaFree(d_b2_);
    if (d_w3_) cudaFree(d_w3_);
    if (d_b3_) cudaFree(d_b3_);
    if (d_stage_frame_) cudaFree(d_stage_frame_);
    if (h_mask_) cudaFreeHost(h_mask_);
    if (h_row_nonzero_) cudaFreeHost(h_row_nonzero_);
    if (h_reqs_) cudaFreeHost(h_reqs_);
    if (h_results_) cudaFreeHost(h_results_);
}

void CudaComputeBackend::ensure_mask_capacity(uint32_t half_w, uint32_t half_h) {
    const size_t need_bytes = static_cast<size_t>(half_w) * half_h;
    if (need_bytes > mask_cap_bytes_) {
        if (h_mask_) cudaFreeHost(h_mask_);
        RM_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_mask_), need_bytes, cudaHostAllocMapped));
        RM_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_mask_), h_mask_, 0));
        mask_cap_bytes_ = need_bytes;
    }
    if (half_h > row_cap_) {
        if (h_row_nonzero_) cudaFreeHost(h_row_nonzero_);
        RM_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_row_nonzero_), half_h, cudaHostAllocMapped));
        RM_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_row_nonzero_), h_row_nonzero_, 0));
        row_cap_ = half_h;
    }
}

const uint8_t* CudaComputeBackend::resolve_device_frame(const FrameSlot& frame) {
    // Jetson NX 零拷贝快速路径（由 LatestFramePool 通过 cudaHostAllocMapped 分配）
    if (frame.device_ptr != nullptr) {
        return frame.device_ptr;
    }
    // 若调用方传入的是常规主机内存，则回退到异步拷贝进中转显存
    if (frame.byte_size > stage_cap_bytes_) {
        if (d_stage_frame_) cudaFree(d_stage_frame_);
        RM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_stage_frame_), frame.byte_size));
        stage_cap_bytes_ = frame.byte_size;
    }
    RM_CUDA_CHECK(cudaMemcpyAsync(d_stage_frame_, frame.host_ptr, frame.byte_size,
                                  cudaMemcpyHostToDevice, stream_));
    return d_stage_frame_;
}

void CudaComputeBackend::preprocess_bayer(const FrameSlot& frame, const PreprocessParams& params,
                                          HalfMaskBuffer& out_mask) {
    if (frame.host_ptr == nullptr || frame.width < 2 || frame.height < 2) {
        out_mask.resize(0, 0);
        return;
    }

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
    ensure_mask_capacity(half_w, half_h);

    const uint8_t* d_src = resolve_device_frame(frame);
    RM_CUDA_CHECK(cudaMemsetAsync(d_row_nonzero_, 0, half_h, stream_));

    switch (frame.format) {
        case PixelFormat::BayerRG8:
            launch_bayer_kernel<PixelFormat::BayerRG8>(
                params.enemy_color, d_src, frame.stride, rx, ry, half_w, half_h,
                params.color_diff_thresh, params.brightness_thresh, d_mask_, d_row_nonzero_, stream_);
            break;
        case PixelFormat::BayerBG8:
            launch_bayer_kernel<PixelFormat::BayerBG8>(
                params.enemy_color, d_src, frame.stride, rx, ry, half_w, half_h,
                params.color_diff_thresh, params.brightness_thresh, d_mask_, d_row_nonzero_, stream_);
            break;
        case PixelFormat::BayerGR8:
            launch_bayer_kernel<PixelFormat::BayerGR8>(
                params.enemy_color, d_src, frame.stride, rx, ry, half_w, half_h,
                params.color_diff_thresh, params.brightness_thresh, d_mask_, d_row_nonzero_, stream_);
            break;
        case PixelFormat::BayerGB8:
            launch_bayer_kernel<PixelFormat::BayerGB8>(
                params.enemy_color, d_src, frame.stride, rx, ry, half_w, half_h,
                params.color_diff_thresh, params.brightness_thresh, d_mask_, d_row_nonzero_, stream_);
            break;
        default:
            out_mask.roi_x = rx;
            out_mask.roi_y = ry;
            out_mask.resize(half_w, half_h);
            std::memset(out_mask.mask.data(), 0, out_mask.mask.size());
            return;
    }

    RM_CUDA_CHECK(cudaStreamSynchronize(stream_));

    out_mask.roi_x = rx;
    out_mask.roi_y = ry;
    out_mask.resize(half_w, half_h);
    std::memcpy(out_mask.mask.data(), h_mask_, static_cast<size_t>(half_w) * half_h);
    std::memcpy(out_mask.row_nonzero.data(), h_row_nonzero_, half_h);
}

void CudaComputeBackend::batch_warp_and_classify(const FrameSlot& frame,
                                                 const WarpPatchRequest* requests, size_t count,
                                                 ClassifyResult* out_results) {
    if (frame.host_ptr == nullptr || requests == nullptr || out_results == nullptr || count == 0) {
        return;
    }

    const size_t n = std::min(count, kMaxBatchArmors);
    std::memcpy(h_reqs_, requests, n * sizeof(WarpPatchRequest));

    const uint8_t* d_src = resolve_device_frame(frame);
    batch_warp_mlp_kernel<<<static_cast<unsigned int>(n), kMlpHidden1, 0, stream_>>>(
        d_src, frame.width, frame.height, frame.stride, d_reqs_,
        d_w1_, d_b1_, d_w2_, d_b2_, d_w3_, d_b3_, d_results_);

    RM_CUDA_CHECK(cudaStreamSynchronize(stream_));
    std::memcpy(out_results, h_results_, n * sizeof(ClassifyResult));
}

#else

CudaComputeBackend::CudaComputeBackend(MlpWeights weights) : cpu_fallback_(std::move(weights)) {}
CudaComputeBackend::~CudaComputeBackend() = default;

void CudaComputeBackend::preprocess_bayer(const FrameSlot& frame, const PreprocessParams& params,
                                          HalfMaskBuffer& out_mask) {
    cpu_fallback_.preprocess_bayer(frame, params, out_mask);
}

void CudaComputeBackend::batch_warp_and_classify(const FrameSlot& frame,
                                                 const WarpPatchRequest* requests, size_t count,
                                                 ClassifyResult* out_results) {
    cpu_fallback_.batch_warp_and_classify(frame, requests, count, out_results);
}

#endif  // RM_AIM_WITH_CUDA

}  // namespace rm_aim::backend
