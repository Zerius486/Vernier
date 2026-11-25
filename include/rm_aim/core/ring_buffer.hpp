#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <type_traits>
#include <vector>

#include "rm_aim/core/types.hpp"

#ifdef RM_AIM_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace rm_aim {

/// 无锁单生产者单消费者（SPSC）环形队列。
/// 容量 Capacity 必须为大于等于 2 的 2 的幂。
template <typename T, size_t Capacity>
class SpscRingBuffer {
    static_assert((Capacity >= 2) && ((Capacity & (Capacity - 1)) == 0),
                  "Capacity must be a power of two >= 2");

public:
    SpscRingBuffer() = default;
    SpscRingBuffer(const SpscRingBuffer&) = delete;
    SpscRingBuffer& operator=(const SpscRingBuffer&) = delete;

    bool try_push(const T& item) noexcept(std::is_nothrow_copy_assignable_v<T>) {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t next = (head + 1) & kMask;
        if (next == tail_.load(std::memory_order_acquire)) {
            return false;
        }
        storage_[head] = item;
        head_.store(next, std::memory_order_release);
        return true;
    }

    bool try_push(T&& item) noexcept(std::is_nothrow_move_assignable_v<T>) {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t next = (head + 1) & kMask;
        if (next == tail_.load(std::memory_order_acquire)) {
            return false;
        }
        storage_[head] = std::move(item);
        head_.store(next, std::memory_order_release);
        return true;
    }

    bool try_pop(T& out) noexcept(std::is_nothrow_move_assignable_v<T>) {
        const size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) {
            return false;
        }
        out = std::move(storage_[tail]);
        tail_.store((tail + 1) & kMask, std::memory_order_release);
        return true;
    }

    bool empty() const noexcept {
        return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
    }

    size_t size() const noexcept {
        const size_t head = head_.load(std::memory_order_acquire);
        const size_t tail = tail_.load(std::memory_order_acquire);
        return (head - tail) & kMask;
    }

    static constexpr size_t capacity() noexcept { return Capacity - 1; }

private:
    static constexpr size_t kMask = Capacity - 1;
    static constexpr size_t kCacheLine = 64;

    alignas(kCacheLine) std::atomic<size_t> head_{0};
    alignas(kCacheLine) std::atomic<size_t> tail_{0};
    alignas(kCacheLine) std::array<T, Capacity> storage_{};
};

/// 零拷贝最新帧三/四缓冲池（SPSC）。
///
/// 生产者始终写入无竞争的空闲槽位，并原子发布为最新帧（丢弃未被消费的旧帧，保证最低延迟）；
/// 消费者始终无阻塞获取最新一帧，不会阻塞相机采集线程。
class LatestFramePool {
public:
    LatestFramePool(size_t num_slots, size_t max_frame_bytes)
        : slots_(num_slots < 3 ? 3 : num_slots) {
        // 按 64 字节对齐容量，适配 SIMD 指令与缓存行
        const size_t aligned_bytes = (max_frame_bytes + 63U) & ~static_cast<size_t>(63U);
        for (auto& slot : slots_) {
            slot.capacity = aligned_bytes;
#ifdef RM_AIM_WITH_CUDA
            void* host_mem = nullptr;
            if (cudaHostAlloc(&host_mem, aligned_bytes, cudaHostAllocMapped) == cudaSuccess) {
                void* dev_mem = nullptr;
                if (cudaHostGetDevicePointer(&dev_mem, host_mem, 0) == cudaSuccess) {
                    slot.host_ptr = static_cast<uint8_t*>(host_mem);
                    slot.device_ptr = static_cast<uint8_t*>(dev_mem);
                    cuda_allocated_ = true;
                } else {
                    cudaFreeHost(host_mem);
                }
            }
#endif
            if (slot.host_ptr == nullptr) {
                slot.host_ptr = static_cast<uint8_t*>(std::aligned_alloc(64, aligned_bytes));
                slot.device_ptr = nullptr;
            }
        }
        write_idx_ = 0;
        latest_idx_.store(-1, std::memory_order_relaxed);
        read_idx_.store(-1, std::memory_order_relaxed);
    }

    ~LatestFramePool() {
        for (auto& slot : slots_) {
            if (slot.host_ptr == nullptr) {
                continue;
            }
#ifdef RM_AIM_WITH_CUDA
            if (cuda_allocated_) {
                cudaFreeHost(slot.host_ptr);
                continue;
            }
#endif
            std::free(slot.host_ptr);
        }
    }

    LatestFramePool(const LatestFramePool&) = delete;
    LatestFramePool& operator=(const LatestFramePool&) = delete;

    /// 生产者侧：获取当前独占的可写帧槽。
    FrameSlot& acquire_write_slot() noexcept { return slots_[write_idx_]; }

    /// 生产者侧：将已写完的帧槽发布为最新帧，并切换到下一个空闲写槽。
    void publish_write_slot() noexcept {
        const int published = write_idx_;
        const int prev_latest = latest_idx_.exchange(published, std::memory_order_acq_rel);
        if (prev_latest != -1) {
            dropped_frames_.fetch_add(1, std::memory_order_relaxed);
        }

        // 选择既不是刚发布的槽位、也不是消费者正在读取的槽位作为下一个写槽
        const int reader = read_idx_.load(std::memory_order_acquire);
        const int n = static_cast<int>(slots_.size());
        for (int i = 0; i < n; ++i) {
            if (i != published && i != reader) {
                write_idx_ = i;
                break;
            }
        }
    }

    /// 消费者侧：尝试获取最新发布的帧槽；若无新帧则返回 nullptr。
    const FrameSlot* try_acquire_latest() noexcept {
        const int idx = latest_idx_.exchange(-1, std::memory_order_acq_rel);
        if (idx < 0) {
            return nullptr;
        }
        read_idx_.store(idx, std::memory_order_release);
        return &slots_[static_cast<size_t>(idx)];
    }

    /// 消费者侧：释放当前持有的读帧槽。
    void release_read_slot() noexcept { read_idx_.store(-1, std::memory_order_release); }

    uint64_t dropped_frames() const noexcept {
        return dropped_frames_.load(std::memory_order_relaxed);
    }

    bool is_cuda_zero_copy() const noexcept { return cuda_allocated_; }

private:
    std::vector<FrameSlot> slots_;
    int write_idx_ = 0;
    bool cuda_allocated_ = false;

    alignas(64) std::atomic<int> latest_idx_{-1};
    alignas(64) std::atomic<int> read_idx_{-1};
    alignas(64) std::atomic<uint64_t> dropped_frames_{0};
};

}  // namespace rm_aim
