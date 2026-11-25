#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rm_aim::backend {

inline constexpr uint32_t kMlpInputW = 20;
inline constexpr uint32_t kMlpInputH = 28;
inline constexpr uint32_t kMlpInputDim = kMlpInputW * kMlpInputH;  // 560
inline constexpr uint32_t kMlpHidden1 = 64;
inline constexpr uint32_t kMlpHidden2 = 32;
inline constexpr uint32_t kMlpNumClasses = 9;  // 0: 负样本/非装甲板，1..8: 装甲板数字类别

#pragma pack(push, 1)
struct MlpFileHeader {
    char magic[8];  // 魔数："VRMLP001"
    uint32_t input_w;
    uint32_t input_h;
    uint32_t hidden1;
    uint32_t hidden2;
    uint32_t num_classes;
    uint32_t reserved;
};
#pragma pack(pop)

/// 超轻量 3 层 MLP 权重，按 [in_dim][out_dim] 行主序存放。
/// 该内存布局确保在 CUDA 中每个线程负责 1 个输出神经元（tid -> out_dim）时，
/// 对全局显存的读取达到 100% 合并访存（Coalesced）。
struct MlpWeights {
    // W1: [560][64], b1: [64]
    std::vector<float> w1;
    std::vector<float> b1;
    // W2: [64][32], b2: [32]
    std::vector<float> w2;
    std::vector<float> b2;
    // W3: [32][9], b3: [9]
    std::vector<float> w3;
    std::vector<float> b3;

    bool valid() const noexcept {
        return w1.size() == kMlpInputDim * kMlpHidden1 && b1.size() == kMlpHidden1 &&
               w2.size() == kMlpHidden1 * kMlpHidden2 && b2.size() == kMlpHidden2 &&
               w3.size() == kMlpHidden2 * kMlpNumClasses && b3.size() == kMlpNumClasses;
    }

    static MlpWeights load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) {
            throw std::runtime_error("Failed to open MLP weights file: " + path);
        }

        MlpFileHeader hdr{};
        f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        if (!f.good() || std::memcmp(hdr.magic, "VRMLP001", 8) != 0) {
            throw std::runtime_error("Invalid MLP weights magic header: " + path);
        }
        if (hdr.input_w != kMlpInputW || hdr.input_h != kMlpInputH ||
            hdr.hidden1 != kMlpHidden1 || hdr.hidden2 != kMlpHidden2 ||
            hdr.num_classes != kMlpNumClasses) {
            throw std::runtime_error("MLP weights dimensions mismatch in: " + path);
        }

        MlpWeights m;
        m.w1.resize(kMlpInputDim * kMlpHidden1);
        m.b1.resize(kMlpHidden1);
        m.w2.resize(kMlpHidden1 * kMlpHidden2);
        m.b2.resize(kMlpHidden2);
        m.w3.resize(kMlpHidden2 * kMlpNumClasses);
        m.b3.resize(kMlpNumClasses);

        auto read_vec = [&](std::vector<float>& v) {
            f.read(reinterpret_cast<char*>(v.data()),
                   static_cast<std::streamsize>(v.size() * sizeof(float)));
        };
        read_vec(m.w1);
        read_vec(m.b1);
        read_vec(m.w2);
        read_vec(m.b2);
        read_vec(m.w3);
        read_vec(m.b3);

        if (!f.good()) {
            throw std::runtime_error("Truncated MLP weights file: " + path);
        }
        return m;
    }

    bool save(const std::string& path) const {
        if (!valid()) return false;
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f.is_open()) return false;

        MlpFileHeader hdr{};
        std::memcpy(hdr.magic, "VRMLP001", 8);
        hdr.input_w = kMlpInputW;
        hdr.input_h = kMlpInputH;
        hdr.hidden1 = kMlpHidden1;
        hdr.hidden2 = kMlpHidden2;
        hdr.num_classes = kMlpNumClasses;
        hdr.reserved = 0;

        f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        auto write_vec = [&](const std::vector<float>& v) {
            f.write(reinterpret_cast<const char*>(v.data()),
                    static_cast<std::streamsize>(v.size() * sizeof(float)));
        };
        write_vec(w1);
        write_vec(b1);
        write_vec(w2);
        write_vec(b2);
        write_vec(w3);
        write_vec(b3);
        return f.good();
    }

    /// 生成确定性测试权重（固定以高置信度预测 `forced_class`），用于未训练真实权重前的单测与压测。
    static MlpWeights make_dummy(uint8_t forced_class = 3) {
        MlpWeights m;
        m.w1.assign(kMlpInputDim * kMlpHidden1, 0.0f);
        m.b1.assign(kMlpHidden1, 0.0f);
        m.w2.assign(kMlpHidden1 * kMlpHidden2, 0.0f);
        m.b2.assign(kMlpHidden2, 0.0f);
        m.w3.assign(kMlpHidden2 * kMlpNumClasses, 0.0f);
        m.b3.assign(kMlpNumClasses, -2.0f);
        if (forced_class < kMlpNumClasses) {
            m.b3[forced_class] = 4.0f;
        }
        return m;
    }
};

}  // namespace rm_aim::backend
