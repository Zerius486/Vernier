#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

#include "rm_aim/core/config.hpp"
#include "rm_aim/core/ring_buffer.hpp"
#include "rm_aim/hal/camera.hpp"
#include "rm_aim/hal/crc.hpp"
#include "rm_aim/hal/protocol.hpp"
#include "rm_aim/hal/serial.hpp"

using namespace rm_aim;
using namespace rm_aim::hal;

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::cerr << "\nCHECK FAILED at " << __FILE__ << ":" << __LINE__ << ": "     \
                      << #cond << "\n";                                                  \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)

namespace {

void test_protocol_and_crc() {
    std::cout << "[TEST] h7_framework protocol & CRC16... ";

    ImuSample sample{};
    sample.mode = 1;
    sample.q = {0.70710678f, 0.0f, 0.70710678f, 0.0f};
    sample.yaw = 0.42f;
    sample.yaw_vel = 1.25f;
    sample.pitch = -0.15f;
    sample.pitch_vel = -0.30f;
    sample.bullet_speed = 24.5f;
    sample.bullet_count = 37;

    const GimbalToVisionPacket pkt = pack_gimbal_state(sample);
    CHECK(sizeof(pkt) == 43);
    CHECK(pkt.head[0] == 'A' && pkt.head[1] == 'B');
    CHECK(check_crc16(reinterpret_cast<const uint8_t*>(&pkt), sizeof(pkt)));

    // 向流式解包器注入含噪声、半包、粘包的混合字节流
    std::vector<uint8_t> stream;
    stream.push_back(0x00);
    stream.push_back('A');
    stream.push_back('X');  // 伪帧头
    stream.push_back('A');
    stream.push_back('B');  // 损坏帧头
    for (int i = 0; i < 5; ++i) stream.push_back(0xFF);

    const auto* raw = reinterpret_cast<const uint8_t*>(&pkt);
    stream.insert(stream.end(), raw, raw + sizeof(pkt));
    stream.insert(stream.end(), raw, raw + sizeof(pkt));

    StreamPacketParser<GimbalToVisionPacket> parser;
    std::vector<GimbalToVisionPacket> decoded;

    // 按非对齐的 7 字节分片喂入，压测拆包重组能力
    for (size_t off = 0; off < stream.size(); off += 7) {
        const size_t n = std::min<size_t>(7, stream.size() - off);
        parser.feed(stream.data() + off, n, [&](const GimbalToVisionPacket& out) {
            decoded.push_back(out);
        });
    }

    CHECK(decoded.size() == 2);
    CHECK(decoded[0].bullet_count == 37);
    CHECK(std::fabs(decoded[1].bullet_speed - 24.5f) < 1e-5f);

    // NaN 防护测试（对齐 h7_framework VisionProtocolFrameIsValid）
    GimbalCmd bad_cmd{};
    bad_cmd.mode = 2;
    bad_cmd.yaw = std::numeric_limits<float>::quiet_NaN();
    const VisionToGimbalPacket safe_pkt = pack_vision_cmd(bad_cmd);
    CHECK(sizeof(safe_pkt) == 30);
    CHECK(safe_pkt.mode == 0);
    CHECK(safe_pkt.fire_rate_hz == 0);
    CHECK(is_valid_packet(safe_pkt));
    CHECK(check_crc16(reinterpret_cast<const uint8_t*>(&safe_pkt), sizeof(safe_pkt)));

    std::cout << "OK\n";
}

void test_spsc_and_frame_pool() {
    std::cout << "[TEST] SpscRingBuffer & LatestFramePool... ";

    SpscRingBuffer<uint64_t, 1024> q;
    constexpr uint64_t kCount = 100000;

    std::thread prod([&] {
        for (uint64_t i = 0; i < kCount; ++i) {
            while (!q.try_push(i)) {
                std::this_thread::yield();
            }
        }
    });

    uint64_t expected = 0;
    while (expected < kCount) {
        uint64_t val = 0;
        if (q.try_pop(val)) {
            CHECK(val == expected);
            ++expected;
        }
    }
    prod.join();

    // 验证 LatestFramePool 丢弃旧帧、只保留最新帧语义
    LatestFramePool pool(3, 640 * 480);
    for (uint64_t i = 1; i <= 5; ++i) {
        FrameSlot& w = pool.acquire_write_slot();
        w.frame_id = i;
        w.width = 640;
        w.height = 480;
        w.byte_size = 640 * 480;
        w.format = PixelFormat::BayerRG8;
        w.host_ptr[0] = static_cast<uint8_t>(i);
        pool.publish_write_slot();
    }

    CHECK(pool.dropped_frames() == 4);
    const FrameSlot* latest = pool.try_acquire_latest();
    CHECK(latest != nullptr);
    CHECK(latest->frame_id == 5);
    CHECK(latest->host_ptr[0] == 5);
    pool.release_read_slot();
    CHECK(pool.try_acquire_latest() == nullptr);

    std::cout << "OK\n";
}

void test_config_loader(const std::string& cfg_path) {
    std::cout << "[TEST] Zero-dependency YAML config loader... ";

    const AppConfig cfg = AppConfig::load(cfg_path);
    CHECK(cfg.enemy_color == EnemyColor::Blue);
    CHECK(cfg.camera.source == "hik");
    CHECK(std::fabs(cfg.camera.exposure_us - 2000.0) < 1e-6);
    CHECK(std::fabs(cfg.camera.camera_matrix[0] - 1776.9477196851155) < 1e-6);
    CHECK(std::fabs(cfg.extrinsics.t_camera2gimbal[0] - 0.1) < 1e-6);
    CHECK(cfg.serial.baudrate == 921600);
    CHECK(cfg.detector.use_dynamic_roi == true);
    CHECK(cfg.aimer.auto_fire == true);

    std::cout << "OK\n";
}

void test_loopback_serial() {
    std::cout << "[TEST] LoopbackSerialLink... ";

    VisionToGimbalPacket received_cmd{};
    bool cmd_ok = false;
    LoopbackSerialLink link([&](const VisionToGimbalPacket& pkt) {
        received_cmd = pkt;
        cmd_ok = true;
    });

    ImuSample received_imu{};
    bool imu_ok = false;
    CHECK(link.start([&](const ImuSample& s) {
        received_imu = s;
        imu_ok = true;
    }));

    ImuSample tx_imu{};
    tx_imu.mode = 1;
    tx_imu.yaw = 0.55f;
    tx_imu.bullet_speed = 25.0f;
    const GimbalToVisionPacket raw_imu = pack_gimbal_state(tx_imu);
    const size_t n =
        link.inject_rx_bytes(reinterpret_cast<const uint8_t*>(&raw_imu), sizeof(raw_imu));
    CHECK(n == 1 && imu_ok);
    CHECK(std::fabs(received_imu.yaw - 0.55f) < 1e-5f);

    GimbalCmd cmd{};
    cmd.mode = 2;
    cmd.yaw = 0.60f;
    cmd.yaw_vel = 1.2f;
    cmd.yaw_acc = 0.4f;
    cmd.pitch = -0.08f;
    cmd.fire_rate_hz = 15;
    CHECK(link.send_command(cmd));
    CHECK(cmd_ok);
    CHECK(received_cmd.mode == 2);
    CHECK(received_cmd.fire_rate_hz == 15);
    CHECK(std::fabs(received_cmd.yaw_acceleration - 0.4f) < 1e-5f);

    link.stop();
    std::cout << "OK\n";
}

void test_synthetic_record_and_replay() {
    std::cout << "[TEST] Synthetic BayerRG8 + IMU -> RMRAW002 -> ReplayCameraSource... ";

    const auto tmp_file = std::filesystem::temp_directory_path() / "rm_aim_core_hal_test.rmraw";

    LatestFramePool pool(3, 320 * 240);
    SyntheticCameraSource synth(320, 240, 0.0, EnemyColor::Blue);
    RawFrameWriter writer;
    CHECK(writer.open(tmp_file.string()));
    CHECK(synth.start(pool));

    int recorded = 0;
    while (recorded < 5) {
        const FrameSlot* slot = pool.try_acquire_latest();
        if (slot != nullptr) {
            CHECK(slot->format == PixelFormat::BayerRG8);
            CHECK(slot->width == 320 && slot->height == 240);

            // 同步写入对应时间戳的 IMU 样本，验证 RMRAW002 交错录包与回放
            ImuSample imu{};
            imu.timestamp_ns = slot->timestamp_ns;
            imu.mode = 1;
            imu.yaw = 0.1f * static_cast<float>(recorded);
            imu.bullet_speed = 24.5f;
            CHECK(writer.write_imu(imu));
            CHECK(writer.write(*slot));

            pool.release_read_slot();
            ++recorded;
        } else {
            std::this_thread::yield();
        }
    }
    synth.stop();
    writer.close();

    // 验证顺序读取器正确识别 RMRAW002 并还原全部帧与 IMU 块
    {
        RawLogReader reader;
        CHECK(reader.open(tmp_file.string()));
        CHECK(reader.version() == 2);
        RawChunkType type = RawChunkType::Frame;
        RawFrameHeader fhdr{};
        std::vector<uint8_t> pixels;
        ImuSample imu{};
        int f_cnt = 0;
        int imu_cnt = 0;
        while (reader.read_next(type, fhdr, pixels, imu)) {
            if (type == RawChunkType::Frame) ++f_cnt;
            if (type == RawChunkType::Imu) ++imu_cnt;
        }
        CHECK(f_cnt == 5);
        CHECK(imu_cnt == 5);
    }

    // 从 .rmraw 离线回放（略微限速以确保消费端稳定取帧）
    LatestFramePool replay_pool(3, 320 * 240);
    ReplayCameraSource replay(tmp_file.string(), 500.0, false);
    std::atomic<int> replayed_imu{0};
    replay.set_imu_callback([&](const ImuSample& s) {
        if (std::fabs(s.bullet_speed - 24.5f) < 1e-4f) {
            replayed_imu.fetch_add(1, std::memory_order_relaxed);
        }
    });
    CHECK(replay.start(replay_pool));

    int replayed = 0;
    const auto deadline = Clock::now() + std::chrono::seconds(2);
    while (replayed < 5 && Clock::now() < deadline) {
        const FrameSlot* slot = replay_pool.try_acquire_latest();
        if (slot != nullptr) {
            CHECK(slot->width == 320 && slot->height == 240);
            CHECK(slot->format == PixelFormat::BayerRG8);
            replay_pool.release_read_slot();
            ++replayed;
        } else {
            std::this_thread::yield();
        }
    }
    replay.stop();
    std::filesystem::remove(tmp_file);
    CHECK(replayed >= 1);
    CHECK(replayed_imu.load(std::memory_order_relaxed) == 5);

    std::cout << "OK (" << replayed << " frames, " << replayed_imu.load() << " IMU replayed)\n";
}

}  // namespace

int main(int argc, char** argv) {
    const std::string cfg_path = (argc > 1) ? argv[1] : "config/example.yaml";
    test_protocol_and_crc();
    test_spsc_and_frame_pool();
    test_config_loader(cfg_path);
    test_loopback_serial();
    test_synthetic_record_and_replay();
    std::cout << "All core_hal tests passed!\n";
    return 0;
}
