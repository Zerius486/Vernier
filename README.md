# Vernier

面向 **High Efficiency（低延迟高帧率）**、**Low Usage（低算力低内存）** 与 **High Accessibility（极简依赖、跨平台易部署）** 重新构建的 RoboMaster 自瞄框架。

## 核心特性

- **零重型依赖**：无 ROS、无 YOLO、无 TensorRT / OpenVINO / ONNXRuntime；核心仅依赖 C++20 + CMake + OpenCV + Eigen3（按需自动探测海康 MVS 与 CUDA）。
- **双平台原生适配**：
  - **Jetson NX (ARM64)**：手写 3 个 CUDA 融合核函数（`bayer_color_thresh_kernel`、`batch_lightbar_pca_grad_kernel`、`batch_warp_mlp_kernel`）+ `cudaHostAllocMapped` 零拷贝帧池（`-DENABLE_CUDA=ON`）。
  - **Intel NUC (x86_64)**：等价 CPU SIMD (AVX2) + Eigen3 后端，无需安装 CUDA Toolkit。
- **海康工业相机 (MVS) 零拷贝采集**：强制 `BayerRG8`（1/3 带宽、免 CPU 全图 Demosaic）、关闭自动白平衡/自动曝光、内置原生 Linux `USBDEVFS_RESET` 断线复位（无需 `libusb`）。
- **亚像素感知与抗脉冲小陀螺估计**：
  - 活跃行跳扫连通域 + 原图局部加权 **PCA 亚像素角点精修** + `28×20` 超轻量 MLP 批量分类 + 动态 ROI 追踪；
  - IPPE PnP + 基于重力/固定倾角的 **1D 黄金分割朝向消歧**；
  - **12 维阻尼角加速（Singer）整车 EKF + 新息自适应 $Q$**（反制非匀速/脉冲变速小陀螺），预留 `IMotionPredictor` 外挂运动模型接口。
- **双端协同火控（对齐 `h7_framework`）**：
  - 统一 POSIX UART / USB-CDC 异步驱动（`termios2` + `epoll` + 断线重枚举自动重连）；
  - 协议与 `h7_framework` 字节级对齐（上行 43B / 下行 30B，含 `yaw/pitch` 位置/速度/加速度前馈与视觉自适应建议射频 `fire_rate_hz`，并监听 `bullet_count` 做弹速闭环滤波）。
- **SSH 无头调试友好**：内置 SSH 终端原地刷新仪表盘、零依赖 MJPEG HTTP 网页推流（无人观看时 0 CPU）、PlotJuggler UDP JSON 遥测、`.rmraw` 原始录包回放与代码级禁火纯跟随测试程序 `vernier_follow`。

## 目录结构

```text
rm_auto_aim/
├── CMakeLists.txt                   # 跨平台自适应构建 (x86_64 / aarch64 / CUDA / MVS)
├── cmake/FindMVS.cmake              # 自动定位 /opt/MVS (lib/64 或 lib/aarch64)
├── config/default.yaml              # 单文件精简配置（内参、外参、串口、检测、火控、调试）
├── scripts/
│   ├── setup_udev.sh                # 一键配置海康相机与 UART/USB-CDC 权限
│   └── train_mlp.py                 # PyTorch 训练 28x20 轻量 MLP 并导出 VRMLP001 权重
├── include/rm_aim/
│   ├── core/                        # 数据类型、无锁 SPSC 队列、零拷贝帧池、YAML 配置
│   ├── hal/                         # CRC16、h7_framework 协议、串口驱动、海康/回放相机
│   ├── backend/                     # IComputeBackend、CPU SIMD+Eigen、CUDA 融合算子、MLP 权重
│   ├── perception/                  # PCA 亚像素检测器、IPPE PnP 与 1D 朝向优化
│   ├── estimation/                  # IMU 四元数 Slerp、IMotionPredictor、12维 Singer EKF、弹道火控
│   └── debug/                       # 零依赖 MJPEG HTTP 推流、PlotJuggler UDP 遥测
├── src/
│   ├── main.cpp                     # vernier 完整自瞄火控主程序
│   └── follow_main.cpp              # vernier_follow 纯瞄准跟随测试程序（代码级强制禁火）
└── tests/                           # Phase 1~3 单元测试、CPU vs CUDA 等价校验与微秒级 Benchmark
```

## 快速构建与测试

### 1. 首次部署环境权限（实车 NUC / NX）

```bash
./scripts/setup_udev.sh
```

### 2. 编译与运行测试

```bash
# Intel NUC / 普通开发机（自动探测 OpenCV / Eigen3 / MVS）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure -V

# Jetson NX（启用手写 CUDA 融合核函数 + 零拷贝 + 自动运行 CPU vs CUDA 数值等价校验）
sudo nvpmodel -m 0 && sudo jetson_clocks
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure -V

# 可选：需要标定内参/外参/时间同步时，开启 RM_AIM_BUILD_CALIB 构建 vernier_calib
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DRM_AIM_BUILD_CALIB=ON
cmake --build build --target vernier_calib -j$(nproc)
```

## 运行与实车（哨兵/步兵）联调指南

### 可执行程序

```bash
# 1. 完整自瞄火控主程序
./build/vernier config/default.yaml

# 2. 纯瞄准跟随测试程序（代码级强制禁火：mode <= 1, fire_rate_hz 恒为 0）
./build/vernier_follow config/default.yaml
./build/vernier_follow config/default.yaml --no-ff --mjpeg

# 3. 可选 SSH 无头标定与校准工具（需 -DRM_AIM_BUILD_CALIB=ON）
./build/vernier_calib intrinsic config/default.yaml --cols 9 --rows 6 --square 0.025
./build/vernier_calib sync config/default.yaml --duration 6
```

### 上车纯跟随 3 步安全联调流程（SSH）

1. **同步固件与配置**：
   - 确保下位机已烧录同步升级后的 30 字节 `h7_framework` 固件，建议首次测试断开摩擦轮/拨弹供电；
   - 在 `config/default.yaml` 中核对 `target.enemy_color`、`serial.device`、`camera_matrix`、`r_camera2gimbal`、`t_camera2gimbal` 与 `exposure_us`。
   - *(纯跟随测试阶段若尚未训练 `models/armor_mlp.bin`，程序会自动启用占位权重直接跑通检测与跟随)*
2. **先看图 + 验串口（关前馈）**：
   ```bash
   ./build/vernier_follow config/default.yaml --no-ff --mjpeg
   ```
   - 终端确认 `Ser(rx=..., OK)` 正常增长且 `FPS` 稳在 150–200+；
   - 浏览器打开 `http://<设备IP>:8080/` 检查黄色装甲板框稳定套住靶板，慢移靶板确认云台左移左转、上移抬头方向无误，并微调 `yaw_offset_deg` / `pitch_offset_deg`。
3. **开前馈测动态跟随**：
   ```bash
   ./build/vernier_follow config/default.yaml --mjpeg
   ```
   快速横移或转动靶板，配合 PlotJuggler（UDP 9870）检查动态跟随与切板平滑度。

### 训练与导出超轻量 MLP 数字分类权重

```bash
python3 scripts/train_mlp.py --data /path/to/dataset --out models/armor_mlp.bin --epochs 25
```

## 通信协议（对齐 `h7_framework`）

世界系约定为标准 **FLU 右手系**（X 前、Y 左、Z 上；Yaw 向左为正，Pitch 抬头为正，水平为 `0 rad`）：

- **上行 `GimbalToVision`（43B）**：`AB` + `mode` + `q[4](wxyz)` + `yaw/pitch` 角度与角速度 + `bullet_speed` + `bullet_count` + `crc16`
- **下行 `VisionToGimbal`（30B）**：`AB` + `mode` (`0:不控制, 1:跟随, 2:跟随并开火`) + `yaw/pitch` 位置/速度/加速度前馈 + `fire_rate_hz`（视觉建议射频，下位机保留热量裁决权）+ `crc16`
