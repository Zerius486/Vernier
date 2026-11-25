# Vernier

面向 **High Efficiency（低延迟高帧率）**、**Low Usage（低算力低内存）** 与 **High Accessibility（极简依赖、跨平台易部署）** 构建的 RoboMaster 自瞄框架。

## 核心特性

- **零重型依赖**：无 ROS、无 YOLO、无 TensorRT / OpenVINO / ONNXRuntime；核心仅依赖 C++20 + CMake + OpenCV + Eigen3（按需自动探测海康 MVS 与 CUDA）。
- **双平台原生适配**：
  - **Jetson NX (ARM64)**：手写 3 个 CUDA 融合核函数（`bayer_color_thresh_kernel`、`batch_lightbar_pca_grad_kernel`、`batch_warp_mlp_kernel`）+ `cudaHostAllocMapped` 零拷贝帧池（`-DENABLE_CUDA=ON`）。
  - **Intel NUC (x86_64)**：等价 CPU SIMD (AVX2) + Eigen3 后端，无需安装 CUDA Toolkit。
- **海康工业相机 (MVS) 零拷贝采集**：强制 `BayerRG8`（1/3 带宽、免 CPU 全图 Demosaic）、关闭自动白平衡/自动曝光、内置原生 Linux `USBDEVFS_RESET` 断线复位（无需 `libusb`）。
- **编译期双特征解耦（`RM_AIM_FEATURE_TYPE`）与亚像素位姿估计**：
  - 通过 `IArmorDetector` 抽象接口与工厂函数 `create_armor_detector` 隔离装甲板视觉特征；默认 `LIGHTBAR` 启用完整传统灯条检测（活跃行跳扫连通域 + 原图局部加权 **PCA 与梯度亚像素端点精修** + `28×20` 超轻量 MLP 批量分类 + 动态 ROI 追踪），`DOTMATRIX` 提供点阵留空占位接口与 3D 尺寸隔离分支，待官方样式公布后直接填充；
  - IPPE PnP + 镜头去畸变 + 基于重力与固定倾角约束的 **1D 黄金分割朝向消歧**与 **4-DoF（x, y, z, yaw）阻尼 Gauss-Newton 联合位姿精修**（`refine_pose_4dof`）；
  - **12 维阻尼角加速（Singer）整车 EKF + 新息自适应 Q**（反制非匀速/脉冲变速小陀螺），预留 `IMotionPredictor` 外挂运动模型接口。
- **双端协同火控（对齐 `h7_framework`）**：
  - 统一 POSIX UART / USB-CDC 异步驱动（`termios2` + `epoll` + 断线重枚举自动重连）；
  - 协议与 `h7_framework` 字节级对齐（上行 43B / 下行 30B，含 `yaw/pitch` 位置/速度/加速度前馈与视觉自适应建议射频 `fire_rate_hz`，并监听 `bullet_count` 做弹速闭环滤波）。
- **SSH 无头调试友好**：内置 SSH 终端原地刷新仪表盘、零依赖 MJPEG HTTP 网页推流（无人观看时 0 CPU）、PlotJuggler UDP JSON 遥测、`.rmraw` 原始录包回放、代码级禁火纯跟随测试程序 `vernier_follow` 及可选无头标定工具 `vernier_calib`。

## 目录结构

```text
rm_auto_aim/
├── CMakeLists.txt                   # 跨平台自适应构建（x86_64 / aarch64 / CUDA / MVS / LIGHTBAR|DOTMATRIX）
├── cmake/FindMVS.cmake              # 自动定位 /opt/MVS (lib/64 或 lib/aarch64)
├── config/default.yaml              # 单文件精简配置（内参、外参、串口、检测、火控、调试）
├── scripts/
│   ├── setup_udev.sh                # 一键配置海康相机与 UART/USB-CDC 权限
│   ├── install_service.sh           # 安装 systemd 无头开机自启与崩溃守护服务
│   └── train_mlp.py                 # PyTorch 训练 28x20 轻量 MLP 并导出 VRMLP001 权重
├── include/rm_aim/
│   ├── core/                        # 数据类型、无锁 SPSC 队列、零拷贝帧池、YAML 配置
│   ├── hal/                         # CRC16、h7_framework 协议、串口驱动、海康/回放相机
│   ├── backend/                     # IComputeBackend、CPU SIMD+Eigen、CUDA 融合算子、MLP 权重
│   ├── perception/                  # IArmorDetector 抽象接口、LightbarDetector、DotMatrixDetector（占位接口）、detector_utils、PnpSolver（4-DoF 精修）
│   ├── estimation/                  # IMU 四元数 Slerp、IMotionPredictor、12 维 Singer EKF、弹道火控
│   └── debug/                       # 零依赖 MJPEG HTTP 推流、PlotJuggler UDP 遥测
├── src/
│   ├── core/                        # 配置解析实现
│   ├── hal/                         # 串口、海康 MVS、.rmraw 录包回放实现
│   ├── backend/                     # CPU AVX2/Eigen 与 CUDA 计算后端实现
│   ├── perception/                  # LightbarDetector、DotMatrixDetector（占位）与 PnpSolver 实现
│   ├── estimation/                  # IMU 缓冲、Singer EKF 与弹道火控实现
│   ├── debug/                       # MJPEG 服务与 UDP 遥测实现
│   ├── main.cpp                     # vernier 完整自瞄火控主程序
│   ├── follow_main.cpp              # vernier_follow 纯瞄准跟随测试程序（代码级强制禁火）
│   └── calib_main.cpp               # vernier_calib SSH 无头内参/外参/时间同步标定与样本导出工具
└── tests/                           # Phase 1~3 单元测试、CPU vs CUDA 等价校验与微秒级 Benchmark
```

## 快速构建与测试

### 1. 首次部署环境权限（实车 NUC / NX）

```bash
./scripts/setup_udev.sh
```

### 2. CMake 编译选项

| CMake 选项 | 默认值 | 可选值 | 说明 |
| :--- | :--- | :--- | :--- |
| `RM_AIM_FEATURE_TYPE` | `LIGHTBAR` | `LIGHTBAR` / `DOTMATRIX` | 编译期选择视觉特征检测器（传统灯条 / 点阵留空占位接口） |
| `ENABLE_CUDA` | `AUTO` | `AUTO` / `ON` / `OFF` | 是否启用手写 CUDA 融合后端（Jetson NX 建议显式 `ON`） |
| `RM_AIM_ENABLE_NATIVE_OPT` | `ON` | `ON` / `OFF` | 在 Release 构建中启用 `-march=native` / `-mcpu=native` 指令优化 |
| `RM_AIM_BUILD_CALIB` | `OFF` | `ON` / `OFF` | 是否构建无头标定与样本导出工具 `vernier_calib` |
| `RM_AIM_BUILD_TESTS` | `ON` | `ON` / `OFF` | 是否构建 Phase 1~3 单元测试与 Benchmark |

### 3. 编译与运行测试

```bash
# Intel NUC / 普通开发机（默认构建 LIGHTBAR 传统灯条版）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DRM_AIM_FEATURE_TYPE=LIGHTBAR
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure -V

# 切换为去灯条点阵留空占位接口版
# 注：当前 DOTMATRIX 仅链接 DotMatrixDetector 留空占位接口（detect() 返回空结果），
#     仅用于验证编译链接与接口契约，待官方样式公布并完成实现前不可用于实车瞄准
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DRM_AIM_FEATURE_TYPE=DOTMATRIX
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

## 点阵装甲板扩展指南（点阵留空占位接口）

待官方公布点阵装甲板样式与物理尺寸后，仅需按以下 4 处接入点完成实现，其余 `core/`、`hal/`、`backend/`、`estimation/`、`debug/` 与 `main.cpp` 均无需改动：

1. **视觉检测与角点提取**：在 `include/rm_aim/perception/dotmatrix_detector.hpp` 与 `src/perception/dotmatrix_detector.cpp` 中实现 `DotMatrixDetector::detect()`，按顺时针 `[左上, 右上, 右下, 左下]` 输出原图角点及 `color` / `class_id` / `confidence` / `is_large`（按需复用 `include/rm_aim/perception/detector_utils.hpp`）。
2. **3D 物理尺寸常量**：在 `include/rm_aim/perception/pnp_solver.hpp` 的 `#ifdef RM_AIM_FEATURE_DOTMATRIX` 分支下替换 `kSmallArmorHalfW`、`kLargeArmorHalfW`、`kArmorHalfH` 占位常量。
3. **专属配置参数**：在 `include/rm_aim/core/config.hpp`、`src/core/config.cpp` 与 `config/default.yaml` 中补充点阵阈值与几何约束参数。
4. **标定导出与单元测试**：按点阵实际角点定义调整 `src/calib_main.cpp` 中 `dump_digits` 的 `#ifdef RM_AIM_FEATURE_DOTMATRIX` 裁剪逻辑，并在 `tests/test_phase2.cpp` 中补充点阵验证用例。

## 运行与实车（哨兵/步兵）联调指南

### 可执行程序

```bash
# 1. 完整自瞄火控主程序
./build/vernier config/default.yaml

# 2. 纯瞄准跟随测试程序（代码级强制禁火：mode <= 1, fire_rate_hz 恒为 0）
./build/vernier_follow config/default.yaml
./build/vernier_follow config/default.yaml --no-ff --mjpeg

# 3. 可选 SSH 无头标定与样本导出工具（需 -DRM_AIM_BUILD_CALIB=ON）
./build/vernier_calib intrinsic config/default.yaml --cols 9 --rows 6 --square 0.025
./build/vernier_calib sync config/default.yaml --duration 6
./build/vernier_calib dump_digits config/default.yaml --input records/capture.rmraw --out dataset/raw

# 4. 实车无头开机自启与崩溃守护（systemd）
./scripts/install_service.sh
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
