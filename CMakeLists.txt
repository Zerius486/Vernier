cmake_minimum_required(VERSION 3.20)
project(vernier VERSION 0.3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---------------------------------------------------------------------------
# 编译选项
# ---------------------------------------------------------------------------
set(ENABLE_CUDA "AUTO" CACHE STRING "是否启用手写 CUDA 融合后端：AUTO / ON / OFF")
set_property(CACHE ENABLE_CUDA PROPERTY STRINGS AUTO ON OFF)

set(RM_AIM_FEATURE_TYPE "LIGHTBAR" CACHE STRING "装甲板视觉特征类型：LIGHTBAR（传统灯条）或 DOTMATRIX（点阵留空占位接口）")
set_property(CACHE RM_AIM_FEATURE_TYPE PROPERTY STRINGS LIGHTBAR DOTMATRIX)

option(RM_AIM_ENABLE_NATIVE_OPT "在 Release 构建中启用 -march=native / SIMD 指令优化" ON)
option(RM_AIM_BUILD_TESTS "构建单元测试与性能基准" ON)
option(RM_AIM_BUILD_CALIB "构建 SSH 无头标定与校准工具 vernier_calib" OFF)

# ---------------------------------------------------------------------------
# 架构检测与编译器标志
# ---------------------------------------------------------------------------
message(STATUS "Target processor: ${CMAKE_SYSTEM_PROCESSOR}")

add_library(rm_aim_compile_options INTERFACE)
target_compile_options(rm_aim_compile_options INTERFACE
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O3>
)

if(RM_AIM_ENABLE_NATIVE_OPT)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        target_compile_options(rm_aim_compile_options INTERFACE
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-march=native -fno-math-errno -fno-trapping-math>
        )
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
        target_compile_options(rm_aim_compile_options INTERFACE
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-mcpu=native -fno-math-errno -fno-trapping-math>
        )
    endif()
endif()

# ---------------------------------------------------------------------------
# 可选依赖探测
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)
find_package(MVS QUIET)
find_package(OpenCV QUIET COMPONENTS core imgproc calib3d imgcodecs videoio)
find_package(Eigen3 QUIET)

# CUDA 探测
set(RM_AIM_WITH_CUDA OFF)
if(ENABLE_CUDA STREQUAL "ON")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(RM_AIM_WITH_CUDA ON)
elseif(ENABLE_CUDA STREQUAL "AUTO")
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            set(RM_AIM_WITH_CUDA ON)
        endif()
    endif()
endif()

if(RM_AIM_WITH_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # 默认覆盖 Jetson Xavier NX (72) 与 Orin NX (87)
        set(CMAKE_CUDA_ARCHITECTURES 72 87)
    endif()
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    message(STATUS "Compute backend: CUDA (${CUDAToolkit_VERSION}) + CPU fallback")
else()
    message(STATUS "Compute backend: CPU (SIMD + Eigen)")
endif()

if(RM_AIM_FEATURE_TYPE STREQUAL "LIGHTBAR")
    set(_rm_aim_detector_src "src/perception/lightbar_detector.cpp")
elseif(RM_AIM_FEATURE_TYPE STREQUAL "DOTMATRIX")
    set(_rm_aim_detector_src "src/perception/dotmatrix_detector.cpp")
else()
    message(FATAL_ERROR "未知的 RM_AIM_FEATURE_TYPE: ${RM_AIM_FEATURE_TYPE}，仅支持 LIGHTBAR 或 DOTMATRIX")
endif()

message(STATUS "Hikvision MVS SDK: ${MVS_FOUND}")
message(STATUS "OpenCV found:      ${OpenCV_FOUND}")
message(STATUS "Eigen3 found:      ${Eigen3_FOUND}")
message(STATUS "Armor feature:     ${RM_AIM_FEATURE_TYPE}")

# ---------------------------------------------------------------------------
# 核心库：Core + HAL + Backend + Perception + Estimation + Debug
# ---------------------------------------------------------------------------
set(RM_AIM_CORE_SOURCES
    src/core/config.cpp
    src/hal/serial.cpp
    src/hal/replay_camera.cpp
    src/backend/cpu_backend.cpp
    ${_rm_aim_detector_src}
    src/perception/pnp_solver.cpp
    src/estimation/imu_buffer.cpp
    src/estimation/ekf_tracker.cpp
    src/estimation/mpc_planner.cpp
    src/estimation/ballistic.cpp
    src/debug/mjpeg_server.cpp
    src/debug/udp_telemetry.cpp
)

if(MVS_FOUND)
    list(APPEND RM_AIM_CORE_SOURCES src/hal/hik_camera.cpp)
endif()

if(RM_AIM_WITH_CUDA)
    list(APPEND RM_AIM_CORE_SOURCES src/backend/cuda_backend.cu)
endif()

add_library(rm_aim_core STATIC ${RM_AIM_CORE_SOURCES})

target_include_directories(rm_aim_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(rm_aim_core PUBLIC
    rm_aim_compile_options
    Threads::Threads
)

if(RM_AIM_FEATURE_TYPE STREQUAL "LIGHTBAR")
    target_compile_definitions(rm_aim_core PUBLIC RM_AIM_FEATURE_LIGHTBAR=1)
elseif(RM_AIM_FEATURE_TYPE STREQUAL "DOTMATRIX")
    target_compile_definitions(rm_aim_core PUBLIC RM_AIM_FEATURE_DOTMATRIX=1)
endif()

if(MVS_FOUND)
    target_link_libraries(rm_aim_core PUBLIC MVS::MVS)
    target_compile_definitions(rm_aim_core PUBLIC RM_AIM_WITH_MVS=1)
endif()

if(RM_AIM_WITH_CUDA)
    target_link_libraries(rm_aim_core PUBLIC CUDA::cudart)
    target_compile_definitions(rm_aim_core PUBLIC RM_AIM_WITH_CUDA=1)
endif()

if(OpenCV_FOUND)
    target_link_libraries(rm_aim_core PUBLIC ${OpenCV_LIBS})
    target_include_directories(rm_aim_core PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(rm_aim_core PUBLIC RM_AIM_WITH_OPENCV=1)
endif()

if(Eigen3_FOUND)
    target_link_libraries(rm_aim_core PUBLIC Eigen3::Eigen)
    target_compile_definitions(rm_aim_core PUBLIC RM_AIM_WITH_EIGEN=1)
endif()

# ---------------------------------------------------------------------------
# 主程序可执行文件
# ---------------------------------------------------------------------------
add_executable(vernier src/main.cpp)
target_link_libraries(vernier PRIVATE rm_aim_core)

add_executable(vernier_follow src/follow_main.cpp)
target_link_libraries(vernier_follow PRIVATE rm_aim_core)

# ---------------------------------------------------------------------------
# 可选标定工具
# ---------------------------------------------------------------------------
if(RM_AIM_BUILD_CALIB)
    if(OpenCV_FOUND AND Eigen3_FOUND)
        add_executable(vernier_calib src/calib_main.cpp)
        target_link_libraries(vernier_calib PRIVATE rm_aim_core)
        message(STATUS "Calibration tool:  ENABLED (vernier_calib)")
    else()
        message(WARNING "RM_AIM_BUILD_CALIB=ON 需要 OpenCV 与 Eigen3，已跳过 vernier_calib")
    endif()
endif()

# ---------------------------------------------------------------------------
# 单元测试与性能基准
# ---------------------------------------------------------------------------
if(RM_AIM_BUILD_TESTS)
    include(CTest)
    enable_testing()

    add_executable(test_core_hal tests/test_core_hal.cpp)
    target_link_libraries(test_core_hal PRIVATE rm_aim_core)
    add_test(NAME core_hal COMMAND test_core_hal "${CMAKE_CURRENT_SOURCE_DIR}/config/example.yaml")

    add_executable(test_perception tests/test_perception.cpp)
    target_link_libraries(test_perception PRIVATE rm_aim_core)
    add_test(NAME perception COMMAND test_perception "${CMAKE_CURRENT_SOURCE_DIR}/config/example.yaml")

    add_executable(test_estimation tests/test_estimation.cpp)
    target_link_libraries(test_estimation PRIVATE rm_aim_core)
    add_test(NAME estimation COMMAND test_estimation "${CMAKE_CURRENT_SOURCE_DIR}/config/example.yaml")
endif()
