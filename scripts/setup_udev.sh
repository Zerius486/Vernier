#!/usr/bin/env bash
set -euo pipefail

# 一键配置海康工业 USB3 相机与 UART/USB-CDC 串口设备权限
RULE_FILE="/etc/udev/rules.d/99-rm-auto-aim.rules"

echo "正在写入 udev 规则到 ${RULE_FILE}（需要 sudo 权限）..."
sudo tee "${RULE_FILE}" >/dev/null <<'EOF'
# 海康威视 USB3 工业相机
SUBSYSTEM=="usb", ATTR{idVendor}=="2bdf", MODE="0666", GROUP="plugdev"

# STM32 USB-CDC (/dev/ttyACM*) 与 USB 转串口 (/dev/ttyUSB*)
SUBSYSTEM=="tty", KERNEL=="ttyACM[0-9]*", MODE="0666", GROUP="dialout"
SUBSYSTEM=="tty", KERNEL=="ttyUSB[0-9]*", MODE="0666", GROUP="dialout"

# Jetson 板载硬件 UART (/dev/ttyTHS*)
SUBSYSTEM=="tty", KERNEL=="ttyTHS[0-9]*", MODE="0666", GROUP="dialout"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger

echo "配置完成。建议同时执行：sudo usermod -aG dialout,plugdev \$USER"
