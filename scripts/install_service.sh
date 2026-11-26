#!/usr/bin/env bash
set -euo pipefail

# 一键安装 Vernier systemd 开机自启与崩溃守护服务
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 ]]; then
    echo "用法: $0 <config.yaml> [bin_path]" >&2
    echo "示例: $0 config/infantry_3.yaml" >&2
    exit 1
fi
CFG_PATH="$(realpath "$1")"
BIN_PATH="${2:-${REPO_DIR}/build/vernier}"
RUN_USER="${SUDO_USER:-$USER}"
SERVICE_FILE="/etc/systemd/system/vernier.service"

if [[ ! -f "${CFG_PATH}" ]]; then
    echo "错误：未找到配置文件 ${CFG_PATH}。" >&2
    exit 1
fi

if [[ ! -x "${BIN_PATH}" ]]; then
    echo "错误：未找到可执行文件 ${BIN_PATH}，请先编译项目。" >&2
    exit 1
fi

echo "正在安装 systemd 服务到 ${SERVICE_FILE}..."
sudo tee "${SERVICE_FILE}" >/dev/null <<EOF
[Unit]
Description=Vernier RoboMaster Auto-Aim Service
After=network.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${REPO_DIR}
ExecStart=${BIN_PATH} ${CFG_PATH}
Restart=always
RestartSec=1
Nice=-10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now vernier.service

echo "Vernier 守护服务已启动！常用命令："
echo "  查看实时状态与日志: journalctl -u vernier -f"
echo "  停止服务:           sudo systemctl stop vernier"
echo "  重启服务:           sudo systemctl restart vernier"
