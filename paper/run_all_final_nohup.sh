#!/usr/bin/env bash
set -euo pipefail

# 后台启动 Clean Run v1（SSH 断开不影响）
# - 日志：outputs/final_run/run_all_final_<timestamp>.log
# - PID ：outputs/final_run/run_all_final_<timestamp>.pid
#
# 用法：
#   cd /root/autodl-tmp/paper
#   bash run_all_final_nohup.sh
#
# 跳过建图（可选）：
#   SKIP_BUILD_GRAPH=1 bash run_all_final_nohup.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${ROOT_DIR}/outputs/final_run"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${ROOT_DIR}/outputs/final_run/run_all_final_${TS}.log"
PID_PATH="${ROOT_DIR}/outputs/final_run/run_all_final_${TS}.pid"

echo ">>> launching in background"
echo "    log: ${LOG_PATH}"

# 关键：用 bash 显式执行，确保环境变量（如 SKIP_BUILD_GRAPH/SPLIT_DATE）生效
nohup bash "${ROOT_DIR}/run_all_final.sh" > "${LOG_PATH}" 2>&1 &
PID="$!"
echo "${PID}" > "${PID_PATH}"

echo ">>> pid: ${PID} (saved to ${PID_PATH})"
echo ">>> follow logs:"
echo "    tail -f \"${LOG_PATH}\""

