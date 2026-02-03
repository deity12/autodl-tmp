#!/usr/bin/env bash
set -euo pipefail

# Clean Run v1 一键脚本（new.md 对齐）
# - 先建图（可用 SKIP_BUILD_GRAPH=1 跳过）
# - 再跑 4 个实验
# - 最后生成 outputs/final_run/summary.csv

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SPLIT_DATE="${SPLIT_DATE:-2021-12-31}"
SKIP_BUILD_GRAPH="${SKIP_BUILD_GRAPH:-0}"

echo ">>> [Clean Run v1] split_date=${SPLIT_DATE}"

if [[ "${SKIP_BUILD_GRAPH}" != "1" ]]; then
  echo ">>> [Step 1] Build graph (strict cutoff)"
  python "${ROOT_DIR}/2_build_graph.py" --split_date "${SPLIT_DATE}"
else
  echo ">>> [Step 1] Build graph skipped (SKIP_BUILD_GRAPH=1)"
fi

echo ">>> [Step 2] Run experiments"

python "${ROOT_DIR}/run_experiment.py" --model rwkv --use_graph --exp_name full
python "${ROOT_DIR}/run_experiment.py" --model rwkv --no-use_graph --exp_name w_o_graph
python "${ROOT_DIR}/run_experiment.py" --model lstm --use_graph --exp_name ablation_lstm_backbone
python "${ROOT_DIR}/run_experiment.py" --model gru --use_graph --exp_name ablation_gru_backbone

echo ">>> [Step 3] Summarize"
python "${ROOT_DIR}/evaluation/summarize_final_run.py"

echo ">>> Done. See: ${ROOT_DIR}/outputs/final_run/summary.csv"

