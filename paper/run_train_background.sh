#!/bin/bash
# 后台训练 + 无缓冲输出，用 tail -f train.log 可实时看进度
set -e

cd /root/autodl-tmp/paper

echo ">>> 检查 GPU..."
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
echo ""

for f in data/processed/Final_Model_Data.csv data/processed/Graph_Adjacency.npy data/processed/Graph_Tickers.json; do
    [ -f "$f" ] || { echo "❌ 缺少: $f"; exit 1; }
done
mkdir -p outputs/checkpoints outputs/logs outputs/figures

# 无缓冲 + 后台运行，保证 tail -f 是实时的
export PYTHONUNBUFFERED=1
nohup python -u 3_train.py \
    --batch_size 1024 \
    --epochs 30 \
    --lr 3e-4 \
    --num_workers 10 \
    --prefetch_factor 4 \
    --use_amp \
    --no-use_compile \
    --temporal_backend rwkv \
    --use_rank_loss \
    --rank_loss_weight 0.1 \
    --rank_loss_type rankic \
    > train.log 2>&1 &

echo $! > train.pid
echo "✅ 训练已在后台启动，PID: $(cat train.pid)"
echo ""
echo "实时查看输出："
echo "  tail -f train.log"
echo ""
echo "查看进程："
echo "  ps -p \$(cat train.pid) -o pid,etime,pcpu,pmem"
