#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3_train.py - Graph-RWKV 全量模型训练脚本（可直接运行）

默认配置已对齐 new.md（训练 2018-01-01~2020-06-30，验证 2020-07-01~2020-12-31，测试 2021-01-01~2023-12-31）。
直接运行即可，无需传参：

  前台：  python 3_train.py
  后台：  nohup python 3_train.py > train.log 2>&1 &

需要改参数时再在命令行传入，例如：--batch_size 512 --epochs 20
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

# 抑制 PyG 可选扩展库加载失败警告（不影响训练）
warnings.filterwarnings("ignore", message=".*An issue occurred while importing.*", category=UserWarning)

from utils.logging_utils import setup_logging

# ================= 默认配置（参考顶会 KDD/AAAI：窄而浅 + 强正则，new.md 时间切分）=================
# Rationale: 金融信噪比低，顶会（MASTER/HIST/StemGNN 等）常用「小隐维 + 高 Dropout + 强 L2」防过拟合。
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV_PATH = os.path.join(_SCRIPT_DIR, "data", "processed", "Final_Model_Data.csv")
GRAPH_PATH = os.path.join(_SCRIPT_DIR, "data", "processed", "Graph_Adjacency.npy")
GRAPH_TICKERS_PATH = os.path.join(_SCRIPT_DIR, "data", "processed", "Graph_Tickers.json")
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "outputs")

# [Model] 窄而浅，防止记忆噪声（顶会常用 64/128 维、2~3 层）
MODEL_N_EMBD = 64        # 384->64，金融特征 <200 维不宜过大隐空间
MODEL_N_LAYERS = 2       # 4->2，浅层泛化更好，避免 over-smoothing
MODEL_GNN_EMBD = 32      # 96->32，图特征不必过大
DROPOUT = 0.3            # 0.08->0.3，强正则强迫学习鲁棒特征

# [Training] 小模型需稍大学习率 + 强 L2，配合 Scheduler 下降
SEQ_LEN = 60             # 保持 60，长序列利于 RWKV
TRAIN_BATCH_SIZE = 1024  # 保持
TRAIN_EPOCHS = 50        # 小模型收敛快，略增轮次微调
TRAIN_LR = 5e-3          # 
WEIGHT_DECAY = 1e-3      # 5e-6->1e-3，强 L2 过滤噪声
EARLY_STOP_PATIENCE = 15 # 给足耐心

TRAIN_NUM_WORKERS = 10
TRAIN_PREFETCH_FACTOR = 4
TRAIN_PIN_MEMORY = True
TRAIN_PERSISTENT_WORKERS = True
USE_AMP = True
USE_COMPILE = False
TEMPORAL_BACKEND = "rwkv"
USE_RANK_LOSS = True
# [Loss] 选股目标对齐：排序为主、MSE 为辅（顶会 Rank-LSTM 等常用 1.0）
RANK_LOSS_WEIGHT = 1.0   # 0.2->1.0，优化目标对齐 IC/RankIC
RANK_LOSS_MAX_PAIRS = 8192
RANK_LOSS_TYPE = "rankic"
RANKIC_TAU = 1.0
RANKIC_MAX_ITEMS = 1024

# 滚动窗口（默认关闭；若开启则与 new.md 一致：训练截止 2020-06-30）
USE_WALK_FORWARD = False
WALK_FORWARD_TRAIN_START = "2018-01-01"
WALK_FORWARD_TRAIN_END = "2020-06-30"   # 与训练集结束一致，防泄露
WALK_FORWARD_TEST_START = "2021-01-01"
WALK_FORWARD_TEST_END = "2023-12-31"
WALK_FORWARD_FREQ = "Q"
# ====================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 Graph-RWKV 全量模型")
    parser.add_argument("--data", type=str, default=DATA_CSV_PATH, help="训练数据 CSV 路径")
    parser.add_argument("--graph", type=str, default=GRAPH_PATH, help="图谱邻接矩阵路径")
    parser.add_argument("--graph_tickers", type=str, default=GRAPH_TICKERS_PATH, help="图谱 tickers 路径")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--batch_size", type=int, default=TRAIN_BATCH_SIZE, help="批大小")
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS, help="训练轮数")
    parser.add_argument("--lr", type=float, default=TRAIN_LR, help="学习率")
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN, help="输入序列长度（日）")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="AdamW 权重衰减")
    parser.add_argument("--early_stop_patience", type=int, default=EARLY_STOP_PATIENCE, help="早停耐心（epoch）")
    parser.add_argument("--num_workers", type=int, default=TRAIN_NUM_WORKERS, help="DataLoader 进程数")
    parser.add_argument("--prefetch_factor", type=int, default=TRAIN_PREFETCH_FACTOR, help="DataLoader 预取因子")
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=TRAIN_PIN_MEMORY, help="是否 pin_memory")
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=TRAIN_PERSISTENT_WORKERS, help="是否 persistent_workers")
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=USE_AMP, help="是否启用 AMP")
    parser.add_argument("--use_compile", action=argparse.BooleanOptionalAction, default=USE_COMPILE, help="是否启用 torch.compile")
    parser.add_argument("--temporal_backend", type=str, default=TEMPORAL_BACKEND, help="时间编码器: rwkv | lstm | gru")
    parser.add_argument("--use_rank_loss", action=argparse.BooleanOptionalAction, default=USE_RANK_LOSS, help="是否启用 RankLoss")
    parser.add_argument("--rank_loss_weight", type=float, default=RANK_LOSS_WEIGHT, help="RankLoss 权重")
    parser.add_argument("--rank_loss_max_pairs", type=int, default=RANK_LOSS_MAX_PAIRS, help="RankNet pairwise max pairs")
    parser.add_argument("--rank_loss_type", type=str, default=RANK_LOSS_TYPE, choices=["pairwise", "rankic"], help="RankLoss 类型")
    parser.add_argument("--rankic_tau", type=float, default=RANKIC_TAU, help="RankIC soft-rank 温度")
    parser.add_argument("--rankic_max_items", type=int, default=RANKIC_MAX_ITEMS, help="RankIC max items")
    parser.add_argument("--use_walk_forward", action=argparse.BooleanOptionalAction, default=USE_WALK_FORWARD, help="是否启用滚动窗口验证")
    parser.add_argument("--wf_train_start", type=str, default=WALK_FORWARD_TRAIN_START, help="滚动窗口训练起始日期")
    parser.add_argument("--wf_train_end", type=str, default=WALK_FORWARD_TRAIN_END, help="滚动窗口训练结束日期")
    parser.add_argument("--wf_test_start", type=str, default=WALK_FORWARD_TEST_START, help="滚动窗口测试起始日期")
    parser.add_argument("--wf_test_end", type=str, default=WALK_FORWARD_TEST_END, help="滚动窗口测试结束日期")
    parser.add_argument("--wf_freq", type=str, default=WALK_FORWARD_FREQ, help="滚动窗口步长频率（如 Q）")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    log_ctx = setup_logging(args.output, "train")
    logger = log_ctx.logger

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from training import train_full

        overrides = {
            "csv_path": args.data,
            "n_embd": MODEL_N_EMBD,
            "n_layers": MODEL_N_LAYERS,
            "gnn_embd": MODEL_GNN_EMBD,
            "dropout": DROPOUT,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "seq_len": args.seq_len,
            "weight_decay": args.weight_decay,
            "early_stop_patience": args.early_stop_patience,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "pin_memory": bool(args.pin_memory),
            "persistent_workers": bool(args.persistent_workers),
            "use_amp": bool(args.use_amp),
            "use_compile": bool(args.use_compile),
            "output_dir": args.output,
            "graph_path": args.graph,
            "graph_tickers_path": args.graph_tickers,
            "experiment_name": "full",
            "checkpoint_name": "best_model.pth",
            "use_graph": True,
            "temporal_backend": args.temporal_backend,
            "use_rank_loss": bool(args.use_rank_loss),
            "rank_loss_weight": float(args.rank_loss_weight),
            "rank_loss_max_pairs": int(args.rank_loss_max_pairs),
            "rank_loss_type": args.rank_loss_type,
            "rankic_tau": float(args.rankic_tau),
            "rankic_max_items": int(args.rankic_max_items),
            "use_walk_forward": bool(args.use_walk_forward),
            "walk_forward_train_start": args.wf_train_start,
            "walk_forward_train_end": args.wf_train_end,
            "walk_forward_test_start": args.wf_test_start,
            "walk_forward_test_end": args.wf_test_end,
            "walk_forward_freq": args.wf_freq,
        }

        logger.info("训练配置覆盖: %s", overrides)
        train_full.CONFIG.update(overrides)
        train_full.main()
    except Exception as exc:
        logger.exception("训练失败: %s", exc)
        raise
    finally:
        log_ctx.restore_print()


if __name__ == "__main__":
    main()
