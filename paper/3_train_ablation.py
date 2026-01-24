#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3_train_ablation.py - 消融实验训练脚本（可直接运行）

支持的消融类型：
  - w/o_graph
  - w/o_semantic
  - w/o_statistical
  - w/o_sentiment
  - all（批量运行）
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from utils.logging_utils import setup_logging

# ================= 配置（可直接修改）=================
DATA_CSV_PATH = "./data/processed/Final_Model_Data.csv"
GRAPH_DIR = "./data/processed"
GRAPH_TICKERS_PATH = "./data/processed/Graph_Tickers.json"
OUTPUT_DIR = "./outputs"

MODEL_N_EMBD = 256
MODEL_N_LAYERS = 3
MODEL_GNN_EMBD = 64
DROPOUT = 0.1

TRAIN_BATCH_SIZE = 1024
TRAIN_EPOCHS = 30
TRAIN_LR = 3e-4
TRAIN_NUM_WORKERS = 10
TRAIN_PREFETCH_FACTOR = 4
TRAIN_PIN_MEMORY = True
TRAIN_PERSISTENT_WORKERS = True
USE_AMP = True
USE_COMPILE = True
TEMPORAL_BACKEND = "rwkv"
# ====================================================


ABLATION_TYPES = ["w/o_graph", "w/o_semantic", "w/o_statistical", "w/o_sentiment", "lstm_hybrid", "gru_hybrid"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph-RWKV 消融实验训练")
    parser.add_argument("--data", type=str, default=DATA_CSV_PATH, help="训练数据 CSV 路径")
    parser.add_argument("--graph_dir", type=str, default=GRAPH_DIR, help="图谱目录")
    parser.add_argument("--graph_tickers", type=str, default=GRAPH_TICKERS_PATH, help="图谱 tickers 路径")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--ablation", type=str, default="all", help="消融类型: w/o_graph | w/o_semantic | w/o_statistical | w/o_sentiment | all")
    parser.add_argument("--batch_size", type=int, default=TRAIN_BATCH_SIZE, help="批大小")
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS, help="训练轮数")
    parser.add_argument("--lr", type=float, default=TRAIN_LR, help="学习率")
    parser.add_argument("--num_workers", type=int, default=TRAIN_NUM_WORKERS, help="DataLoader 进程数")
    parser.add_argument("--prefetch_factor", type=int, default=TRAIN_PREFETCH_FACTOR, help="DataLoader 预取因子")
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=TRAIN_PIN_MEMORY, help="是否 pin_memory")
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=TRAIN_PERSISTENT_WORKERS, help="是否 persistent_workers")
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=USE_AMP, help="是否启用 AMP")
    parser.add_argument("--use_compile", action=argparse.BooleanOptionalAction, default=USE_COMPILE, help="是否启用 torch.compile")
    parser.add_argument("--temporal_backend", type=str, default=TEMPORAL_BACKEND, help="时间编码器: rwkv | lstm | gru")
    return parser.parse_args()


def _graph_path(graph_dir: str, name: str) -> str:
    return os.path.join(graph_dir, name)


def _ablation_runs(graph_dir: str) -> dict[str, dict]:
    return {
        "w/o_graph": {
            "use_graph": False,
            "graph_path": _graph_path(graph_dir, "Graph_Adjacency.npy"),
        },
        "w/o_semantic": {
            "use_graph": True,
            "graph_path": _graph_path(graph_dir, "Graph_Adjacency_stat.npy"),
        },
        "w/o_statistical": {
            "use_graph": True,
            "graph_path": _graph_path(graph_dir, "Graph_Adjacency_semantic.npy"),
        },
        "w/o_sentiment": {
            "use_graph": True,
            "graph_path": _graph_path(graph_dir, "Graph_Adjacency_semantic_nosent.npy"),
        },
        "lstm_hybrid": {
            "use_graph": True,
            "graph_path": _graph_path(graph_dir, "Graph_Adjacency.npy"),
            "temporal_backend": "lstm",
        },
        "gru_hybrid": {
            "use_graph": True,
            "graph_path": _graph_path(graph_dir, "Graph_Adjacency.npy"),
            "temporal_backend": "gru",
        },
    }


def _update_ablation_summary(output_dir: str, summary_path: str) -> None:
    """将单次训练摘要合并进 ablation_summary.json。"""
    if not os.path.exists(summary_path):
        return
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    merged_path = os.path.join(results_dir, "ablation_summary.json")

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        return

    try:
        if os.path.exists(merged_path):
            with open(merged_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        else:
            existing = []
    except Exception:
        existing = []

    # 用 experiment_name 去重
    exp_name = summary.get("experiment_name")
    existing = [e for e in existing if e.get("experiment_name") != exp_name]
    existing.append(summary)

    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = _parse_args()
    log_ctx = setup_logging(args.output, "train_ablation")
    logger = log_ctx.logger

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from training import train_full

        runs = _ablation_runs(args.graph_dir)
        to_run = ABLATION_TYPES if args.ablation == "all" else [args.ablation]

        for ablation in to_run:
            if ablation not in runs:
                logger.error("未知消融类型: %s", ablation)
                continue

            run_cfg = runs[ablation]
            graph_path = run_cfg["graph_path"]
            if run_cfg["use_graph"] and not os.path.exists(graph_path):
                logger.warning("缺少图谱文件: %s，跳过 %s", graph_path, ablation)
                continue

            overrides = {
                "csv_path": args.data,
                "n_embd": MODEL_N_EMBD,
                "n_layers": MODEL_N_LAYERS,
                "gnn_embd": MODEL_GNN_EMBD,
                "dropout": DROPOUT,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor,
                "pin_memory": bool(args.pin_memory),
                "persistent_workers": bool(args.persistent_workers),
                "use_amp": bool(args.use_amp),
                "use_compile": bool(args.use_compile),
                "output_dir": args.output,
                "graph_path": graph_path,
                "graph_tickers_path": args.graph_tickers,
                "experiment_name": f"ablation_{ablation}",
                "checkpoint_name": f"best_model_ablation_{ablation}.pth",
                "use_graph": bool(run_cfg["use_graph"]),
                "temporal_backend": run_cfg.get("temporal_backend", args.temporal_backend),
            }

            logger.info("开始消融训练: %s", ablation)
            logger.info("训练配置覆盖: %s", overrides)
            train_full.CONFIG.update(overrides)
            train_full.main()

            summary_path = os.path.join(
                args.output, "results", f"experiment_{overrides['experiment_name']}.json"
            )
            _update_ablation_summary(args.output, summary_path)
    except Exception as exc:
        logger.exception("消融训练失败: %s", exc)
        raise
    finally:
        log_ctx.restore_print()


if __name__ == "__main__":
    main()
