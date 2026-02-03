#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_experiment.py - Clean Run v1 统一实验入口（new.md 对齐）

目标：
  - 统一入口：支持 --model rwkv|lstm|gru 与 --use_graph/--no-use_graph
  - 固化 Clean Run v1 默认时间切分与超参（具体默认在 training/train_full.py 中）
  - 结果输出到 outputs/final_run/<exp_name>/（避免不同实验互相覆盖）

示例：
  python run_experiment.py --model rwkv --use_graph
  python run_experiment.py --model rwkv --no-use_graph
  python run_experiment.py --model lstm --use_graph
  python run_experiment.py --model gru --use_graph
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean Run v1 experiment runner (Graph-RWKV)")
    p.add_argument(
        "--model",
        type=str,
        default="rwkv",
        choices=["rwkv", "lstm", "gru"],
        help="时间编码器: rwkv|lstm|gru",
    )
    p.add_argument(
        "--use_graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用图结构（GAT）。w/o_graph 设为 --no-use_graph",
    )
    p.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="实验名（默认自动生成）",
    )
    p.add_argument(
        "--run_root",
        type=str,
        default=None,
        help="输出根目录（默认：paper/outputs/final_run）",
    )
    p.add_argument(
        "--data",
        type=str,
        default=None,
        help="Final_Model_Data.csv 路径（默认：paper/data/processed/Final_Model_Data.csv）",
    )
    p.add_argument(
        "--graph",
        type=str,
        default=None,
        help="Graph_Adjacency.npy 路径（默认：paper/data/processed/Graph_Adjacency.npy）",
    )
    p.add_argument(
        "--graph_tickers",
        type=str,
        default=None,
        help="Graph_Tickers.json 路径（默认：paper/data/processed/Graph_Tickers.json）",
    )
    p.add_argument(
        "--feature_columns",
        type=str,
        default=None,
        help="feature_columns.json 路径（默认：paper/data/processed/feature_columns.json）",
    )
    p.add_argument(
        "--features_parquet",
        type=str,
        default=None,
        help="sp500_alpha158_features.parquet 路径（默认：paper/data/processed/sp500_alpha158_features.parquet）",
    )
    return p.parse_args()


def _default_exp_name(model: str, use_graph: bool) -> str:
    if model == "rwkv" and use_graph:
        return "full"
    if model == "rwkv" and (not use_graph):
        return "w_o_graph"
    if model == "lstm" and use_graph:
        return "ablation_lstm_backbone"
    if model == "gru" and use_graph:
        return "ablation_gru_backbone"
    # 兜底：仍能运行
    return f"{model}_{'graph' if use_graph else 'no_graph'}"


def main() -> None:
    args = _parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data", "processed")

    exp_name = args.exp_name or _default_exp_name(args.model, bool(args.use_graph))

    run_root = args.run_root or os.path.join(script_dir, "outputs", "final_run")
    out_dir = os.path.join(run_root, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = args.data or os.path.join(data_dir, "Final_Model_Data.csv")
    graph_path = args.graph or os.path.join(data_dir, "Graph_Adjacency.npy")
    graph_tickers_path = args.graph_tickers or os.path.join(data_dir, "Graph_Tickers.json")
    feature_columns_path = args.feature_columns or os.path.join(data_dir, "feature_columns.json")
    features_path = args.features_parquet or os.path.join(data_dir, "sp500_alpha158_features.parquet")

    # 训练入口：复用 training/train_full.py
    sys.path.insert(0, script_dir)
    from training import train_full  # noqa: WPS433

    # 覆盖配置：只覆盖与“实验可复现”强相关的项；其余默认留在 train_full.PAPER_CONFIG 中
    overrides = {
        # IO
        "csv_path": csv_path,
        "graph_path": graph_path,
        "graph_tickers_path": graph_tickers_path,
        "feature_columns_path": feature_columns_path,
        # dataset 会自动从 parquet 合并缺失特征；这里提供路径
        # （train_full 会把 CONFIG 传到 dataset；dataset 默认同目录查找，这里显式写入便于复现）
        "features_path": features_path,
        # output isolation
        "output_dir": out_dir,
        "experiment_name": exp_name,
        "checkpoint_name": "best_model.pth",
        # ablation switches
        "use_graph": bool(args.use_graph),
        "temporal_backend": str(args.model).lower(),
        # 防止意外偏离 Clean Run：显式使用三阶段日期切分
        "use_date_split": True,
    }

    # 把 overrides 记录到输出目录，便于排查（不依赖 logger）
    try:
        import json

        with open(os.path.join(out_dir, "overrides.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "overrides": overrides,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    except Exception:
        pass

    # 应用并启动训练
    train_full.CONFIG.update(overrides)
    train_full.main()


if __name__ == "__main__":
    main()

