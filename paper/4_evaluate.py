#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4_evaluate.py - 单模型评估脚本（可直接运行）

用法：
  方式1：直接运行（默认配置）
  方式2：命令行覆盖参数
    python 4_evaluate.py --checkpoint ./outputs/checkpoints/best_model.pth --test_data ./data/processed/Final_Model_Data.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

from utils.logging_utils import setup_logging

# ================= 配置（可直接修改）=================
CHECKPOINT_PATH = "./outputs/checkpoints/best_model.pth"
DATA_CSV_PATH = "./data/processed/Final_Model_Data.csv"
GRAPH_PATH = "./data/processed/Graph_Adjacency.npy"
GRAPH_TICKERS_PATH = "./data/processed/Graph_Tickers.json"

BATCH_SIZE = 2048
USE_GRAPH = True
TOP_K = 10
ANNUALIZATION = 252

MODEL_N_EMBD = 256
MODEL_N_LAYERS = 3
MODEL_GNN_EMBD = 64
SEQ_LEN = 30
# ====================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 Graph-RWKV 模型")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH, help="模型权重路径")
    parser.add_argument("--test_data", type=str, default=DATA_CSV_PATH, help="测试数据 CSV 路径")
    parser.add_argument("--graph", type=str, default=GRAPH_PATH, help="图谱邻接矩阵路径")
    parser.add_argument("--graph_tickers", type=str, default=GRAPH_TICKERS_PATH, help="图谱 tickers 路径")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="评估批大小")
    parser.add_argument("--use_graph", action=argparse.BooleanOptionalAction, default=USE_GRAPH, help="是否使用图模型")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Top-K Long-Short 的 K")
    parser.add_argument("--annualization", type=int, default=ANNUALIZATION, help="年化系数（交易日）")
    return parser.parse_args()


def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)

    true_dir = np.sign(y_true)
    pred_dir = np.sign(y_pred)
    dir_acc = float(np.mean(true_dir == pred_dir))

    ic = None
    rank_ic = None
    try:
        ic_val, _ = pearsonr(y_pred, y_true)
        ic = float(ic_val)
    except Exception:
        pass
    try:
        ric_val, _ = spearmanr(y_pred, y_true)
        rank_ic = float(ric_val)
    except Exception:
        pass

    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
        "dir_acc": dir_acc,
        "ic": ic,
        "rank_ic": rank_ic,
        "n_samples": int(len(y_true)),
    }


def _calc_backtest_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: list[str],
    top_k: int = 10,
    annualization: int = 252,
) -> dict:
    """Top-K Long-Short 回测指标。"""
    if not dates:
        return {}

    df = {
        "date": dates,
        "y_true": y_true.flatten(),
        "y_pred": y_pred.flatten(),
    }
    df = np.rec.fromarrays([df["date"], df["y_true"], df["y_pred"]], names="date,y_true,y_pred")
    df = np.array(df)

    daily_returns = []
    unique_dates = np.unique(df["date"])
    for d in unique_dates:
        mask = df["date"] == d
        day = df[mask]
        if len(day) < top_k * 2:
            continue
        order = np.argsort(day["y_pred"])
        short_idx = order[:top_k]
        long_idx = order[-top_k:]
        long_ret = np.expm1(day["y_true"][long_idx]).mean()
        short_ret = np.expm1(day["y_true"][short_idx]).mean()
        daily_returns.append(long_ret - short_ret)

    if not daily_returns:
        return {}

    daily_returns = np.array(daily_returns, dtype=np.float64)
    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std(ddof=1) if daily_returns.size > 1 else 0.0

    cumulative = np.cumprod(1.0 + daily_returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak

    ann_return = cumulative[-1] ** (annualization / len(daily_returns)) - 1.0
    sharpe = (mean_ret / std_ret) * np.sqrt(annualization) if std_ret > 1e-12 else None
    max_dd = float(drawdown.min()) if drawdown.size else 0.0

    return {
        "topk_k": int(top_k),
        "annual_return": float(ann_return),
        "sharpe": float(sharpe) if sharpe is not None else None,
        "max_drawdown": max_dd,
        "n_days": int(len(daily_returns)),
    }


def _load_model_config_from_log(checkpoint_path: str) -> dict:
    """尝试从训练日志中读取模型配置，避免评估维度不匹配。"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path)))
    log_path = os.path.join(base_dir, "logs", "training_losses_full.json")
    if not os.path.exists(log_path):
        return {}
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = data.get("config", {}) if isinstance(data, dict) else {}
        return {
            "n_embd": cfg.get("n_embd"),
            "n_layers": cfg.get("n_layers"),
            "gnn_embd": cfg.get("gnn_embd"),
            "seq_len": cfg.get("seq_len"),
        }
    except Exception:
        return {}


def main() -> None:
    args = _parse_args()
    log_ctx = setup_logging(os.path.dirname(args.checkpoint) or ".", "evaluate")
    logger = log_ctx.logger

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from dataProcessed.dataset import FinancialDataset
        from models.gnn_model import GraphRWKV_GNN_Model
        from models.base_model import GraphRWKV_Model

        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"找不到模型文件: {args.checkpoint}")
        if not os.path.exists(args.test_data):
            raise FileNotFoundError(f"找不到测试数据: {args.test_data}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("设备: %s", device)

        cfg_override = _load_model_config_from_log(args.checkpoint)
        n_embd = int(cfg_override.get("n_embd") or MODEL_N_EMBD)
        n_layers = int(cfg_override.get("n_layers") or MODEL_N_LAYERS)
        gnn_embd = int(cfg_override.get("gnn_embd") or MODEL_GNN_EMBD)
        seq_len = int(cfg_override.get("seq_len") or SEQ_LEN)

        train_dataset = FinancialDataset(args.test_data, seq_len=seq_len, mode="train")
        test_dataset = FinancialDataset(
            args.test_data,
            seq_len=seq_len,
            mode="test",
            scaler=train_dataset.scaler,
            vol_stats=getattr(train_dataset, "vol_stats", None),
        )
        input_dim = len(train_dataset.feature_cols)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(8, max(2, (os.cpu_count() or 12) - 2)),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            prefetch_factor=4,
        )

        if args.use_graph:
            if not os.path.exists(args.graph):
                raise FileNotFoundError(f"找不到图谱文件: {args.graph}")
            adj_matrix = np.load(args.graph)
            num_nodes = adj_matrix.shape[0]
            model = GraphRWKV_GNN_Model(
                input_dim=input_dim,
                n_embd=n_embd,
                n_layers=n_layers,
                num_nodes=num_nodes,
                adj_matrix=adj_matrix,
                gnn_embd=gnn_embd,
            ).to(device)
        else:
            model = GraphRWKV_Model(
                input_dim=input_dim,
                n_embd=n_embd,
                n_layers=n_layers,
            ).to(device)

        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()

        all_preds = []
        all_labels = []
        all_dates = []

        with torch.inference_mode():
            for batch in test_loader:
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                vol = batch.get("vol")
                if vol is not None:
                    vol = vol.to(device, non_blocking=True)
                node_idx = batch.get("node_indices")
                if node_idx is not None:
                    node_idx = node_idx.to(device, non_blocking=True)
                dates = batch.get("target_date")
                if dates is not None:
                    all_dates.extend(list(dates))

                if args.use_graph:
                    preds = model(x, vol=vol, node_indices=node_idx)
                else:
                    preds = model(x, vol=vol)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_labels, axis=0)
        metrics = _calc_metrics(y_true, y_pred)
        backtest = _calc_backtest_metrics(
            y_true,
            y_pred,
            all_dates,
            top_k=args.top_k,
            annualization=args.annualization,
        )

        logger.info("评估完成: %s", metrics)
        if backtest:
            logger.info("回测指标: %s", backtest)
    except Exception as exc:
        logger.exception("评估失败: %s", exc)
        raise
    finally:
        log_ctx.restore_print()


if __name__ == "__main__":
    main()
