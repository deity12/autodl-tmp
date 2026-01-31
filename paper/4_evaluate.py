#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4_evaluate.py - 单模型评估脚本（可直接运行）

用法：
  方式1：直接运行（默认配置）
 方式2：命令行覆盖参数
    python 4_evaluate.py --checkpoint ./outputs/checkpoints/best_model.pth --test_data ./paper/data/processed/Final_Model_Data.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Optional

# 抑制 PyG 可选扩展库加载失败警告（不影响主流程，仅部分算子回退到 PyTorch）
warnings.filterwarnings("ignore", message=".*An issue occurred while importing.*", category=UserWarning)

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

from utils.logging_utils import setup_logging

# ================= 配置（可直接修改）=================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(_SCRIPT_DIR, "outputs", "checkpoints", "best_model.pth")
DATA_CSV_PATH = os.path.join(_SCRIPT_DIR, "data", "processed", "Final_Model_Data.csv")
GRAPH_PATH = os.path.join(_SCRIPT_DIR, "data", "processed", "Graph_Adjacency.npy")
GRAPH_TICKERS_PATH = os.path.join(_SCRIPT_DIR, "data", "processed", "Graph_Tickers.json")

BATCH_SIZE = 2048
USE_GRAPH = True
TOP_K = 30  # 与论文一致：Top-30 Long-Short
ANNUALIZATION = 252

# 与 train_full 一致：scaler 用训练期，测试集用 2021-01-01~2023-12-31（论文报告区间）
TRAIN_START = "2018-01-01"
TRAIN_END = "2020-06-30"
TEST_START = "2021-01-01"
TEST_END = "2023-12-31"

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
    parser.add_argument("--train_start", type=str, default=TRAIN_START, help="训练期起始（用于 fit scaler）")
    parser.add_argument("--train_end", type=str, default=TRAIN_END, help="训练期结束")
    parser.add_argument("--test_start", type=str, default=TEST_START, help="测试集起始（论文报告区间）")
    parser.add_argument("--test_end", type=str, default=TEST_END, help="测试集结束")
    return parser.parse_args()


def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, dates: list[str] = None) -> dict:
    """计算评估指标，包括每日 IC/RankIC 和 ICIR/RankICIR（符合顶会标准）"""
    from collections import defaultdict
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)

    true_dir = np.sign(y_true)
    pred_dir = np.sign(y_pred)
    dir_acc = float(np.mean(true_dir == pred_dir))

    # 计算整体 IC/RankIC
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

    # 计算每日 IC/RankIC，然后聚合为 ICIR/RankICIR（顶会标准做法）
    icir = None
    rank_icir = None
    daily_ic = None
    daily_rankic = None
    
    if dates is not None and len(dates) == len(y_true):
        buckets_true = defaultdict(list)
        buckets_pred = defaultdict(list)
        for t, p, d in zip(y_true, y_pred, dates):
            buckets_true[d].append(float(t))
            buckets_pred[d].append(float(p))
        
        ic_list = []
        rankic_list = []
        for d in buckets_true.keys():
            yt = np.asarray(buckets_true[d], dtype=np.float64)
            yp = np.asarray(buckets_pred[d], dtype=np.float64)
            if yt.size < 2:
                continue
            try:
                ic_val, _ = pearsonr(yp, yt)
                ic_list.append(float(ic_val))
            except Exception:
                pass
            try:
                ric_val, _ = spearmanr(yp, yt)
                rankic_list.append(float(ric_val))
            except Exception:
                pass
        
        if ic_list:
            daily_ic = float(np.mean(ic_list))
            if np.std(ic_list) > 1e-8:
                icir = float(np.mean(ic_list) / np.std(ic_list))
        
        if rankic_list:
            daily_rankic = float(np.mean(rankic_list))
            if np.std(rankic_list) > 1e-8:
                rank_icir = float(np.mean(rankic_list) / np.std(rankic_list))

    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
        "dir_acc": dir_acc,
        "ic": ic,
        "rank_ic": rank_ic,
        "daily_ic": daily_ic,        # 每日 IC 均值
        "daily_rankic": daily_rankic, # 每日 RankIC 均值
        "icir": icir,                # IC 稳定性指标
        "rank_icir": rank_icir,      # RankIC 稳定性指标
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

        train_dataset = FinancialDataset(
            args.test_data,
            seq_len=seq_len,
            mode="train",
            start_date=args.train_start,
            end_date=args.train_end,
        )
        test_dataset = FinancialDataset(
            args.test_data,
            seq_len=seq_len,
            mode="test",
            scaler=train_dataset.scaler,
            vol_stats=getattr(train_dataset, "vol_stats", None),
            start_date=args.test_start,
            end_date=args.test_end,
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
        metrics = _calc_metrics(y_true, y_pred, all_dates)  # 传入 dates 以计算 ICIR
        backtest = _calc_backtest_metrics(
            y_true,
            y_pred,
            all_dates,
            top_k=args.top_k,
            annualization=args.annualization,
        )

        # 格式化输出（符合顶会论文表格格式）
        print("\n" + "=" * 60)
        print("📊 Graph-RWKV 模型评估结果（测试集）")
        print(f"   测试集区间: {args.test_start} ~ {args.test_end}（与 train_full 一致）")
        print("=" * 60)
        print(f"\n【预测能力指标】")
        print(f"  IC (每日均值):        {metrics.get('daily_ic', 'N/A'):.4f}" if metrics.get('daily_ic') else "  IC (每日均值):        N/A")
        print(f"  RankIC (每日均值):    {metrics.get('daily_rankic', 'N/A'):.4f}" if metrics.get('daily_rankic') else "  RankIC (每日均值):    N/A")
        print(f"  ICIR:                 {metrics.get('icir', 'N/A'):.4f}" if metrics.get('icir') else "  ICIR:                 N/A")
        print(f"  RankICIR:             {metrics.get('rank_icir', 'N/A'):.4f}" if metrics.get('rank_icir') else "  RankICIR:             N/A")
        print(f"\n【回归指标】")
        print(f"  MSE:                  {metrics['mse']:.6f}")
        print(f"  RMSE:                 {metrics['rmse']:.6f}")
        print(f"  R²:                   {metrics['r2']:.4f}")
        print(f"  方向准确率:           {metrics['dir_acc']:.2%}")
        print(f"  样本数:               {metrics['n_samples']}")
        
        if backtest:
            print(f"\n【Top-{args.top_k} Long-Short 回测】")
            print(f"  年化收益率:           {backtest.get('annual_return', 'N/A'):.2%}" if backtest.get('annual_return') is not None else "  年化收益率:           N/A")
            print(f"  夏普比率:             {backtest.get('sharpe', 'N/A'):.4f}" if backtest.get('sharpe') else "  夏普比率:             N/A")
            print(f"  最大回撤:             {backtest.get('max_drawdown', 'N/A'):.2%}" if backtest.get('max_drawdown') is not None else "  最大回撤:             N/A")
            print(f"  交易天数:             {backtest.get('n_days', 'N/A')}")
        
        print("=" * 60)
        
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
