#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Alpha158-like 特征工程（独立脚本）
================================

目标（来自 new.md / 用户确认的 Specifications）：
  - 不依赖 Qlib 自动下载器 / provider / .bin 数据
  - 以 FNSPID / 对齐后的 OHLCV 为输入，计算 Alpha158-like 技术因子
  - 优先使用 pandas-ta；若环境不支持（例如本机 Py3.9），则回退纯 pandas 实现（会在日志中明确提示）
  - 输出：
      1) sp500_alpha158_features.parquet（或用户指定路径）
      2) feature_columns.json（特征列清单，供 DataLoader 读取）

备注：
  - 默认输出 158 个 Alpha158-like 因子（与 Alpha158 论文维度更贴近）；如需更少可通过参数调小
  - 输出仅包含 [Date, Ticker, features...]，不重复写入原始 OHLCV
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd


# 默认窗口（更贴近常见 Alpha158 风格窗口设置）
DEFAULT_WINDOWS = [5, 10, 20, 30, 40, 50, 60]


def _norm_ticker(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.replace("-", ".", regex=False)


def _require_ohlcv(df: pd.DataFrame) -> None:
    need = {"Date", "Ticker", "Open", "High", "Low", "Close", "Volume"}
    missing = sorted(list(need - set(df.columns)))
    if missing:
        raise ValueError(f"缺少必要列: {missing}（需要 Date/Ticker/Open/High/Low/Close/Volume）")


def _write_parquet_or_raise(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception as e:
        raise RuntimeError(
            f"写入 Parquet 失败: {path}. "
            "请安装 pyarrow 或 fastparquet（推荐 pyarrow）。"
        ) from e


def _compute_features_pandas(df: pd.DataFrame, windows: List[int]) -> Tuple[pd.DataFrame, List[str]]:
    """
    纯 pandas 实现（fallback）：覆盖 Momentum/Volatility/Trend/Volume 的核心指标。
    说明：为保证跨环境可跑，这里不依赖 pandas-ta / ta-lib。
    """
    _require_ohlcv(df)
    out = df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"])
    out["Ticker"] = _norm_ticker(out["Ticker"])
    out = out.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    g = out.groupby("Ticker", sort=False)

    close = out["Close"].astype(float).replace(0, np.nan)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    vol = out["Volume"].astype(float)

    # 基础动量 / 波动（与后续 rolling 特征复用）
    # 显式指定 fill_method=None，避免未来 Pandas 默认行为变更导致结果漂移
    out["ret_1d"] = g["Close"].pct_change(1, fill_method=None).replace([np.inf, -np.inf], np.nan)
    out["_log_close"] = g["Close"].transform(lambda x: np.log(x.replace(0, np.nan)))
    out["log_ret_1d"] = g["_log_close"].diff()
    out["hl_range"] = (high - low) / close
    out["oc_ret"] = (out["Close"].astype(float) - out["Open"].astype(float)) / out["Open"].astype(float).replace(0, np.nan)

    feature_cols: List[str] = ["ret_1d", "log_ret_1d", "hl_range", "oc_ret"]

    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    # True Range / ATR 预备量
    prev_close = g["Close"].shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["_tr"] = tr

    typical_price = (high + low + close) / 3.0
    out["_tp"] = typical_price

    # 为了减少重复 groupby/rolling 的开销，提前准备一些序列
    prev_high = g["High"].shift(1)
    prev_low = g["Low"].shift(1)

    for w in windows:
        # Trend
        sma = g["Close"].transform(lambda x: x.rolling(w, min_periods=w).mean())
        ema = g["Close"].transform(lambda x: x.ewm(span=w, adjust=False).mean())
        out[f"close_sma_{w}"] = sma
        out[f"close_ema_{w}"] = ema
        out[f"close_sma_ratio_{w}"] = out["Close"].astype(float) / sma - 1.0
        out[f"close_ema_ratio_{w}"] = out["Close"].astype(float) / ema - 1.0

        # Momentum
        out[f"roc_{w}"] = g["Close"].pct_change(w, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out[f"mom_{w}"] = g["Close"].diff(w)
        out[f"rsi_{w}"] = g["Close"].transform(lambda x, _w=w: _rsi(x.astype(float), _w))

        # Volatility
        out[f"ret_mean_{w}"] = g["ret_1d"].transform(lambda x: x.rolling(w, min_periods=w).mean())
        out[f"ret_std_{w}"] = g["ret_1d"].transform(lambda x: x.rolling(w, min_periods=w).std())
        out[f"atr_{w}"] = g["_tr"].transform(lambda x: x.ewm(alpha=1 / w, adjust=False).mean())

        # Bollinger（用 close rolling mean/std 近似）
        c_std = g["Close"].transform(lambda x: x.rolling(w, min_periods=w).std())
        upper = sma + 2.0 * c_std
        lower = sma - 2.0 * c_std
        out[f"bb_width_{w}"] = (upper - lower) / sma
        out[f"bb_percent_{w}"] = (out["Close"].astype(float) - lower) / (upper - lower)

        # Volume
        out[f"vol_mean_{w}"] = g["Volume"].transform(lambda x: x.rolling(w, min_periods=w).mean())
        out[f"vol_std_{w}"] = g["Volume"].transform(lambda x: x.rolling(w, min_periods=w).std())
        tpv = out["_tp"] * out["Volume"].astype(float)
        tpv_sum = tpv.groupby(out["Ticker"], sort=False).transform(lambda x: x.rolling(w, min_periods=w).sum())
        vol_sum = out["Volume"].astype(float).groupby(out["Ticker"], sort=False).transform(lambda x: x.rolling(w, min_periods=w).sum())
        out[f"vwap_{w}"] = tpv_sum / vol_sum.replace(0, np.nan)

        # ========= Alpha158-like 关键增强：更多常见指标（用于补足到 158 维）=========
        # 1) MFI（Money Flow Index）
        tp = out["_tp"].astype(float)
        mf = tp * out["Volume"].astype(float)
        tp_diff = g["_tp"].diff()
        pos_mf = mf.where(tp_diff > 0, 0.0)
        neg_mf = mf.where(tp_diff < 0, 0.0).abs()
        pos_sum = pos_mf.groupby(out["Ticker"], sort=False).transform(lambda x: x.rolling(w, min_periods=w).sum())
        neg_sum = neg_mf.groupby(out["Ticker"], sort=False).transform(lambda x: x.rolling(w, min_periods=w).sum())
        mfr = pos_sum / neg_sum.replace(0, np.nan)
        out[f"mfi_{w}"] = (100.0 - (100.0 / (1.0 + mfr))).clip(lower=0.0, upper=100.0)

        # 2) CCI（Commodity Channel Index）
        tp_sma = g["_tp"].transform(lambda x: x.rolling(w, min_periods=w).mean())
        tp_mad = g["_tp"].transform(
            lambda x: x.rolling(w, min_periods=w).apply(lambda a: np.mean(np.abs(a - a.mean())), raw=True)
        )
        out[f"cci_{w}"] = (tp - tp_sma) / (0.015 * tp_mad.replace(0, np.nan))

        # 3) Williams %R（WILLR）
        hh = g["High"].transform(lambda x: x.rolling(w, min_periods=w).max())
        ll = g["Low"].transform(lambda x: x.rolling(w, min_periods=w).min())
        denom_hl = (hh - ll).replace(0, np.nan)
        out[f"willr_{w}"] = (-100.0 * (hh - out["Close"].astype(float)) / denom_hl).clip(lower=-100.0, upper=0.0)

        # 4) ADX / DI+ / DI-（简化版：rolling 平滑）
        up_move = (out["High"].astype(float) - prev_high.astype(float))
        down_move = (prev_low.astype(float) - out["Low"].astype(float))
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        tr_sum = g["_tr"].transform(lambda x: x.rolling(w, min_periods=w).sum()).replace(0, np.nan)
        plus_dm_sum = plus_dm.groupby(out["Ticker"], sort=False).transform(lambda x: x.rolling(w, min_periods=w).sum())
        minus_dm_sum = minus_dm.groupby(out["Ticker"], sort=False).transform(lambda x: x.rolling(w, min_periods=w).sum())
        plus_di = 100.0 * plus_dm_sum / tr_sum
        minus_di = 100.0 * minus_dm_sum / tr_sum
        out[f"di_plus_{w}"] = plus_di
        out[f"di_minus_{w}"] = minus_di
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        out[f"adx_{w}"] = dx.groupby(out["Ticker"], sort=False).transform(lambda x: x.rolling(w, min_periods=w).mean())

        # 5) CMF（Chaikin Money Flow）
        high_low = (out["High"].astype(float) - out["Low"].astype(float)).replace(0, np.nan)
        mfm = ((out["Close"].astype(float) - out["Low"].astype(float)) - (out["High"].astype(float) - out["Close"].astype(float))) / high_low
        mfv = (mfm.fillna(0.0) * out["Volume"].astype(float))
        mfv_sum = mfv.groupby(out["Ticker"], sort=False).transform(lambda x: x.rolling(w, min_periods=w).sum())
        vol_sum2 = out["Volume"].astype(float).groupby(out["Ticker"], sort=False).transform(lambda x: x.rolling(w, min_periods=w).sum())
        out[f"cmf_{w}"] = mfv_sum / vol_sum2.replace(0, np.nan)

        feature_cols.extend(
            [
                f"close_sma_{w}",
                f"close_ema_{w}",
                f"close_sma_ratio_{w}",
                f"close_ema_ratio_{w}",
                f"roc_{w}",
                f"mom_{w}",
                f"rsi_{w}",
                f"ret_mean_{w}",
                f"ret_std_{w}",
                f"atr_{w}",
                f"bb_width_{w}",
                f"bb_percent_{w}",
                f"vol_mean_{w}",
                f"vol_std_{w}",
                f"vwap_{w}",
                f"mfi_{w}",
                f"cci_{w}",
                f"willr_{w}",
                f"di_plus_{w}",
                f"di_minus_{w}",
                f"adx_{w}",
                f"cmf_{w}",
            ]
        )

    # MACD（固定参数）
    ema12 = g["Close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = g["Close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    macd = ema12 - ema26
    signal = macd.groupby(out["Ticker"], sort=False).transform(lambda x: x.ewm(span=9, adjust=False).mean())
    out["macd"] = macd
    out["macd_signal"] = signal
    out["macd_hist"] = macd - signal
    feature_cols.extend(["macd", "macd_signal", "macd_hist"])

    # Stochastic (14,3)
    low14 = g["Low"].transform(lambda x: x.rolling(14, min_periods=14).min())
    high14 = g["High"].transform(lambda x: x.rolling(14, min_periods=14).max())
    stoch_k = (out["Close"].astype(float) - low14) / (high14 - low14) * 100.0
    out["stoch_k_14"] = stoch_k
    out["stoch_d_14"] = stoch_k.groupby(out["Ticker"], sort=False).transform(lambda x: x.rolling(3, min_periods=3).mean())
    feature_cols.extend(["stoch_k_14", "stoch_d_14"])

    # OBV
    direction = np.sign(g["Close"].diff()).fillna(0.0)
    out["obv"] = (direction * vol).groupby(out["Ticker"], sort=False).cumsum()
    feature_cols.append("obv")

    # 清理中间列
    out = out.drop(columns=["Open", "High", "Low", "Close", "Volume", "_tr", "_tp", "_log_close"], errors="ignore")

    # 仅返回真正存在的特征列（避免重复/缺失）
    feature_cols = [c for c in feature_cols if c in out.columns]
    return out, feature_cols


def _compute_features_pandas_ta(df: pd.DataFrame, windows: List[int]) -> Tuple[pd.DataFrame, List[str]]:
    """
    pandas-ta 实现（优先路径）。
    输出列名由 pandas-ta 决定；为便于统一，最终会去除 OHLCV 只保留因子列。
    """
    _require_ohlcv(df)
    import pandas_ta as ta  # noqa: F401  # type: ignore[import-not-found]

    work = df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]].copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work = work.dropna(subset=["Date"])
    work["Ticker"] = _norm_ticker(work["Ticker"])
    work = work.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    feats = []
    for t, gdf in work.groupby("Ticker", sort=False):
        gdf = gdf.sort_values("Date").copy()
        gdf = gdf.set_index("Date", drop=False)

        # 核心指标（跨四类：Momentum/Volatility/Trend/Volume）
        for w in windows:
            gdf.ta.rsi(length=w, append=True)
            gdf.ta.roc(length=w, append=True)
            gdf.ta.ema(length=w, append=True)
            gdf.ta.sma(length=w, append=True)
            gdf.ta.atr(length=w, append=True)
        gdf.ta.macd(append=True)
        gdf.ta.bbands(length=20, append=True)
        gdf.ta.stoch(append=True)
        gdf.ta.obv(append=True)
        gdf.ta.mfi(append=True)

        # 只保留 Date/Ticker + 新增指标列
        base = {"Date", "Ticker", "Open", "High", "Low", "Close", "Volume"}
        cols = [c for c in gdf.columns if c not in base]
        gdf = gdf.reset_index(drop=True)[["Date", "Ticker"] + cols]
        feats.append(gdf)

    out = pd.concat(feats, ignore_index=True) if feats else work[["Date", "Ticker"]].copy()
    feature_cols = [c for c in out.columns if c not in ("Date", "Ticker")]
    return out, feature_cols


def build_feature_file(
    prices_csv: str,
    output_parquet: str,
    feature_columns_path: str,
    n_features: int = 100,
    use_pandas_ta: bool = True,
    windows: List[int] | None = None,
) -> Tuple[str, List[str]]:
    # 如果用户指定的 prices_csv 不存在，则回退到项目内的 data/processed/Stock_Prices.csv
    if not os.path.exists(prices_csv):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        fallback = os.path.join(project_root, "data", "processed", "Stock_Prices.csv")
        if os.path.exists(fallback):
            print(f"[WARN] 未找到指定文件 {prices_csv}，回退使用 {fallback}")
            prices_csv = fallback
        else:
            raise FileNotFoundError(f"未找到指定的 prices_csv: {prices_csv}，且回退文件不存在: {fallback}")

    df = pd.read_csv(prices_csv, low_memory=False)
    windows = windows or DEFAULT_WINDOWS

    if use_pandas_ta:
        try:
            feat_df, feat_cols = _compute_features_pandas_ta(df, windows)
            print(f">>> pandas-ta 可用：生成因子 {len(feat_cols)} 列")
        except Exception as e:
            print(f"[WARN] pandas-ta 不可用/失败，回退纯 pandas 实现：{e}")
            feat_df, feat_cols = _compute_features_pandas(df, windows)
    else:
        feat_df, feat_cols = _compute_features_pandas(df, windows)

    # 控制维度到 n_features（默认 158）
    # 这里保持“构造顺序”作为默认优先级（不要按字母排序），更符合因子工程的直觉。
    feat_cols = list(dict.fromkeys(feat_cols))  # 去重并保留顺序
    if n_features > 0:
        feat_cols = feat_cols[: int(n_features)]
        keep = ["Date", "Ticker"] + feat_cols
        feat_df = feat_df[keep].copy()

    # 【关键增强】截面 Rank（Cross-sectional Rank）
    # 将原始数值转化为每日的百分比排名（pct rank），并中心化到 [-0.5, 0.5]
    # 这在量化预测中常用于提升不同股票间的可比性，帮助模型更快收敛。
    if feat_cols:
        feat_df[feat_cols] = feat_df.groupby("Date", sort=False)[feat_cols].rank(pct=True) - 0.5

    # 在写入前填充缺失值：先向前填充，再将剩余 NaN 置为 0，避免 downstream 出现 NaN 导致加载或训练失败
    feat_df = feat_df.ffill().fillna(0.0)

    os.makedirs(os.path.dirname(output_parquet) or ".", exist_ok=True)
    _write_parquet_or_raise(feat_df, output_parquet)

    os.makedirs(os.path.dirname(feature_columns_path) or ".", exist_ok=True)
    with open(feature_columns_path, "w", encoding="utf-8") as f:
        json.dump(feat_cols, f, indent=2, ensure_ascii=False)

    return output_parquet, feat_cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha158-like feature engineering (pandas-ta preferred).")
    parser.add_argument("--prices_csv", type=str, default="./paper/data/processed/Stock_Prices_sp500.csv")
    parser.add_argument("--output", type=str, default="./paper/data/processed/sp500_alpha158_features.parquet")
    parser.add_argument("--feature_columns", type=str, default="./paper/data/processed/feature_columns.json")
    parser.add_argument("--n_features", type=int, default=158)
    parser.add_argument("--no_pandas_ta", action="store_true", help="强制不用 pandas-ta（使用纯 pandas 实现）")
    args = parser.parse_args()

    out_path, cols = build_feature_file(
        prices_csv=args.prices_csv,
        output_parquet=args.output,
        feature_columns_path=args.feature_columns,
        n_features=args.n_features,
        use_pandas_ta=not args.no_pandas_ta,
    )
    print(f">>> features_out = {out_path}")
    print(f">>> feature_dim = {len(cols)}")


if __name__ == "__main__":
    main()
