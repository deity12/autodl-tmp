#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S&P 500 静态交集过滤（Static Intersection）
========================================

目标：
  Universe = set(FNSPID_Tickers) ∩ set(Static_SP500_List)

约束（来自 new.md / 用户确认的 Specifications）：
  - 不做“历史动态成分进出”建模（不追踪每日成分变化）
  - 仅在预处理阶段剔除非 S&P 500 的股票
  - S&P 500 列表来源：优先本地 sp500_list.txt；否则可在线获取（Wikipedia）

输出：
  - 过滤后的 Stock_Prices.csv / Stock_News.csv（默认写新文件，避免覆盖）
  - sp500_universe_tickers.txt（最终 Universe 列表，便于审计/复现）
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Optional, Set

import pandas as pd


WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def normalize_ticker(t: str) -> str:
    # 常见差异：BRK.B vs BRK-B；统一为 '.' 形式，后续两边都做同样归一化。
    return str(t).strip().upper().replace("-", ".")


def load_sp500_list(sp500_list_path: Optional[str] = None, source: str = "auto") -> Set[str]:
    """
    加载“静态 S&P 500 列表”（当前成分股）。

    source:
      - auto: 优先本地文件；否则 Wikipedia
      - file: 仅本地文件
      - wikipedia: 仅在线抓取
    """
    source = (source or "auto").strip().lower()

    if source not in {"auto", "file", "wikipedia"}:
        raise ValueError(f"Unknown source: {source}")

    if source in {"auto", "file"}:
        if sp500_list_path and os.path.exists(sp500_list_path):
            with open(sp500_list_path, "r", encoding="utf-8") as f:
                tickers = {
                    normalize_ticker(line)
                    for line in f
                    if line.strip() and not line.strip().startswith("#")
                }
            if tickers:
                return tickers
        if source == "file":
            raise FileNotFoundError(f"sp500_list.txt not found: {sp500_list_path}")

    # Wikipedia fallback（yfinance 本身没有稳定的“成分股列表 API”，这里用公开页面更可控）
    try:
        tables = pd.read_html(WIKI_SP500_URL)
    except Exception as e:
        raise RuntimeError(
            "Failed to fetch S&P 500 constituents from Wikipedia. "
            "Please provide a local sp500_list.txt (preferred), or install HTML parsers (e.g. lxml)."
        ) from e
    if not tables:
        raise RuntimeError("Failed to load S&P 500 table from Wikipedia.")

    df = tables[0]
    symbol_col = None
    for c in df.columns:
        if str(c).strip().lower() in {"symbol", "ticker symbol", "ticker"}:
            symbol_col = c
            break
    if symbol_col is None:
        # Wikipedia 当前表的第一列通常就是 Symbol
        symbol_col = df.columns[0]

    tickers = {normalize_ticker(x) for x in df[symbol_col].dropna().astype(str).tolist()}
    if not tickers:
        raise RuntimeError("Empty S&P 500 ticker list from Wikipedia.")
    return tickers


def _iter_filtered_chunks(
    input_csv: str,
    tickers: Set[str],
    ticker_col: str = "Ticker",
    chunksize: int = 200_000,
) -> Iterable[pd.DataFrame]:
    reader = pd.read_csv(input_csv, chunksize=chunksize, low_memory=False, on_bad_lines="skip")
    for chunk in reader:
        if ticker_col not in chunk.columns:
            raise KeyError(f"Missing column '{ticker_col}' in {input_csv}")
        chunk[ticker_col] = chunk[ticker_col].astype(str).map(normalize_ticker)
        yield chunk[chunk[ticker_col].isin(tickers)]


def filter_csv_by_tickers(
    input_csv: str,
    output_csv: str,
    tickers: Set[str],
    ticker_col: str = "Ticker",
    chunksize: int = 200_000,
) -> int:
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    total = 0
    first = True
    for chunk in _iter_filtered_chunks(input_csv, tickers, ticker_col=ticker_col, chunksize=chunksize):
        if chunk.empty:
            continue
        chunk.to_csv(output_csv, index=False, mode="w" if first else "a", header=first)
        first = False
        total += len(chunk)
    return total


def _load_fns_tickers_from_prices(prices_csv: str) -> Set[str]:
    df = pd.read_csv(prices_csv, usecols=["Ticker"])
    return {normalize_ticker(t) for t in df["Ticker"].dropna().astype(str).tolist()}


def filter_sp500(
    prices_csv: str,
    news_csv: Optional[str],
    prices_out: str,
    news_out: Optional[str],
    sp500_list_path: Optional[str] = None,
    sp500_source: str = "auto",
    universe_out: Optional[str] = None,
) -> Set[str]:
    fns_tickers = _load_fns_tickers_from_prices(prices_csv)
    sp500 = load_sp500_list(sp500_list_path=sp500_list_path, source=sp500_source)
    universe = set(sorted(fns_tickers.intersection(sp500)))

    if universe_out:
        os.makedirs(os.path.dirname(universe_out) or ".", exist_ok=True)
        with open(universe_out, "w", encoding="utf-8") as f:
            for t in sorted(universe):
                f.write(t + "\n")

    filter_csv_by_tickers(prices_csv, prices_out, universe, ticker_col="Ticker")
    if news_csv and news_out and os.path.exists(news_csv):
        filter_csv_by_tickers(news_csv, news_out, universe, ticker_col="Ticker")

    return universe


def default_sp500_list_path() -> Optional[str]:
    candidates = [
        "./sp500_list.txt",
        "./paper/data/raw/FNSPID/sp500_list.txt",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter FNSPID data to S&P 500 static intersection.")
    parser.add_argument("--prices_in", type=str, default="./paper/data/processed/Stock_Prices.csv")
    parser.add_argument("--news_in", type=str, default="./paper/data/processed/Stock_News.csv")
    parser.add_argument("--prices_out", type=str, default="./paper/data/processed/Stock_Prices_sp500.csv")
    parser.add_argument("--news_out", type=str, default="./paper/data/processed/Stock_News_sp500.csv")
    parser.add_argument("--sp500_list", type=str, default=None, help="Local sp500_list.txt path (preferred).")
    parser.add_argument("--sp500_source", type=str, default="auto", choices=["auto", "file", "wikipedia"])
    parser.add_argument("--universe_out", type=str, default="./paper/data/processed/sp500_universe_tickers.txt")
    args = parser.parse_args()

    sp500_list = args.sp500_list or default_sp500_list_path()
    universe = filter_sp500(
        prices_csv=args.prices_in,
        news_csv=args.news_in,
        prices_out=args.prices_out,
        news_out=args.news_out,
        sp500_list_path=sp500_list,
        sp500_source=args.sp500_source,
        universe_out=args.universe_out,
    )
    print(f">>> Universe size = {len(universe)}")
    print(f">>> prices_out = {args.prices_out}")
    if args.news_in and os.path.exists(args.news_in):
        print(f">>> news_out = {args.news_out}")
    if args.universe_out:
        print(f">>> universe_out = {args.universe_out}")


if __name__ == "__main__":
    main()
