#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2_build_graph.py - 构建图谱入口脚本（可直接运行）

功能：
  基于新闻数据构建语义图 + 统计相关性图，并融合为混合图。

用法：
  方式1：直接运行（默认配置）
  方式2：命令行覆盖参数
    python 2_build_graph.py --news ./paper/data/processed/Stock_News_sp500.csv --model_data ./paper/data/processed/Final_Model_Data.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from utils.logging_utils import setup_logging

# ================= 配置（可直接修改）=================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NEWS_FILE = os.path.join(_SCRIPT_DIR, "data", "processed", "Stock_News_sp500.csv")
MODEL_DATA_FILE = os.path.join(_SCRIPT_DIR, "data", "processed", "Final_Model_Data.csv")
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "data", "processed") + os.sep

USE_LLM = True
MAX_PER_TICKER = 200
MAX_TOTAL = 100000
USE_SP500_ONLY = True
# ====================================================

def _parse_split_date(value: str) -> pd.Timestamp:
    split_date = pd.to_datetime(value, errors="coerce")
    if pd.isna(split_date):
        raise ValueError(f"Invalid split_date: {value}")
    if split_date.tz is not None:
        split_date = split_date.tz_convert(None)
    return split_date


def _filter_news_by_split_date(
    input_path: str,
    output_dir: str,
    split_date: pd.Timestamp,
    logger,
) -> str:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"News file not found: {input_path}")
    df_news = pd.read_csv(input_path, low_memory=False)
    if "Date" not in df_news.columns:
        raise ValueError("News file missing 'Date' column; cannot enforce split_date.")
    df_news["Date"] = pd.to_datetime(df_news["Date"], errors="coerce")
    before = len(df_news)
    df_news = df_news[df_news["Date"] < split_date].copy()

    os.makedirs(output_dir, exist_ok=True)
    split_tag = split_date.strftime("%Y%m%d")
    news_base = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{news_base}.before_{split_tag}.csv")
    df_news.to_csv(output_path, index=False)
    logger.info(
        "Strict Data Leakage Prevention: Filtered news %d -> %d",
        before,
        len(df_news),
    )
    return output_path


def _filter_relations_by_split_date(
    input_path: str | None,
    output_dir: str,
    split_date: pd.Timestamp,
    logger,
) -> str | None:
    if not input_path or not os.path.exists(input_path):
        return None

    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".parquet":
        df_rel = pd.read_parquet(input_path)
    else:
        df_rel = pd.read_csv(input_path, low_memory=False)

    date_col = None
    for cand in ("date", "Date"):
        if cand in df_rel.columns:
            date_col = cand
            break
    if date_col is None:
        raise ValueError("Relations file missing date column; cannot enforce split_date.")

    df_rel[date_col] = pd.to_datetime(df_rel[date_col], errors="coerce")
    before = len(df_rel)
    df_rel = df_rel[df_rel[date_col] < split_date].copy()

    os.makedirs(output_dir, exist_ok=True)
    split_tag = split_date.strftime("%Y%m%d")
    rel_base = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{rel_base}.before_{split_tag}{ext}")
    if ext == ".parquet":
        df_rel.to_parquet(output_path, index=False)
    else:
        df_rel.to_csv(output_path, index=False)

    logger.info(
        "Strict Data Leakage Prevention: Filtered relations %d -> %d",
        before,
        len(df_rel),
    )
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建股票关系图谱（语义图 + 统计图）")
    parser.add_argument("--news", type=str, default=NEWS_FILE, help="新闻 CSV 路径")
    parser.add_argument("--model_data", type=str, default=MODEL_DATA_FILE, help="Final_Model_Data.csv 路径")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--llm", action=argparse.BooleanOptionalAction, default=USE_LLM, help="是否启用 LLM")
    parser.add_argument("--max_per_ticker", type=int, default=MAX_PER_TICKER, help="每个股票最多采样新闻数")
    parser.add_argument("--max_total", type=int, default=MAX_TOTAL, help="新闻总采样上限")
    parser.add_argument("--all_stocks", action="store_true", help="使用全量股票（默认仅 S&P500）")
    parser.add_argument(
        "--split_date",
        type=str,
        default="2020-12-31",
        help="Graph cut-off date (YYYY-MM-DD). News/relations with timestamp >= split_date will be excluded.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    log_ctx = setup_logging(args.output_dir, "build_graph")
    logger = log_ctx.logger

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from dataProcessed import build_graph

        os.makedirs(args.output_dir, exist_ok=True)

        split_date = _parse_split_date(args.split_date)
        split_date_str = split_date.strftime("%Y-%m-%d")
        logger.info(
            "Strict Data Leakage Prevention: Graph cut-off date set to %s",
            split_date_str,
        )

        # 覆盖 build_graph 模块的路径配置
        filtered_news = _filter_news_by_split_date(
            args.news,
            args.output_dir,
            split_date,
            logger,
        )
        build_graph.INPUT_NEWS = filtered_news
        build_graph.INPUT_MODEL_DATA = args.model_data
        build_graph.OUTPUT_GRAPH = os.path.join(args.output_dir, "Graph_Adjacency.npy")
        build_graph.OUTPUT_TICKERS = os.path.join(args.output_dir, "Graph_Tickers.json")
        filtered_relations = _filter_relations_by_split_date(
            build_graph.RELATIONS_PARQUET_PATH,
            args.output_dir,
            split_date,
            logger,
        )
        if filtered_relations:
            build_graph.RELATIONS_PARQUET_PATH = filtered_relations

        logger.info("开始构建图谱")
        logger.info("新闻文件: %s", build_graph.INPUT_NEWS)
        logger.info("模型数据: %s", build_graph.INPUT_MODEL_DATA)
        logger.info("输出目录: %s", args.output_dir)
        logger.info("LLM: %s", "启用" if args.llm else "禁用")
        logger.info("每股票采样上限: %d", args.max_per_ticker)
        logger.info("总采样上限: %d", args.max_total)
        logger.info("股票范围: %s", "全量" if args.all_stocks else "S&P 500 成分股")

        build_graph.build_dynamic_graph(
            use_llm=bool(args.llm),
            max_per_ticker=int(args.max_per_ticker),
            max_total=int(args.max_total),
            use_sp500=not bool(args.all_stocks),
        )
    except Exception as exc:
        logger.exception("建图失败: %s", exc)
        raise
    finally:
        log_ctx.restore_print()


if __name__ == "__main__":
    main()
