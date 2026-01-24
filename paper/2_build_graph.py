#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2_build_graph.py - 构建图谱入口脚本（可直接运行）

功能：
  基于新闻数据构建语义图 + 统计相关性图，并融合为混合图。

用法：
  方式1：直接运行（默认配置）
  方式2：命令行覆盖参数
    python 2_build_graph.py --news ./data/processed/Stock_News.csv --model_data ./data/processed/Final_Model_Data.csv
"""

from __future__ import annotations

import argparse
import os
import sys

from utils.logging_utils import setup_logging

# ================= 配置（可直接修改）=================
NEWS_FILE = "./data/processed/Stock_News.csv"
MODEL_DATA_FILE = "./data/processed/Final_Model_Data.csv"
OUTPUT_DIR = "./data/processed/"

USE_LLM = True
MAX_PER_TICKER = 200
MAX_TOTAL = 100000
USE_SP500_ONLY = True
# ====================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建股票关系图谱（语义图 + 统计图）")
    parser.add_argument("--news", type=str, default=NEWS_FILE, help="新闻 CSV 路径")
    parser.add_argument("--model_data", type=str, default=MODEL_DATA_FILE, help="Final_Model_Data.csv 路径")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--llm", action=argparse.BooleanOptionalAction, default=USE_LLM, help="是否启用 LLM")
    parser.add_argument("--max_per_ticker", type=int, default=MAX_PER_TICKER, help="每个股票最多采样新闻数")
    parser.add_argument("--max_total", type=int, default=MAX_TOTAL, help="新闻总采样上限")
    parser.add_argument("--all_stocks", action="store_true", help="使用全量股票（默认仅 S&P500）")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    log_ctx = setup_logging(args.output_dir, "build_graph")
    logger = log_ctx.logger

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from dataProcessed import build_graph

        os.makedirs(args.output_dir, exist_ok=True)

        # 覆盖 build_graph 模块的路径配置
        build_graph.INPUT_NEWS = args.news
        build_graph.INPUT_MODEL_DATA = args.model_data
        build_graph.OUTPUT_GRAPH = os.path.join(args.output_dir, "Graph_Adjacency.npy")
        build_graph.OUTPUT_TICKERS = os.path.join(args.output_dir, "Graph_Tickers.json")

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
