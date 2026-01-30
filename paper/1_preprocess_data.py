#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1_preprocess_data.py - 数据预处理入口脚本（可直接运行）

功能：
  1) 合并股价 CSV（Stock_Prices.csv）
  2) 流式清洗新闻（Stock_News.csv）

用法：
  方式1：直接运行（使用默认配置）
 方式2：命令行覆盖参数
    python 1_preprocess_data.py --input_dir ./paper/data/raw/FNSPID/full_history --news_file ./paper/data/raw/FNSPID/nasdaq_exteral_data.csv
"""

from __future__ import annotations

import argparse
import os
import sys

from utils.logging_utils import setup_logging

# ================= 配置（可直接修改）=================
# 使用脚本所在目录，使从 paper/ 或项目根运行都能正确找到 data/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PRICES_DIR = os.path.join(_SCRIPT_DIR, "data", "raw", "FNSPID", "full_history") + os.sep
RAW_NEWS_FILE = os.path.join(_SCRIPT_DIR, "data", "raw", "FNSPID", "nasdaq_exteral_data.csv")
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "data", "processed") + os.sep
MARKET_INDEX_PATH = os.path.join(_SCRIPT_DIR, "data", "processed", "SP500_Index.csv")
FINAL_OUTPUT_FILE = os.path.join(_SCRIPT_DIR, "data", "processed", "Final_Model_Data.csv")
ALPHA_OUTPUT_FILE = os.path.join(_SCRIPT_DIR, "data", "processed", "sp500_alpha158_features.parquet")
FEATURE_COLUMNS_PATH = os.path.join(_SCRIPT_DIR, "data", "processed", "feature_columns.json")

START_DATE = "2018-01-01"
END_DATE = "2023-12-31"

CUTOFF_HOUR = 16
CUTOFF_MINUTE = 0
RUN_ALIGN = True
USE_ALPHA158 = True
USE_QLIB_ALPHA158 = False
QLIB_PROVIDER_URI = os.environ.get("QLIB_PROVIDER_URI", "")
QLIB_REGION = os.environ.get("QLIB_REGION", "US")
FILTER_SP500 = True
# ====================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="数据预处理：合并股价 + 清洗新闻")
    parser.add_argument("--input_dir", type=str, default=RAW_PRICES_DIR, help="股价 CSV 目录")
    parser.add_argument("--news_file", type=str, default=RAW_NEWS_FILE, help="新闻大文件路径")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--start_date", type=str, default=START_DATE, help="起始日期（YYYY-MM-DD）")
    parser.add_argument("--end_date", type=str, default=END_DATE, help="结束日期（YYYY-MM-DD）")
    parser.add_argument("--cutoff_hour", type=int, default=CUTOFF_HOUR, help="16:00 Cut-off 小时")
    parser.add_argument("--cutoff_minute", type=int, default=CUTOFF_MINUTE, help="16:00 Cut-off 分钟")
    parser.add_argument("--market_index", type=str, default=MARKET_INDEX_PATH, help="S&P500 指数数据路径")
    parser.add_argument("--final_output", type=str, default=FINAL_OUTPUT_FILE, help="对齐后的最终输出 CSV")
    parser.add_argument("--alpha_output", type=str, default=ALPHA_OUTPUT_FILE, help="Alpha158-like 因子输出（Parquet）")
    parser.add_argument("--feature_columns", type=str, default=FEATURE_COLUMNS_PATH, help="特征列清单输出路径")
    parser.add_argument("--run_align", action=argparse.BooleanOptionalAction, default=RUN_ALIGN, help="是否在 ETL 后执行对齐")
    parser.add_argument("--use_alpha158", action=argparse.BooleanOptionalAction, default=USE_ALPHA158, help="是否生成 Alpha158 特征")
    parser.add_argument("--use_qlib_alpha158", action=argparse.BooleanOptionalAction, default=USE_QLIB_ALPHA158, help="是否优先使用 Qlib Alpha158")
    parser.add_argument("--qlib_provider_uri", type=str, default=QLIB_PROVIDER_URI, help="Qlib provider URI (可以为空)")
    parser.add_argument("--qlib_region", type=str, default=QLIB_REGION, help="Qlib region (US/CN)")
    parser.add_argument("--filter_sp500", action=argparse.BooleanOptionalAction, default=FILTER_SP500, help="是否启用 S&P500 静态交集过滤")
    parser.add_argument("--sp500_list", type=str, default=None, help="本地 sp500_list.txt 路径（推荐）")
    parser.add_argument("--sp500_source", type=str, default="auto", choices=["auto", "file", "wikipedia"], help="S&P500 列表来源")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    log_ctx = setup_logging(args.output_dir, "preprocess")
    logger = log_ctx.logger

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from dataProcessed import etl, align

        # 覆盖 ETL 模块配置
        etl.RAW_PRICES_DIR = args.input_dir
        etl.RAW_NEWS_FILE = args.news_file
        etl.OUTPUT_DIR = args.output_dir
        etl.START_DATE = args.start_date
        etl.END_DATE = args.end_date
        etl.CUTOFF_HOUR = args.cutoff_hour
        etl.CUTOFF_MINUTE = args.cutoff_minute

        os.makedirs(etl.OUTPUT_DIR, exist_ok=True)

        logger.info("开始预处理：合并股价 + 清洗新闻")
        logger.info("输入股价目录: %s", etl.RAW_PRICES_DIR)
        logger.info("输入新闻文件: %s", etl.RAW_NEWS_FILE)
        logger.info("输出目录: %s", etl.OUTPUT_DIR)
        logger.info("时间范围: %s ~ %s", etl.START_DATE, etl.END_DATE)
        logger.info("Cut-off: %02d:%02d", etl.CUTOFF_HOUR, etl.CUTOFF_MINUTE)
        logger.info("Alpha158: %s", "启用" if args.use_alpha158 else "关闭")
        logger.info("Alpha158(Qlib): %s", "启用" if args.use_qlib_alpha158 else "关闭")

        tickers = etl.merge_stock_prices()

        # S&P 500 静态交集过滤：Universe = FNSPID_Tickers ∩ Static_SP500_List
        # - 优先本地 sp500_list.txt；否则可在线抓取（Wikipedia）
        # - 不处理历史动态成分进出；仅做静态交集
        price_csv = os.path.join(etl.OUTPUT_DIR, "Stock_Prices.csv")
        news_tickers = tickers
        price_for_align = price_csv
        if args.filter_sp500 and tickers:
            try:
                from dataProcessed import filter_sp500 as sp500_filter

                sp500_list = args.sp500_list or sp500_filter.default_sp500_list_path()
                sp500_set = sp500_filter.load_sp500_list(sp500_list_path=sp500_list, source=args.sp500_source)
                fns_set = {sp500_filter.normalize_ticker(t) for t in tickers}
                universe = sorted(fns_set.intersection(sp500_set))

                if universe:
                    universe_out = os.path.join(etl.OUTPUT_DIR, "sp500_universe_tickers.txt")
                    with open(universe_out, "w", encoding="utf-8") as f:
                        for t in universe:
                            f.write(t + "\n")
                    logger.info("S&P500 静态交集 Universe: %d", len(universe))
                    logger.info("Universe 已保存: %s", universe_out)

                    # 1) 新闻清洗阶段直接用 Universe 过滤（避免处理/落盘非 S&P500 的新闻）
                    news_tickers = universe

                    # 2) 价格数据输出一份 filtered 版本，供后续对齐/训练使用
                    price_sp500_csv = os.path.join(etl.OUTPUT_DIR, "Stock_Prices_sp500.csv")
                    sp500_filter.filter_csv_by_tickers(price_csv, price_sp500_csv, set(universe), ticker_col="Ticker")
                    price_for_align = price_sp500_csv
                    logger.info("S&P500 过滤后的价格已保存: %s", price_sp500_csv)
                else:
                    logger.warning("S&P500 静态交集为空：将回退使用全量 tickers。")
            except Exception as e:
                logger.warning("S&P500 过滤失败，将回退使用全量 tickers。原因: %s", e)

        if tickers:
            etl.process_huge_news_file(news_tickers)
        else:
            logger.warning("未找到有效股票数据，跳过新闻清洗。")

        if args.run_align:
            if not os.path.exists(args.market_index):
                logger.warning("未找到大盘指数文件: %s，跳过对齐。", args.market_index)
            else:
                align.PROCESSED_PRICE_PATH = price_for_align
                align.PROCESSED_NEWS_PATH = os.path.join(etl.OUTPUT_DIR, "Stock_News.csv")
                align.MARKET_INDEX_PATH = args.market_index
                align.OUTPUT_FILE = args.final_output
                align.USE_ALPHA158 = bool(args.use_alpha158)
                align.USE_QLIB_ALPHA158 = bool(args.use_qlib_alpha158)
                align.QLIB_PROVIDER_URI = args.qlib_provider_uri
                align.QLIB_REGION = args.qlib_region
                align.ALPHA_OUTPUT_FILE = args.alpha_output
                align.FEATURE_COLUMNS_PATH = args.feature_columns
                logger.info("开始对齐数据（Final_Model_Data.csv）")
                align.align_all_data()
    except Exception as exc:
        logger.exception("预处理失败: %s", exc)
        raise
    finally:
        log_ctx.restore_print()


if __name__ == "__main__":
    main()
