"""
数据对齐与特征工程脚本（Step 2）
==========================

目标：
  将 ETL 阶段生成的个股行情与新闻数据，与 S&P 500 指数数据按日期对齐，
  生成训练/测试通用的模型输入表 `Final_Model_Data.csv`。

输入（本仓库统一放在 paper/data/processed/ 下）：
  - `paper/data/processed/Stock_Prices_sp500.csv`（S&P500 过滤后的价格）
  - `paper/data/processed/Stock_News_sp500.csv`（S&P500 过滤后的新闻）
  - `paper/data/processed/SP500_Index.csv`（大盘指数：Date, Market_Close, Market_Vol）

输出：
  - `data/processed/Final_Model_Data.csv`

关键产物字段：
  - `Log_Ret`：对数收益率（预测目标）
  - `Volatility_20d`：20 日滚动波动率（用于量子门控/分组评估）
  - `News_Text`：同一股票同一天的新闻标题聚合（用 `|` 拼接）

注意：
  - 对新闻数据做“同日同股聚合”，避免一日多条新闻导致行重复。
  - 合并前会做时区统一（tz-aware -> tz-naive），避免 pandas merge 报错。
"""

import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

# 保证以绝对路径直接运行 align.py 时也能找到 dataProcessed 包
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PAPER_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PAPER_ROOT not in sys.path:
    sys.path.insert(0, _PAPER_ROOT)

# ================= 配置路径 (根据你之前的运行结果修改) =================
# 1. 价格与新闻：均使用 paper/data/processed/ 下的过滤后文件
PROCESSED_PRICE_PATH = "./paper/data/processed/Stock_Prices_sp500.csv"
PROCESSED_NEWS_PATH = "./paper/data/processed/Stock_News_sp500.csv"

# 2. 大盘指数路径（你当前已放在 paper/data/processed/）
MARKET_INDEX_PATH = "./paper/data/processed/SP500_Index.csv"

# 3. 最终输出路径
OUTPUT_FILE = "./paper/data/processed/Final_Model_Data.csv"

# Alpha158（轻量）特征配置（默认启用）
USE_ALPHA158 = True
ALPHA_OUTPUT_FILE = "./paper/data/processed/sp500_alpha158_features.parquet"
FEATURE_COLUMNS_PATH = "./paper/data/processed/feature_columns.json"
FEATURE_TARGET_DIM = 158  # 默认输出 158 个 Alpha158-like 因子

# ================= 主逻辑：数据对齐 =================
def align_all_data():
    """
    主流程：读取 ETL 产物并对齐生成 `Final_Model_Data.csv`。

    步骤概览：
      1) 读取个股行情（按 Ticker+Date 排序）
      2) 读取大盘指数（按 Date 排序）
      3) 读取并聚合新闻（同日同股拼接为 `News_Text`，可选）
      4) 合并：prices LEFT JOIN market（Date）再 LEFT JOIN news（Date+Ticker）
      5) 生成标签/特征：`Log_Ret` 与 `Volatility_20d`
      6) 保存到 `OUTPUT_FILE`
    """
    print(">>> [Step 1/3] 读取预处理后的数据...")
    
    # Check paths
    if not os.path.exists(PROCESSED_PRICE_PATH):
        print(f"❌ 找不到股价文件: {PROCESSED_PRICE_PATH}")
        return
    if not os.path.exists(MARKET_INDEX_PATH):
        print(f"❌ 找不到大盘文件: {MARKET_INDEX_PATH}")
        return

    # 1. 读取股价
    print("正在读取股价数据...")
    df_price = pd.read_csv(PROCESSED_PRICE_PATH)
    df_price['Date'] = pd.to_datetime(df_price['Date'])
    if "Ticker" in df_price.columns:
        df_price["Ticker"] = (
            df_price["Ticker"]
            .astype(str)
            .str.upper()
            .str.replace("-", ".", regex=False)
        )
    # 确保按Date和Ticker排序，便于后续计算
    df_price = df_price.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    # 2. 读取大盘
    print("正在读取大盘数据...")
    df_market = pd.read_csv(MARKET_INDEX_PATH)
    df_market['Date'] = pd.to_datetime(df_market['Date'])
    # 确保按Date排序
    df_market = df_market.sort_values('Date').reset_index(drop=True)
    
    # 3. 读取新闻
    print("正在读取新闻数据...")
    if os.path.exists(PROCESSED_NEWS_PATH):
        df_news = pd.read_csv(PROCESSED_NEWS_PATH, low_memory=False)  # 修复 mixed types 警告
        df_news['Date'] = pd.to_datetime(df_news['Date'])
        if "Ticker" in df_news.columns:
            df_news["Ticker"] = (
                df_news["Ticker"]
                .astype(str)
                .str.upper()
                .str.replace("-", ".", regex=False)
            )
        
        # 【关键修复】统一时区格式：如果Date列带时区，转换为无时区的datetime
        # 这样可以避免合并时的时区不匹配错误
        if df_news['Date'].dt.tz is not None:
            # 如果有时区，转换为无时区（只保留日期部分）
            df_news['Date'] = df_news['Date'].dt.tz_localize(None)
        
        # 【关键】聚合新闻：同一天同一个股票的多条新闻合并成一条
        # 注意：这里的 Date 已经在 etl.py 中按 16:00 cut-off 规则对齐
        # 即：16:00 之前的新闻归入当日，16:00 之后的归入次日
        print("正在聚合每日新闻 (这可能需要一点时间)...")
        print("  [提示] 新闻日期已按 16:00 cut-off 规则对齐（在 etl.py 中完成）")
        # 自动识别新闻列名 (兼容 Headline 或 Article_title)
        text_col = 'Headline' if 'Headline' in df_news.columns else 'Article_title'
        
        if text_col in df_news.columns:
            # 填充空值防止报错
            df_news[text_col] = df_news[text_col].fillna("")
            # 按对齐后的日期和股票代码聚合新闻
            # 这样确保用于预测 T+1 日收益的新闻都是 T 日 16:00 之前发布的
            df_news_agg = df_news.groupby(['Date', 'Ticker'])[text_col].apply(
                lambda x: ' | '.join(x.astype(str))
            ).reset_index()
            df_news_agg = df_news_agg.rename(columns={text_col: 'News_Text'})
            
            # 确保聚合后的Date列也是无时区的datetime
            if df_news_agg['Date'].dt.tz is not None:
                df_news_agg['Date'] = df_news_agg['Date'].dt.tz_localize(None)
        else:
            print("⚠️ 警告: 新闻文件中没找到 Headline 列，将跳过新闻合并。")
            df_news_agg = None
    else:
        print("⚠️ 未找到新闻文件，将只合并价格和大盘。")
        df_news_agg = None
    
    print(f">>> [Step 2/3] 开始全量对齐 (Merge)...")
    
    # 4. 合并流程
    # 以股价表为主表 (Left Join)，确保每个个股的每个交易日都在
    merged_df = pd.merge(df_price, df_market, on='Date', how='left') # 加入大盘
    
    if df_news_agg is not None:
        # 【额外检查】确保合并前时区格式一致（双重保险）
        if merged_df['Date'].dt.tz is not None:
            merged_df['Date'] = merged_df['Date'].dt.tz_localize(None)
        if df_news_agg['Date'].dt.tz is not None:
            df_news_agg['Date'] = df_news_agg['Date'].dt.tz_localize(None)
        
        merged_df = pd.merge(merged_df, df_news_agg, on=['Date', 'Ticker'], how='left') # 加入新闻
        merged_df['News_Text'] = merged_df['News_Text'].fillna("") # 无新闻填空
    else:
        merged_df['News_Text'] = "" # 如果没新闻文件，这就给个空列
    
    # 确保合并后的数据按Ticker和Date排序（重要：用于后续的shift和rolling计算）
    merged_df = merged_df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    # 5. 缺失值处理
    # 市场数据前向填充 (比如美股节假日可能个股有数据但大盘没数据，或者反之)
    cols_to_ffill = ['Market_Close', 'Market_Vol']
    # 检查列是否存在再填充
    existing_cols = [c for c in cols_to_ffill if c in merged_df.columns]
    if existing_cols:
        merged_df[existing_cols] = merged_df[existing_cols].ffill()
        merged_df[existing_cols] = merged_df[existing_cols].fillna(0) # 开头如果是NaN填0
    
    # 6. 计算波动率 (高性能向量化版本)
    print(">>> [Step 3/3] 计算波动率标签 (Volatility Label)...")
    print("正在计算 Log Return 和 Rolling Volatility (向量化优化版本)...")
    
    # 1. 处理 0 值和缺失值 (全表操作，极快)
    merged_df['Close'] = merged_df['Close'].replace(0, np.nan)
    merged_df['Close'] = merged_df.groupby('Ticker')['Close'].ffill()  # 组内填充
    
    # 2. 计算 Log Return (利用 GroupBy 的 transform，比 apply 快)
    # log(P_t / P_{t-1}) = log(P_t) - log(P_{t-1})
    # 这里先取log再diff，数学上是等价的，且计算更快
    merged_df['Log_Close'] = np.log(merged_df['Close'])
    merged_df['Log_Ret'] = merged_df.groupby('Ticker')['Log_Close'].diff()
    
    # 3. 计算 Rolling Volatility
    merged_df['Volatility_20d'] = merged_df.groupby('Ticker')['Log_Ret'].transform(lambda x: x.rolling(20).std())
    
    # 4. 清理中间变量和NaN
    merged_df = merged_df.drop(columns=['Log_Close'])
    merged_df['Volatility_20d'] = merged_df['Volatility_20d'].fillna(0)
    merged_df['Log_Ret'] = merged_df['Log_Ret'].fillna(0)

    # 保存基础数据
    print(f"正在保存最终文件至 {OUTPUT_FILE} ...")
    merged_df.to_csv(OUTPUT_FILE, index=False)
    print("✅ 基础数据已保存。")

    # 可选：生成 Alpha158-like 因子（独立落盘为 Parquet，供 DataLoader 读取并 merge）
    if USE_ALPHA158:
        try:
            from dataProcessed.feature_engineering import build_feature_file

            print(">>> 开始生成 Alpha158-like 因子（pandas-ta 优先）...")
            out_path, cols = build_feature_file(
                prices_csv=PROCESSED_PRICE_PATH,
                output_parquet=ALPHA_OUTPUT_FILE,
                feature_columns_path=FEATURE_COLUMNS_PATH,
                n_features=FEATURE_TARGET_DIM,
                use_pandas_ta=True,
            )
            print(f"✅ Alpha158-like 因子已保存: {out_path} (dim={len(cols)})")
            print(f"✅ 特征列清单已保存: {FEATURE_COLUMNS_PATH}")
        except Exception as e:
            print(f"⚠️ Alpha158-like 因子生成失败: {e}")

    print("✅ 全部完成！你的数据已经准备好进入模型了。")
    print("包含列:", merged_df.columns.tolist())
    print(f"数据总行数: {len(merged_df)}")

if __name__ == "__main__":
    align_all_data()
