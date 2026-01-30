"""
ETL脚本：股票价格和新闻数据清洗与合并

功能：
1. 合并数千个个股CSV文件为统一的股价数据文件
2. 流式处理23GB新闻大文件，提取指定时间范围和股票代码的新闻

输出：
- Stock_Prices.csv: 合并后的股价数据（包含Date, Ticker, Open, Close, High, Low, Volume）
- Stock_News.csv: 清洗后的新闻数据（包含Date, Ticker, Headline, Publisher）
"""

import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

# ================= 配置路径 =================
# 使用基于脚本位置的路径，避免依赖当前工作目录导致的路径不一致
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_PRICES_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'FNSPID', 'full_history')  # 股价CSV文件所在目录
RAW_NEWS_FILE = os.path.join(PROJECT_ROOT, 'data', 'raw', 'FNSPID', 'nasdaq_exteral_data.csv')  # 新闻数据大文件路径
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')  # 处理后的数据输出目录

# 自动创建输出目录（如果不存在）
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ================= 时间范围配置 =================
# 设定数据提取的时间范围（与论文研究期间一致）
START_DATE = '2018-01-01'  # 起始日期
END_DATE = '2023-12-31'    # 结束日期

# ================= 16:00 Cut-off 时间配置 =================
# 【核心创新点】严格的前瞻偏差规避：严禁使用未来信息
# 根据论文要求，设定每日 16:00（收盘时间）为界：
# - T 日 16:00 之前发布的新闻 → 构建 T 日的图 → 参与预测 T+1 日收益
# - T 日 16:00 之后发布的新闻 → 归入 T+1 日数据流
CUTOFF_HOUR = 16  # 美股收盘时间（16:00 ET）
CUTOFF_MINUTE = 0

# ================= 任务1：合并股价数据 =================
def merge_stock_prices():
    """
    合并所有个股CSV文件为统一的股价数据文件
    
    处理流程：
    1. 扫描目录下所有CSV文件
    2. 对每个文件进行数据清洗和标准化
    3. 提取股票代码（Ticker）并统一格式
    4. 过滤指定时间范围的数据
    5. 合并所有数据并保存
    
    Returns:
        list: 有效的股票代码列表（用于后续新闻数据过滤）
    """
    print(">>> 正在合并几千个个股CSV文件...")
    
    # 获取所有CSV文件列表
    all_files = glob.glob(os.path.join(RAW_PRICES_DIR, "*.csv"))
    # 排除 MAC 系统隐藏文件（以 ._ 开头的文件）
    valid_files = [f for f in all_files if "._" not in f]
    
    print(f"检测到 {len(valid_files)} 个股票文件。")
    
    df_list = []  # 存储所有清洗后的数据框
    
    # 使用 tqdm 显示处理进度
    for filename in tqdm(valid_files):
        try:
            # 读取单个CSV文件
            df = pd.read_csv(filename)
            
            # 【优化】: 强制列名转小写，解决不同文件间 Date/date 大小写不一致问题
            # 这样可以统一处理，避免因大小写导致的列名匹配失败
            df.columns = [c.lower() for c in df.columns]
            
            # 从文件名提取股票代码（Ticker）
            # 例如：AAPL.csv -> AAPL，并统一转换为大写格式
            ticker = os.path.basename(filename).replace('.csv', '').upper().replace("-", ".")
            df['Ticker'] = ticker  # 新增的列默认首字母大写，保持一致性
            
            # 核心数据清洗流程
            if 'date' in df.columns:
                # 将日期列转换为datetime类型，无效日期转为NaN
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                # 删除日期为NaN的行（无效日期数据）
                df = df.dropna(subset=['date'])
                
                # 时间范围过滤：只保留指定时间范围内的数据
                mask = (df['date'] >= START_DATE) & (df['date'] <= END_DATE)
                df = df.loc[mask]
                
                # 统一列名为首字母大写格式（标准化列名，便于后续处理）
                df = df.rename(columns={
                    'date': 'Date', 
                    'open': 'Open', 
                    'close': 'Close', 
                    'high': 'High', 
                    'low': 'Low', 
                    'volume': 'Volume'
                })
                
                # 如果过滤后数据不为空，则添加到列表中
                if not df.empty:
                    # 只保留标准列，防止有些文件包含额外的乱七八糟的列
                    keep_cols = ['Date', 'Ticker', 'Open', 'Close', 'High', 'Low', 'Volume']
                    # 只选择实际存在的列（防止某些文件缺少某些列）
                    actual_cols = [c for c in keep_cols if c in df.columns]
                    df_list.append(df[actual_cols])
                    
        except Exception as e:
            # 如果文件损坏或读取失败，跳过并打印错误信息
            print(f"跳过损坏文件 {filename}: {e}")

    # 合并所有数据框
    if df_list:
        # 使用 concat 合并所有数据框，ignore_index=True 重新生成索引
        full_price_df = pd.concat(df_list, ignore_index=True)
        output_path = os.path.join(OUTPUT_DIR, 'Stock_Prices.csv')
        # 保存为CSV文件，不包含行索引
        full_price_df.to_csv(output_path, index=False)
        print(f"✅ 股价合并完成！已保存至: {output_path}")
        print(f"总数据量: {len(full_price_df)} 行")
        # 返回所有唯一的股票代码列表（用于后续新闻数据过滤）
        return list(full_price_df['Ticker'].unique())
    else:
        print("❌ 未找到有效股价数据，请检查路径！")
        return []

# ================= 任务2：清洗 23GB 新闻大文件 =================
def process_huge_news_file(valid_tickers):
    """
    流式处理23GB新闻大文件，提取指定时间范围和股票代码的新闻
    【修正版】：针对无时间戳数据实施"保守型 T+1 偏移"策略，解决前瞻偏差。
    """
    print("\n>>> 正在流式处理 23GB 新闻大文件 (启动保守型防泄露策略)...")
    
    if not os.path.exists(RAW_NEWS_FILE):
        print(f"❌ 找不到新闻文件: {RAW_NEWS_FILE}")
        return

    # 加载交易日历
    trading_days = None
    price_path = os.path.join(OUTPUT_DIR, "Stock_Prices.csv")
    if os.path.exists(price_path):
        try:
            df_dates = pd.read_csv(price_path, usecols=["Date"])
            td = pd.to_datetime(df_dates["Date"], errors="coerce").dropna().dt.normalize().unique()
            trading_days = np.sort(td)
        except Exception as e:
            print(f"⚠️ 无法加载交易日历: {e}")

    CHUNK_SIZE = 100000 
    cleaned_chunks = []
    
    # 【修复】添加 low_memory=False 解决 DtypeWarning
    reader = pd.read_csv(RAW_NEWS_FILE, chunksize=CHUNK_SIZE, on_bad_lines='skip', low_memory=False)
    
    # 状态标记
    has_warned_resolution = False

    for i, chunk in enumerate(reader):
        if 'Date' in chunk.columns:
            # 1. 基础转换
            ts_utc = pd.to_datetime(chunk["Date"], errors="coerce", utc=True)
            valid_ts = ts_utc.notna()
            
            if valid_ts.any():
                chunk = chunk.loc[valid_ts].copy()
                ts_utc = ts_utc.loc[valid_ts]

                # 【核心修正】检测分钟级精度丢失 (00:00:00)
                # 如果是 00:00:00，说明是日级别数据，必须强制 +1 天以防泄露
                is_daily_res = (ts_utc.dt.hour == 0) & (ts_utc.dt.minute == 0) & (ts_utc.dt.second == 0)
                
                # 仅打印一次提示
                if is_daily_res.mean() > 0.9 and not has_warned_resolution:
                    print("\n[INFO] 检测到数据主要是日级别精度 (00:00:00)。")
                    print("       >>> 已启动【保守型防泄露策略】：强制将此类新闻归入 T+1 日。")
                    has_warned_resolution = True

                # 2. 转美东时间
                ts_et = ts_utc.dt.tz_convert("US/Eastern")

                # 3. 计算 16:00 Cut-off
                cutoff_seconds = int(CUTOFF_HOUR) * 3600 + int(CUTOFF_MINUTE) * 60
                secs = (
                    ts_et.dt.hour * 3600
                    + ts_et.dt.minute * 60
                    + ts_et.dt.second
                    + ts_et.dt.microsecond / 1_000_000.0
                )
                is_before_cutoff = secs <= cutoff_seconds

                # 4. 对齐逻辑
                aligned_day_et = ts_et.dt.normalize()
                # 正常逻辑：收盘后 -> T+1
                aligned_day_et = aligned_day_et.where(is_before_cutoff, aligned_day_et + pd.Timedelta(days=1))
                
                # 【Plan B 修正】：对于 00:00:00 的数据，强制再加一天
                # 这样即使该新闻是当天盘后发生的，也会被安全地归入明天
                aligned_day_et[is_daily_res] += pd.Timedelta(days=1)

                aligned_day = aligned_day_et.dt.tz_localize(None)

                # 5. 交易日历对齐
                if trading_days is not None and len(trading_days) > 0:
                    idx = np.searchsorted(trading_days, aligned_day.values.astype("datetime64[ns]"), side="left")
                    out = np.full(shape=idx.shape, fill_value=np.datetime64("NaT"), dtype="datetime64[ns]")
                    ok = idx < len(trading_days)
                    out[ok] = trading_days[idx[ok]]
                    chunk["Date"] = out
                else:
                    chunk["Date"] = aligned_day.values.astype("datetime64[ns]")

                chunk = chunk.dropna(subset=["Date"])
                chunk = chunk[(chunk["Date"] >= START_DATE) & (chunk["Date"] <= END_DATE)]
            else:
                chunk = chunk.iloc[0:0]
        
        # 股票代码过滤
        if 'Stock_symbol' in chunk.columns and valid_tickers:
            chunk['Stock_symbol'] = chunk['Stock_symbol'].astype(str).str.upper().str.replace("-", ".", regex=False)
            chunk = chunk[chunk['Stock_symbol'].isin(valid_tickers)]
            chunk = chunk.rename(columns={'Stock_symbol': 'Ticker'})
            
        if not chunk.empty:
            keep_cols = ['Date', 'Ticker', 'Article_title', 'Publisher'] 
            actual_cols = [c for c in keep_cols if c in chunk.columns]
            temp_df = chunk[actual_cols].copy()
            if 'Article_title' in temp_df.columns:
                temp_df = temp_df.rename(columns={'Article_title': 'Headline'})
            cleaned_chunks.append(temp_df)
            
        if i % 10 == 0:
            print(f"已处理 {i * CHUNK_SIZE} 行...", end='\r')

    if cleaned_chunks:
        full_news_df = pd.concat(cleaned_chunks, ignore_index=True)
        output_path = os.path.join(OUTPUT_DIR, 'Stock_News.csv')
        full_news_df.to_csv(output_path, index=False)
        print(f"\n✅ 新闻清洗完成！已保存至: {output_path}")
        print(f"提取新闻条数: {len(full_news_df)}")
    else:
        print("\n⚠️ 未提取到任何有效新闻。")

# ================= 主程序入口 =================
if __name__ == "__main__":
    """
    主执行流程：
    1. 先合并股价数据，获取有效的股票代码列表
    2. 如果股价处理成功（返回了有效的股票代码列表），再处理新闻数据
    3. 新闻数据会根据股价数据中的股票代码进行过滤，确保数据一致性
    """
    # 步骤1：合并股价数据，返回有效的股票代码列表
    valid_tickers = merge_stock_prices()
    
    # 步骤2：如果股价处理成功，再处理新闻数据
    # 这样可以确保新闻数据只包含有对应股价数据的股票
    if valid_tickers:
        process_huge_news_file(valid_tickers)
