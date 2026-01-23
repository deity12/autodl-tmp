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
import os
import glob
from tqdm import tqdm

# ================= 配置路径 =================
# 【注意】请根据你的实际文件夹层级调整这里，如果有两层 full_history 记得加上
RAW_PRICES_DIR = './data/raw/FNSPID/full_history/'  # 股价CSV文件所在目录
RAW_NEWS_FILE = './data/raw/FNSPID/nasdaq_exteral_data.csv'  # 新闻数据大文件路径
OUTPUT_DIR = './data/processed/'  # 处理后的数据输出目录

# 自动创建输出目录（如果不存在）
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ================= 时间范围配置 =================
# 设定数据提取的时间范围（与论文研究期间一致）
START_DATE = '2019-01-01'  # 起始日期
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
            ticker = os.path.basename(filename).replace('.csv', '').upper()
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
    
    采用分块读取（chunk processing）方式，避免一次性加载整个文件导致内存溢出
    
    Args:
        valid_tickers (list): 有效的股票代码列表（从股价数据中提取）
    
    处理流程：
    1. 分块读取大文件（每次10万行）
    2. 对每个数据块进行时间过滤
    3. 对每个数据块进行股票代码过滤
    4. 只保留关键列，节省存储空间
    5. 合并所有清洗后的数据块并保存
    """
    print("\n>>> 正在流式处理 23GB 新闻大文件...")
    
    # 检查文件是否存在
    if not os.path.exists(RAW_NEWS_FILE):
        print(f"❌ 找不到新闻文件: {RAW_NEWS_FILE}")
        return

    # 设置分块大小：每次读取10万行，防止内存溢出
    CHUNK_SIZE = 100000 
    cleaned_chunks = []  # 存储所有清洗后的数据块
    
    # 创建分块读取器：chunksize指定每块大小，on_bad_lines='skip'跳过格式错误的行
    reader = pd.read_csv(RAW_NEWS_FILE, chunksize=CHUNK_SIZE, on_bad_lines='skip')
    
    # 逐块处理数据
    for i, chunk in enumerate(reader):
        # 1. 时间处理：过滤指定时间范围的数据，并实现 16:00 cut-off 对齐
        if 'Date' in chunk.columns:
            # 将日期列转换为datetime类型，无效日期转为NaN
            # 注意：如果原始数据包含时间戳（如 '2023-01-01 15:30:00'），pd.to_datetime 会自动解析
            chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce')
            # 删除日期为NaN的行
            chunk = chunk.dropna(subset=['Date'])
            
            # 【关键创新】16:00 Cut-off 时间对齐逻辑
            # 如果 Date 列包含时间信息（datetime），则根据时间判断归属日期
            # 如果 Date 列只有日期信息（date），则默认归入当日（假设为收盘前发布）
            if chunk['Date'].dtype == 'datetime64[ns]':
                # 提取日期部分（不包含时间）
                chunk['Date_Only'] = chunk['Date'].dt.date
                # 提取时间部分（小时和分钟）
                chunk['Time'] = chunk['Date'].dt.time
                
                # 判断每条新闻是否在 16:00 之前发布
                # 如果时间 >= 16:00，则归入次日；否则归入当日
                cutoff_time = pd.Timestamp(f"{CUTOFF_HOUR:02d}:{CUTOFF_MINUTE:02d}").time()
                chunk['Is_Before_Cutoff'] = chunk['Time'] < cutoff_time
                
                # 根据 cut-off 规则调整日期归属
                # 16:00 之前的新闻归入当日，16:00 之后的归入次日
                chunk['Aligned_Date'] = chunk['Date_Only'].copy()
                # 16:00 之后的新闻，日期加1天
                mask_after_cutoff = ~chunk['Is_Before_Cutoff']
                if mask_after_cutoff.any():
                    chunk.loc[mask_after_cutoff, 'Aligned_Date'] = (
                        pd.to_datetime(chunk.loc[mask_after_cutoff, 'Date_Only']) + pd.Timedelta(days=1)
                    ).dt.date
                
                # 使用对齐后的日期替换原始 Date 列
                chunk['Date'] = pd.to_datetime(chunk['Aligned_Date'])
                
                # 清理临时列
                chunk = chunk.drop(columns=['Date_Only', 'Time', 'Is_Before_Cutoff', 'Aligned_Date'], errors='ignore')
            else:
                # 如果只有日期信息，默认归入当日（假设为收盘前发布）
                chunk['Date'] = pd.to_datetime(chunk['Date'])
            
            # 只保留指定时间范围内的数据（使用对齐后的日期）
            chunk = chunk[(chunk['Date'] >= START_DATE) & (chunk['Date'] <= END_DATE)]
        
        # 2. 股票代码处理：只保留有对应股价数据的股票新闻
        if 'Stock_symbol' in chunk.columns and valid_tickers:
            # 统一股票代码为大写格式（与股价数据中的格式一致）
            chunk['Stock_symbol'] = chunk['Stock_symbol'].astype(str).str.upper()
            # 只保留在有效股票代码列表中的新闻
            chunk = chunk[chunk['Stock_symbol'].isin(valid_tickers)]
            # 重命名为 Ticker，与股价数据保持一致
            chunk = chunk.rename(columns={'Stock_symbol': 'Ticker'})
            
        # 3. 保存清洗后的数据块
        if not chunk.empty:
            # 【优化】: 这里的列名需要对应原始CSV里的列名
            # 如果原始列名是 Article_title，这里就要写 Article_title
            # 只保留关键列，丢弃全文内容以节省存储空间
            keep_cols = ['Date', 'Ticker', 'Article_title', 'Publisher'] 
            # 只选择实际存在的列（防止某些列不存在）
            actual_cols = [c for c in keep_cols if c in chunk.columns]
            
            # 创建临时数据框副本，避免修改原始chunk
            temp_df = chunk[actual_cols].copy()
            # 【关键】: 重命名 Article_title -> Headline，为了对接后续分析代码
            # 统一列名便于后续处理和分析
            if 'Article_title' in temp_df.columns:
                temp_df = temp_df.rename(columns={'Article_title': 'Headline'})
            
            # 将清洗后的数据块添加到列表
            cleaned_chunks.append(temp_df)
            
        # 每处理10个数据块，打印一次进度（使用\r实现同一行更新）
        if i % 10 == 0:
            print(f"已处理 {i * CHUNK_SIZE} 行...", end='\r')

    # 合并所有清洗后的数据块
    if cleaned_chunks:
        # 使用 concat 合并所有数据块，ignore_index=True 重新生成索引
        full_news_df = pd.concat(cleaned_chunks, ignore_index=True)
        output_path = os.path.join(OUTPUT_DIR, 'Stock_News.csv')
        # 保存为CSV文件，不包含行索引
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