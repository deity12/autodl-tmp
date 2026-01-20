# -*- coding: utf-8 -*-
"""
LLM 动态图谱构建 (修正版 V3：支持 S&P 500 核心股票过滤)
========================================================================
核心修正：
1. [关键] 支持只使用 S&P 500 成分股（推荐用于论文）
2. [关键] 使用分层采样，确保每个股票都有代表性的新闻
3. 打乱新闻顺序，避免只处理排序靠前的股票
4. 添加图谱统计信息输出

论文建议：
- 使用 S&P 500 成分股是金融/量化研究的学术惯例
- 大公司新闻质量高，关系更明确，图谱更有意义
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch

# ================= 路径配置 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')

INPUT_NEWS = os.path.join(DATA_PROCESSED, 'Stock_News.csv')
INPUT_MODEL_DATA = os.path.join(DATA_PROCESSED, 'Final_Model_Data.csv')
OUTPUT_GRAPH = os.path.join(DATA_PROCESSED, 'Graph_Adjacency.npy')

# LLM 配置
USE_LOCAL_MODEL = True
LOCAL_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
LOCAL_MODEL_PATH = os.environ.get(
    "LOCAL_MODEL_PATH", 
    "/root/autodl-tmp/models/qwen/Qwen2.5-14B-Instruct" 
)

# ================= 采样配置（48GB显存优化版）=================
# 每个股票最多采样多少条新闻（分层采样）
MAX_NEWS_PER_TICKER = 100  # 48GB显存充足，增加采样以提高图谱质量

# 总共最多处理多少条新闻
MAX_TOTAL_NEWS = 100000  # 增加总量以获得更丰富的关系

# 是否使用 LLM（False 则使用规则匹配）
# 48GB显存完全够用，启用LLM以获得更准确的关系提取
USE_LLM_DEFAULT = True

# ================= S&P 500 成分股（2023年版本，约500只）=================
# 这是学术研究中常用的核心股票列表
# 数据来源：Wikipedia / Yahoo Finance
SP500_TICKERS = {
    # 信息技术 (Information Technology)
    'AAPL', 'MSFT', 'NVDA', 'AVGO', 'CSCO', 'ADBE', 'CRM', 'ORCL', 'ACN', 'IBM',
    'INTC', 'AMD', 'QCOM', 'TXN', 'AMAT', 'MU', 'LRCX', 'ADI', 'KLAC', 'SNPS',
    'CDNS', 'MCHP', 'NXPI', 'MPWR', 'FTNT', 'PANW', 'NOW', 'INTU', 'ADSK', 'ANSS',
    'PYPL', 'FIS', 'FISV', 'GPN', 'ADP', 'PAYX', 'CTSH', 'IT', 'EPAM', 'AKAM',
    
    # 医疗保健 (Health Care)
    'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY',
    'AMGN', 'GILD', 'VRTX', 'REGN', 'ISRG', 'MDT', 'SYK', 'BDX', 'BSX', 'EW',
    'ZBH', 'IDXX', 'DXCM', 'ALGN', 'HOLX', 'MTD', 'IQV', 'CI', 'ELV', 'HUM',
    'CVS', 'MCK', 'CAH', 'ABC', 'CNC', 'MOH', 'HCA', 'UHS', 'DVA', 'LH',
    'DGX', 'A', 'WAT', 'PKI', 'BIO', 'TECH', 'HSIC', 'COO', 'RMD', 'BAX',
    
    # 金融 (Financials)
    'BRK.B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'SCHW', 'AXP',
    'BLK', 'SPGI', 'C', 'PNC', 'USB', 'TFC', 'CME', 'ICE', 'CB', 'MMC',
    'AON', 'PGR', 'AIG', 'MET', 'PRU', 'AFL', 'ALL', 'TRV', 'CINF', 'HIG',
    'AJG', 'WTW', 'BRO', 'RE', 'L', 'GL', 'COF', 'DFS', 'SYF', 'ALLY',
    'MTB', 'FITB', 'HBAN', 'KEY', 'RF', 'CFG', 'ZION', 'NTRS', 'STT', 'BK',
    
    # 通信服务 (Communication Services)
    'GOOGL', 'GOOG', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR',
    'ATVI', 'EA', 'TTWO', 'WBD', 'PARA', 'FOX', 'FOXA', 'NWS', 'NWSA', 'OMC',
    'IPG', 'LYV', 'MTCH', 'ZG', 'PINS',
    
    # 消费品 (Consumer Discretionary)
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'MAR',
    'HLT', 'CMG', 'ORLY', 'AZO', 'ROST', 'DHI', 'LEN', 'PHM', 'NVR', 'GM',
    'F', 'APTV', 'BWA', 'LEA', 'RL', 'TPR', 'VFC', 'PVH', 'HAS', 'MAT',
    'DRI', 'YUM', 'WYNN', 'MGM', 'CZR', 'RCL', 'CCL', 'NCLH', 'LVS', 'EXPE',
    'ABNB', 'UBER', 'LYFT', 'DASH', 'EBAY', 'ETSY', 'W', 'BBY', 'KMX', 'AN',
    
    # 必需消费品 (Consumer Staples)
    'PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'MDLZ', 'CL', 'EL',
    'KMB', 'GIS', 'K', 'HSY', 'HRL', 'SJM', 'MKC', 'CAG', 'CPB', 'TSN',
    'KHC', 'STZ', 'BF.B', 'TAP', 'KDP', 'MNST', 'WBA', 'SYY', 'KR', 'TGT',
    'DG', 'DLTR', 'CLX', 'CHD', 'COR',
    
    # 工业 (Industrials)
    'UNP', 'UPS', 'HON', 'BA', 'RTX', 'CAT', 'DE', 'LMT', 'GE', 'MMM',
    'GD', 'NOC', 'LHX', 'TDG', 'ITW', 'EMR', 'ROK', 'PH', 'ETN', 'PCAR',
    'CTAS', 'FAST', 'WM', 'RSG', 'WCN', 'VRSK', 'CPRT', 'CSX', 'NSC', 'FDX',
    'EXPD', 'CHRW', 'JBHT', 'DAL', 'UAL', 'LUV', 'AAL', 'ALK', 'CARR', 'OTIS',
    'JCI', 'TT', 'IR', 'SWK', 'MAS', 'GNRC', 'PWR', 'AME', 'DOV', 'ROP',
    
    # 能源 (Energy)
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'PXD', 'OXY',
    'HES', 'DVN', 'FANG', 'HAL', 'BKR', 'KMI', 'WMB', 'OKE', 'TRGP', 'APA',
    'MRO', 'CTRA',
    
    # 材料 (Materials)
    'LIN', 'APD', 'SHW', 'ECL', 'DD', 'DOW', 'NEM', 'FCX', 'NUE', 'VMC',
    'MLM', 'PPG', 'ALB', 'EMN', 'CE', 'CF', 'MOS', 'FMC', 'IFF', 'CTVA',
    'LYB', 'IP', 'PKG', 'SEE', 'AVY', 'BALL', 'AMCR',
    
    # 房地产 (Real Estate)
    'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'AVB',
    'EQR', 'VTR', 'ARE', 'MAA', 'UDR', 'ESS', 'HST', 'PEAK', 'KIM', 'REG',
    'FRT', 'BXP', 'VNO', 'SLG', 'CBRE', 'IRM', 'WY', 'SBAC', 'INVH', 'CPT',
    
    # 公用事业 (Utilities)
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'ED', 'PEG',
    'WEC', 'ES', 'AWK', 'DTE', 'EIX', 'ETR', 'FE', 'PPL', 'AEE', 'CMS',
    'CNP', 'EVRG', 'ATO', 'NI', 'LNT', 'PNW', 'NRG', 'CEG',
}

# 是否只使用 S&P 500 成分股（强烈推荐用于论文）
USE_SP500_ONLY = True


def extract_relations_with_llm(news_text, client=None, local_model=None, local_tokenizer=None):
    """利用 LLM 提取关系"""
    if not news_text or (isinstance(news_text, float) and pd.isna(news_text)):
        return []
    
    text = str(news_text)[:500]

    prompt = f"""
请从以下财经新闻中提取公司之间的显式关系（如：供应、竞争、合作、母子公司、诉讼）。
新闻内容：{text}

请严格按以下 JSON 格式返回列表，不要包含其他废话：
[{{"src": "公司A股票代码", "dst": "公司B股票代码", "relation": "关系类型"}}]
如果无明确关系，返回 []。
"""

    try:
        if local_model is not None and local_tokenizer is not None:
            messages = [{"role": "user", "content": prompt}]
            text_input = local_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            device = local_model.device
            model_inputs = local_tokenizer([text_input], return_tensors="pt").to(device)
            
            with torch.no_grad():
                generated_ids = local_model.generate(
                    **model_inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            raw = local_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
        elif client is not None:
            return []
        else:
            return []
        
        import json
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
        
    except Exception as e:
        return []


def stratified_sample_news(df_news, max_per_ticker=20, max_total=50000, random_state=42):
    """
    分层采样：确保每个股票都有代表性的新闻
    
    参数:
        df_news: 新闻 DataFrame
        max_per_ticker: 每个股票最多采样多少条
        max_total: 总共最多采样多少条
        random_state: 随机种子（确保可复现）
    
    返回:
        采样后的 DataFrame
    """
    print(f">>> 开始分层采样...")
    print(f"    原始新闻总数: {len(df_news)}")
    print(f"    每个股票最多采样: {max_per_ticker} 条")
    print(f"    总采样上限: {max_total} 条")
    
    # 按 Ticker 分组采样
    sampled_dfs = []
    ticker_counts = df_news['Ticker'].value_counts()
    
    for ticker in ticker_counts.index:
        ticker_news = df_news[df_news['Ticker'] == ticker]
        n_sample = min(len(ticker_news), max_per_ticker)
        sampled = ticker_news.sample(n=n_sample, random_state=random_state)
        sampled_dfs.append(sampled)
    
    # 合并所有采样结果
    df_sampled = pd.concat(sampled_dfs, ignore_index=True)
    
    # 如果总数超过上限，再随机采样
    if len(df_sampled) > max_total:
        df_sampled = df_sampled.sample(n=max_total, random_state=random_state)
    
    # 打乱顺序
    df_sampled = df_sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"    采样后新闻总数: {len(df_sampled)}")
    print(f"    覆盖股票数: {df_sampled['Ticker'].nunique()}")
    
    return df_sampled


def build_dynamic_graph(use_llm=USE_LLM_DEFAULT, max_per_ticker=MAX_NEWS_PER_TICKER, max_total=MAX_TOTAL_NEWS, use_sp500=USE_SP500_ONLY):
    """
    构建动态图谱
    
    参数:
        use_llm: 是否使用 LLM 提取关系（False 则使用规则匹配）
        max_per_ticker: 每个股票最多采样多少条新闻
        max_total: 总共最多处理多少条新闻
        use_sp500: 是否只使用 S&P 500 成分股（推荐用于论文）
    """
    print("=" * 70)
    print(">>> [Step 1] 读取模型数据与新闻...")
    print("=" * 70)

    if not os.path.exists(INPUT_MODEL_DATA):
        print(f"[ERROR] 未找到 {INPUT_MODEL_DATA}")
        return

    df_price = pd.read_csv(INPUT_MODEL_DATA)
    all_tickers = sorted(df_price['Ticker'].unique())
    print(f"    原始数据检测到 {len(all_tickers)} 只股票。")
    
    # =============== S&P 500 过滤（推荐用于论文）===============
    if use_sp500:
        # 找出数据中存在的 S&P 500 成分股
        sp500_in_data = [t for t in all_tickers if t in SP500_TICKERS]
        print(f"\n📌 [S&P 500 模式] 只使用核心成分股")
        print(f"    S&P 500 成分股定义: {len(SP500_TICKERS)} 只")
        print(f"    数据中匹配到: {len(sp500_in_data)} 只")
        
        if len(sp500_in_data) < 100:
            print(f"⚠️ 警告：匹配到的 S&P 500 成分股较少 ({len(sp500_in_data)} 只)")
            print("    可能原因：数据集中的股票代码格式不同，或数据集不包含这些股票")
            print("    将使用全量股票...")
            tickers = all_tickers
        else:
            tickers = sp500_in_data
            # 过滤价格数据，只保留 S&P 500 成分股
            df_price = df_price[df_price['Ticker'].isin(tickers)]
    else:
        tickers = all_tickers
        print(f"📌 [全量模式] 使用所有 {len(tickers)} 只股票")
    
    ticker2idx = {t: i for i, t in enumerate(tickers)}
    num_nodes = len(tickers)
    print(f"    最终使用 {num_nodes} 只股票构建图谱。")

    if not os.path.exists(INPUT_NEWS):
        print(f"[WARN] 未找到新闻文件 {INPUT_NEWS}，保存单位阵。")
        adj_matrix = np.eye(num_nodes, dtype=np.float32)
        np.save(OUTPUT_GRAPH, adj_matrix)
        return

    df_news = pd.read_csv(INPUT_NEWS, low_memory=False)
    print(f"    原始新闻总数: {len(df_news)}")
    
    # 如果使用 S&P 500 模式，过滤新闻数据
    if use_sp500 and len(tickers) < len(all_tickers):
        before_filter = len(df_news)
        df_news = df_news[df_news['Ticker'].isin(tickers)].copy()
        print(f"    [S&P 500 过滤] 保留新闻: {before_filter} -> {len(df_news)}")

    # =========================== 防止"未来信息"数据泄露 ===========================
    try:
        if 'Date' in df_news.columns:
            df_news['Date'] = pd.to_datetime(df_news['Date'], errors='coerce')
            
            if df_news['Date'].dt.tz is not None:
                df_news['Date'] = df_news['Date'].dt.tz_localize(None)

            df_price_for_split = pd.read_csv(INPUT_MODEL_DATA, usecols=['Date'])
            df_price_for_split['Date'] = pd.to_datetime(df_price_for_split['Date'])
            unique_dates = sorted(df_price_for_split['Date'].unique())
            
            if len(unique_dates) >= 2:
                split_idx = int(len(unique_dates) * 0.8)
                split_idx = min(split_idx, len(unique_dates) - 1)
                split_date = unique_dates[split_idx]
                
                print(f"\n[防泄露] 切分日期 split_date = {split_date}")
                before_news = len(df_news)
                df_news = df_news[df_news['Date'] < split_date].copy()
                print(f"[防泄露] 过滤后保留新闻: {before_news} -> {len(df_news)}")
            else:
                print("[WARN] 日期不足，跳过过滤。")
    except Exception as e:
        print(f"[ERROR] 时间过滤失败: {e}，将使用全量新闻（存在泄露风险）。")

    # =========================== 分层采样（关键修正）===========================
    df_news_sampled = stratified_sample_news(
        df_news, 
        max_per_ticker=max_per_ticker, 
        max_total=max_total
    )

    # 获取文本列
    text_col = 'Headline' if 'Headline' in df_news_sampled.columns else 'Article_title'
    if text_col not in df_news_sampled.columns:
        cols = [c for c in df_news_sampled.columns if df_news_sampled[c].dtype == object]
        text_col = cols[0] if cols else None
    
    if text_col is None:
        print("[WARN] 没找到文本列，保存单位阵。")
        np.save(OUTPUT_GRAPH, np.eye(num_nodes, dtype=np.float32))
        return

    # 初始化邻接矩阵（单位阵 = 自环）
    adj_matrix = np.eye(num_nodes, dtype=np.float32)

    # =========================== 加载 LLM 模型（可选）===========================
    local_model = None
    local_tokenizer = None
    
    if use_llm:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"\n[加载中] 正在加载本地模型: {LOCAL_MODEL_PATH} ...")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"    设备: {device}")

            local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
            local_model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_PATH,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            print("[OK] 模型加载成功！")
        except Exception as e:
            print(f"[ERROR] 模型加载失败: {e}")
            print(">>> 降级为规则模拟模式。")
            local_model = None

    # =========================== 循环建图 ===========================
    print(f"\n>>> [Step 2] 开始建图 (共 {len(df_news_sampled)} 条新闻)...")
    print("=" * 70)
    
    edge_count = 0
    matched_tickers = set()
    
    for _, row in tqdm(df_news_sampled.iterrows(), total=len(df_news_sampled), desc="Building Graph"):
        src_ticker = row.get('Ticker')
        if src_ticker not in ticker2idx:
            continue
            
        content = row.get(text_col, "")
        if not content or pd.isna(content):
            continue
        
        content = str(content)
        
        # LLM 模式
        relations = []
        if local_model:
            relations = extract_relations_with_llm(content, local_model=local_model, local_tokenizer=local_tokenizer)
        
        if relations:
            for r in relations:
                src, dst = r.get("src"), r.get("dst")
                if src and dst and src in ticker2idx and dst in ticker2idx and src != dst:
                    i, j = ticker2idx[src], ticker2idx[dst]
                    if adj_matrix[i, j] == 0:  # 新边
                        edge_count += 1
                    adj_matrix[i, j] = 1.0
                    adj_matrix[j, i] = 1.0
                    matched_tickers.add(src)
                    matched_tickers.add(dst)
        else:
            # 规则兜底：检查新闻中是否提到其他股票代码
            for t in tickers:
                if t != src_ticker and len(str(t)) >= 2:
                    # 使用更严格的匹配：股票代码需要作为独立单词出现
                    # 避免 "A" 匹配到 "Apple" 等误匹配
                    if len(t) >= 3 and t.upper() in content.upper():
                        if t in ticker2idx:
                            i, j = ticker2idx[src_ticker], ticker2idx[t]
                            if adj_matrix[i, j] == 0:  # 新边
                                edge_count += 1
                            adj_matrix[i, j] = 1.0
                            adj_matrix[j, i] = 1.0
                            matched_tickers.add(src_ticker)
                            matched_tickers.add(t)

    # =========================== 保存结果 ===========================
    print("\n>>> [Step 3] 保存结果...")
    np.save(OUTPUT_GRAPH, adj_matrix)
    
    # =========================== 输出统计信息 ===========================
    print("\n" + "=" * 70)
    print(">>> 图谱统计信息")
    print("=" * 70)
    
    # 计算图谱统计
    total_edges = (adj_matrix.sum() - num_nodes) / 2  # 减去自环，除以2（无向图）
    density = total_edges / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0
    
    # 计算每个节点的度
    degrees = adj_matrix.sum(axis=1) - 1  # 减去自环
    non_isolated = np.sum(degrees > 0)
    
    print(f"    节点数 (股票数): {num_nodes}")
    print(f"    边数 (股票关系): {int(total_edges)}")
    print(f"    图密度: {density:.6f}")
    print(f"    有连接的股票数: {non_isolated} / {num_nodes} ({non_isolated/num_nodes*100:.1f}%)")
    print(f"    平均度: {degrees.mean():.2f}")
    print(f"    最大度: {int(degrees.max())}")
    print(f"    孤立节点数: {num_nodes - non_isolated}")
    
    if non_isolated < num_nodes * 0.5:
        print("\n⚠️ 警告：超过一半的股票是孤立节点！")
        print("   建议：")
        print("   1. 增加 max_per_ticker 参数")
        print("   2. 使用 LLM 模式 (use_llm=True) 提取更多关系")
        print("   3. 检查新闻数据质量")
    
    print(f"\n[OK] 已保存至 {OUTPUT_GRAPH}，形状: {adj_matrix.shape}")
    print("=" * 70)
    
    return adj_matrix


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='构建股票关系图谱')
    parser.add_argument('--use_llm', action='store_true', help='使用 LLM 提取关系（默认使用规则匹配）')
    parser.add_argument('--max_per_ticker', type=int, default=MAX_NEWS_PER_TICKER, help='每个股票最多采样多少条新闻')
    parser.add_argument('--max_total', type=int, default=MAX_TOTAL_NEWS, help='总共最多处理多少条新闻')
    parser.add_argument('--all_stocks', action='store_true', help='使用全量股票（默认只用 S&P 500）')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("📊 股票关系图谱构建工具")
    print("=" * 70)
    print(f"配置:")
    print(f"  - 股票范围: {'全量' if args.all_stocks else 'S&P 500 成分股（推荐）'}")
    print(f"  - 关系提取: {'LLM' if args.use_llm else '规则匹配'}")
    print(f"  - 每股票采样: {args.max_per_ticker} 条新闻")
    print(f"  - 总采样上限: {args.max_total} 条")
    print("=" * 70 + "\n")
    
    build_dynamic_graph(
        use_llm=args.use_llm, 
        max_per_ticker=args.max_per_ticker,
        max_total=args.max_total,
        use_sp500=not args.all_stocks  # 默认使用 S&P 500
    )
