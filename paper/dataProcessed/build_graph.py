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
import warnings

# 关闭与本项目无关/不美观的环境警告（不影响LLM建图结果）
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
warnings.filterwarnings(
    "ignore",
    message=r"Failed to load image Python extension:.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*`torch_dtype` is deprecated.*",
)

# 降低 transformers 的日志噪声（不影响结果）
try:
    from transformers.utils import logging as _hf_logging
    _hf_logging.set_verbosity_error()
except Exception:
    pass

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

# ================= 采样配置（批处理优化版）=================
# 每个股票最多采样多少条新闻（分层采样）
# 批处理后速度提升15倍，可以处理更多新闻
MAX_NEWS_PER_TICKER = 200  # 适度采样，确保质量

# 总共最多处理多少条新闻
# 批处理模式：耗时取决于 batch / 推理参数与GPU吞吐（通常为数小时量级）
MAX_TOTAL_NEWS = 100000  # 平衡质量与时间

# 是否使用 LLM（False 则使用规则匹配）
# 48GB显存完全够用，启用LLM以获得更准确的关系提取
USE_LLM_DEFAULT = True  # ⚠️ 确保启用LLM模式

# ================= LLM 推理加速配置（可用环境变量覆盖）=================
# 说明：
# - batch 越大吞吐越高，但显存占用也越大；48GB 通常可以从 32 开始尝试
# - 关系抽取只需要很短的 JSON 输出，不需要 256 token + 采样
LLM_BATCH_SIZE_DEFAULT = int(os.environ.get("LLM_BATCH_SIZE", "64"))
LLM_MAX_INPUT_TOKENS_DEFAULT = int(os.environ.get("LLM_MAX_INPUT_TOKENS", "1536"))
LLM_MAX_NEW_TOKENS_DEFAULT = int(os.environ.get("LLM_MAX_NEW_TOKENS", "96"))
LLM_DO_SAMPLE_DEFAULT = os.environ.get("LLM_DO_SAMPLE", "0") == "1"

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


def _normalize_llm_relations(parsed):
    """
    将 LLM 返回的 JSON 解析结果规整为统一格式：
    List[{"src": str, "dst": str, "relation": Optional[str]}]
    
    兼容常见“跑偏”格式：
    - [{"src":"AAPL","dst":"QCOM","relation":"supply"}]
    - [["AAPL","QCOM","supply"], ["TSLA","GM","competition"]]
    - {"relations": [...]} / {"data": [...]} 等包装
    """
    if parsed is None:
        return []

    # 有些模型会多包一层 dict
    if isinstance(parsed, dict):
        for k in ("relations", "relation", "edges", "triples", "items", "data", "results"):
            if k in parsed:
                parsed = parsed.get(k)
                break

    if not isinstance(parsed, list):
        return []

    norm = []
    for item in parsed:
        src = dst = rel = None

        if isinstance(item, dict):
            src = item.get("src") or item.get("source") or item.get("from")
            dst = item.get("dst") or item.get("target") or item.get("to")
            rel = item.get("relation") or item.get("type")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            src, dst = item[0], item[1]
            rel = item[2] if len(item) >= 3 else None
        else:
            continue

        if src is None or dst is None:
            continue

        src = str(src).strip().upper()
        dst = str(dst).strip().upper()
        rel = str(rel).strip() if rel is not None else None

        # 过滤空字符串
        if not src or not dst:
            continue

        norm.append({"src": src, "dst": dst, "relation": rel})

    return norm


def extract_relations_with_llm_batch(
    news_texts,
    local_model=None,
    local_tokenizer=None,
    batch_size=8,
    max_input_tokens=LLM_MAX_INPUT_TOKENS_DEFAULT,
    max_new_tokens=LLM_MAX_NEW_TOKENS_DEFAULT,
    do_sample=LLM_DO_SAMPLE_DEFAULT,
):
    """批处理LLM提取关系 - 保持高质量Prompt，通过批处理提速"""
    if local_model is None or local_tokenizer is None:
        return [[] for _ in news_texts]
    
    results = []
    
    # 批处理
    for i in range(0, len(news_texts), batch_size):
        batch = news_texts[i:i+batch_size]
        batch_prompts = []
        
        for text in batch:
            if not text or (isinstance(text, float) and pd.isna(text)):
                batch_prompts.append(None)
                continue
            
            text = str(text)[:500]
            
            # 使用完整的高质量prompt（与原版一致）
            prompt = f"""你是一个专业的金融关系抽取专家。请从以下财经新闻标题中提取公司之间的**显式关系**。

新闻标题：{text}

关系类型（仅限以下类型）：
1. 供应链关系 (supply): 供应商、采购、订单、合同
2. 竞争关系 (competition): 竞争对手、市场争夺、价格战
3. 合作关系 (cooperation): 合作、联盟、合资、战略伙伴
4. 并购关系 (merger): 收购、兼并、重组、出售资产
5. 诉讼关系 (lawsuit): 起诉、诉讼、法律纠纷、侵权
6. 投资关系 (investment): 投资、入股、持股、战略投资

输出要求：
1. 只提取**明确提到两家公司**且关系清晰的内容
2. 股票代码必须是**美股代码**（如AAPL、TSLA、MSFT等）
3. 如果新闻只提到一家公司，返回 []
4. 如果关系不属于以上6类，返回 []

严格按以下JSON格式输出（不要有任何其他文字）：
[{{"src": "公司A代码", "dst": "公司B代码", "relation": "关系类型"}}]

示例：
- "苹果与高通达成5年芯片供应协议" → [{{"src":"AAPL","dst":"QCOM","relation":"supply"}}]
- "特斯拉与通用汽车竞争电动车市场" → [{{"src":"TSLA","dst":"GM","relation":"competition"}}]
- "微软完成对暴雪娱乐的收购" → [{{"src":"MSFT","dst":"ATVI","relation":"merger"}}]
- "苹果发布新款iPhone" → []

现在请分析上述新闻标题："""
            
            batch_prompts.append(prompt)
        
        # 批量推理
        valid_prompts = [p for p in batch_prompts if p is not None]
        if valid_prompts:
            try:
                device = local_model.device
                
                # 批量编码所有prompt
                inputs = []
                for prompt in valid_prompts:
                    messages = [{"role": "user", "content": prompt}]
                    text_input = local_tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs.append(text_input)
                
                # 批量tokenize（关键加速点）
                model_inputs = local_tokenizer(
                    inputs, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=max_input_tokens
                ).to(device)
                
                # 批量生成
                with torch.inference_mode():
                    generated_ids = local_model.generate(
                        **model_inputs,
                        max_new_tokens=max_new_tokens,  # 关系抽取只需要很短输出
                        do_sample=do_sample,
                        temperature=0.0 if not do_sample else 0.1,
                        pad_token_id=getattr(local_tokenizer, "pad_token_id", None),
                        eos_token_id=getattr(local_tokenizer, "eos_token_id", None),
                    )
                
                # 批量解码
                valid_idx = 0
                for j, prompt in enumerate(batch_prompts):
                    if prompt is None:
                        results.append([])
                    else:
                        output_ids = generated_ids[valid_idx]
                        input_len = model_inputs.input_ids[valid_idx].shape[0]
                        generated = output_ids[input_len:]
                        raw = local_tokenizer.decode(generated, skip_special_tokens=True)
                        
                        try:
                            import json
                            if "```" in raw:
                                raw = raw.split("```")[1]
                                if raw.startswith("json"):
                                    raw = raw[4:]
                            parsed = json.loads(raw)
                            results.append(_normalize_llm_relations(parsed))
                        except:
                            results.append([])
                        
                        valid_idx += 1
                        
            except Exception as e:
                # 批处理失败时，用空结果填充
                for prompt in batch_prompts:
                    results.append([])
    
    return results


def extract_relations_with_llm(news_text, client=None, local_model=None, local_tokenizer=None):
    """单条提取（保持向后兼容）"""
    result = extract_relations_with_llm_batch([news_text], local_model, local_tokenizer, batch_size=1)
    return result[0] if result else []


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
            # decoder-only 模型必须使用 left padding，否则可能影响生成结果并产生警告
            local_tokenizer.padding_side = "left"
            if getattr(local_tokenizer, "pad_token", None) is None:
                local_tokenizer.pad_token = local_tokenizer.eos_token

            dtype = torch.float16 if device == "cuda" else torch.float32
            model_kwargs = dict(
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )
            # 尝试启用更快的注意力实现（若环境不支持会自动回退）
            if device == "cuda":
                model_kwargs["attn_implementation"] = "flash_attention_2"
            # 兼容不同 transformers 版本：优先使用新参数 dtype=（可消除 torch_dtype deprecation 提示）
            try:
                local_model = AutoModelForCausalLM.from_pretrained(
                    LOCAL_MODEL_PATH,
                    dtype=dtype,
                    **model_kwargs,
                )
            except Exception:
                # 回退：移除 flash_attention_2 或 dtype 参数差异
                model_kwargs.pop("attn_implementation", None)
                local_model = AutoModelForCausalLM.from_pretrained(
                    LOCAL_MODEL_PATH,
                    torch_dtype=dtype,
                    **model_kwargs,
                )

            # 同步 pad_token_id，避免generate阶段的padding问题
            try:
                local_model.config.pad_token_id = local_tokenizer.pad_token_id
            except Exception:
                pass
            try:
                local_model.eval()
            except Exception:
                pass
            print("[OK] 模型加载成功！")
        except Exception as e:
            print(f"[ERROR] 模型加载失败: {e}")
            print(">>> 降级为规则模拟模式。")
            local_model = None

    # =========================== 循环建图（批处理优化）===========================
    print(f"\n>>> [Step 2] 开始建图 (共 {len(df_news_sampled)} 条新闻)...")
    print("=" * 70)
    
    # 进度保存配置
    CHECKPOINT_INTERVAL = 10000
    checkpoint_path = OUTPUT_GRAPH.replace('.npy', '_checkpoint.npy')
    BATCH_SIZE = int(os.environ.get("LLM_BATCH_SIZE", str(LLM_BATCH_SIZE_DEFAULT)))
    MAX_INPUT_TOKENS = int(os.environ.get("LLM_MAX_INPUT_TOKENS", str(LLM_MAX_INPUT_TOKENS_DEFAULT)))
    MAX_NEW_TOKENS = int(os.environ.get("LLM_MAX_NEW_TOKENS", str(LLM_MAX_NEW_TOKENS_DEFAULT)))
    DO_SAMPLE = os.environ.get("LLM_DO_SAMPLE", "1" if LLM_DO_SAMPLE_DEFAULT else "0") == "1"
    
    edge_count = 0
    matched_tickers = set()
    
    if local_model:
        print(f"[批处理模式] batch={BATCH_SIZE}, max_input_tokens={MAX_INPUT_TOKENS}, max_new_tokens={MAX_NEW_TOKENS}, do_sample={DO_SAMPLE}")
        
        # 批处理LLM推理
        batch_news = []
        batch_tickers = []
        
        for idx, row in tqdm(df_news_sampled.iterrows(), total=len(df_news_sampled), desc="Building Graph"):
            src_ticker = row.get('Ticker')
            if src_ticker not in ticker2idx:
                continue
                
            content = row.get(text_col, "")
            if not content or pd.isna(content):
                continue
            
            batch_news.append(str(content))
            batch_tickers.append(src_ticker)
            
            # 达到批次大小或最后一批
            if len(batch_news) >= BATCH_SIZE or idx == df_news_sampled.index[-1]:
                # 批量推理（遇到OOM自动降低batch再重试）
                while True:
                    try:
                        batch_relations = extract_relations_with_llm_batch(
                            batch_news,
                            local_model,
                            local_tokenizer,
                            batch_size=BATCH_SIZE,
                            max_input_tokens=MAX_INPUT_TOKENS,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=DO_SAMPLE,
                        )
                        break
                    except torch.cuda.OutOfMemoryError:
                        if BATCH_SIZE <= 4:
                            raise
                        torch.cuda.empty_cache()
                        BATCH_SIZE = max(4, BATCH_SIZE // 2)
                        print(f"\n[OOM] 显存不足，自动降低 batch_size -> {BATCH_SIZE} 后继续")
                
                # 处理结果
                for src_ticker, relations in zip(batch_tickers, batch_relations):
                    if relations:
                        for r in relations:
                            # 兼容：r 可能是 dict 或 list/tuple（LLM 输出偶发跑偏）
                            if isinstance(r, dict):
                                src, dst = r.get("src"), r.get("dst")
                            elif isinstance(r, (list, tuple)) and len(r) >= 2:
                                src, dst = r[0], r[1]
                            else:
                                continue
                            if src and dst and src in ticker2idx and dst in ticker2idx and src != dst:
                                i, j = ticker2idx[src], ticker2idx[dst]
                                if adj_matrix[i, j] == 0:
                                    edge_count += 1
                                adj_matrix[i, j] = 1.0
                                adj_matrix[j, i] = 1.0
                                matched_tickers.add(src)
                                matched_tickers.add(dst)
                
                # 清空批次
                batch_news = []
                batch_tickers = []
                
                # 进度保存
                if (idx + 1) % CHECKPOINT_INTERVAL == 0:
                    np.save(checkpoint_path, adj_matrix)
                    print(f"\n[进度保存] 已处理 {idx+1}/{len(df_news_sampled)} 条 (边数: {int((adj_matrix.sum()-num_nodes)/2)})")
    else:
        # 规则模式（不变）
        for idx, row in tqdm(df_news_sampled.iterrows(), total=len(df_news_sampled), desc="Building Graph"):
            src_ticker = row.get('Ticker')
            if src_ticker not in ticker2idx:
                continue
                
            content = row.get(text_col, "")
            if not content or pd.isna(content):
                continue
            
            content = str(content)
            
            # 规则匹配
            for t in tickers:
                if t != src_ticker and len(str(t)) >= 3 and t.upper() in content.upper():
                    if t in ticker2idx:
                        i, j = ticker2idx[src_ticker], ticker2idx[t]
                        if adj_matrix[i, j] == 0:
                            edge_count += 1
                        adj_matrix[i, j] = 1.0
                        adj_matrix[j, i] = 1.0
                        matched_tickers.add(src_ticker)
                        matched_tickers.add(t)
            
            if (idx + 1) % CHECKPOINT_INTERVAL == 0:
                np.save(checkpoint_path, adj_matrix)
                print(f"\n[进度保存] 已处理 {idx+1}/{len(df_news_sampled)} 条 (边数: {int((adj_matrix.sum()-num_nodes)/2)})")

    # =========================== 保存最终结果 ===========================
    print("\n>>> [Step 3] 保存最终结果...")
    np.save(OUTPUT_GRAPH, adj_matrix)
    
    # 删除checkpoint文件
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"[清理] 已删除临时checkpoint文件")
    
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
    # 兼容旧参数：--use_llm / --no_llm，同时提供更清晰的 --llm/--no-llm
    parser.add_argument('--use_llm', action='store_true', help='(兼容) 强制启用LLM')
    parser.add_argument('--no_llm', action='store_true', help='(兼容) 强制禁用LLM（规则匹配）')
    parser.add_argument('--llm', action=argparse.BooleanOptionalAction, default=USE_LLM_DEFAULT,
                        help=f'是否使用LLM（默认: {USE_LLM_DEFAULT}）')
    parser.add_argument('--max_per_ticker', type=int, default=MAX_NEWS_PER_TICKER, help='每个股票最多采样多少条新闻')
    parser.add_argument('--max_total', type=int, default=MAX_TOTAL_NEWS, help='总共最多处理多少条新闻')
    parser.add_argument('--all_stocks', action='store_true', help='使用全量股票（默认只用 S&P 500）')
    
    args = parser.parse_args()
    
    # LLM开关：默认取 --llm 的值，但旧参数可覆盖
    use_llm_mode = bool(args.llm)
    if args.no_llm:
        use_llm_mode = False
    if args.use_llm:
        use_llm_mode = True
    
    print("\n" + "=" * 70)
    print("📊 股票关系图谱构建工具")
    print("=" * 70)
    print(f"配置:")
    print(f"  - 股票范围: {'全量' if args.all_stocks else 'S&P 500 成分股（推荐）'}")
    print(f"  - 关系提取: {'🧠 LLM语义提取 (Qwen2.5-14B)' if use_llm_mode else '📋 规则匹配'}")
    print(f"  - 每股票采样: {args.max_per_ticker} 条新闻")
    print(f"  - 总采样上限: {args.max_total} 条")
    if use_llm_mode:
        bs = int(os.environ.get("LLM_BATCH_SIZE", str(LLM_BATCH_SIZE_DEFAULT)))
        mi = int(os.environ.get("LLM_MAX_INPUT_TOKENS", str(LLM_MAX_INPUT_TOKENS_DEFAULT)))
        mn = int(os.environ.get("LLM_MAX_NEW_TOKENS", str(LLM_MAX_NEW_TOKENS_DEFAULT)))
        ds = os.environ.get("LLM_DO_SAMPLE", "1" if LLM_DO_SAMPLE_DEFAULT else "0")
        print(f"  - 批处理大小: {bs} 条/批（可用环境变量 LLM_BATCH_SIZE 调整）")
        print(f"  - 推理参数: max_input_tokens={mi}, max_new_tokens={mn}, do_sample={ds}")
    print("=" * 70 + "\n")
    
    build_dynamic_graph(
        use_llm=use_llm_mode, 
        max_per_ticker=args.max_per_ticker,
        max_total=args.max_total,
        use_sp500=not args.all_stocks  # 默认使用 S&P 500
    )
