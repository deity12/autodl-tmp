# -*- coding: utf-8 -*-
"""
LLM 动态图谱构建 (最终修正版)
========================================================================
修正点：
1. [关键] 模型路径修正为 Qwen2.5 (带点)，指向真实下载的权重文件
2. [关键] 修复了时区比较 Bug (Invalid comparison)，防止防泄露逻辑失效
3. 强制使用本地模型
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch # 提前导入，防止延迟加载报错

# ================= 路径配置 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')

INPUT_NEWS = os.path.join(DATA_PROCESSED, 'Stock_News.csv')
INPUT_MODEL_DATA = os.path.join(DATA_PROCESSED, 'Final_Model_Data.csv')
OUTPUT_GRAPH = os.path.join(DATA_PROCESSED, 'Graph_Adjacency.npy')

# LLM 配置
API_KEY = None
BASE_URL = None

# 本地模型配置
USE_LOCAL_MODEL = True
LOCAL_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

# 【关键修正】这里必须指向那个带点的文件夹！
LOCAL_MODEL_PATH = os.environ.get(
    "LOCAL_MODEL_PATH", 
    "/root/autodl-tmp/models/qwen/Qwen2.5-14B-Instruct" 
)

# 规则模拟时，最多用多少条新闻来建图
MAX_NEWS_FOR_RULE = 2000


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
        # 本地模型模式
        if local_model is not None and local_tokenizer is not None:
            messages = [{"role": "user", "content": prompt}]
            text_input = local_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # 自动获取设备
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


def build_dynamic_graph(use_llm=False, max_news=MAX_NEWS_FOR_RULE):
    print(">>> [Step 1] 读取模型数据与新闻...")

    if not os.path.exists(INPUT_MODEL_DATA):
        print(f"[ERROR] 未找到 {INPUT_MODEL_DATA}")
        return

    df_price = pd.read_csv(INPUT_MODEL_DATA)
    tickers = sorted(df_price['Ticker'].unique())
    ticker2idx = {t: i for i, t in enumerate(tickers)}
    num_nodes = len(tickers)
    print(f"    检测到 {num_nodes} 只股票。")

    if not os.path.exists(INPUT_NEWS):
        adj_matrix = np.eye(num_nodes, dtype=np.float32)
        np.save(OUTPUT_GRAPH, adj_matrix)
        return

    df_news = pd.read_csv(INPUT_NEWS, low_memory=False)

    # =========================== 防止“未来信息”数据泄露 (修正版) ===========================
    try:
        if 'Date' in df_news.columns:
            # 1. 转换时间格式
            df_news['Date'] = pd.to_datetime(df_news['Date'], errors='coerce')
            
            # 2. 【关键】强制去除时区信息，防止 UTC 与 None 比较报错
            if df_news['Date'].dt.tz is not None:
                df_news['Date'] = df_news['Date'].dt.tz_localize(None)

            # 3. 获取切分点
            df_price_for_split = pd.read_csv(INPUT_MODEL_DATA, usecols=['Date'])
            df_price_for_split['Date'] = pd.to_datetime(df_price_for_split['Date'])
            unique_dates = sorted(df_price_for_split['Date'].unique())
            
            if len(unique_dates) >= 2:
                split_idx = int(len(unique_dates) * 0.8)
                split_idx = min(split_idx, len(unique_dates) - 1)
                split_date = unique_dates[split_idx]
                
                print(f"[防泄露] 切分日期 split_date = {split_date}")
                before_news = len(df_news)
                df_news = df_news[df_news['Date'] < split_date].copy()
                print(f"[防泄露] 过滤后保留新闻: {before_news} -> {len(df_news)}")
            else:
                print("[WARN] 日期不足，跳过过滤。")
    except Exception as e:
        print(f"[ERROR] 时间过滤失败: {e}，将使用全量新闻（存在泄露风险）。")
    # =================================================================================

    text_col = 'Headline' if 'Headline' in df_news.columns else 'Article_title'
    if text_col not in df_news.columns:
        cols = [c for c in df_news.columns if df_news[c].dtype == object]
        text_col = cols[0] if cols else None
    
    if text_col is None:
        print("[WARN] 没找到文本列，保存单位阵。")
        np.save(OUTPUT_GRAPH, np.eye(num_nodes, dtype=np.float32))
        return

    adj_matrix = np.eye(num_nodes, dtype=np.float32)

    # 加载模型
    local_model = None
    local_tokenizer = None
    
    if use_llm:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"[加载中] 正在加载本地模型: {LOCAL_MODEL_PATH} ...")
            print("注意：因为你显存很大(48G)，这里直接加载 FP16 全精度模型，速度会比较快！")
            
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
            print("请检查路径是否正确，或尝试删除 'Qwen2___5' 那个空文件夹。")
            print(">>> 降级为规则模拟模式。")
            local_model = None

    # 循环建图
    n_news = min(max_news, len(df_news))
    print(f">>> [Step 2] 开始建图 (共 {n_news} 条)...")
    
    it = df_news.head(n_news).iterrows()
    for _, row in tqdm(it, total=n_news):
        src_ticker = row.get('Ticker')
        if src_ticker not in ticker2idx:
            continue
            
        content = row.get(text_col, "")
        
        relations = []
        if local_model:
            relations = extract_relations_with_llm(content, local_model=local_model, local_tokenizer=local_tokenizer)
        
        if relations:
            for r in relations:
                src, dst = r.get("src"), r.get("dst")
                if src and dst and src in ticker2idx and dst in ticker2idx and src != dst:
                    i, j = ticker2idx[src], ticker2idx[dst]
                    adj_matrix[i, j] = 1.0
                    adj_matrix[j, i] = 1.0
        else:
            # 规则兜底
            for t in tickers:
                if t != src_ticker and len(str(t)) >= 2 and str(t) in str(content):
                    if t in ticker2idx:
                        i, j = ticker2idx[src_ticker], ticker2idx[t]
                        adj_matrix[i, j] = 1.0
                        adj_matrix[j, i] = 1.0

    print(">>> [Step 3] 保存结果...")
    np.save(OUTPUT_GRAPH, adj_matrix)
    print(f"[OK] 已保存至 {OUTPUT_GRAPH}，形状: {adj_matrix.shape}")


if __name__ == "__main__":
    # 强制启用本地模型
    build_dynamic_graph(use_llm=True, max_news=MAX_NEWS_FOR_RULE)