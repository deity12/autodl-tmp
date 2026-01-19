# -*- coding: utf-8 -*-
"""
LLM 动态图谱构建 (对应论文 2.1 模块一：基于LLM的动态图谱)
========================================================================
功能：
    1. 从财经新闻（Stock_News.csv）中提取公司之间的关系（供应、竞争、合作等）
    2. 利用 LLM（可选）或规则从新闻文本中识别"公司A-关系-公司B"
    3. 构建邻接矩阵 Graph_Adjacency.npy，供 GNN 模块使用

【论文创新点：Headline-only 策略】
    考虑到全量新闻处理的算力成本与噪音问题，本项目创新性地采用"仅标题 (Headline-only)"策略。
    仅使用新闻标题进行关系抽取，大幅降低计算成本，同时标题通常包含最核心的实体关系信息。

输出：
    - ./data/processed/Graph_Adjacency.npy：邻接矩阵，形状 (N, N)，N=股票数量
    - 矩阵与 Final_Model_Data.csv 中的 Ticker 顺序一致（sorted），便于 dataset/model 对齐

注意：
    - 若无可用的 LLM API Key 或本地模型，会使用"规则模拟"从新闻中匹配股票代码来建边，保证流程可跑通
    - 优先使用本地模型（Qwen2.5-14B-Instruct），其次使用 API，最后使用规则模拟
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# ================= 路径配置（相对项目根目录，兼容从 dataProcessed 或项目根运行）=================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')

# 输入：新闻数据、用于对齐 Ticker 的模型数据
INPUT_NEWS = os.path.join(DATA_PROCESSED, 'Stock_News.csv')
INPUT_MODEL_DATA = os.path.join(DATA_PROCESSED, 'Final_Model_Data.csv')
# 输出：邻接矩阵
OUTPUT_GRAPH = os.path.join(DATA_PROCESSED, 'Graph_Adjacency.npy')

# LLM 配置（可选，若不用 LLM 则保持 None，会走规则模拟）
API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or "你的API_KEY"
BASE_URL = os.environ.get("LLM_BASE_URL") or "https://api.deepseek.com"

# 本地模型配置（优先使用本地模型）
USE_LOCAL_MODEL = os.environ.get("USE_LOCAL_MODEL", "").lower() == "true"
LOCAL_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"  # HuggingFace 模型名
# 如果模型已下载到本地，可以使用本地路径（根据实际路径调整）
LOCAL_MODEL_PATH = "/root/autodl-tmp/models/qwen/Qwen2___5-14B-Instruct"

# 规则模拟时，最多用多少条新闻来建图（控制运行时间）
MAX_NEWS_FOR_RULE = 2000


def extract_relations_with_llm(news_text, client=None, local_model=None, local_tokenizer=None):
    """
    利用 LLM 从新闻文本中提取实体关系（公司-关系-公司）。
    返回格式: [{"src": "Ticker1", "dst": "Ticker2", "relation": "关系类型"}, ...]

    支持两种模式：
    1. API 模式：使用 client（OpenAI 兼容接口）
    2. 本地模型模式：使用 local_model 和 local_tokenizer

    若 LLM 调用失败或未配置，返回 []，由上层用规则模拟替代。
    """
    if not news_text or (isinstance(news_text, float) and pd.isna(news_text)):
        return []
    # 【Headline-only 策略】标题通常较短，截断到 500 字符足够（标题一般 < 200 字符）
    # 此截断主要防止异常长标题，符合论文"仅标题"的轻量化设计
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
            import torch
            messages = [{"role": "user", "content": prompt}]
            # 使用 tokenizer 的 apply_chat_template 格式化输入
            text_input = local_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # 获取模型设备（兼容 device_map="auto" 和手动指定设备的情况）
            if hasattr(local_model, 'device'):
                device = local_model.device
            elif hasattr(local_model, 'hf_device_map') and local_model.hf_device_map:
                # 对于多设备模型，使用第一个设备的第一个参数
                first_device = next(iter(local_model.hf_device_map.values()))
                device = torch.device(first_device)
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        # API 模式
        elif client is not None:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            raw = response.choices[0].message.content
        else:
            return []
        
        # 尝试解析 JSON（这里简化，实际建议用 json.loads + 正则兜底）
        import json
        # 去掉 markdown 代码块
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        print(f"  [LLM] 解析或调用失败: {e}")
        return []


def build_dynamic_graph(use_llm=False, max_news=MAX_NEWS_FOR_RULE):
    """
    主流程：读取新闻与模型数据 -> 建 Ticker 索引 -> 从新闻建边 -> 保存邻接矩阵。

    参数:
        use_llm: 是否调用 LLM 做关系抽取；False 时用规则（新闻中出现另一只股票代码即建边）
        max_news: 规则模式下最多处理的新闻条数
    """
    print(">>> [Step 1] 读取模型数据与新闻...")

    if not os.path.exists(INPUT_MODEL_DATA):
        print(f"[ERROR] 未找到 {INPUT_MODEL_DATA}，请先运行 align.py 生成 Final_Model_Data.csv")
        return

    # 从 Final_Model_Data 读取 Ticker，与 dataset.py 使用的股票集合、顺序对齐
    df_price = pd.read_csv(INPUT_MODEL_DATA)
    tickers = sorted(df_price['Ticker'].unique())
    ticker2idx = {t: i for i, t in enumerate(tickers)}
    num_nodes = len(tickers)
    print(f"    检测到 {num_nodes} 只股票，Ticker 顺序与 Dataset 一致（sorted）。")

    if not os.path.exists(INPUT_NEWS):
        print(f"[WARN] 未找到 {INPUT_NEWS}，将生成仅带自环的邻接矩阵（单位阵）。")
        adj_matrix = np.eye(num_nodes, dtype=np.float32)
        np.save(OUTPUT_GRAPH, adj_matrix)
        print(f">>> 已保存至 {OUTPUT_GRAPH}，形状: {adj_matrix.shape}")
        return

    df_news = pd.read_csv(INPUT_NEWS, low_memory=False)
    # 【论文创新点：Headline-only 策略】优先使用标题列，符合论文"仅标题"的轻量化设计
    # 标题包含核心实体关系信息，且处理成本远低于全文
    text_col = 'Headline' if 'Headline' in df_news.columns else 'Article_title'
    if text_col not in df_news.columns:
        text_col = [c for c in df_news.columns if df_news[c].dtype == object][0] if any(df_news[c].dtype == object for c in df_news.columns) else None
    if text_col is None:
        print("[WARN] 新闻中未找到文本列，将保存单位阵。")
        adj_matrix = np.eye(num_nodes, dtype=np.float32)
        np.save(OUTPUT_GRAPH, adj_matrix)
        print(f">>> 已保存至 {OUTPUT_GRAPH}，形状: {adj_matrix.shape}")
        return

    # 初始化邻接矩阵：自环 1，其余 0
    adj_matrix = np.eye(num_nodes, dtype=np.float32)

    # 是否使用 LLM（支持本地模型和 API 模式）
    client = None
    local_model = None
    local_tokenizer = None
    
    if use_llm:
        # 优先使用本地模型
        if USE_LOCAL_MODEL:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                print(f"[加载中] 正在加载本地模型 {LOCAL_MODEL_NAME}（首次运行需下载，约 14GB）...")
                
                # 检查本地路径是否存在，如果存在则使用本地路径，否则使用模型名（会从 HuggingFace 下载）
                model_path = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else LOCAL_MODEL_NAME
                
                # 检查设备（优先使用 GPU）
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"    使用设备: {device}")
                
                # 加载 tokenizer
                local_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                
                # 加载模型
                local_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=True
                )
                
                if device == "cpu":
                    local_model = local_model.to(device)
                
                local_model.eval()  # 设置为评估模式
                print(f"[OK] 本地模型加载完成（设备: {device}）。")
            except Exception as e:
                print(f"[WARN] 本地模型加载失败: {e}")
                print("    将尝试使用 API 模式或规则模拟。")
                local_model = None
                local_tokenizer = None
        
        # 如果本地模型未启用或加载失败，尝试使用 API
        if local_model is None and API_KEY and API_KEY != "你的API_KEY":
            try:
                from openai import OpenAI
                client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
                print("    使用 API 模式进行关系抽取。")
            except ImportError:
                print("    未安装 openai，将使用规则模拟。")
                client = None
            except Exception as e:
                print(f"    API 初始化失败: {e}，将使用规则模拟。")
                client = None
        
        # 如果既没有本地模型也没有 API，给出警告
        if local_model is None and client is None:
            print("[WARN] 未配置本地模型或 API_KEY，将使用规则模拟。")

    # 限制处理条数，便于快速跑通
    n_news = min(max_news, len(df_news))
    it = df_news.head(n_news).iterrows()

    print(f">>> [Step 2] 从新闻构建边（共 {n_news} 条，规则模拟='提及某股票代码即建边'）...")
    for _, row in tqdm(it, total=n_news):
        src_ticker = row.get('Ticker', None)
        if src_ticker not in ticker2idx:
            continue
        content = row.get(text_col, "")
        if pd.isna(content):
            content = ""

        relations = []
        if local_model is not None and local_tokenizer is not None:
            relations = extract_relations_with_llm(content, local_model=local_model, local_tokenizer=local_tokenizer)
        elif client is not None:
            relations = extract_relations_with_llm(content, client=client)

        if relations:
            # 使用 LLM 返回的结构化关系
            for r in relations:
                src = r.get("src", "").strip().upper()
                dst = r.get("dst", "").strip().upper()
                if src in ticker2idx and dst in ticker2idx and src != dst:
                    i, j = ticker2idx[src], ticker2idx[dst]
                    adj_matrix[i, j] = 1.0
                    adj_matrix[j, i] = 1.0
        else:
            # 规则模拟：新闻正文中若出现另一只股票的 Ticker，则建无向边
            for t in tickers:
                if t != src_ticker and len(t) >= 2 and t in str(content):
                    if src_ticker in ticker2idx and t in ticker2idx:
                        i, j = ticker2idx[src_ticker], ticker2idx[t]
                        adj_matrix[i, j] = 1.0
                        adj_matrix[j, i] = 1.0

    print(">>> [Step 3] 保存邻接矩阵...")
    np.save(OUTPUT_GRAPH, adj_matrix)
    print(f"[OK] 邻接矩阵已保存至 {OUTPUT_GRAPH}，形状: {adj_matrix.shape}")


if __name__ == "__main__":
    # 如果设置了 USE_LOCAL_MODEL 环境变量，自动启用 LLM
    # 否则默认使用规则模拟；若配置了 API_KEY 且希望用 LLM，可改为 build_dynamic_graph(use_llm=True)
    use_llm = USE_LOCAL_MODEL or (API_KEY and API_KEY != "你的API_KEY")
    build_dynamic_graph(use_llm=use_llm, max_news=MAX_NEWS_FOR_RULE)
