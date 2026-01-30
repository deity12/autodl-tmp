import pandas as pd
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 配置
MODEL_PATH = "/root/autodl-tmp/models/qwen/Qwen2.5-14B-Instruct"
NEWS_PATH = "/root/autodl-tmp/paper/data/processed/Graph_Adjacency_news_sampled.csv"
TICKERS_PATH = "/root/autodl-tmp/paper/data/processed/Graph_Tickers.json"

def debug_extraction():
    # 1. 加载数据
    df = pd.read_csv(NEWS_PATH).head(20) # 只看前20条
    with open(TICKERS_PATH, 'r') as f:
        valid_tickers = set(json.load(f)['tickers'])
    
    print(f"有效股票代码库示例: {list(valid_tickers)[:5]}")

    # 2. 加载模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)

    # 3. 这里的 Prompt 和你 build_graph.py 里的一模一样
    for txt in df['Headline'].tolist():
        prompt = f"""你是一个专业的金融关系抽取专家。请从以下财经新闻标题中提取公司之间的**显式关系**和**情感极性**。

新闻标题：{txt}

关系类型（仅限以下类型）：
1. 供应链关系 (supply)
2. 竞争关系 (competition)
3. 合作关系 (cooperation)
4. 并购关系 (merger)
5. 诉讼关系 (lawsuit)
6. 投资关系 (investment)
7. 共同事件关系 (co-event)

输出要求：
1. 只提取**明确提到两家公司**且关系清晰的内容
2. 股票代码必须是**美股代码**（如AAPL、TSLA、MSFT等）
3. 必须为每条关系提供 sentiment_score

严格按以下JSON格式输出：
[{{"src": "公司A代码", "dst": "公司B代码", "relation": "关系类型", "sentiment_score": 0.5}}]

现在请分析上述新闻标题："""

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=96, do_sample=False)
        output = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

        print("-" * 50)
        print(f"新闻: {txt}")
        print(f"LLM 原文输出: {output}")
        
        # 模拟你的过滤逻辑
        try:
            # 尝试提取 JSON
            start = output.find('[')
            end = output.rfind(']') + 1
            if start != -1 and end != -1:
                json_str = output[start:end]
                data = json.loads(json_str)
                for item in data:
                    src = str(item.get('src', '')).strip().upper()
                    dst = str(item.get('dst', '')).strip().upper()
                    
                    # 关键判定
                    is_src_valid = src in valid_tickers
                    is_dst_valid = dst in valid_tickers
                    
                    print(f"解析: src={src} ({'✅ 有效' if is_src_valid else '❌ 无效 -> 被丢弃'}), dst={dst} ({'✅ 有效' if is_dst_valid else '❌ 无效 -> 被丢弃'})")
        except:
            print("解析 JSON 失败")
            
debug_extraction()