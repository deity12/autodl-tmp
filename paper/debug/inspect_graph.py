import numpy as np
import json
import os

# 路径配置
DATA_DIR = "/root/autodl-tmp/paper/data/processed"
ADJ_PATH = os.path.join(DATA_DIR, "Graph_Adjacency.npy")
TICKER_PATH = os.path.join(DATA_DIR, "Graph_Tickers.json")

def inspect_graph():
    # 1. 检查文件是否存在
    if not os.path.exists(ADJ_PATH):
        print(f"❌ 还没生成 {ADJ_PATH}")
        return

    # 2. 加载数据
    print(f"正在加载图谱数据...")
    adj = np.load(ADJ_PATH)
    with open(TICKER_PATH, 'r') as f:
        data = json.load(f)
        # 兼容两种格式：可能是 list 也可能是 dict
        if isinstance(data, list):
            tickers = data
        else:
            tickers = data.get('tickers', list(data.keys()))
            
    # 建立索引映射: 0 -> AAPL, 1 -> MSFT
    idx2ticker = {i: t for i, t in enumerate(tickers)}
    
    # 3. 统计和展示
    print("-" * 50)
    print(f"📊 图谱概览")
    print(f"节点数 (股票): {adj.shape[0]}")
    print(f"边数 (连接): {np.sum(adj > 0) / 2:.0f} (无向边)")
    print("-" * 50)
    
    print("🔍 具体的连接关系 (前 50 条):")
    count = 0
    # 遍历矩阵的上三角 (避免重复输出 A-B 和 B-A)
    rows, cols = np.where(np.triu(adj, k=1) > 0)
    
    for r, c in zip(rows, cols):
        t1 = idx2ticker[r]
        t2 = idx2ticker[c]
        weight = adj[r, c]
        
        # 你的权重可能是 1.0 (语义) 或 0.x (统计)
        # 如果只想看 LLM 的边，通常权重是整数或者特定的值
        print(f"{count+1}. {t1} <--> {t2}  (权重: {weight:.4f})")
        
        count += 1
        if count >= 50:
            print("\n... (后面还有，这里只显示前50条) ...")
            break
            
    if count == 0:
        print("⚠️ 图谱里没有任何边！(是不是还没跑完？)")

if __name__ == "__main__":
    inspect_graph()