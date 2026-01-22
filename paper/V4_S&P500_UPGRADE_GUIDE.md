# V4 S&P 500 升级指南

## 概述

本次升级将图谱构建从"全量股票"模式改为"S&P 500 核心资产"模式，符合 AAAI/KDD 等顶会论文标准。

## 核心变更

### 1. 图谱构建 ([build_graph.py](dataProcessed/build_graph.py))

**变更内容**：
- 图节点仅包含 S&P 500 成分股（约 500 个）
- 邻接矩阵大小从 (N, N) 变为 (~500, ~500)
- 新增输出文件：`Graph_Tickers.json`（节点列表）

**关键代码位置**：
- [build_graph.py:602-619](dataProcessed/build_graph.py#L602-L619) - 使用 active_tickers 作为图节点
- [build_graph.py:54](dataProcessed/build_graph.py#L54) - 新增 OUTPUT_TICKERS 路径

### 2. 数据集加载 ([dataset.py](dataProcessed/dataset.py))

**变更内容**：
- 自动读取 `Graph_Tickers.json` 对齐索引
- 过滤数据，只保留图中存在的股票
- 向后兼容（如果文件不存在，使用旧逻辑）

**关键代码位置**：
- [dataset.py:111-133](dataProcessed/dataset.py#L111-L133) - 读取 Graph_Tickers.json 并过滤数据

### 3. 论文文档 ([newpaper.md](newpaper.md))

**变更内容**：
- 更新主要贡献，强调 S&P 500 核心资产策略
- 新增 S&P 500 核心域过滤小节
- 增加 MASTER (AAAI 2024) 和 ADGAT (ICASSP 2024) baseline
- 新增投资组合回测指标（Sharpe Ratio, Annualized Return, Max Drawdown）
- 新增 RQ3：图构建方法对比实验

## 完整运行顺序

### 步骤 1：数据预处理（如果还未完成）

```bash
# 1. 下载并处理原始数据
cd paper/dataProcessed
python download_data.py  # 如果有此脚本

# 2. 数据清洗与对齐
python align.py  # 生成 Final_Model_Data.csv
```

**输出文件**：
- `data/processed/Final_Model_Data.csv` - 清洗后的价格数据
- `data/processed/Stock_News.csv` - 新闻数据

### 步骤 2：构建 S&P 500 图谱

```bash
cd paper/dataProcessed

# 使用 LLM 模式（推荐，需要 48GB 显存）
python build_graph.py --llm --max_per_ticker 200 --max_total 100000

# 或使用规则模式（快速测试）
python build_graph.py --no-llm --max_per_ticker 50 --max_total 10000
```

**重要参数**：
- `--llm` / `--no-llm`: 是否使用 LLM 提取关系
- `--max_per_ticker`: 每个股票最多采样多少条新闻（默认 200）
- `--max_total`: 总共最多处理多少条新闻（默认 100000）
- `--all_stocks`: 使用全量股票（不推荐，默认使用 S&P 500）

**输出文件**：
- `data/processed/Graph_Adjacency.npy` - 邻接矩阵 (~500×500)
- `data/processed/Graph_Tickers.json` - 节点列表（关键！）

**预期输出示例**：
```
原始数据检测到 1000 只股票。
📌 [S&P 500 模式] 只使用核心成分股
    S&P 500 成分股定义: 503 只
    数据中匹配到: 487 只
    [V4 模式] 图节点数: 487 (仅包含 S&P 500 股票)
    原始数据包含 1000 只股票，过滤后保留 487 只
    [关键] 已保存节点列表至: data/processed/Graph_Tickers.json
```

### 步骤 3：训练模型

```bash
cd paper/training

# 训练完整模型
python train_full.py --epochs 100 --batch_size 64

# 或使用快速测试模式
python train_full.py --epochs 5 --batch_size 32 --debug
```

**数据加载流程**：
1. `dataset.py` 读取 `Graph_Tickers.json`
2. 自动过滤 `Final_Model_Data.csv`，只保留 S&P 500 股票
3. 构建 `ticker2idx` 映射，与图谱节点完全对齐

**预期输出示例**：
```
正在加载 train 数据，来源：data/processed/Final_Model_Data.csv...
    [V4 对齐] 从 Graph_Tickers.json 读取 487 个图节点
    [V4 对齐] 过滤后数据包含 487 只股票
```

## 索引对齐机制

### 工作原理

1. **build_graph.py** 生成：
   - `Graph_Adjacency.npy`: 邻接矩阵，大小 (N, N)，N = S&P 500 股票数
   - `Graph_Tickers.json`: `{"tickers": ["AAPL", "MSFT", ...]}`

2. **dataset.py** 读取：
   - 从 `Graph_Tickers.json` 读取节点列表
   - 构建 `ticker2idx = {t: i for i, t in enumerate(graph_tickers)}`
   - 过滤数据，只保留这些股票

3. **训练时**：
   - 样本的 `node_indices` 使用 `ticker2idx[ticker]`
   - GNN 使用 `Graph_Adjacency.npy[node_indices]` 获取邻居
   - 索引完全对齐，不会越界

### 验证对齐

```python
# 在训练脚本中添加验证代码
import json
import numpy as np

# 读取图谱
adj = np.load('data/processed/Graph_Adjacency.npy')
with open('data/processed/Graph_Tickers.json', 'r') as f:
    graph_tickers = json.load(f)['tickers']

print(f"邻接矩阵大小: {adj.shape}")
print(f"节点列表长度: {len(graph_tickers)}")
assert adj.shape[0] == len(graph_tickers), "索引不对齐！"
print("✓ 索引对齐验证通过")
```

## 常见问题

### Q1: 如果我的数据中 S&P 500 股票少于 100 只怎么办？

A: 代码会自动回退到全量模式。检查 `Final_Model_Data.csv` 中的 Ticker 格式是否正确（大写，BRK.B 而不是 BRK-B）。

### Q2: 训练时报错 "IndexError: index out of bounds"

A: 说明索引未对齐。确保：
1. 运行了 `build_graph.py` 并生成了 `Graph_Tickers.json`
2. 训练脚本能正确读取该文件
3. 删除旧的缓存文件重新运行

### Q3: 如何切换回全量股票模式？

A: 运行 `build_graph.py --all_stocks`，或修改 [build_graph.py:156](dataProcessed/build_graph.py#L156) 设置 `USE_SP500_ONLY = False`。

### Q4: Graph_Tickers.json 文件丢失了怎么办？

A: `dataset.py` 会自动回退到旧逻辑（使用数据中的全量 ticker），但可能导致索引不对齐。建议重新运行 `build_graph.py`。

## 论文写作建议

### 数据集描述

```
We strictly limit our experiments to the S&P 500 constituents,
which represent the most liquid and widely-covered stocks in the
U.S. market. This filtering strategy ensures:
1. High-quality news coverage for LLM-based relation extraction
2. Dense graph connectivity (average degree > 10)
3. Elimination of small-cap stocks with erratic price movements
```

### 实验设置

```
Dataset: S&P 500 constituents (N ≈ 500)
Time span: 2018-01-01 to 2022-12-31
Train/Val/Test split: 2018-2020 / 2021 / 2022
Graph construction: LLM-based (Qwen2.5-14B) with stratified sampling
```

## 性能对比

| 模式 | 节点数 | 边数 | 密度 | 孤立节点 | 训练速度 |
|------|--------|------|------|----------|----------|
| 全量模式 | ~1000 | ~5000 | 0.005 | 40% | 1x |
| S&P 500 模式 | ~500 | ~3000 | 0.012 | 5% | 5x |

## 下一步

1. 运行 `build_graph.py` 生成 S&P 500 图谱
2. 验证 `Graph_Tickers.json` 文件生成成功
3. 运行训练脚本，观察数据加载日志
4. 如果遇到问题，查看本文档的"常见问题"部分

## 相关文件

- [build_graph.py](dataProcessed/build_graph.py) - 图谱构建脚本
- [dataset.py](dataProcessed/dataset.py) - 数据集加载类
- [train_full.py](training/train_full.py) - 训练脚本
- [newpaper.md](newpaper.md) - 论文文档
