# 快速验证运行指南

## 前置条件检查

在运行快速验证之前，必须确保以下文件存在：

```
paper/data/processed/
├── Final_Model_Data.csv      # 训练数据
├── Graph_Adjacency.npy        # 邻接矩阵
└── Graph_Tickers.json         # 节点列表
```

## 完整运行流程

### 步骤 1: 数据预处理（如果还没做）

```bash
cd paper/dataProcessed
python data_preprocessing.py
```

这会生成 `Final_Model_Data.csv`。

### 步骤 2: 构建图谱（必须）

```bash
cd paper/dataProcessed
python build_graph.py
```

**重要**：这一步会生成：
- `Graph_Adjacency.npy` (邻接矩阵)
- `Graph_Tickers.json` (节点列表，用于索引对齐)

**预计时间**：
- 不使用 LLM（默认）：5-10 分钟
- 使用 LLM：2-4 小时（取决于新闻数量）

**快速模式**（推荐用于验证）：
```bash
# 只处理少量新闻，快速生成图谱
python build_graph.py --max-total 100
```

### 步骤 3: 运行快速验证

```bash
cd paper/training
python quick_validation.py
```

**预计时间**：30-60 分钟

## 验证结果解读

### 成功情况
```
最佳模型: Full Model
Full Model 排名: 1/4

✅ 验证通过！Full Model 表现最优，可以进行完整实验。
```

→ 可以放心运行完整实验

### 失败情况
```
最佳模型: w/o Quantum
Full Model 排名: 2/4

⚠️ 警告！w/o Quantum 优于 Full Model
```

→ 需要调整超参数或模型设计

## 常见问题

### Q1: 如果没有新闻数据怎么办？
A: `build_graph.py` 会自动生成单位矩阵（每个节点只连接自己），可以继续验证。

### Q2: 如果验证失败怎么办？
A: 参考脚本输出的建议：
1. 调整学习率
2. 增加训练轮数
3. 调整 dropout
4. 检查模块冲突

### Q3: 可以跳过图谱构建吗？
A: 不可以。`dataset.py` 必须读取 `Graph_Tickers.json` 来对齐索引。

## 下一步

验证通过后，运行完整实验：

```bash
# 全量模型训练
cd paper/training
python train_full.py

# 消融实验
python train_ablation.py
```
