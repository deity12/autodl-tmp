# Codex 代码优化提示词（Graph-RWKV 项目）

## 📋 项目背景

**项目名称**：基于大语言模型动态图谱与 Graph-RWKV 的时空解耦金融预测研究

**当前状态**：
- 已完成从旧架构（QL-MATCC-GNN with Quantum）到新架构（Graph-RWKV）的代码重构
- 已移除 Quantum、MATCC、MarketGuidance 等不需要的组件
- 核心架构：RWKV 时间序列编码器 + 动态图注意力网络（GAT）
- 数据流：FNSPID 新闻数据 → LLM 提取关系+情感 → 混合图构建 → Graph-RWKV 训练

## 🖥️ 服务器配置（重要！）

**硬件资源**：
- **GPU**: 48GB VRAM (vGPU-48GB-350W) × 1
- **CPU**: 12 vCPU Intel Xeon Platinum 8260 @ 2.40GHz
- **RAM**: 90GB
- **Disk**: 50GB 数据盘（需注意空间管理）

**软件环境**：
- **PyTorch**: 2.1.2
- **Python**: 3.10
- **CUDA**: 11.8
- **OS**: Ubuntu 22.04

**优化重点**：
- 充分利用 48GB GPU：大 batch size、长序列、大模型
- 充分利用 90GB RAM：数据缓存、内存映射、预加载
- 12 vCPU：多进程数据加载（num_workers=8-10）
- 50GB 磁盘：优化存储格式，避免冗余文件

---

## 🏗️ 核心架构说明

### 1. **时间维：RWKV 时间序列编码器**
- **组件**：`GraphRWKV_Model` (位于 `paper/models/base_model.py`)
- **核心**：`RWKV_TimeMixing` - 线性 Attention 机制，O(1) 推理复杂度
- **结构**：
  ```
  输入 (B, T, F) 
  → Linear(input_dim → n_embd)
  → N 层 RWKV Block:
     - RWKV_TimeMixing (时间维线性注意力)
     - Classical_ChannelMixing (经典 FFN)
  → 输出最后一个时间步的特征向量 (B, n_embd)
  ```

### 2. **空间维：动态图注意力网络（GAT）**
- **组件**：`GraphRWKV_GNN_Model` (位于 `paper/models/gnn_model.py`)
- **输入图**：LLM 增强的情感加权混合图
  - 语义图：LLM 提取的关系 + 情感极性分数 (-1.0 到 1.0)
  - 统计图：30 天收益率 Pearson 相关性（|ρ| > 0.6）
  - 混合公式：`A_t^final = Norm(A_t^semantic + λ · A_t^stat)`
- **结构**：
  ```
  时序特征 (B, n_embd)
  → GAT 空间聚合 (基于混合图)
  → 时空特征融合 (Concat + Linear)
  → 收益率预测
  ```

### 3. **数据对齐机制**
- **16:00 Cut-off**：T 日 16:00 之前的新闻 → 构建 T 日图 → 预测 T+1 日收益
- **严格前瞻偏差规避**：确保不使用未来信息

---

## 🎯 核心设计原则（最重要！）

### **1. 傻瓜式运行**
- **每个脚本都是独立的 `.py` 文件，可以直接运行**
- **命令行参数简洁明了，一眼就知道是干什么的**
- **示例**：
  ```bash
  # 数据预处理
  python 1_preprocess_data.py --input_dir ./data/raw --output_dir ./data/processed
  
  # 构建图谱
  python 2_build_graph.py --news_file ./data/processed/Stock_News.csv --output ./data/processed/graph.npy
  
  # 训练模型
  python 3_train.py --gpu 0  # 配置在脚本顶部，命令行参数可选
  
  # 评估模型
  python 4_evaluate.py --checkpoint ./outputs/best_model.pth --test_data ./data/processed/test.csv
  ```

### **2. 脚本命名规范**
- **数字前缀**：表示执行顺序（1_preprocess, 2_build_graph, 3_train, 4_evaluate）
- **功能明确**：文件名直接说明功能
- **单一职责**：每个脚本只做一件事

### **3. 日志系统（重要！）**
- **每次运行自动保存日志到文件**
- **日志文件命名包含时间戳，方便区分不同运行**
- **日志包含：时间戳、步骤、进度、错误信息、资源使用等**
- **日志格式清晰，方便检查和调试**
- **示例**：
  ```python
  # 日志文件命名：train_20260123_143025.log（日期_时间）
  # 日志内容：
  # 2026-01-23 14:30:25 - INFO - 开始训练 Graph-RWKV 模型
  # 2026-01-23 14:30:26 - INFO - 加载数据: ./data/processed/Final_Model_Data.csv
  # 2026-01-23 14:30:30 - INFO - 数据加载完成，训练集: 10000 样本，测试集: 2000 样本
  # 2026-01-23 14:30:31 - INFO - 初始化模型: n_embd=256, n_layers=3
  # 2026-01-23 14:30:35 - INFO - 开始训练，Epoch 1/30
  # 2026-01-23 14:32:10 - INFO - Epoch 1 完成，Train Loss: 0.0234, Val Loss: 0.0198
  # ...
  ```

### **3. 配置管理**
- **配置直接写在 py 文件里**（不使用 config.yaml）
- **命令行参数只覆盖关键参数（如 GPU ID、数据路径）**
- **默认配置适合 48GB GPU + 90GB RAM**

## 🎯 代码优化方向

### **优先级 0：脚本简化和易用性（最高优先级）**

1. **重构为独立可运行脚本**
   - 将现有模块化代码整合为可直接运行的脚本
   - 每个脚本包含完整的错误处理和进度显示
   - 添加 `if __name__ == "__main__"` 入口

2. **命令行接口设计**
   ```python
   # ✅ 好的设计
   python 3_train.py \
       --data ./data/processed/Final_Model_Data.csv \
       --graph ./data/processed/Graph_Adjacency.npy \
       --output ./outputs \
       --batch_size 1024 \
       --epochs 30
   
   # ❌ 避免的设计
   python 3_train.py --batch_size 2048  # 只覆盖需要修改的参数
   ```

3. **进度显示和日志**
   - 使用 `tqdm` 显示进度条
   - 关键步骤打印清晰的中文提示
   - 错误信息友好且包含解决建议

### **优先级 1：性能优化（针对 48GB GPU + 90GB RAM）**

1. **RWKV 时间混合层优化（针对 48GB GPU）**
   - 当前使用 `torch.jit.script` 编译 `rwkv_linear_attention_cpu`
   - **优化方向**：
     - 使用 `torch.compile` (PyTorch 2.1.2 支持) 加速整个模型
     - 优化 GPU 内存访问模式，支持更大的 batch size（1024+）
     - 支持更长的序列长度（seq_len=60+，充分利用 GPU 内存）
     - 使用 TF32 精度（`torch.set_float32_matmul_precision('high')`）

2. **GAT 图注意力计算优化（针对大图）**
   - 当前使用稀疏注意力（仅对有边的节点对计算）
   - **优化方向**：
     - 使用 PyTorch Geometric (PyG) 的 `GATv2Conv` 替代手动实现
     - 支持边权重（情感分数）作为注意力权重
     - 批量图处理优化（充分利用 48GB GPU）
     - 使用 `torch.sparse` 优化稀疏矩阵乘法

3. **数据加载优化（针对 90GB RAM + 12 vCPU）**
   - **优化方向**：
     - `num_workers=8-10`（充分利用 12 vCPU）
     - `pin_memory=True`, `prefetch_factor=4`（减少 GPU 等待）
     - 使用内存映射文件（`np.memmap`）处理大 CSV
     - 数据预处理结果缓存到 RAM（90GB 足够大）
     - 使用 `torchdata` 优化数据管道

### **优先级 2：代码质量优化**

1. **类型注解完善**
   - 所有函数添加完整的类型注解（`typing` 模块）
   - 使用 `torch.jit.script` 兼容的类型注解
   - 添加 `__annotations__` 检查

2. **错误处理增强**
   - 添加输入验证（shape、dtype、device 检查）
   - 添加友好的错误消息（包含修复建议）
   - 添加数值稳定性检查（NaN/Inf 检测）

3. **文档字符串标准化**
   - 使用 Google/NumPy 风格文档字符串
   - 添加参数说明、返回值说明、示例代码
   - 添加数学公式的 LaTeX 注释

### **优先级 3：架构优化**

1. **模块化改进**
   - 将 RWKV Block 提取为独立模块
   - 将图构建逻辑进一步模块化
   - 添加配置管理模块（使用 `hydra` 或 `omegaconf`）

2. **可扩展性**
   - 支持多 GPU 训练（`torch.nn.DataParallel` 或 `DistributedDataParallel`）
   - 支持混合精度训练（AMP，已部分实现）
   - 支持模型检查点恢复和断点续训

3. **实验管理**
   - 集成 `wandb` 或 `tensorboard` 进行实验跟踪
   - 添加超参数搜索支持（`optuna` 或 `ray.tune`）
   - 添加模型版本管理

---

## 📝 代码规范要求

### **1. Python 代码风格**
- 遵循 **PEP 8** 规范
- 使用 **Black** 格式化（行长度 100）
- 使用 **isort** 排序导入
- 使用 **mypy** 进行类型检查

### **2. 命名规范**
- **类名**：`PascalCase`（如 `GraphRWKV_Model`）
- **函数/变量名**：`snake_case`（如 `build_dynamic_graph`）
- **常量**：`UPPER_SNAKE_CASE`（如 `CUTOFF_HOUR`）
- **私有成员**：前缀 `_`（如 `_normalize_llm_relations`）

### **3. 注释规范**
- **中文注释**：所有注释使用中文（项目要求）
- **模块级注释**：每个文件开头有模块说明
- **函数注释**：包含功能说明、参数说明、返回值说明
- **关键逻辑**：复杂算法添加行内注释

### **4. 错误处理**
```python
# ✅ 好的做法
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"操作失败: {e}")
    raise ValueError(f"友好的错误消息: {e}") from e

# ❌ 避免的做法
try:
    result = risky_operation()
except:
    pass  # 静默失败
```

### **5. 类型注解示例**
```python
from typing import Optional, Tuple, Dict, List
import torch

def forward(
    self,
    x: torch.Tensor,  # (B, T, F)
    vol: Optional[torch.Tensor] = None,  # (B, 1)
    node_indices: Optional[torch.Tensor] = None,  # (B,)
) -> torch.Tensor:  # (B, 1)
    """前向传播。
    
    Args:
        x: 输入序列，形状 (B, T, F)
        vol: 波动率（保留以兼容接口，新方向中不使用）
        node_indices: 节点索引，用于 GAT 聚合
    
    Returns:
        收益率预测，形状 (B, 1)
    """
    pass
```

---

## 🔧 具体优化任务清单

### **任务 0：脚本重构为独立可运行文件（最高优先级）**

**目标**：将现有代码重构为 5 个独立可运行的脚本

1. **`1_preprocess_data.py`** - 数据预处理
   ```python
   """
   功能：数据清洗、对齐、特征工程
   输入：原始数据目录
   输出：Final_Model_Data.csv
   命令行：python 1_preprocess_data.py --input_dir ./data/raw --output_dir ./data/processed
   """
   ```

2. **`2_build_graph.py`** - 构建图谱
   ```python
   """
   功能：LLM 提取关系+情感，构建混合图
   输入：Stock_News.csv, Stock_Prices.csv
   输出：Graph_Adjacency.npy, Graph_Adjacency_tickers.json
   命令行：python 2_build_graph.py --news ./data/processed/Stock_News.csv --prices ./data/processed/Stock_Prices.csv --output ./data/processed
   """
   ```

3. **`3_train.py`** - 全量模型训练
   ```python
   """
   功能：训练 Graph-RWKV 全量模型（完整架构：RWKV + GAT + 混合图）
   输入：Final_Model_Data.csv, Graph_Adjacency.npy
   输出：checkpoints/best_model.pth, training_logs.json
   命令行：python 3_train.py --data ./data/processed/Final_Model_Data.csv --graph ./data/processed/Graph_Adjacency.npy --output ./outputs --batch_size 1024 --epochs 30
   """
   ```

4. **`3_train_ablation.py`** - 消融实验训练
   ```python
   """
   功能：运行消融实验，验证各组件的有效性
   支持的消融实验：
     - w/o_graph: 去掉 GAT 图聚合（只用 RWKV）
     - w/o_semantic: 只用统计图，不用语义图
     - w/o_statistical: 只用语义图，不用统计图
     - w/o_sentiment: 语义图不用情感分数（只用关系）
   
   输入：Final_Model_Data.csv, Graph_Adjacency.npy
   输出：checkpoints/best_model_ablation_*.pth, ablation_results.json
   命令行：python 3_train_ablation.py --data ./data/processed/Final_Model_Data.csv --graph ./data/processed/Graph_Adjacency.npy --output ./outputs --ablation w/o_graph --batch_size 1024
   
   批量运行所有消融实验：
   python 3_train_ablation.py --data ./data/processed/Final_Model_Data.csv --graph ./data/processed/Graph_Adjacency.npy --output ./outputs --ablation all
   """
   ```

5. **`4_evaluate.py`** - 评估模型
   ```python
   """
   功能：训练 Graph-RWKV 模型
   输入：Final_Model_Data.csv, Graph_Adjacency.npy
   输出：checkpoints/best_model.pth, training_logs.json
   命令行：python 3_train.py --data ./data/processed/Final_Model_Data.csv --graph ./data/processed/Graph_Adjacency.npy --output ./outputs --batch_size 1024 --epochs 30
   """
   ```

4. **`4_evaluate.py`** - 评估模型
   ```python
   """
   功能：评估模型性能（IC, RankIC, Sharpe Ratio）
   输入：checkpoint, test_data
   输出：evaluation_results.csv, plots/
   命令行：python 4_evaluate.py --checkpoint ./outputs/best_model.pth --test_data ./data/processed/test.csv --output ./outputs/results
   """
   ```

**要求**：
- **可以直接点击运行**：使用默认配置，无需命令行参数
- **命令行参数可选**：用于覆盖默认配置（argparse with defaults）
- **完善的日志系统**：每次运行自动保存日志到文件
- **日志文件命名**：包含时间戳，如 `train_20260123_143025.log`
- **日志内容**：时间戳、步骤、进度、错误、资源使用等
- **进度显示**：使用 `tqdm` 显示进度条
- **错误处理**：友好的错误提示和异常捕获
- **关键步骤**：打印清晰的中文说明（同时输出到控制台和日志文件）

### **任务 1：RWKV 时间混合层性能优化（针对 48GB GPU）**
```python
# 文件：paper/models/base_model.py
# 类：RWKV_TimeMixing
# 优化点（针对 48GB GPU）：
# 1. 使用 torch.compile 加速（PyTorch 2.1.2 支持）
# 2. 启用 TF32 精度（torch.set_float32_matmul_precision('high')）
# 3. 优化 GPU 内存访问，支持 batch_size=1024+
# 4. 支持更长序列（seq_len=60+）
# 5. 参考 RWKV-LM 官方实现的优化技巧
```

### **任务 2：GAT 层使用 PyG 重构（参考 ADGAT）**
```python
# 文件：paper/models/gnn_model.py
# 类：GraphAttentionLayer
# 优化点（参考 ADGAT 实现）：
# 1. 使用 torch_geometric.nn.GATv2Conv 替代手动实现
# 2. 支持边权重（情感分数作为边权重）
# 3. 优化 batch 处理（充分利用 48GB GPU）
# 4. 参考 ADGAT 的金融图处理技巧
# 5. 使用 torch.sparse 优化稀疏矩阵操作
```

### **任务 3：全量模型训练脚本（重构为 3_train.py）**
```python
# 文件：3_train.py（重构自 paper/training/train_full.py）
# 功能：训练 Graph-RWKV 全量模型（完整架构）
# 优化点（针对 48GB GPU + 90GB RAM）：
# 1. 默认 batch_size=1024（充分利用 GPU）
# 2. num_workers=8-10（充分利用 12 vCPU）
# 3. 混合精度训练（AMP）完整支持
# 4. torch.compile 模型加速
# 5. 启用 TF32 精度
# 6. 添加 wandb/tensorboard 集成（可选）
# 7. 模型检查点自动保存和恢复
# 8. 简洁的命令行接口
# 9. 只训练全量模型，不包含消融实验逻辑
```

### **任务 3.5：消融实验训练脚本（新建 3_train_ablation.py）**
```python
# 文件：3_train_ablation.py（新建，参考 paper/training/train_ablation.py）
# 功能：运行消融实验，验证各组件的有效性
# 设计要点：
# 1. 支持多种消融实验类型（--ablation 参数）
# 2. 可以单独运行某个消融实验，也可以批量运行所有
# 3. 每个消融实验保存独立的 checkpoint 和日志
# 4. 输出消融实验结果对比表
# 5. 命令行接口简洁明了
# 6. 复用 3_train.py 的训练逻辑（避免代码重复）
```

### **任务 4：数据加载优化（针对 90GB RAM）**
```python
# 文件：paper/dataProcessed/dataset.py
# 类：FinancialDataset
# 优化点（充分利用 90GB RAM）：
# 1. 数据预加载到 RAM（90GB 足够大）
# 2. 使用内存映射文件（np.memmap）处理大 CSV
# 3. num_workers=8-10（充分利用 12 vCPU）
# 4. pin_memory=True, prefetch_factor=4
# 5. 数据预处理结果缓存到 RAM
# 6. 使用 torchdata 优化数据管道
```

### **任务 5：图构建优化（重构为 2_build_graph.py）**
```python
# 文件：2_build_graph.py（重构自 paper/dataProcessed/build_graph.py）
# 优化点：
# 1. LLM 批处理优化（减少 API 调用，参考 FinGPT）
# 2. 统计相关性计算使用 GPU 加速（48GB GPU）
# 3. 图构建结果缓存到磁盘（避免重复构建）
# 4. 多进程处理（充分利用 12 vCPU）
# 5. 简洁的命令行接口
# 6. 进度显示（tqdm）
```

---

## 🚀 优化示例代码模板

### **示例 0：全量模型训练脚本模板（3_train.py）- 可直接点击运行**
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3_train.py - Graph-RWKV 全量模型训练脚本

功能：训练 Graph-RWKV 全量模型（完整架构：RWKV + GAT + 混合图）
用法：
  方式1（推荐）：直接点击运行，使用默认配置
  方式2：命令行运行，覆盖部分参数
    python 3_train.py --batch_size 2048 --epochs 50
"""

import argparse
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import torch
from tqdm import tqdm
import yaml
from models.gnn_model import GraphRWKV_GNN_Model

def setup_logging(output_dir: str) -> logging.Logger:
    """设置日志系统，同时输出到控制台和文件。"""
    # 创建日志目录
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 日志文件名包含时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'train_{timestamp}.log'
    
    # 配置日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # 文件输出
            logging.StreamHandler(sys.stdout)  # 控制台输出
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件已创建: {log_file}")
    return logger

# ==================== 配置参数（直接写在文件里，方便修改）====================
# 数据配置
DATA_CSV_PATH = './data/processed/Final_Model_Data.csv'
GRAPH_PATH = './data/processed/Graph_Adjacency.npy'

# 模型配置
MODEL_N_EMBD = 256      # 嵌入维度（适合 48GB GPU）
MODEL_N_LAYERS = 3      # RWKV 层数
MODEL_GNN_EMBD = 64     # GAT 嵌入维度
MODEL_DROPOUT = 0.1     # Dropout 率

# 训练配置
TRAIN_BATCH_SIZE = 1024      # 批大小（适合 48GB GPU）
TRAIN_EPOCHS = 30            # 训练轮数
TRAIN_LR = 3e-4              # 学习率
TRAIN_NUM_WORKERS = 10       # 数据加载进程数（适合 12 vCPU）
TRAIN_PIN_MEMORY = True      # 固定内存
TRAIN_PREFETCH_FACTOR = 4    # 预取因子

# 输出配置
OUTPUT_DIR = './outputs'
CHECKPOINT_DIR = './outputs/checkpoints'
LOG_DIR = './outputs/logs'

# GPU 配置
GPU_ID = 0                   # GPU ID
# ============================================================================

def main():
    # 解析命令行参数（可选，用于覆盖脚本中的配置）
    parser = argparse.ArgumentParser(description='训练 Graph-RWKV 全量模型')
    parser.add_argument('--data', type=str, default=DATA_CSV_PATH, 
                       help=f'训练数据 CSV 路径（默认: {DATA_CSV_PATH}）')
    parser.add_argument('--graph', type=str, default=GRAPH_PATH,
                       help=f'图谱邻接矩阵路径（默认: {GRAPH_PATH}）')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                       help=f'输出目录（默认: {OUTPUT_DIR}）')
    parser.add_argument('--batch_size', type=int, default=TRAIN_BATCH_SIZE,
                       help=f'批大小（默认: {TRAIN_BATCH_SIZE}，适合48GB GPU）')
    parser.add_argument('--epochs', type=int, default=TRAIN_EPOCHS,
                       help=f'训练轮数（默认: {TRAIN_EPOCHS}）')
    parser.add_argument('--lr', type=float, default=TRAIN_LR,
                       help=f'学习率（默认: {TRAIN_LR}）')
    parser.add_argument('--gpu', type=int, default=GPU_ID, 
                       help=f'GPU ID（默认: {GPU_ID}）')
    
    args = parser.parse_args()
    
    # 使用命令行参数或脚本中的默认配置
    data_path = args.data
    graph_path = args.graph
    output_dir = args.output
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    gpu_id = args.gpu
    
    # 设置日志
    logger = setup_logging(args.output)
    logger.info("="*60)
    logger.info("开始训练 Graph-RWKV 全量模型")
    logger.info("="*60)
    logger.info(f"配置参数:")
    logger.info(f"  - 数据路径: {args.data}")
    logger.info(f"  - 图谱路径: {args.graph}")
    logger.info(f"  - 输出目录: {args.output}")
    logger.info(f"  - Batch Size: {args.batch_size}")
    logger.info(f"  - Epochs: {args.epochs}")
    logger.info(f"  - Learning Rate: {args.lr}")
    
    # 设置 GPU
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
        logger.info(f"GPU 内存: {torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3:.1f} GB")
    
    # 启用 TF32（针对 A100/H100）
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    logger.info("已启用 TF32 和 cuDNN benchmark")
    
    try:
        # 加载数据
        logger.info("加载数据...")
        # ... 数据加载代码 ...
        logger.info(f"数据加载完成，训练集: {len(train_dataset)} 样本，测试集: {len(test_dataset)} 样本")
        
        # 初始化模型（全量模型：RWKV + GAT + 混合图）
        logger.info("初始化 Graph-RWKV 全量模型...")
        logger.info(f"模型配置: n_embd={MODEL_N_EMBD}, n_layers={MODEL_N_LAYERS}, gnn_embd={MODEL_GNN_EMBD}")
        model = GraphRWKV_GNN_Model(
            input_dim=8,
            n_embd=MODEL_N_EMBD,
            n_layers=MODEL_N_LAYERS,
            gnn_embd=MODEL_GNN_EMBD,
            dropout=MODEL_DROPOUT,
        ).to(device)
        
        # 使用 torch.compile 加速（PyTorch 2.1.2）
        logger.info("使用 torch.compile 加速模型...")
        model = torch.compile(model, mode='reduce-overhead')
        
        # 训练循环
        logger.info("开始训练全量模型...")
        for epoch in tqdm(range(epochs), desc='训练进度'):
            # ... 训练代码 ...
            train_loss = 0.0  # 示例
            val_loss = 0.0    # 示例
            
            logger.info(f"Epoch {epoch+1}/{args.epochs} 完成，Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        logger.info("="*60)
        logger.info("全量模型训练完成！")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.warning("用户中断训练，正在保存检查点...")
        # 保存当前状态
    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
```

### **示例 0.5：消融实验训练脚本模板（3_train_ablation.py）**
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3_train_ablation.py - Graph-RWKV 消融实验训练脚本

功能：运行消融实验，验证各组件的有效性
用法：
  单个消融实验：python 3_train_ablation.py --data ./data/processed/Final_Model_Data.csv --graph ./data/processed/Graph_Adjacency.npy --ablation w/o_graph
  所有消融实验：python 3_train_ablation.py --data ./data/processed/Final_Model_Data.csv --graph ./data/processed/Graph_Adjacency.npy --ablation all
"""

import argparse
import torch
from tqdm import tqdm
from models.gnn_model import GraphRWKV_GNN_Model

# 支持的消融实验类型
ABLATION_TYPES = {
    'w/o_graph': {'use_graph': False, 'use_semantic': True, 'use_statistical': True},
    'w/o_semantic': {'use_graph': True, 'use_semantic': False, 'use_statistical': True},
    'w/o_statistical': {'use_graph': True, 'use_semantic': True, 'use_statistical': False},
    'w/o_sentiment': {'use_graph': True, 'use_semantic': True, 'use_statistical': True, 'use_sentiment': False},
}

def train_ablation(ablation_type: str, args):
    """训练单个消融实验。"""
    config = ABLATION_TYPES[ablation_type]
    print(f"\n开始训练消融实验: {ablation_type}")
    print(f"配置: {config}")
    
    # 加载数据
    # ... 数据加载代码 ...
    
    # 初始化模型（根据消融配置）
    if not config['use_graph']:
        # 只用 RWKV，不用 GAT
        from models.base_model import GraphRWKV_Model
        model = GraphRWKV_Model(...).to(device)
    else:
        # 使用 GAT，但根据配置调整图结构
        model = GraphRWKV_GNN_Model(
            ...,
            use_semantic_graph=config.get('use_semantic', True),
            use_statistical_graph=config.get('use_statistical', True),
            use_sentiment=config.get('use_sentiment', True),
        ).to(device)
    
    # 训练循环（复用 3_train.py 的训练逻辑）
    # ... 训练代码 ...
    
    # 保存结果
    output_path = f"{args.output}/checkpoints/best_model_ablation_{ablation_type}.pth"
    torch.save(model.state_dict(), output_path)
    print(f"消融实验 {ablation_type} 完成，模型已保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='运行 Graph-RWKV 消融实验')
    parser.add_argument('--data', type=str, required=True, help='训练数据 CSV 路径')
    parser.add_argument('--graph', type=str, required=True, help='图谱邻接矩阵路径')
    parser.add_argument('--output', type=str, default='./outputs', help='输出目录')
    parser.add_argument('--ablation', type=str, required=True, 
                       choices=list(ABLATION_TYPES.keys()) + ['all'],
                       help='消融实验类型：w/o_graph, w/o_semantic, w/o_statistical, w/o_sentiment, 或 all（运行所有）')
    parser.add_argument('--batch_size', type=int, default=1024, help='批大小')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    
    args = parser.parse_args()
    
    # 设置 GPU
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 启用 TF32
    torch.set_float32_matmul_precision('high')
    
    # 运行消融实验
    if args.ablation == 'all':
        print("运行所有消融实验...")
        for ablation_type in ABLATION_TYPES.keys():
            train_ablation(ablation_type, args)
        print("\n所有消融实验完成！")
    else:
        train_ablation(args.ablation, args)

if __name__ == '__main__':
    main()
```

### **示例 1：优化的 RWKV TimeMixing（针对 48GB GPU）**
```python
@torch.jit.script
def rwkv_linear_attention_optimized(
    time_decay: torch.Tensor,
    time_first: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """优化的 RWKV 线性注意力计算（针对 48GB GPU）。
    
    参考 RWKV-LM 官方实现优化。
    使用原地操作和预分配内存减少分配开销。
    支持大 batch size (1024+) 和长序列 (60+)。
    """
    B, T, C = k.shape
    device = k.device
    
    # 预分配输出张量（GPU 内存充足）
    wkv = torch.zeros(B, T, C, device=device, dtype=k.dtype)
    
    # 初始化状态
    aa = torch.zeros(B, C, device=device, dtype=k.dtype)
    bb = torch.zeros(B, C, device=device, dtype=k.dtype)
    pp = torch.full((B, C), -1e9, device=device, dtype=k.dtype)
    
    # 优化的循环（GPU 友好）
    for t in range(T):
        kt = k[:, t, :]
        vt = v[:, t, :]
        ww = time_decay + pp
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        
        # 原地操作（减少内存分配）
        aa.mul_(e1).add_(e2 * vt)
        bb.mul_(e1).add_(e2)
        pp.copy_(p)
        
        wkv[:, t, :] = aa / bb.clamp(min=1e-8)
    
    return wkv

# 在模型初始化时启用 torch.compile
class GraphRWKV_Model(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ... 初始化代码 ...
    
    def forward(self, x, vol=None):
        # ... 前向传播 ...
        pass

# 使用方式（在训练脚本中）
model = GraphRWKV_Model(...).to(device)
model = torch.compile(model, mode='reduce-overhead')  # PyTorch 2.1.2
```

### **示例 2：使用 PyG 的 GAT（参考 ADGAT）**
```python
from torch_geometric.nn import GATv2Conv

class OptimizedGraphAttentionLayer(nn.Module):
    """使用 PyG 优化的 GAT 层（参考 ADGAT 实现）。
    
    支持边权重（情感分数），充分利用 48GB GPU。
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gat = GATv2Conv(
            in_channels=in_features,
            out_channels=out_features // num_heads,
            heads=num_heads,
            dropout=dropout,
            edge_dim=1,  # 支持边权重（情感分数）
            concat=True,
            add_self_loops=True,  # 添加自环
        )
        self.layer_norm = nn.LayerNorm(out_features)
    
    def forward(
        self,
        x: torch.Tensor,  # (N, in_features)
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: Optional[torch.Tensor] = None,  # (E, 1) 情感分数
    ) -> torch.Tensor:
        """前向传播。
        
        Args:
            x: 节点特征
            edge_index: 边索引
            edge_attr: 边权重（情感分数，-1.0 到 1.0）
        
        Returns:
            聚合后的节点特征
        """
        h = self.gat(x, edge_index, edge_attr=edge_attr)
        return self.layer_norm(h)
```

### **示例 3：数据加载优化（针对 90GB RAM）**
```python
from torch.utils.data import DataLoader
import numpy as np

class OptimizedFinancialDataset:
    """优化的金融数据集（充分利用 90GB RAM）。"""
    
    def __init__(self, csv_path: str, seq_len: int = 30):
        # 使用内存映射文件（适合大文件）
        self.data = np.memmap(csv_path, mode='r', dtype=np.float32)
        
        # 预加载到 RAM（90GB 足够大）
        print("预加载数据到 RAM...")
        self.data = np.array(self.data)  # 从 memmap 转为普通数组
        
        # ... 其他初始化 ...
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # ... 返回样本 ...
        pass

# 数据加载器配置（充分利用 12 vCPU）
train_loader = DataLoader(
    dataset,
    batch_size=1024,  # 大 batch size（48GB GPU）
    shuffle=True,
    num_workers=10,  # 充分利用 12 vCPU
    pin_memory=True,  # GPU 加速
    prefetch_factor=4,  # 预取更多数据
    persistent_workers=True,  # 保持 worker 进程
)
```

### **示例 4：训练脚本优化（针对 48GB GPU）**
```python
import torch
from torch.cuda.amp import autocast, GradScaler

def train_optimized(model, train_loader, val_loader, config):
    """优化的训练函数（针对 48GB GPU + 90GB RAM）。"""
    
    device = torch.device('cuda:0')
    
    # 启用 TF32（A100/H100）
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    
    # 使用 torch.compile 加速（PyTorch 2.1.2）
    model = torch.compile(model, mode='reduce-overhead')
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=1e-5,
    )
    
    # 训练循环
    for epoch in range(config['epochs']):
        model.train()
        for batch in train_loader:
            x = batch['x'].to(device, non_blocking=True)
            y = batch['y'].to(device, non_blocking=True)
            node_indices = batch['node_indices'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 混合精度前向传播
            with autocast():
                pred = model(x, node_indices=node_indices)
                loss = criterion(pred, y)
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        
        # 验证
        val_loss = validate(model, val_loader)
        
        # 保存检查点
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), 'best_model.pth')
```

---

## 📊 性能基准测试要求（针对 48GB GPU + 90GB RAM）

优化后需要提供以下性能指标：

1. **训练速度**（目标）：
   - 每个 epoch 的训练时间：< 5 分钟（batch_size=1024）
   - 每秒处理的样本数：> 2000 samples/sec
   - GPU 利用率：> 90%
   - GPU 内存使用：充分利用（30-40GB）

2. **内存占用**（目标）：
   - 峰值 GPU 内存：30-40GB（充分利用 48GB）
   - 峰值 CPU 内存：< 50GB（90GB RAM 充足）
   - 数据加载内存：< 20GB（RAM 缓存）

3. **模型性能**（目标）：
   - 验证集 Loss：持续下降
   - IC (Information Coefficient)：> 0.05
   - RankIC：> 0.08
   - 方向准确率：> 52%

4. **推理速度**（目标）：
   - 单个样本推理时间：< 1ms
   - 批量推理吞吐量：> 5000 samples/sec（batch_size=1024）

---

## ✅ 验收标准

优化后的代码需要满足：

1. **傻瓜式运行**（最重要）：
   - ✅ 5 个独立可运行的脚本（1_preprocess_data.py, 2_build_graph.py, 3_train.py, 3_train_ablation.py, 4_evaluate.py）
   - ✅ 全量模型训练和消融实验分开（3_train.py 和 3_train_ablation.py）
   - ✅ 命令行参数简洁明了，一眼就知道是干什么的
   - ✅ 每个脚本都有完整的错误处理和进度显示
   - ✅ 关键步骤打印清晰的中文提示

2. **功能正确性**：
   - ✅ 所有测试通过
   - ✅ 模型输出形状正确
   - ✅ 训练可以正常进行
   - ✅ 数据流完整（预处理 → 建图 → 训练 → 评估）

3. **性能提升**（针对 48GB GPU + 90GB RAM）：
   - ✅ 训练速度：batch_size=1024 时，每个 epoch < 5 分钟
   - ✅ GPU 利用率：> 90%
   - ✅ GPU 内存使用：30-40GB（充分利用）
   - ✅ 数据加载：num_workers=10，充分利用 12 vCPU

4. **代码质量**：
   - ✅ 通过 `mypy` 类型检查
   - ✅ 通过 `pylint` 代码检查（分数 ≥ 8.0）
   - ✅ 所有函数有完整的文档字符串（中文）
   - ✅ 代码结构清晰，易于理解

5. **可维护性**：
   - ✅ 代码结构清晰
   - ✅ 注释完整（中文）
   - ✅ 易于扩展
   - ✅ 配置直接写在 py 文件里（脚本顶部）

---

## 🎓 参考资源（论文公开源码）

根据论文参考文献，以下开源项目可以参考：

1. **RWKV 官方实现**：
   - GitHub: https://github.com/BlinkDL/RWKV-LM
   - **参考点**：RWKV 时间混合层的优化实现、JIT 编译技巧

2. **ADGAT (Attribute-Driven Graph Attention Networks)**：
   - GitHub: https://github.com/RuichengFIC/ADGAT
   - **参考点**：金融图神经网络的实现、GAT 优化技巧
   - **论文**：Cheng, D., et al. (2022). "Modeling the Momentum Spillover Effect for Stock Prediction via Attribute-Driven Graph Attention Networks (ADGAT)." AAAI

3. **PyTorch Geometric**：
   - GitHub: https://github.com/pyg-team/pytorch_geometric
   - **参考点**：使用 `GATv2Conv` 替代手动实现、批量图处理

4. **FinGPT**：
   - GitHub: https://github.com/AI4Finance-Foundation/FinGPT
   - **参考点**：金融文本处理、LLM 在金融领域的应用

5. **PyTorch 性能优化指南**：
   - https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
   - **参考点**：48GB GPU 的优化配置、混合精度训练

6. **MATCC (参考趋势解耦思路，但不使用)**：
   - GitHub: https://github.com/caozhiy/MATCC
   - **注意**：新方向不使用 MATCC，但可以参考其数据处理思路

---

## 📌 注意事项

1. **傻瓜式设计**：
   - 每个脚本都是独立的，可以直接运行
   - 命令行参数简洁明了，不要过度复杂
   - 添加清晰的进度显示和错误提示
   - 关键步骤打印中文说明

2. **充分利用硬件**：
   - 48GB GPU：batch_size=1024+，seq_len=60+，启用 TF32
   - 90GB RAM：数据预加载，内存映射文件
   - 12 vCPU：num_workers=8-10，多进程处理

3. **保持兼容性**：
   - 优化时不要破坏现有接口
   - 保留 `QL_MATCC_Model` 等兼容性别名
   - 配置文件向后兼容

4. **参考论文源码**：
   - RWKV-LM：RWKV 时间混合层优化
   - ADGAT：GAT 层实现和金融图处理
   - FinGPT：LLM 批处理优化

5. **渐进式优化**：
   - 先完成脚本重构（任务 0）
   - 再优化性能（任务 1-5）
   - 每次优化后测试验证

6. **文档更新**：
   - 优化后更新 README
   - 更新代码注释
   - 添加使用示例

---

---

## 💡 额外优化建议

### **1. 项目结构优化**

建议的项目目录结构：
```
paper/
├── 1_preprocess_data.py      # 数据预处理（独立脚本）
├── 2_build_graph.py           # 构建图谱（独立脚本）
├── 3_train.py                 # 全量模型训练（独立脚本）
├── 3_train_ablation.py        # 消融实验训练（独立脚本）
├── 4_evaluate.py              # 评估模型（独立脚本）
├── requirements.txt           # Python 依赖（明确版本）
├── README.md                  # 使用说明
├── models/                    # 模型定义
│   ├── base_model.py
│   └── gnn_model.py
├── dataProcessed/             # 数据处理模块（被脚本调用）
│   ├── etl.py
│   ├── align.py
│   ├── build_graph.py
│   └── dataset.py
├── data/                      # 数据目录
│   ├── raw/                   # 原始数据
│   └── processed/             # 处理后的数据
└── outputs/                   # 输出目录
    ├── checkpoints/           # 模型检查点
    ├── logs/                  # 训练日志
    ├── figures/               # 图表
    └── results/               # 评估结果
```

### **2. 依赖管理优化**

创建 `requirements.txt`，明确版本：
```txt
# 核心框架
torch==2.1.2
torchvision==0.16.2
torchaudio==2.1.2

# 图神经网络
torch-geometric==2.4.0

# 数据处理
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2

# LLM 相关
transformers==4.36.2
accelerate==0.25.0

# 可视化
matplotlib==3.8.2
seaborn==0.13.0
tqdm==4.66.1

# 工具
pyyaml==6.0.1
```

**建议**：
- 使用 `pip freeze > requirements.txt` 锁定版本
- 在 README 中说明如何安装依赖
- 考虑使用 `conda` 环境管理（可选）

### **3. 日志系统（重要！每次运行自动保存）**

每个脚本都应该有完善的日志系统，**每次运行自动保存日志到文件**：

```python
import logging
import sys
from datetime import datetime
from pathlib import Path

def setup_logging(output_dir: str, script_name: str) -> logging.Logger:
    """设置日志系统，每次运行创建新的日志文件。
    
    Args:
        output_dir: 输出目录
        script_name: 脚本名称（用于日志文件名）
    
    Returns:
        Logger 对象
    """
    # 创建日志目录
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 日志文件名包含时间戳：script_name_YYYYMMDD_HHMMSS.log
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{script_name}_{timestamp}.log'
    
    # 配置日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 配置日志（同时输出到文件和控制台）
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # 文件输出
            logging.StreamHandler(sys.stdout)  # 控制台输出
        ],
        force=True  # 强制重新配置（避免重复运行时的日志混乱）
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_file}")
    logger.info("="*60)
    return logger

def main():
    # 设置日志（每次运行创建新文件）
    logger = setup_logging('./outputs', 'train')
    
    try:
        # 记录关键信息
        logger.info("开始训练 Graph-RWKV 模型")
        logger.info(f"配置参数: batch_size={args.batch_size}, epochs={args.epochs}")
        
        # 检查输入文件
        if not os.path.exists(args.data):
            logger.error(f"数据文件不存在: {args.data}")
            raise FileNotFoundError(f"数据文件不存在: {args.data}")
        
        # 检查 GPU
        if not torch.cuda.is_available():
            logger.warning("CUDA 不可用，将使用 CPU（训练会很慢）")
        else:
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # 检查磁盘空间（50GB 数据盘）
        disk_usage = shutil.disk_usage(args.output)
        if disk_usage.free < 10 * 1024**3:  # 小于 10GB
            logger.warning(f"磁盘空间不足，剩余: {disk_usage.free / 1024**3:.1f} GB")
        
        # 记录每个步骤
        logger.info("步骤 1/5: 加载数据...")
        # ... 加载数据 ...
        logger.info(f"数据加载完成，训练集: {len(train_dataset)} 样本")
        
        logger.info("步骤 2/5: 初始化模型...")
        # ... 初始化模型 ...
        logger.info("模型初始化完成")
        
        logger.info("步骤 3/5: 开始训练...")
        for epoch in range(args.epochs):
            # ... 训练代码 ...
            logger.info(f"Epoch {epoch+1}/{args.epochs} 完成，Loss: {loss:.6f}")
        
        logger.info("训练完成！")
        
    except KeyboardInterrupt:
        logger.warning("用户中断训练，正在保存检查点...")
        # 保存当前状态
    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)  # exc_info=True 记录完整堆栈
        raise
    finally:
        logger.info("="*60)
        logger.info(f"日志已保存到: {log_file}")
```

**日志文件命名规则**：
- 格式：`{script_name}_{YYYYMMDD}_{HHMMSS}.log`
- 示例：`train_20260123_143025.log`, `build_graph_20260123_150130.log`
- 好处：每次运行都有独立的日志文件，方便查看和对比

**日志内容要求**：
- 时间戳：每条日志都有时间戳
- 步骤：清晰标注当前执行步骤（如 "步骤 1/5: 加载数据..."）
- 进度：记录训练进度（epoch、loss 等）
- 错误：完整的错误信息和堆栈跟踪
- 资源：GPU、内存使用情况
- 配置：记录使用的配置参数

### **4. 性能监控和资源使用**

添加资源监控功能：

```python
import psutil
import GPUtil

def monitor_resources():
    """监控 CPU、内存、GPU 使用情况。"""
    # CPU 使用率
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # 内存使用
    memory = psutil.virtual_memory()
    
    # GPU 使用
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        logger.info(f"CPU: {cpu_percent}% | "
                   f"RAM: {memory.percent}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB) | "
                   f"GPU: {gpu.load*100:.1f}% | "
                   f"GPU Memory: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB")
```

### **5. 数据管理优化（针对 50GB 磁盘）**

**建议**：
1. **压缩存储**：使用压缩格式存储中间文件
   ```python
   # 使用压缩的 NumPy 格式
   np.savez_compressed('graph.npz', adj_matrix=adj_matrix)
   
   # 使用 Parquet 格式存储 CSV（更小）
   df.to_parquet('data.parquet', compression='snappy')
   ```

2. **清理临时文件**：脚本运行后自动清理
   ```python
   import tempfile
   import shutil
   
   # 使用临时目录
   temp_dir = tempfile.mkdtemp()
   try:
       # ... 处理逻辑 ...
   finally:
       shutil.rmtree(temp_dir)  # 清理临时文件
   ```

3. **数据检查点**：支持断点续跑
   ```python
   # 保存处理进度
   checkpoint_file = 'data/processed/.preprocess_checkpoint.json'
   if os.path.exists(checkpoint_file):
       # 从检查点恢复
       with open(checkpoint_file) as f:
           checkpoint = json.load(f)
   ```

### **6. 配置管理优化**

使用 `config.yaml` 统一管理配置，脚本只覆盖关键参数：

```yaml
# config.yaml
data:
  input_dir: "./data/raw"
  output_dir: "./data/processed"
  seq_len: 30

model:
  n_embd: 256
  n_layers: 3
  gnn_embd: 64
  dropout: 0.1

training:
  batch_size: 1024      # 针对 48GB GPU
  epochs: 30
  lr: 3e-4
  num_workers: 10       # 针对 12 vCPU
  pin_memory: true
  prefetch_factor: 4

hardware:
  gpu_id: 0
  enable_tf32: true
  enable_compile: true   # torch.compile
  mixed_precision: true  # AMP
```

脚本中加载配置：
```python
import yaml

def load_config(config_path='config.yaml'):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

# 命令行参数可以覆盖配置
config = load_config()
if args.batch_size:
    config['training']['batch_size'] = args.batch_size
```

### **7. 实验管理和版本控制**

**建议**：
1. **实验命名**：使用时间戳或描述性名称
   ```python
   from datetime import datetime
   exp_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
   # 或
   exp_name = f"graph_rwkv_embd{MODEL_N_EMBD}_layers{MODEL_N_LAYERS}"
   ```

2. **保存实验配置**：每次训练保存完整配置到日志
   ```python
   experiment_dir = f"outputs/experiments/{exp_name}"
   os.makedirs(experiment_dir, exist_ok=True)
   
   # 保存配置（以 JSON 格式，不使用 yaml）
   config_dict = {
       'n_embd': MODEL_N_EMBD,
       'n_layers': MODEL_N_LAYERS,
       'batch_size': TRAIN_BATCH_SIZE,
       'epochs': TRAIN_EPOCHS,
       # ... 其他配置
   }
   with open(f"{experiment_dir}/config.json", 'w') as f:
       json.dump(config_dict, f, indent=2)
   
   # 保存命令行参数
   with open(f"{experiment_dir}/args.txt", 'w') as f:
       f.write(str(args))
   ```

3. **Git 集成**：记录代码版本
   ```python
   import subprocess
   
   def get_git_hash():
       try:
           return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
       except:
           return "unknown"
   
   config['git_hash'] = get_git_hash()
   ```

### **8. 快速验证脚本**

创建一个快速验证脚本，检查环境是否正确：

```python
# check_environment.py
"""快速检查环境和依赖。"""

import sys

def check_python_version():
    if sys.version_info < (3, 10):
        print("❌ Python 版本需要 >= 3.10")
        return False
    print(f"✅ Python 版本: {sys.version}")
    return True

def check_torch():
    try:
        import torch
        print(f"✅ PyTorch 版本: {torch.__version__}")
        print(f"✅ CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    except ImportError:
        print("❌ PyTorch 未安装")
        return False

def check_dependencies():
    deps = ['pandas', 'numpy', 'sklearn', 'torch_geometric', 'transformers']
    for dep in deps:
        try:
            __import__(dep)
            print(f"✅ {dep} 已安装")
        except ImportError:
            print(f"❌ {dep} 未安装")
            return False
    return True

if __name__ == '__main__':
    print("检查环境...")
    all_ok = True
    all_ok &= check_python_version()
    all_ok &= check_torch()
    all_ok &= check_dependencies()
    
    if all_ok:
        print("\n✅ 环境检查通过！")
    else:
        print("\n❌ 环境检查失败，请安装缺失的依赖")
        sys.exit(1)
```

### **9. README 完善**

README 应该包含：

1. **快速开始**：
   ```markdown
   ## 快速开始
   
   ### 1. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```
   
   ### 2. 检查环境
   ```bash
   python check_environment.py
   ```
   
   ### 3. 运行完整流程
   ```bash
   # 数据预处理
   python 1_preprocess_data.py --input_dir ./data/raw --output_dir ./data/processed
   
   # 构建图谱
   python 2_build_graph.py --news ./data/processed/Stock_News.csv --output ./data/processed
   
   # 训练全量模型
   python 3_train.py --data ./data/processed/Final_Model_Data.csv --graph ./data/processed/Graph_Adjacency.npy --output ./outputs
   
   # 评估模型
   python 4_evaluate.py --checkpoint ./outputs/best_model.pth --test_data ./data/processed/test.csv
   ```
   ```

2. **常见问题**：
   ```markdown
   ## 常见问题
   
   ### Q: 训练时 GPU 内存不足？
   A: 减小 batch_size，或减小 n_embd、n_layers
   
   ### Q: 数据预处理很慢？
   A: 检查 num_workers 设置，充分利用 12 vCPU
   
   ### Q: 磁盘空间不足？
   A: 清理 outputs/ 目录的旧实验，或使用压缩格式
   ```

### **10. 性能优化检查清单**

在代码中添加性能检查：

```python
def check_performance_settings():
    """检查性能优化设置。"""
    checks = []
    
    # 检查 TF32
    if torch.cuda.is_available():
        if torch.backends.cuda.matmul.allow_tf32:
            checks.append("✅ TF32 已启用")
        else:
            checks.append("⚠️ TF32 未启用（建议启用）")
    
    # 检查 cuDNN benchmark
    if torch.backends.cudnn.benchmark:
        checks.append("✅ cuDNN benchmark 已启用")
    else:
        checks.append("⚠️ cuDNN benchmark 未启用（建议启用）")
    
    # 检查数据加载
    if num_workers >= 8:
        checks.append(f"✅ num_workers={num_workers}（充分利用 CPU）")
    else:
        checks.append(f"⚠️ num_workers={num_workers}（建议 >= 8）")
    
    for check in checks:
        logger.info(check)
```

### **11. 一键运行脚本**

创建一个主脚本，可以一键运行完整流程：

```python
# run_all.py
"""一键运行完整流程。"""

import subprocess
import sys

def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"执行: {description}")
    print(f"命令: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"❌ {description} 失败")
        sys.exit(1)
    print(f"✅ {description} 完成")

if __name__ == '__main__':
    # 1. 数据预处理
    run_command(
        ['python', '1_preprocess_data.py', '--input_dir', './data/raw', '--output_dir', './data/processed'],
        '数据预处理'
    )
    
    # 2. 构建图谱
    run_command(
        ['python', '2_build_graph.py', '--news', './data/processed/Stock_News.csv', '--output', './data/processed'],
        '构建图谱'
    )
    
    # 3. 训练模型
    run_command(
        ['python', '3_train.py', '--data', './data/processed/Final_Model_Data.csv', 
         '--graph', './data/processed/Graph_Adjacency.npy', '--output', './outputs'],
        '训练模型'
    )
    
    # 4. 评估模型
    run_command(
        ['python', '4_evaluate.py', '--checkpoint', './outputs/best_model.pth', 
         '--test_data', './data/processed/test.csv'],
        '评估模型'
    )
    
    print("\n✅ 完整流程执行成功！")
```

### **12. 输出结果整理**

自动整理实验结果：

```python
# 训练完成后自动生成报告
def generate_training_report(output_dir, metrics):
    report = f"""
# 训练报告

## 模型配置
- n_embd: {config['n_embd']}
- n_layers: {config['n_layers']}
- batch_size: {config['batch_size']}

## 训练结果
- 最佳验证 Loss: {metrics['best_val_loss']:.6f}
- IC: {metrics['ic']:.4f}
- RankIC: {metrics['rank_ic']:.4f}

## 资源使用
- 训练时间: {metrics['training_time']:.1f} 分钟
- GPU 峰值内存: {metrics['gpu_memory']:.1f} GB
"""
    with open(f"{output_dir}/report.md", 'w') as f:
        f.write(report)
```

---

## 📋 最终检查清单

优化完成后，确保：

- [ ] **5 个脚本都可以直接点击运行**（无需命令行参数）
- [ ] **每次运行自动保存日志到文件**（包含时间戳）
- [ ] **日志内容完整**：时间戳、步骤、进度、错误、资源使用
- [ ] 命令行参数可选（用于覆盖默认配置）
- [ ] 有完整的错误处理和日志系统
- [ ] 有 `requirements.txt`（不使用 config.yaml，配置直接写在 py 文件里）
- [ ] 有 `check_environment.py` 检查脚本
- [ ] README 包含快速开始和常见问题
- [ ] 代码通过类型检查（mypy）
- [ ] 所有函数有中文文档字符串
- [ ] 充分利用 48GB GPU 和 90GB RAM
- [ ] 输出结果清晰易读

## 🎯 关键特性总结

### **1. 直接点击运行**
- ✅ 每个脚本都有 `if __name__ == '__main__'` 入口
- ✅ 使用脚本中的默认配置（配置直接写在 py 文件里）
- ✅ 命令行参数可选，用于覆盖配置
- ✅ 适合在 IDE 中直接运行

### **2. 完善的日志系统**
- ✅ 每次运行自动创建新的日志文件
- ✅ 日志文件命名：`{script_name}_{YYYYMMDD}_{HHMMSS}.log`
- ✅ 同时输出到文件和控制台
- ✅ 记录时间戳、步骤、进度、错误、资源使用
- ✅ 方便检查和调试代码

---

**最后更新**：2026-01-23
**维护者**：项目开发团队
