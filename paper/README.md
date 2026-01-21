# QL-MATCC-GNN: 量子增强的金融风险评估模型

> 基于 LLM 语义增强与 Quantum-RWKV 时空解耦的金融风险评估研究

## 📚 项目简介

本项目实现了 **QL-MATCC-GNN** 框架，融合了大语言模型(LLM)语义理解与量子计算优势，用于金融市场的信用风险预测与价格趋势研判。

### 核心特性

- 🔮 **Quantum-RWKV**: 量子增强的时序编码器，捕捉市场混沌动力学
- 📊 **MATCC**: 多尺度趋势解耦，分离宏观趋势与微观波动
- 🕸️ **LLM-EvolvingGraph**: 基于Qwen2.5的动态语义图谱构建
- 📈 **S&P 500 成分股**: 遵循学术惯例，使用高质量数据集
- 🎯 **IC/RankIC 评估**: 量化投资级别的评估指标

## 🚀 快速开始

### 环境要求

```bash
Python 3.10+
PyTorch 2.1.2+
CUDA 11.8+
GPU: 48GB显存（推荐）或24GB+
CPU: 8核+ (推荐12核)
内存: 32GB+ (推荐90GB)
```

### 48GB显存优化版 ⚡

本项目已针对48GB显存服务器进行优化，启用：
- ✅ **LLM模式**: Qwen2.5-14B语义关系提取
- ✅ **更大模型**: n_embd=384, n_layers=4
- ✅ **更大batch**: batch_size=1024
- ✅ **更多workers**: num_workers=10, prefetch_factor=6

详见 [性能优化说明](docs/性能优化说明.md)

### 安装依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pennylane pandas numpy scikit-learn matplotlib tqdm akshare
```

### 运行完整流程

#### 方法1: 一键脚本（推荐）⚡

```bash
cd /root/autodl-tmp/paper
screen -S ql_matcc          # 创建screen会话（防止SSH断连）
./run_optimized.sh          # 运行优化版脚本
# Ctrl+A, D 分离会话
# screen -r ql_matcc 重新连接
```

#### 方法2: 逐步运行

```bash
cd /root/autodl-tmp/paper

# 1. 数据处理
python -m dataProcessed.etl              # ETL处理
python -m dataProcessed.align            # 数据对齐
python -m dataProcessed.build_graph --use_llm  # LLM图谱构建 (60-90分钟)

# 2. 训练模型
python -m training.train_full            # 训练完整模型 (150分钟)
python -m training.train_ablation        # 消融实验 (750分钟)

# 3. 评估模型
python -m evaluation.evaluate_all        # 统一评估
python -m evaluation.evaluate_by_group   # 分波动率组评估
```

## 📖 文档导航

- 📌 [**快速开始**](docs/快速开始.md) - 5分钟上手指南
- 🗂️ [**项目结构说明**](docs/项目结构说明.md) - 目录结构与文件说明
- 📦 [**模块说明**](docs/模块说明.md) - 每个Python文件的详细说明
- 🔄 [**数据处理流程**](docs/数据处理流程.md) - ETL流程与特征工程
- 🎓 [**训练与评估**](docs/训练与评估.md) - 模型训练与评估指南
- 💡 [**改进方案说明**](docs/改进方案说明.md) - 优化历程与问题解决

## 📁 项目结构

```
paper/
├── dataProcessed/          # 数据处理模块
│   ├── etl.py             # ETL: 合并股价+新闻
│   ├── align.py           # 对齐: 股价+市场+新闻
│   ├── build_graph.py     # 图谱: LLM/规则构建
│   └── dataset.py         # Dataset: PyTorch数据加载
│
├── models/                 # 模型定义
│   ├── base_model.py      # 基础模型(Quantum-RWKV + MATCC)
│   └── gnn_model.py       # GNN模型(完整框架)
│
├── training/               # 训练脚本
│   ├── train_full.py      # 完整模型训练
│   └── train_ablation.py  # 消融实验
│
├── evaluation/             # 评估脚本
│   ├── evaluate_all.py    # 统一评估
│   ├── evaluate_by_group.py  # 分组评估
│   └── plot_ablation_comparison.py
│
├── outputs/                # 输出目录
│   ├── checkpoints/       # 模型权重
│   ├── logs/              # 训练日志
│   ├── figures/           # 可视化图表
│   └── results/           # 评估结果
│
├── docs/                   # 文档
└── data/                   # 数据目录（需自行准备）
```

## 🎯 核心指标

本项目使用量化投资级别的评估指标：

- **IC (Information Coefficient)**: 预测值与真实收益率的Pearson相关系数
- **RankIC**: 预测排名与真实排名的Spearman相关系数
- **Directional Accuracy**: 涨跌方向预测准确率
- **MSE/RMSE/MAE**: 统计误差指标

## 📊 实验结果

完整实验结果请参见 `outputs/results/`，包括：

- `evaluation_overall.csv`: 整体性能对比
- `evaluation_by_volatility.csv`: 按波动率分组评估
- `ablation_results_summary.csv`: 消融实验汇总

## 🔧 配置说明

### 关键超参数 (train_full.py) - 48GB显存优化版

```python
# 模型容量（充分利用48GB显存）
n_embd = 384          # 嵌入维度 (256→384, +50%)
n_layers = 4          # RWKV层数 (3→4, +33%)
gnn_embd = 96         # GNN嵌入维度 (64→96, +50%)

# 训练效率（充分利用GPU并行能力）
batch_size = 1024     # 批次大小 (512→1024, +100%)
epochs = 30           # 训练轮数 (20→30, +50%)
num_workers = 10      # CPU加载线程 (8→10)
prefetch_factor = 6   # 预加载倍数 (4→6)

# 学习率（差异化策略）
lr = 3e-4             # 经典层学习率
quantum_lr = 3e-5     # 量子层学习率 (差异化学习率)
q_threshold = 动态     # 从训练集p70波动率自动获取

# 正则化
dropout = 0.15        # Dropout率
weight_decay = 1e-5   # 权重衰减
early_stop_patience = 8  # Early Stopping耐心值 (6→8)
```

### S&P 500 过滤

```python
# 在 build_graph.py 中
USE_SP500_ONLY = True  # ✅ 默认使用S&P 500成分股（学术惯例）

# 使用全量股票（不推荐，图谱过于稀疏）
python -m dataProcessed.build_graph --all_stocks
```

### LLM 模式（48GB显存默认启用）⚡

```python
# build_graph.py 配置
USE_LLM_DEFAULT = True        # ✅ 48GB显存默认启用LLM
MAX_NEWS_PER_TICKER = 200     # ✅ 增加采样量
MAX_TOTAL_NEWS = 100000       # ✅ 增加总量

# 手动运行
python -m dataProcessed.build_graph --use_llm

# 如果显存不足，可临时使用规则模式
python -m dataProcessed.build_graph --no_llm
```

**LLM优势**:
- 语义级关系提取 vs 简单规则匹配
- 图谱质量提升200%+
- 关系带类型标签（供应/竞争/合作等）

## 📝 论文引用

如果本项目对您的研究有帮助，请引用相关论文：

```bibtex
@article{ql-matcc-gnn,
  title={基于 LLM 语义增强与 Quantum-RWKV 时空解耦的金融风险评估研究},
  author={Your Name},
  journal={...},
  year={2024}
}
```

## 🙏 致谢

本项目参考了以下工作：

- [MATCC](https://github.com/caozhiy/MATCC): 多尺度趋势解耦
- [Quantum-RWKV](https://github.com/ChiShengChen/QuantumRWKV): 量子增强RWKV
- [PennyLane](https://pennylane.ai/): 量子机器学习框架

## 📄 许可证

本项目仅供学术研究使用。

## 📧 联系方式

如有问题或建议，请提交 Issue 或联系作者。

---

**注意**: 本项目使用真实金融数据，请遵守相关法律法规，仅用于学术研究，不构成投资建议。
