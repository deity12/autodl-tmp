# QL-MATCC-GNN：LLM 建图 + Quantum-RWKV 时序建模 + MATCC 趋势解耦 + GNN 融合（金融收益预测/风险评估）

本目录 `paper/` 是一套可复现实验工程：从 FNSPID 原始行情/新闻出发，完成 **数据清洗与对齐 → LLM 语义关系建图 → QL-MATCC-GNN 训练 → 评估与出图**。

> 这份 README 以“代码真实实现”为准：每一步对应到具体脚本、明确输入/输出文件、给出可直接运行的命令，并指出最常见的坑（例如图谱 tickers 顺序不一致、评估脚本模型维度不匹配等）。

---

## 你应该从哪里开始

- 想快速理解每个文件做什么：先看 `docs/模块说明.md`（逐文件说明）、`docs/项目结构说明.md`（目录结构）。
- 想跑通完整流程：按本文 “快速开始/分步运行”。
- 想了解最近的重构与一致性修复：看 `REFACTORING_SUMMARY.md`。

---

## 环境与依赖

### 硬件建议

- **训练**：GPU 24GB+（推荐 48GB），CPU 8 核+，内存 32GB+（推荐 90GB）
- **LLM 建图（Qwen2.5-14B）**：GPU 48GB 更稳；脚本支持 OOM 自动降 batch，但整体耗时可能为数小时

### 软件版本建议

- Python 3.10+
- PyTorch 2.1+（CUDA 11.8 常见）

### 安装依赖（示例）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pennylane pandas numpy scikit-learn matplotlib tqdm akshare transformers
```

> 说明：量子模块依赖 `pennylane`，如遇到 `autoray` 兼容问题，`models/base_model.py` 会提示你用 `pip install autoray==0.6.5 pennylane --upgrade` 修复。

---

## 数据准备（必须）

本项目**不包含原始数据**。你需要按以下路径放置数据（或按你环境修改脚本里的路径常量）。

### 目录约定

```text
paper/data/raw/FNSPID/
  ├─ full_history/                 # 个股历史行情 CSV（很多文件）
  ├─ nasdaq_exteral_data.csv        # 23GB 新闻大文件
  └─ SP500_Index.csv                # S&P 500 指数（可用脚本下载生成）
```

### S&P500 指数获取

如果你没有 `SP500_Index.csv`，可运行：

```bash
cd /root/autodl-tmp/paper
python -m dataProcessed.download_market_index
```

输出：`paper/data/raw/FNSPID/SP500_Index.csv`

---

## 快速开始（能跑通的推荐方式）

所有命令建议在 `paper/` 下执行：

```bash
cd /root/autodl-tmp/paper
```

### 方式 A：分步运行（推荐，最清晰、最稳）

```bash
# 1) ETL：合并个股行情 + 从新闻大文件抽取（输出 Stock_Prices.csv / Stock_News.csv）
python -m dataProcessed.etl

# 2) 对齐：个股 + 大盘 + 新闻，并生成标签/特征（输出 Final_Model_Data.csv）
python -m dataProcessed.align

# 3) 建图：从训练期新闻抽取股票关系（输出 Graph_Adjacency.npy + tickers.json）
python -m dataProcessed.build_graph --use_llm

# 4) 训练 Full Model（输出 best_model_full.pth + logs/figures）
python -m training.train_full

# 5) 消融实验（默认 3 组：no_quantum / no_matcc / no_graph）
python -m training.train_ablation

# 6) 评估与出图
python -m evaluation.evaluate_all
python -m evaluation.evaluate_by_group
```

### 方式 B：只跑 LLM 建图（长任务，建议 screen/tmux）

`run_llm_graph.sh` 会检查你是否在 screen/tmux 中运行，并保存日志与图谱统计信息：

```bash
cd /root/autodl-tmp/paper
screen -S llm_graph
./run_llm_graph.sh
```

你也可以用环境变量控制 LLM 批处理大小/输入长度（OOM 会自动降）：

```bash
export LLM_BATCH_SIZE=48
export LLM_MAX_INPUT_TOKENS=1536
export LLM_MAX_NEW_TOKENS=96
export LLM_DO_SAMPLE=0
python -m dataProcessed.build_graph --use_llm
```

### 方式 C：一键脚本（可用，但请以训练脚本实际配置为准）

```bash
cd /root/autodl-tmp/paper
./run_optimized.sh
```

说明：`run_optimized.sh` 是“流程编排脚本”，但其注释/提示中个别超参描述可能与当前训练脚本默认值不一致；**最终以 `training/train_full.py` / `training/train_ablation.py` 中的 `CONFIG/BASE_CONFIG` 为准**。

---

## 代码到底在做什么（按目录解释）

### `dataProcessed/`：数据清洗、对齐、建图、Dataset

- `etl.py`：合并 `full_history/*.csv` → `data/processed/Stock_Prices.csv`；流式读取 `nasdaq_exteral_data.csv` → `data/processed/Stock_News.csv`
- `align.py`：把个股行情 + S&P500 指数 + 新闻聚合对齐到同一张表，并生成
  - `Log_Ret`：对数收益率（预测目标）
  - `Volatility_20d`：20 日滚动波动率（量子门控输入）
  输出 `data/processed/Final_Model_Data.csv`
- `build_graph.py`：从训练期新闻构建关系图，输出
  - `data/processed/Graph_Adjacency.npy`：邻接矩阵（无向，含自环）
  - `data/processed/Graph_Adjacency_tickers.json`：节点顺序（用于与 Dataset 的 `ticker2idx` 对齐）
  关键机制：S&P500 过滤、分层采样、LLM 批处理抽取、断点续跑、自动 OOM 降级、严格防未来泄露
- `dataset.py`：`FinancialDataset`，把 `Final_Model_Data.csv` 变成训练样本：
  - 输入 `x`：过去 `seq_len(=30)` 天 × 8 个特征
  - 标签 `y`：目标日 `Log_Ret`
  - `vol`：波动率
  - `node_indices`：股票在图中的节点编号
  - `target_date`：用于“按日期截面”的 IC/RankIC 或按日分组 batch

### `models/`：模型定义

- `base_model.py`：时序主干（不含图）
  - **MATCC**：因果滑动平均做趋势/波动解耦
  - **RWKV**：时间混合层（JIT 加速），使用完整线性层保证表达能力
  - **Quantum_ChannelMixing**：按波动率门控，高波动样本走量子分支（8量子比特 + 4层纠缠，256维希尔伯特空间）
- `gnn_model.py`：完整模型 `QL_MATCC_GNN_Model`
  - 先用 `QL_MATCC_Model` 提取时序表征
  - 再在 `Graph_Adjacency.npy` 上做双头稀疏 GAT 聚合（含 mini-batch 邻居扩展/诱导子图）
  - 融合后输出收益率预测

### `training/`：训练与消融

- `train_full.py`：训练完整模型（默认配置：`n_embd=256, n_layers=3, n_qubits=8, gnn_embd=64, batch=512, epochs=30, dropout=0.1`）
  - 从 `FinancialDataset` 的 `vol_stats['p70']` 自动取 `q_threshold`（量子门控阈值）
  - 支持 AMP、梯度裁剪、早停、差异化学习率、按日期分组 batch、可选 RankNet 排序损失
- `train_ablation.py`：消融实验（与论文/代码一致的 3 组）
  - `no_quantum`：去掉量子模块
  - `no_matcc`：去掉趋势解耦
  - `no_graph`：去掉 GNN 图聚合
- `date_batch_sampler.py`：按 `target_date` 做 batch 的采样器（股票排序/RankIC 口径常用）

### `evaluation/`：评估与可视化

- `evaluate_all.py`：统一评估 Full/消融模型，输出 `outputs/results/evaluation_overall.csv` 等
- `evaluate_by_group.py`：按波动率分组对比（验证“量子模块在高波动更有效”）
- `plot_ablation_comparison.py`：读取 `losses_*.json` 重画消融对比图与汇总表

---

## 输出文件说明（跑完你会得到什么）

### 数据阶段（`paper/data/processed/`）

- `Stock_Prices.csv`：合并后的个股行情
- `Stock_News.csv`：过滤后的新闻
- `Final_Model_Data.csv`：对齐后的训练表（特征 + 标签 + 新闻聚合）
- `Graph_Adjacency.npy`：关系图邻接矩阵
- `Graph_Adjacency_tickers.json`：邻接矩阵对应的 ticker 顺序（**非常关键**）

### 训练阶段（`paper/outputs/`）

- `outputs/checkpoints/`
  - `best_model_full.pth`
  - `best_model_no_quantum.pth`
  - `best_model_no_matcc.pth`
  - `best_model_no_graph.pth`
- `outputs/logs/`：训练过程与 loss 历史（json）
- `outputs/figures/`：训练曲线、对比图（png）
- `outputs/results/`：评估结果表（csv）

---

## 配置与超参数：在哪里改（非常重要）

### 训练配置（当前版本没有 CLI 参数）

本工程的训练超参来自脚本内的 `CONFIG/BASE_CONFIG` 字典：

- Full Model：`training/train_full.py` 的 `CONFIG = {...}`
- 消融：`training/train_ablation.py` 的 `BASE_CONFIG = {...}`

如果你想跑“更大模型/更大 batch/更多 epoch”，请直接修改这些 dict（或后续再加 argparse）。

### LLM 建图配置

`dataProcessed/build_graph.py` 的关键默认值：

- `USE_LLM_DEFAULT = True`（默认启用 LLM）
- `USE_SP500_ONLY = True`（默认 S&P500 模式）
- `MAX_NEWS_PER_TICKER = 200`
- `MAX_TOTAL_NEWS = 100000`

以及可通过环境变量覆盖的推理参数：

- `LLM_BATCH_SIZE`
- `LLM_MAX_INPUT_TOKENS`
- `LLM_MAX_NEW_TOKENS`
- `LLM_DO_SAMPLE`

本地模型路径默认：

- `LOCAL_MODEL_PATH=/root/autodl-tmp/models/qwen/Qwen2.5-14B-Instruct`

---

## 常见问题（跑不通时先看这里）

### 1) 图谱节点顺序错位（最危险：可能不报错但结果全错）

症状：训练时报错提示 graph tickers 与 dataset tickers 不一致，或你怀疑图聚合不对。

解决：

- 删除旧图并用同一份 `Final_Model_Data.csv` 重新生成：
  - 删除 `data/processed/Graph_Adjacency.npy` 和 `data/processed/Graph_Adjacency_tickers.json`
  - 重新运行 `python -m dataProcessed.build_graph --use_llm`

### 2) LLM 建图 OOM 或太慢

优先降低 batch：

```bash
export LLM_BATCH_SIZE=16
python -m dataProcessed.build_graph --use_llm
```

脚本内部也会自动降 batch/输入长度重试，但手动先降更省时间。

### 3) 评估脚本加载模型报 “size mismatch”

原因：`evaluation/evaluate_all.py` 与 `evaluation/evaluate_by_group.py` 内部有一份 **硬编码的 `MODEL_CONFIG`**，必须与训练时一致。

如果你用默认训练配置（256/3/64），请将评估脚本中的 `MODEL_CONFIG` 同步为训练一致的维度（否则无法加载 `.pth`）。

---

## 指标口径（你论文里怎么写）

- 回归误差：MSE / MAE / RMSE / \(R^2\)
- 方向准确率：`Directional Accuracy`
- 因子评估核心：IC / RankIC
  - 推荐口径：**按日期截面（cross-section）计算 IC/RankIC，再对日期取平均**（你的训练/工具模块已支持这种口径）

---

## 文档导航

- `docs/快速开始.md`：快速跑通
- `docs/项目结构说明.md`：目录结构与文件关系
- `docs/模块说明.md`：每个 Python 文件做什么
- `docs/训练与评估.md`：训练/评估口径与建议

---

## 论文引用与致谢

若用于学术写作，可在此基础上补充你的真实作者信息与期刊/会议。

```bibtex
@article{ql-matcc-gnn,
  title={基于 LLM 语义增强与 Quantum-RWKV 时空解耦的金融风险评估研究},
  author={Your Name},
  journal={...},
  year={2024}
}
```

参考工作（方法/灵感来源）：

- `https://github.com/caozhiy/MATCC`
- `https://github.com/ChiShengChen/QuantumRWKV`
- `https://pennylane.ai/`

---

## 免责声明

本项目使用真实金融数据与指标，仅用于学术研究，不构成任何投资建议。数据与模型输出的使用请遵守相关法律法规。
