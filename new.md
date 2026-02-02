---

# 硕士学位论文实施方案与深度研究报告 (Final Implementation Baseline)

**论文题目**：基于大语言模型动态图谱与 Graph-RWKV 的时空解耦金融预测研究
**英文题目**：Research on Financial Spatiotemporal Forecasting based on LLM-Driven Dynamic Graphs and Graph-RWKV

---

## 1. 研究背景与立题依据 (Introduction)

### 1.1 问题陈述 (Problem Statement)

金融市场是一个非线性、高噪声的复杂自适应系统，其资产价格波动受**时间维度**（历史量价趋势）和**空间维度**（产业链、竞争关系、宏观事件）的双重驱动。当前深度学习在金融时空预测中面临三大核心瓶颈：

1. **长序列建模的效率悖论**：
* 主流 Transformer 架构的计算复杂度随序列长度呈二次方增长（Vaswani et al., 2017），难以高效处理跨度数年（Sequence Length > 60 甚至几百）的日线级特征，限制了模型捕捉长周期宏观趋势的能力。
* RNN/LSTM 类模型虽然推理快，但无法并行训练，且存在长程梯度消失问题。


2. **关联关系的静态化与稀疏性**：
* 传统图神经网络（GNN）多依赖静态行业分类或纯历史价格相关性建图，缺乏对突发新闻事件的实时响应能力。
* 在封闭股票池（如 S&P 500）中，纯新闻驱动的图往往极度稀疏（Disconnected），导致 GNN 出现“信息孤岛”或“过度平滑”现象。


3. **特征工程的同质化**：
* 仅依赖基础 OHLCV 数据难以捕捉深层市场信号，缺乏系统性的因子挖掘（如 Alpha158）和截面标准化处理，导致模型输入信噪比低。



### 1.2 本研究的解决方案 (Proposed Solution)

本研究提出 **Graph-RWKV** 架构，融合两大前沿技术以实现高效的时空解耦建模：

1. **RWKV (Receptance Weighted Key Value)**：作为时间序列编码器，结合了 RNN 的线性推理效率（O(L)）和 Transformer 的并行训练能力，能够高效处理长周期金融时序特征（Peng et al., 2023）。
2. **LLM 增强的混合动态图 (LLM-Enhanced Hybrid Graph)**：利用轻量化大语言模型 (Qwen2.5-14B) 离线提取新闻中的语义关联，并结合量价统计相关性，构建“语义+统计”的双层动态图谱，解决稀疏性问题。

---

## 2. 核心方法论与架构设计 (Methodology)

### 2.1 模块一：混合动态图谱构建 (Hybrid Graph Construction)

针对图稀疏性和关系方向性问题，采用“混合致密化 + 严格防泄露”策略。

**策略 A：LLM 语义关系与情感提取（显式层）**

* **输入数据**：经过 ETL 清洗的 `Stock_News.csv`（或 S&P 500 过滤后的 `Stock_News_sp500.csv`）。
* **模型选型**：Qwen2.5-14B-Instruct（4-bit/8-bit 量化或 FP16，运行于 48GB VRAM 环境）。
* **关系类型**：限定 7 类——supply（供应链）、competition（竞争）、cooperation（合作）、merger（并购）、lawsuit（诉讼）、investment（投资）、co-event（共同事件），便于复现与消融。
* **处理流程**：
1. **抽取**：从新闻 Headline/Content 中提取三元组 `{Source, Target, Relation, Sentiment}`，仅保留 src/dst 均在图节点（S&P 500 成分股）内的边。
2. **截断 (Strict Cutoff)**：设置 `--split_date 2021-12-31`（与训练集结束日期一致，见 3.3 节）。构图**仅使用此日期前的历史新闻**，严禁使用验证集/测试集期间的信息构建图结构，彻底杜绝 Look-ahead Bias。
3. **时序衰减**：采用指数衰减因子（本实现 α=0.995，半衰期约 138 天），按日累积边权，确保旧闻的影响力随时间降低；取值依据见本文 3.3 节时间切分与 2.1 节「旧闻影响力随时间降低」。



**策略 B：统计相关性修正（隐式层）**

* **补全逻辑**：计算 S&P 500 成分股**过去 N 天**收益率的皮尔逊相关系数，保留强相关边（本实现 N=30 天，|ρ|>0.6），捕捉资金面的隐式联动，解决新闻缺失时的图连通性问题。

**最终产物**：

* 邻接矩阵：`Graph_Adjacency.npy`（使用 `split_date=2021-12-31` 前数据一次性构建的累积混合图，训练/验证/测试阶段**静态使用**，无未来信息）。
* 节点映射：`Graph_Tickers.json`（与 dataset 的 ticker2idx 顺序严格一致，防止索引错位）。
* 消融用图：`Graph_Adjacency_semantic.npy`（仅语义图）、`Graph_Adjacency_stat.npy`（仅统计图）。

### 2.2 模块二：Graph-RWKV 时空编码器 (Spatiotemporal Encoder)

**1. 节点对齐 (Node Alignment)**

* **全集定义**：构建 S&P 500 成分股的静态并集 (Static Union)。
* **动态掩码 (Dynamic Masking)**：若某股票在当日停牌或数据缺失，其特征向量置零，且在计算 Loss 时被 Mask 掉，不影响梯度更新。

**2. 时间维：RWKV-TimeSeries Encoder**

* 每个股票 i 的**精选 26 维特征序列**（基于 Alpha158-like 因子池筛选与去噪）独立进入 RWKV 模块。
* 利用线性 Attention 机制捕捉长程依赖，输出时间嵌入。

**3. 空间维：广播优化版图注意力 (Broadcast GAT)**

* **内存优化**：针对全连接或稠密图场景，不构造巨大的边对注意力中间张量。
* **实现机制**：通过 Broadcasting 机制直接计算注意力分数，将显存复杂度从 O(E·d) 降低至 O(N·d)，适配 48GB 显存环境（N=节点数，E=边数，d=特征维）。

---

## 3. 详细实施指南 (Implementation Details)

### 3.1 数据集准备与清洗 (ETL Pipeline)

1. **S&P 500 样本过滤**：
* **基准**：基于 `sp500_list.txt` 从全量 FNSPID 数据中筛选。
* **产出**：`Stock_Prices_sp500.csv` 和 `Stock_News.csv`。


2. **新闻清洗与防泄露 (Anti-Leakage Strategy)**：
* **问题**：原始数据中存在大量 `00:00:00` 时间戳的新闻。
* **对策**：实施 **"Conservative T+1 Shift"** 策略。若新闻时间无法精确到盘中，强制归入 **T+1 日** 的交易决策。宁可牺牲部分实时性，也要绝对保证无未来信息泄露。


3. **特征工程 (Feature Engineering)**：
* **脚本**：`feature_engineering.py`
* **方法**：基于纯 Pandas 实现（fallback 机制）或 Pandas-TA。
* **初始池**：基于 Alpha158-like 因子库生成 **158 维**量价因子（Momentum / Volatility / Price-Volume / Overlap 等）。
* **特征筛选 (Refined Feature Set)**：考虑金融数据低信噪比特性，全量特征噪声大且不利于收敛；最终通过特征筛选与去噪，构建**26 维精选特征集**（以 1d/5D 窗口为主），覆盖动量、波动率、量价趋势、强弱指标等维度，例如：
  - **动量/收益**：`ret_1d`, `log_ret_1d`, `roc_5`, `mom_5`, `oc_ret`
  - **趋势/均线**：`close_sma_5`, `close_ema_5`, `close_sma_ratio_5`, `close_ema_ratio_5`
  - **波动率**：`hl_range`, `ret_std_5`, `atr_5`, `bb_width_5`, `bb_percent_5`
  - **量价/资金**：`vol_mean_5`, `vol_std_5`, `vwap_5`, `mfi_5`, `cmf_5`
  - **强弱/趋势强度**：`rsi_5`, `cci_5`, `willr_5`, `di_plus_5`, `di_minus_5`, `adx_5`
* **预处理**：对特征进行截面标准化（Rank-based），执行 `groupby('Date').rank(pct=True) - 0.5`，将特征映射至 **[-0.5, 0.5]**，降低量纲差异与异常值影响。
* **存储**：全量特征存为 `sp500_alpha158_features.parquet`；最终实验通过 `input_dim=26` 使用上述精选子集。





### 3.2 架构解耦与动态加载 (Architecture Decoupling)

* **存储解耦**：
* 基础数据（OHLCV + Text + Label）存为 CSV。
* Alpha158-like 特征池（158 维）存为 Parquet；训练/评估阶段通过 `input_dim=26` 选择 26 维精选特征。


* **动态缝合**：
* 在 `FinancialDataset` 初始化时，根据 `Date` 和 `Ticker` 索引，将 CSV 与 Parquet 数据在内存中动态合并 (Merge)。
* 优势：避免了生成体积巨大的单一 CSV 文件，且方便灵活调整特征维度。



### 3.3 科学时间切分策略 (Train/Valid/Test Split)

为确保实验的严谨性并符合顶会论文标准（参考 AAAI 2024 MDGNN、Qlib Alpha158 基准），采用**抗周期切分法（4:1:1）**。**关键原则**：

1. **图谱仅使用训练集截止日期前的数据构建**（`split_date = 2021-12-31`），训练/验证/测试阶段**共用同一静态图**，从源头杜绝未来信息泄露。
2. **验证集独立于测试集**：验证集仅用于早停和超参调优，**不参与最终指标报告**；最终论文只报告测试集指标。
3. **验证集选用 2022 年熊市**：在最难行情里选模型，保证选出的模型具备抗跌能力，而非“只会做多”。

| 数据集划分 | 时间范围 | 样本量 | 用途 |
| --- | --- | --- | --- |
| **训练集 (Train)** | **2018-01-01 至 2021-12-31** | 4 年 | 模型参数更新、梯度下降；涵盖 2018 震荡、2019 慢牛、2020 熔断反转、2021 牛市 |
| **验证集 (Valid)** | **2022-01-01 至 2022-12-31** | 1 年 | 超参搜索、早停监控、最佳 checkpoint 保存；2022 熊市，用于挑选抗跌模型 |
| **测试集 (Test)** | **2023-01-01 至 2023-12-31** | 1 年 | **最终性能评估**（论文报告此区间指标）；OOD 区间，检验模型适应新风格能力 |

**图谱截止日期**：`2021-12-31`（与训练集结束日期一致）。建图命令：`python 2_build_graph.py --split_date 2021-12-31`。


## 4. 实验设计与评估 (Experimental Design)

**实验配置（Clean Run v1，更新：2026-02）**
- **数据切分**：4:1:1 抗周期切分（Train 2018–2021.12，Val 2022，Test 2023），图谱 `split_date=2021-12-31`
- **特征**：基于 Alpha158-like 生成 158 维因子，并筛选为 **26 维精选特征集**用于最终实验
- **序列长度**：`seq_len=60`
- **训练**：`epochs=50`, `batch_size=1024`, `lr=1e-3`, `optimizer=AdamW`, `dropout=0.3`, `early_stop=10`
- **模型**：`n_embd=64`, `n_layers=2`, `gnn_embd=32`；启用 RankIC Loss，`rank_loss_weight=1.0`
- **实验脚本**：统一入口 `run_experiment.py`（`--model rwkv|lstm|gru`，`--use_graph`），一键队列 `run_all_final.sh`；结果输出至 `outputs/final_run/<exp_name>/`，汇总表 `outputs/final_run/summary.csv`


### 4.1 评估指标 (Evaluation Metrics)

遵循顶会标准（参考 AAAI 2024 MDGNN、KDD 2020 MTGNN、Qlib 基准），采用以下指标体系：

**1. 截面预测能力指标（Cross-sectional Metrics）**

| 指标 | 定义 | 说明 |
| --- | --- | --- |
| **IC (Information Coefficient)** | 每日 Pearson(ŷ_t, r_t) 的均值 | 衡量预测值与真实收益的线性相关性 |
| **RankIC** | 每日 Spearman(ŷ_t, r_t) 的均值 | 衡量排序一致性，对异常值更鲁棒 |
| **ICIR** | mean(IC) / std(IC) | 衡量预测能力的**稳定性**，越高越好 |
| **RankICIR** | mean(RankIC) / std(RankIC) | RankIC 版本的稳定性指标 |

**计算方式**（代码实现）：
```python
# 按日期分组，每日计算一个截面 IC/RankIC
for date in unique_dates:
    day_pred, day_true = get_samples(date)
    ic_list.append(pearsonr(day_pred, day_true))
    rankic_list.append(spearmanr(day_pred, day_true))
IC = mean(ic_list)
ICIR = mean(ic_list) / std(ic_list)
```

**2. 回归指标（Regression Metrics）**

| 指标 | 说明 |
| --- | --- |
| **MSE / RMSE** | 均方误差，评估预测精度 |
| **方向准确率 (Directional Accuracy)** | sign(ŷ) == sign(r) 的比例 |

**3. 投资组合回测指标 (Portfolio Backtest)**

| 策略 | 说明 |
| --- | --- |
| **Top-K Long-Short** | 每日做多预测得分最高的 K 只，做空得分最低的 K 只 |
| **参数** | K = 30（符合 S&P 500 约 6% 的选股比例） |

| 指标 | 定义 |
| --- | --- |
| **年化收益率 (ARR)** | 累计收益年化，系数 252 交易日 |
| **夏普比率 (Sharpe)** | mean(daily_ret) / std(daily_ret) × √252 |
| **最大回撤 (MaxDD)** | max(peak - trough) / peak |


### 4.2 Baseline 对比 (Baselines)

为验证方法有效性，与以下 SOTA 方法对比（参考顶会论文实验设计）：

| 类别 | 方法 | 来源 | 说明 |
| --- | --- | --- | --- |
| **时序模型** | LSTM | Baseline | 经典 RNN，无图结构 |
| | GRU | Baseline | 轻量 RNN |
| | Transformer | NIPS 2017 | 自注意力机制 |
| **图模型** | GAT | ICLR 2018 | 静态图注意力 |
| | MTGNN | KDD 2020 | 多变量时序+自适应图 |
| **混合模型** | MDGNN | AAAI 2024 | 多关系动态图+Transformer |
| **本文** | **Graph-RWKV** | Ours | RWKV + LLM 混合图 + 广播 GAT |


### 4.3 消融实验 (Ablation Studies)

为验证各模块的有效性，设计以下对比实验：

| 消融变体 | 配置 | 验证目标 |
| --- | --- | --- |
| **w/o Graph** | `use_graph=False` | 验证图谱的空间聚合增益 |
| **w/o Semantic** | 仅使用统计相关性图 | 验证 LLM 语义图的价值 |
| **w/o Statistical** | 仅使用 LLM 语义图 | 验证统计图的补全作用 |
| **w/o RankIC Loss** | `use_rank_loss=False` | 验证排序损失的贡献 |
| **LSTM Backbone** | `temporal_backend=lstm` | 验证 RWKV 相比 LSTM 的优势 |
| **GRU Backbone** | `temporal_backend=gru` | 验证 RWKV 相比 GRU 的优势 |

---


### 4.4 主实验结果 (Main Results)

实验采用 **Clean Run** 统一配置：4:1:1 切分（Train 2018–2021.12，Val 2022，Test 2023），`seq_len=60`, `lr=1e-3`, `dropout=0.3`, `rank_loss_weight=1.0`, `epochs=50`，**26 维精选特征集**。跑完 `run_all_final.sh` 后，从 `outputs/final_run/summary.csv` 取数填入下表，作为论文 **Table 1**。

**表 1：不同模型在 S&P 500 数据集上的性能对比（Clean Run v1，Val=2022 熊市，Test=2023）**

| 模型 | Graph | Backbone | 特征数 | Val RankIC (2022) | Test RankIC (2023) | Test IC |
| --- | --- | --- | --- | --- | --- | --- |
| **Graph-RWKV (Ours)** | **✓** | **RWKV** | **26** | （待填） | （待填） | （待填） |
| RWKV w/o Graph | ✗ | RWKV | 26 | （待填） | （待填） | （待填） |
| Graph-LSTM | ✓ | LSTM | 26 | （待填） | （待填） | （待填） |
| Graph-GRU | ✓ | GRU | 26 | （待填） | （待填） | （待填） |

**预期与解读**：
1. **图结构的有效性**：Graph-RWKV 相对 RWKV w/o Graph 在 Val（2022 熊市）与 Test（2023）上应有明显提升，说明动态图谱能捕捉股票间隐式关联。
2. **抗周期设计**：在 2022 熊市上选出的最佳模型，在 2023 OOD 测试上可检验是否具备抗跌与适应新风格能力。
3. **基线对比**：Graph-LSTM / Graph-GRU 作为对比基线，与 Graph-RWKV 使用完全相同的配置与数据切分，仅时间编码器不同。

### 4.5 讨论 (Discussion)

Clean Run 采用 2022 熊市作验证集，选出的模型应更抗跌；2023 作为完全独立的测试集，用于报告最终 RankIC/IC。若 Graph-RWKV 在 Val 与 Test 上均优于无图基线与 LSTM/GRU 基线，可支撑“图结构 + RWKV”的有效性；若在 Test 上存在基线略高的情况，可在 Discussion 中从稳健性、容量与过拟合角度进行讨论。

### 4.6 实验流程与可视化 (Clean Run & Visualization)

**论文 v1 实验流程（Clean Run）**：
1. 建图：`python 2_build_graph.py --split_date 2021-12-31`
2. 一键跑 4 个实验：`bash run_all_final.sh`（或依次执行 `run_experiment.py --model rwkv --use_graph` 等）
3. 汇总：从 `outputs/final_run/summary.csv` 取 Val/Test RankIC 填入表 1

**可视化**：
1. **训练曲线**：从 `outputs/final_run/<exp_name>/` 下各实验的日志或 JSON 绘制 Val RankIC 随 Epoch 变化，对比四条曲线。
2. **结果对比图**：用 Test RankIC / Test IC 做柱状图；若需回测，可补充轻量 `backtest.py` 形成累计收益曲线。
## 5. 参考文献 (References)

**Category A: Transformer & RWKV (时序编码器)**

1. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). "Attention Is All You Need." *NeurIPS*.
2. Peng, B., Alcaide, E., Anthony, Q., Albalak, A., Arcadinho, S., et al. (2023). "RWKV: Reinventing RNNs for the Transformer Era." *Findings of EMNLP 2023*, pp. 14048–14077.
3. Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., et al. (2023). "Retentive Network: A Successor to Transformer for Large Language Models." *arXiv:2307.08621*.

**Category B: Graph Neural Networks & Financial Forecasting (Baseline 对比)**

4. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). "Graph Attention Networks." *ICLR*. [GAT Baseline]
5. Wu, Z., Pan, S., Long, G., Jiang, J., Chang, X., & Zhang, C. (2020). "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks." *KDD 2020*. [MTGNN Baseline]
6. Yin, Q., Yang, C., Chen, Q., et al. (2024). "MDGNN: Multi-Relational Dynamic Graph Neural Network for Comprehensive and Dynamic Stock Investment Prediction." *AAAI 2024*. [SOTA Baseline]
7. Cheng, D., Yang, F., Xiang, S., & Liu, J. (2022). "Financial Time Series Forecasting with Multi-Modality Graph Neural Networks." *Pattern Recognition*, 121, 108218.

**Category C: LLM & Knowledge Graph**

8. Romanou, A., et al. (2024). "FinDKG: Dynamic Knowledge Graphs with Large Language Models for Detecting Global Trends in Financial Markets." *arXiv:2407.10909*.

**Category D: Quantitative Finance & Evaluation**

9. Yang, L., et al. (2020). "Qlib: An AI-oriented Quantitative Investment Platform." *arXiv:2009.11189*. (Alpha158 定义来源, IC/ICIR 评估标准).
10. Goodwin, T. (2011). "The Information Ratio." *CFA Institute*. (ICIR 定义来源).

---

## 6. 创新点与总结 (Conclusion & Contributions)

**核心贡献**（对齐顶会论文表述）：

1. **架构创新 (Novelty)**：
   - 首次将 **RWKV**（O(L) 复杂度线性注意力）引入金融时空预测领域
   - 提出 **Broadcast GAT** 优化策略，在 48GB 显存限制下实现对 S&P 500 全量成分股的长周期（seq_len ≥ 30）时空建模
   - 设计 **LLM 增强的混合动态图**（语义层 + 统计层），解决纯新闻图的稀疏性问题

2. **实验严谨性 (Rigor)**：
   - 采用 **抗周期切分法（4:1:1）**：Train 2018–2021.12、Val 2022 熊市、Test 2023 OOD；在最难行情中选模型，保证抗跌与可复现
   - 图谱仅使用训练集截止日期（`split_date=2021-12-31`）前的数据构建，从数据源头杜绝 Look-ahead Bias
   - 提出 **"Conservative T+1 Shift"** 策略处理新闻时间戳不确定性

3. **评估全面性 (Comprehensive Evaluation)**：
   - 采用 **IC/RankIC/ICIR** 等量化金融标准指标（参考 Qlib 基准）
   - 与 **LSTM/GRU/GAT/MTGNN/MDGNN** 等 SOTA 方法对比
   - 设计 6 组消融实验验证各模块有效性

4. **可复现性 (Reproducibility)**：
   - 完整开源实现：数据清洗（`1_preprocess_data.py`）、LLM 建图（`2_build_graph.py --split_date 2021-12-31`）、统一实验（`run_experiment.py` + `run_all_final.sh`）、评估脚本（`4_evaluate.py`）
   - 论文 v1 封板采用 Clean Run 流程，超参与数据切分与代码一一对应，可直接复现论文结果
