
------

# 硕士学位论文实施方案与深度研究报告

**论文题目**：基于大语言模型动态图谱与 Graph-RWKV 的时空解耦金融预测研究

**英文题目**：Research on Financial Spatiotemporal Forecasting based on LLM-Driven Dynamic Graphs and Graph-RWKV

------

## 1. 研究背景与立题依据 (Introduction)

### 1.1 问题陈述 (Problem Statement)

金融市场是一个复杂的非线性动态系统，股票价格的波动受**时间维度**（历史量价趋势）和**空间维度**（产业链、竞争关系、宏观事件）的双重影响。当前深度学习模型在金融预测中面临两大核心瓶颈：

1. **序列建模的效率与记忆瓶颈**：
   - 主流 Transformer 架构计算复杂度为 $O(L^2)$，难以高效处理超长历史序列（如过去 3-5 年的日线数据），限制了模型捕捉长周期宏观趋势的能力。
   - RNN/LSTM 类模型虽然推理快，但难以并行训练，且存在长程梯度消失问题。
2. **关联关系的静态化与稀疏性**：
   - 传统图神经网络（GNN）多依赖静态行业分类或历史价格相关性建图，缺乏对突发新闻事件的实时响应能力。
   - 引入 LLM（如 Qwen2.5）提取语义关系虽然前沿，但在 S&P 500 等封闭集合中，纯新闻驱动的图往往极度稀疏（Sparse/Disconnected），导致 GNN 出现“信息孤岛”或“过度平滑”现象。

### 1.2 本研究的解决方案 (Proposed Solution)

本研究提出 **Graph-RWKV** 架构，融合两大核心技术以实现时空解耦建模：

1. **RWKV (Receptance Weighted Key Value)**：作为时间序列编码器，结合了 RNN 的线性推理效率 $O(1)$ 和 Transformer 的并行训练能力，能够高效处理长周期金融时序特征。
2. **LLM 增强的情感加权混合图 (LLM-Enhanced Sentiment-Aware Hybrid Graph)**：利用 Qwen2.5-14B 提取新闻中的因果关系与**情感极性**，并结合量价统计相关性，构建“语义+统计”的双层动态图，解决图稀疏问题并赋予边权重以方向性。

------

## 2. 核心方法论与架构设计 (Methodology)

### 2.1 模块一：基于 Qwen2.5 的动态图谱构建 (Dynamic Graph Construction)

针对图稀疏性和关系方向性问题，采用“混合致密化 + 情感加权”策略：

**策略 A：LLM 语义关系与情感提取（显式层）**

- **模型**：Qwen2.5-14B-Instruct。

- **输入**：FNSPID 数据集每日新闻 Headline + Content。

- **Prompt 优化**：不仅提取关系，还需输出情感极性。

  - *Prompt 示例*：“分析新闻，提取 S&P 500 公司间关系。若两公司受同一事件影响，建立 'Co-Event' 关系。同时评估事件对 Target 公司的情感影响分数（-1为极度利空，1为极度利好）。输出 JSON: `{source, target, relation, sentiment_score}`。”

- **时间衰减累积 (Temporal Decay)**：

  $$A_t^{semantic} = \alpha \cdot A_{t-1}^{semantic} + (1-\alpha) \cdot (E_t \odot S_t)$$

  其中 $E_t$ 为关系矩阵，$S_t$ 为情感分数矩阵，$\alpha$ 为衰减因子（如 0.9），保证图谱的时间连续性。

**策略 B：统计相关性修正（隐式层）**

- **补全逻辑**：计算 S&P 500 成分股过去 30 天收益率的皮尔逊相关系数，保留强相关边（$|\rho| > 0.6$），捕捉资金面的隐式联动。

  $$A_t^{stat} = \mathbb{I}(|\text{Corr}(X_i, X_j)| > \epsilon)$$

**最终混合图 (Final Hybrid Graph)**：

$$A_t^{final} = \text{Norm}(A_t^{semantic} + \lambda \cdot A_t^{stat})$$

既包含新闻的可解释性（Alpha），又保证图的连通性（Beta）。

### 2.2 模块二：Graph-RWKV 时空编码器 (Spatiotemporal Encoder)

**时间维：RWKV-TimeSeries Encoder**

- 每个股票 $i$ 的特征序列 $X_i \in \mathbb{R}^{T \times F}$ 独立进入 RWKV 模块。

- 利用线性 Attention 机制捕捉长程依赖：

  $$h_t = \text{RWKV_Block}(x_t, h_{t-1})$$

- 输出：包含时间上下文的节点嵌入 $H_t \in \mathbb{R}^{N \times D}$。

**空间维：动态图注意力 (Dynamic GAT)**

- 在时间步 $t$，利用构建好的混合图 $A_t^{final}$ 进行空间聚合，使预测考虑到供应链上下游及竞争对手的实时状态：

  $$H'_t = \text{GAT}(H_t, A_t^{final})$$

------

## 3. 详细实施指南 (Implementation Guide)

### 3.1 数据集准备

1. **FNSPID**：用于提取新闻语义图及情感因子。
2. **S&P 500 历史行情 (Alpha158)**：使用 Qlib 库生成 158 维量价因子（开高低收、均线、波动率等）。

### 3.2 严格的数据对齐与前瞻偏差规避 (Crucial Step)

**原则**：严禁使用未来信息。

- **Cut-off Time**：设定每日 16:00（收盘时间）为界。
- **逻辑**：
  - $T$ 日 16:00 **之前**发布的新闻 $\rightarrow$ 构建 $T$ 日的图 $\rightarrow$ 参与预测 $T+1$ 日收益。
  - $T$ 日 16:00 **之后**发布的新闻 $\rightarrow$ 归入 $T+1$ 日数据流。

### 3.3 训练与验证策略 (Training Strategy)

**滚动窗口验证 (Rolling Window / Walk-Forward Validation)**：

为适应金融市场风格切换（Regime Shift），不采用静态划分，而是采用滚动方式：

- **阶段 1**：Train (2018-2020) $\rightarrow$ Test (2021 Q1)
- **阶段 2**：Train (2018-2020 + 2021 Q1) $\rightarrow$ Test (2021 Q2)
- **阶段 3**：...以此类推。

### 3.4 模型搭建细节

- **Input Layer**: Linear 投影 ($158 \to 128$)。

- **Backbone**: 2-4 层 RWKV Block（基于 `rwkv.model` 官方代码修改，去除 NLP 特定 Token 处理）。

- **Graph Layer**: PyG GATv2Conv。

- **Loss Function**: RankIC Loss (侧重排序能力)。

  $$Loss = - \text{PearsonCorr}(\text{Pred}_{\text{rank}}, \text{Target}_{\text{rank}})$$

------

## 4. 实验设计与预期结果 (Experimental Design)

### 4.1 评估指标 (Evaluation Metrics)

除了常规的 MSE/RMSE，重点考察**金融投资价值**：

1. **预测能力指标**：IC (Information Coefficient), RankIC。
2. **投资组合回测 (Top-K Long-Short)**：
   - **年化收益率 (Annualized Return)**：衡量绝对收益。
   - **夏普比率 (Sharpe Ratio)**：衡量风险调整后收益（核心指标）。
   - **最大回撤 (Max Drawdown)**：衡量风控能力。

### 4.2 消融实验 (Ablation Studies)

设计以下变体以证明创新点的有效性：

| **模型变体**          | **图结构构建方式**         | **时序编码器** | **实验目的**                       |
| --------------------- | -------------------------- | -------------- | ---------------------------------- |
| **Graph-RWKV (Ours)** | **Hybrid (Semantic+Stat)** | **RWKV**       | **验证完整方案优越性**             |
| Variant A             | Only Statistical           | RWKV           | 验证 LLM 提取的新闻语义图的增益    |
| Variant B             | Only Semantic              | RWKV           | 验证统计图补全稀疏性的必要性       |
| Variant C             | Hybrid                     | LSTM/GRU       | 验证 RWKV 在长序列建模上的优势     |
| Variant D             | None (No Graph)            | RWKV           | 验证引入空间依赖（图结构）的必要性 |

------

## 5. 参考文献 (References)（可以参考他们公开的源码，如果用的上的话）

**Category A: RWKV & Linear Attention**

1. Peng, B., et al. (2023). "RWKV: Reinventing RNNs for the Transformer Era." arXiv preprint. (GitHub: BlinkDL/RWKV-LM)
2. Sun, Y., et al. (2023). "Retentive Network: A Successor to Transformer for Large Language Models." (GitHub: microsoft/torchscale)

**Category B: Financial Graph Learning (Baselines)**

3. Cheng, D., et al. (2022). "Modeling the Momentum Spillover Effect for Stock Prediction via Attribute-Driven Graph Attention Networks (ADGAT)." AAAI. (GitHub: RuichengFIC/ADGAT)
4. Ye, J., et al. (2024). "Time-Series Graph Neural Networks for Stock Prediction." IEEE TKDE.

**Category C: LLM in Finance**

5. Li, X., et al. (2023). "FinGPT: Open-Source Financial Large Language Models." arXiv. (GitHub: AI4Finance-Foundation/FinGPT)
6. Zhang, X., et al. (2024). "Graph-ToolFormer: To Reason with Graph Self-Correction." (GitHub: jwzhanggy/Graph-ToolFormer)

------

## 6. 创新点总结 (Conclusion & Contributions)

1. **架构创新**：首次将 **RWKV** 引入金融时空预测。相比 Transformer，RWKV 允许输入更长的历史窗口（如 1 年以上），从而捕捉宏观长周期趋势，且训练显存占用大幅降低。
2. **图谱构建创新**：提出 **LLM 驱动的情感-语义-统计多模态图融合策略**。通过 LLM 提取显式因果与情感极性，结合统计隐式关联，彻底解决了 S&P 500 核心资产池中的图稀疏与孤岛问题，同时赋予了量化黑盒模型“可解释性”。
3. **应用价值**：通过严格的滚动窗口回测与夏普比率评估，验证了模型在实际投资组合管理中的超额收益能力。