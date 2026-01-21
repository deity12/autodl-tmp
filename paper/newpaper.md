这份草稿的质量已经非常高，逻辑清晰、技术点明确，且很好地融合了我们之前讨论的“静态累积图谱”和“S&P 500”的合理性论述。

不过，如果要完全达到**优秀硕士学位论文（或开题报告）**的标准，还需要在以下几个方面进行**“学术升维”**：

1. **语气与措辞 (Tone & Diction)**：部分工程细节（如“服务器上下载好了”）过于口语化，需要转化为学术表达（如“本地化部署”）。
2. **创新点凝练 (Contributions)**：评审老师最看重“你到底创新在哪”。需要显式地列出“本文主要贡献”。
3. **理论深度 (Theoretical Depth)**：在描述量子部分时，要强调其数学意义（希尔伯特空间映射），而不仅仅是“用了这个模型”。
4. **引用规范**：GitHub 链接通常不直接放在正文中，而是作为脚注或参考文献。

以下是经过**深度润色和学术化重构**的版本，你可以直接用于**开题报告书**或**论文正文**。

---

# [论文/开题报告] 基于 LLM 结构先验与 Quantum-RWKV 时空解耦的金融风险评估研究

## 一、 立题依据与研究意义 (Introduction)

### 1.1 研究背景

随着金融市场的日益复杂化，信用风险预测与价格趋势研判已从单一的量价分析转向多模态数据融合。然而，现有基于深度学习的量化方法在实际应用中仍面临三大核心挑战：

1. **宏观与微观的纠缠 (Entanglement of Factors)**：个股价格受系统性风险（Beta）与特质性风险（Alpha）的共同驱动，传统模型难以在非平稳市场中有效剥离两者，导致模型在牛熊转换期的泛化能力不足。
2. **长序列中的混沌动力学 (Chaos in Long Sequences)**：金融时间序列具有高度非线性与混沌特征。经典神经网络在捕捉长周期中隐藏的微小扰动及“黑天鹅”信号时存在理论瓶颈。
3. **语义信息的结构化缺失**：非结构化财经新闻蕴含着关键的风险传导路径（如供应链断裂），但传统方法难以将此类语义信息转化为高质量的图结构先验。

### 1.2 本文主要贡献 (Main Contributions)

针对上述痛点，本项目提出 **QL-MATCC-GNN** 框架，主要创新点如下：

1. **提出基于 LLM 的“因果累积式”图谱构建策略**：利用 Qwen2.5 的语义理解能力与分层采样机制，从海量新闻中蒸馏出长期稳定的供应链与竞争关系，构建全局结构先验，有效解决了动态图稀疏与噪声问题。
2. **设计 Quantum-RWKV 量子增强时序编码器**：创新性地在 RWKV 架构中嵌入变分量子线路 (VQC)，利用量子纠缠与高维希尔伯特空间映射特性，突破了经典神经网络在拟合高波动混沌信号时的性能瓶颈。
3. **构建多尺度趋势解耦与自适应混合推理机制**：显式分离宏观趋势与微观波动，并基于波动率阈值动态调度量子计算资源，在保证精度的同时实现了计算效率的优化。

---

## 二、 研究内容与技术路线 (Methodology)

### 2.1 模块一：基于 LLM 的结构化语义图谱构建 (Structural Semantic Graph)

为解决非结构化文本利用率低的问题，本项目提出一种**“LLM 驱动的全局因果累积图谱”**构建方法。

* **零样本关系抽取 (Zero-shot Relation Extraction)**：
利用 **Qwen2.5-14B-Instruct** 大模型，对 FNSPID 数据集中的财经新闻标题进行语义解析。通过构建特定的 Prompt，抽取显式实体关系三元组（如 `<Entity A, Supply, Entity B>`）。为确保学术实验的可复现性，推理阶段采用确定性解码策略（Greedy Decoding），并约束输出为严格的 JSON 格式。
* **分层采样策略 (Stratified Sampling Strategy)**：
针对原始数据中存在的长尾分布问题，采用分层采样策略构建图谱数据源：对每只 S&P 500 成分股，设定采样上限（200条）并进行全局随机打乱。该策略有效消除了字母序偏差，将图谱的节点连接率提升至预期水平，确保了图神经网络的信息传递效率。
* **全局因果累积机制 (Global Causal Accumulation)**：
考虑到供应链与竞争关系具有**长期稳定性 (Long-term Stability)**，且每日动态图极其稀疏，本项目摒弃了高噪的“快照式”动态图构建方式，转而构建**全局演化邻接矩阵**。
* **机制**：仅聚合**训练截点 (Cut-off Date)** 之前的所有历史关系。
* **优势**：该矩阵作为一种**全局结构先验 (Global Structural Prior)**，刻画了市场中稳固的风险传导骨架，既符合风控业务逻辑，又严格遵守时间因果律，杜绝了未来信息泄露 (Look-ahead Bias)。



### 2.2 模块二：基于 MATCC 的多尺度趋势解耦 (Trend Decoupling)

为解决金融数据的非平稳性 (Non-stationarity)，引入多尺度解耦机制：

* **宏观引导 (Market Guidance)**：利用 1D-CNN 提取 S&P 500 指数及波动率的宏观表征，作为全局上下文注入模型。
* **正交分解**：利用因果滑动平均 (Causal Moving Average) 将个股序列  分解为趋势项 （反映基本面）与波动项 （反映市场情绪）。其中，波动项将被送入量子模块进行深度特征挖掘。

### 2.3 模块三：Quantum-RWKV 量子时序编码器 (Quantum-Enhanced Encoder)

这是本项目的核心理论突破点。针对波动项中的高维混沌特征，设计量子-经典混合架构。

* **骨干网络**：采用 **RWKV** 架构，继承了 Transformer 的并行训练优势与 RNN 的线性推理复杂度。
* **量子通道混合 (Quantum Channel Mixing)**：在 RWKV 的通道混合层引入 **变分量子线路 (VQC)**：
* **高维映射**：利用 `AngleEmbedding` 将经典特征映射至高维量子希尔伯特空间 (Hilbert Space)。
* **纠缠演化**：通过 `StronglyEntanglingLayers` 实现量子比特间的纠缠，捕捉特征间复杂的非线性相关性。
* **测量坍缩**：测量 Pauli-Z 算符期望值，将其转化回经典特征向量。此过程赋予了模型超越传统 MLP 的非线性拟合能力。



### 2.4 模块四：时空特征融合与自适应推理 (Spatiotemporal Fusion)

* **自适应混合推理 (Adaptive Hybrid Inference)**：
设计基于波动率的门控机制。在训练阶段统计波动率分布，选取 **70% 分位数** 作为阈值。推理时，仅对高波动（High Volatility）样本激活量子通道，其余样本使用经典通路。该策略实现了“好钢用在刀刃上”，平衡了量子模拟的高计算成本与模型性能。
* **时空聚合**：利用 **图注意力网络 (GAT)** 在累积图谱上聚合邻居特征，捕捉风险的空间传导效应；最终通过 Late Fusion 融合时序、空间及宏观特征，输出预测结果。

---

## 三、 实验方案与预期结果 (Experimental Design)

### 3.1 数据集与实验设置

* **数据来源**：FNSPID (Financial News and Stock Price Integration Dataset)。
* **研究对象**：**S&P 500 成分股**。选取理由如下：(1) 数据质量高，新闻覆盖全面；(2) 供应链关系公开透明，便于图谱验证；(3) 具有较高的图谱连接密度，适合 GNN 建模。
* **实验环境**：PyTorch 2.1 + PennyLane (量子框架) + RTX 3090。
* **基线模型**：LSTM, Transformer, RWKV (纯经典版), MATCC (SOTA), GAT。

### 3.2 评价指标体系

为了满足学术严谨性与量化投资实战要求，建立多维评价体系：

1. **拟合精度**：MSE, RMSE, MAE —— 衡量回归误差。
2. **方向预测**：Directional Accuracy (DA) —— 衡量择时能力。
3. **量化因子表现 (核心指标)**：
* **IC (Information Coefficient)**：预测值与真实值的皮尔逊相关系数。
* **RankIC (Rank Information Coefficient)**：预测排名的斯皮尔曼秩相关系数。
* *注：RankIC 是衡量模型在投资组合管理中排序能力的关键指标，直接反映选股策略的有效性。*



### 3.3 实施计划与可行性分析

1. **工程可行性**：项目组已完成 Qwen2.5 的本地化部署及 FNSPID 数据的预处理清洗；MATCC 与 Quantum-RWKV 的基础代码已跑通。
2. **算力保障**：实验室配备 RTX 3090 服务器，满足混合精度训练与量子模拟的显存需求。

---

## 四、 参考文献与代码资源 (References)

*(在正式论文中，请使用标准的文献引用格式，如下仅为说明)*

1. **MATCC Model**: Cao, Z., et al. "MATCC: A Novel Approach...". GitHub: `caozhiy/MATCC` (Reference approach for trend decoupling).
2. **Quantum-RWKV**: Chen, C. "Quantum-Enhanced Channel Mixing...". GitHub: `ChiShengChen/QuantumRWKV` (Reference approach for VQC integration).

---

## 五、 工程实现补充：与顶会常见做法的对齐（Implementation Notes）

### 5.1 mini-batch 下的图聚合：邻居扩展 + 诱导子图

在训练阶段，样本以 mini-batch 方式进入模型。若仅在 batch 内切片邻接矩阵（即只让 batch 内节点互相消息传递），会导致：

- batch 内往往 **不包含某个股票的真实邻居**，图聚合退化为“几乎无图信息”的 MLP；
- 同一股票在一个 batch 中出现多次时，若把每条样本当作不同节点，会引入语义错误。

因此，本项目在 `models/gnn_model.py` 中采用更常见的工程实现：

- **Batch 节点去重**：对 batch 内 `node_indices` 去重，并对重复 ticker 的时序表征做 mean pooling；
- **邻居扩展**：对每个 batch 节点从全图邻接中取最多 \(K\) 个邻居，构造诱导子图（induced subgraph）；
- **静态节点表示**：对“batch 外但在子图中的邻居节点”，使用可学习的 `node_embedding` 作为初始特征（训练时会被更新），用于提供稳定的结构先验输入；
- 在诱导子图上做一次 GAT 聚合，最后再映射回 batch 样本得到 `H_graph` 与 `H_temporal` 进行 late fusion。

### 5.2 RankIC/IC 的计算口径：按日期截面（cross-section）统计

股票排序类工作通常按日期计算横截面相关，再对日期取均值（而不是把所有时间点混在一起做相关）。本项目已在工程上做对齐：

- `dataProcessed/dataset.py` 为每条样本返回 `target_date`（与预测目标对齐的日期）；
- `training/train_full.py` 在验证阶段默认采用 **按日期计算 IC 与 RankIC 并取平均** 的口径。

### 5.3 可选：回归 + 排序（Learning-to-Rank）混合目标

为更贴近“选股排序”任务，训练可使用 MSE 与排序损失的混合目标：

\[
\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda \cdot \mathcal{L}_{\text{RankNet}}
\]

其中 RankNet 使用同一日期截面内的股票对进行 pairwise 约束（训练脚本中可通过 `use_rank_loss` 开启，并用 `rank_loss_weight` 控制 \(\lambda\)）。