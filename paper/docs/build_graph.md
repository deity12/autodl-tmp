# `build_graph.py` 文档：股票关系图谱构建（LLM/规则 + 可断点续跑）

本文件对应模块：`autodl-tmp/paper/dataProcessed/build_graph.py`  
作用：从新闻文本中抽取“股票-股票”的关系边，构建**无向邻接矩阵** `Graph_Adjacency.npy`，供后续 `QL_MATCC_GNN_Model` 做图聚合使用。

---

## 1. 输入与输出

### 1.1 输入文件

- **价格/训练数据**：`data/processed/Final_Model_Data.csv`  
  - 必须包含列：`Date`, `Ticker`（其余列不影响建图）
  - 用途：
    - 定义图节点全集 `all_tickers = sorted(unique(Ticker))`
    - 计算训练/测试切分日期（防止新闻“未来信息泄露”）

- **新闻数据**：`data/processed/Stock_News.csv`  
  - 必须包含列：`Ticker`  
  - 文本列优先级：`Headline` > `Article_title` > 第一个 object 列
  - 可选列：`Date`（强烈建议有，用于防泄露过滤）

### 1.2 输出文件（核心）

- **邻接矩阵**：`data/processed/Graph_Adjacency.npy`
  - 形状：`(N, N)`，其中 `N = len(all_tickers)`
  - 矩阵含义：
    - 对角线为 1（自环）
    - 若抽取到 `i <-> j` 有关系，则 `A[i,j]=A[j,i]=1`

### 1.3 输出文件（复现/断点续跑/审计）

- **节点顺序元数据**：`data/processed/Graph_Tickers.json`
  - 内容：`{"tickers": [ ... all_tickers ... ]}`
  - 用途：保证和训练阶段 `dataset.py` 的 `ticker2idx` 对齐；也便于论文复现实验。

- **采样新闻固化**：`data/processed/Graph_Adjacency_news_sampled.csv`
  - 用途：
    - 固定分层采样后的新闻集合，保证可复现
    - 支持断点续跑时不因重新采样造成“进度不一致”

- **关系统计（LLM模式更有用）**：`data/processed/Graph_Adjacency_relation_stats.json`
  - `relation_type_counts`：各关系类型计数（如 supply/competition/...）
  - `top_edges`：出现频次最高的若干边（用于检查/论文展示）

- **断点续跑 checkpoint**：`data/processed/Graph_Adjacency_checkpoint.npz`
  - 内容：邻接矩阵 `adj` + meta（json字符串）
  - meta 记录：
    - `next_pos`：下一条要处理的采样新闻行号（精确断点）
    - `batch_size/max_input_tokens/max_new_tokens/do_sample`：当时推理参数
  - 运行成功结束后会自动删除（防止误用旧 checkpoint）

---

## 2. 关键设计约束（非常重要）

### 2.1 图节点顺序必须与训练一致

`dataset.py` 在训练时构建 `ticker2idx` 的逻辑是：

- `all_tickers = sorted(self.df['Ticker'].unique())`

因此 `build_graph.py` 也固定使用 `Final_Model_Data.csv` 的 `Ticker` 作为**节点全集**与**节点顺序**，并写入 `Graph_Tickers.json`。  
即使开启 `S&P 500` 模式，也**不会改变邻接矩阵的节点数量与顺序**，只会影响：

- 哪些新闻参与建边（新闻会先按 S&P 500 过滤）
- LLM/规则抽取出的边是否允许落在“非 S&P 500”的节点上（默认要求边两端都在 active 集合内）

这样做的目的：避免训练阶段 `node_indices` 与邻接矩阵索引错位导致的崩溃或隐性错误。

### 2.2 防止未来信息泄露（训练集新闻）

如果新闻数据包含 `Date` 列，脚本会：

- 从 `Final_Model_Data.csv` 的日期序列取 80% 分位位置作为 `split_date`
- 只保留 `news.Date < split_date` 的新闻用于建图

这是为了让图结构只基于训练期信息，避免把测试期新闻“泄露”到图里。

---

## 3. 运行方式

在项目根目录（`autodl-tmp/paper/`）下运行：

### 3.1 LLM模式（推荐，论文用）

```bash
python -m dataProcessed.build_graph --use_llm
```

### 3.2 规则模式（快速验证/无LLM环境）

```bash
python -m dataProcessed.build_graph --no_llm
```

### 3.3 股票范围：S&P 500 vs 全量

- 默认：S&P 500（用于论文更常见，噪声更低）
- 全量：加 `--all_stocks`

```bash
python -m dataProcessed.build_graph --use_llm --all_stocks
```

### 3.4 调整采样规模（控制时间/质量）

```bash
python -m dataProcessed.build_graph --use_llm --max_per_ticker 200 --max_total 100000
```

---

## 4. LLM 推理与参数（环境变量）

脚本支持通过环境变量调参（更适合脚本/服务器跑通宵）：

- **`LOCAL_MODEL_PATH`**：本地模型目录（默认 `/root/autodl-tmp/models/qwen/Qwen2.5-14B-Instruct`）
- **`LLM_BATCH_SIZE`**：批大小（默认 64；OOM 会自动降级）
- **`LLM_MAX_INPUT_TOKENS`**：输入最大长度（默认 1536；OOM 时会进一步降低）
- **`LLM_MAX_NEW_TOKENS`**：输出最大长度（默认 96；OOM 时会进一步降低）
- **`LLM_DO_SAMPLE`**：是否采样（默认 0，确定性输出更可复现）

示例（Linux/bash）：

```bash
export LLM_BATCH_SIZE=48
export LLM_MAX_INPUT_TOKENS=1536
export LLM_MAX_NEW_TOKENS=96
export LLM_DO_SAMPLE=0
python -m dataProcessed.build_graph --use_llm
```

**Windows PowerShell**:
```powershell
$env:LLM_BATCH_SIZE="48"
$env:LLM_MAX_INPUT_TOKENS="1536"
$env:LLM_MAX_NEW_TOKENS="96"
$env:LLM_DO_SAMPLE="0"
python -m dataProcessed.build_graph --use_llm
```

---

## 5. 断点续跑机制（防止“开一晚钱花了但中断”）

### 5.1 断点续跑依赖的文件

- `Graph_Adjacency_news_sampled.csv`：固定采样新闻（保证进度一致）
- `Graph_Adjacency_checkpoint.npz`：保存邻接矩阵 + `next_pos`

### 5.2 中断后怎么恢复

无需额外参数，直接重复原命令即可：

```bash
python -m dataProcessed.build_graph --use_llm
```

脚本会自动打印类似：

- `[Resume] 检测到已保存的采样新闻 ...`
- `[Resume] 从 checkpoint 恢复：next_pos=..., batch_size=...`

### 5.3 想“从头重建”怎么办

删除以下文件后重跑（谨慎：会丢掉断点）：

- `data/processed/Graph_Adjacency.npy`
- `data/processed/Graph_Adjacency_checkpoint.npz`
- `data/processed/Graph_Adjacency_news_sampled.csv`

---

## 6. OOM/异常容错策略（为什么这次不容易翻车）

### 6.1 OOM 自动降级顺序

当出现 `torch.cuda.OutOfMemoryError` 时，脚本会按顺序尝试：

1. 降低 `batch_size`（直到 1）
2. 降低 `max_input_tokens`（最低到 512）
3. 降低 `max_new_tokens`（最低到 48）
4. 若仍无法恢复：**跳过该 batch**，计入失败数，但不中断整次任务

### 6.2 异常时强制落盘

任何未捕获异常会触发：

- 尽量 flush 当前 batch
- 保存 `Graph_Adjacency_checkpoint.npz`（含 traceback）
- 抛出异常退出（便于你在日志中定位原因）

---

## 7. 关系抽取范式（与论文/顶会常见做法对齐）

该脚本的构图方式属于“文本驱动的事件/关系抽取建图”，常见于：

- 金融风险传染 / 供应链冲击 / 竞争与并购事件驱动图
- 以“公司—公司”为节点对，文本抽取关系边，再用于图神经网络聚合

LLM Prompt 固定输出 JSON 列表，关系类型限定为 6 类：

- `supply`（供应链）
- `competition`（竞争）
- `cooperation`（合作）
- `merger`（并购）
- `lawsuit`（诉讼）
- `investment`（投资）

脚本同时兼容模型输出“跑偏”的常见格式，并尽可能从输出中提取 JSON 子串。

---

## 8. 论文表述建议：为何采用“静态累积图”（全局因果图谱），不做“动态图序列”

### 8.1 推荐定义（与当前实现一致）

建议在论文中将本项目的图谱定义为：

- **捕捉长期稳定关系的全局因果图谱（Global Causal Graph）**：  
  以训练期内的财经新闻标题为证据，抽取企业间的供应链/竞争/并购等显式关系，并进行累积聚合，得到一个**静态、全局、可复现**的邻接矩阵。  
  “因果”体现在严格遵守时间因果律：**只使用训练期（`split_date` 之前）的新闻构图**，避免未来信息泄露。

这类图更符合金融风控业务逻辑：风控关注的是**长期关联与结构性传导路径**（例如稳定的供应链依赖、长期竞争格局、资本关系），而不是“某一天突发新闻”带来的短期噪声。

### 8.2 为什么不构建“按日动态图序列”（常见但不一定更好）

将图按天/按周切成序列通常会遇到：

- **稀疏性更严重**：每天可抽取到的“明确两家公司且关系清晰”的新闻很少，图会非常稀疏，GNN 聚合信号弱且方差大。
- **噪声更高**：突发新闻、标题党、转述等会导致边大幅波动，训练更不稳定，效果可能反而更差。
- **工程成本更高**：需要额外维护按时间索引的图序列，并在训练时按样本日期取子图，复杂度上升但收益不确定。

因此，本项目采用静态累积图作为“结构先验”：提供稳定的空间聚合路径，让时序模型（MATCC + Quantum-RWKV）去负责“短期变化/混沌波动”的表达。

## 8. 最终质量检查（跑完一定要看）

运行结束会输出图统计：

- 节点数、边数、密度
- 有连接的股票占比、最大度、孤立节点数

常用 sanity check：

1. `Graph_Adjacency.npy` 存在且形状为 `(N,N)`  
2. `Graph_Tickers.json` 的 `tickers` 长度等于 `N`  
3. 若 LLM 模式：`Graph_Adjacency_relation_stats.json` 中 `relation_type_counts` 非空（否则说明抽取失败率高或数据问题）

---

## 9. 常见问题（FAQ）

### 9.1 为什么我选了 S&P 500，但输出邻接矩阵还是很大？

这是**刻意设计**：邻接矩阵的节点顺序必须与训练阶段 `dataset.py` 一致。  
S&P 500 模式只影响建边数据与边的允许范围，不改变 `Graph_Adjacency.npy` 的 `N`。

### 9.2 如何确保训练代码不会索引越界？

只要满足：

- 训练数据的 `ticker2idx` 来自同一份 `Final_Model_Data.csv`
- 图谱由同目录下的 `build_graph.py` 生成（会写入 `Graph_Tickers.json`）

则 `node_indices` 一定与邻接矩阵对齐。

### 9.3 checkpoint 没被删除正常吗？

- 正常完成：脚本会删除 `Graph_Adjacency_checkpoint.npz`
- 非正常退出：会保留该文件用于断点续跑/排错

---

## 10. 相关文件

- 构图脚本：`dataProcessed/build_graph.py`
- 数据对齐：`dataProcessed/align.py`（生成 `Final_Model_Data.csv`）
- 数据集：`dataProcessed/dataset.py`（构建 `ticker2idx`，产出 `node_indices`）
- 训练：`training/train_full.py`（加载 `Graph_Adjacency.npy`）
- 模型：`models/gnn_model.py`（使用邻接矩阵做 batch 子图聚合）

---

## 11. 与“顶会/股票排序论文”更一致的训练与评估口径（重要）

很多股票预测/选股类论文（尤其以 **RankIC/排序能力** 为核心）在工程上通常采用两个实践：

### 11.1 按日期截面（cross-section）组织 batch

原因：

- **RankIC/IC 的定义本质上是“同一天横截面”上的相关性**（即同一天对股票排序是否正确），而不是把所有日期混在一起做相关。
- 因此训练时让一个 batch 尽量来自同一天，才能在 batch 内构造合理的排序学习目标（Learning-to-Rank）。

本项目对应实现：

- `dataProcessed/dataset.py` 在每个样本返回 `target_date`（字符串 `YYYY-MM-DD`）。
- `training/train_full.py` 默认启用按日期分组的 batch（`use_date_grouped_batch=True`），使用 `training/date_batch_sampler.py`。

### 11.2 可选：加入排序损失（Learning-to-Rank），与 MSE 混合

很多工作会用 “回归 + 排序” 的混合目标：

- **MSE**：保证数值拟合能力
- **Rank loss（例如 RankNet pairwise loss）**：直接优化“排序正确性”，更贴近 RankIC 指标

本项目对应实现：

- `training/train_full.py` 中 `use_rank_loss=True` 时，会在 batch 同一天的前提下计算 `RankNet` 风格的 pairwise loss，并按 `rank_loss_weight` 与 MSE 线性混合。


