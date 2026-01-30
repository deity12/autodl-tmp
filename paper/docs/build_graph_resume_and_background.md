# build_graph.py 断点续跑与后台运行说明

## 一、本次失败时已保存的内容

脚本在失败前会保存以下内容，**下次直接重新运行即可从断点续跑**：

| 文件 | 路径 | 说明 |
|------|------|------|
| **断点 checkpoint** | `paper/data/processed/Graph_Adjacency_checkpoint.npz` | 邻接矩阵 + 进度（next_pos、batch_size 等） |
| **采样新闻** | `paper/data/processed/Graph_Adjacency_news_sampled.csv` | 固化后的采样新闻列表，避免重新采样 |
| **节点列表** | `paper/data/processed/Graph_Tickers.json` | 图节点（S&P 500 tickers） |

### 当前断点状态（你这次跑的结果）

- **已处理进度**：`next_pos = 40000`（约 4 万条新闻已跑完）
- **总新闻数**：约 **85,878** 条（采样文件行数 - 1 表头）
- **剩余**：约 **45,878** 条（约 53% 未完成）
- **checkpoint 间隔**：每 1 万条保存一次，下次启动会从 40000 继续

**结论**：这些文件都还在，直接重新运行脚本就会从第 40000 条新闻接着跑，不会从头再来。

---

## 二、后台运行命令（推荐）

在项目根目录 `/root/autodl-tmp` 下执行：

```bash
# 1. 进入项目目录
cd /root/autodl-tmp

# 2. 后台运行，日志写入 build_graph.log，断点续跑
nohup /root/miniconda3/bin/python paper/dataProcessed/build_graph.py --split_date 2020-12-31 > build_graph.log 2>&1 &

# 3. 记下进程 ID，方便后续查看
echo $!
```

### 查看是否在跑、进度如何

```bash
# 看进程是否还在
ps aux | grep build_graph

# 实时看日志最后几行（进度条会在日志里）
tail -f /root/autodl-tmp/build_graph.log

# 看日志最后 50 行
tail -50 /root/autodl-tmp/build_graph.log
```

### 再次确认会从断点续跑

重新运行后，日志里应出现类似：

```text
[Resume] 检测到已保存的采样新闻: .../Graph_Adjacency_news_sampled.csv (n=85878)
[Resume] 从 checkpoint 恢复：next_pos=40000, batch_size=64
```

看到 `next_pos=40000` 就说明是从 4 万条处续跑。

---

## 三、跑完后会生成什么

- `paper/data/processed/Graph_Adjacency.npy` — 最终混合图
- `paper/data/processed/Graph_Adjacency_semantic.npy` — 语义图
- `paper/data/processed/Graph_Adjacency_stat.npy` — 统计图
- `paper/data/processed/Graph_Adjacency_relation_stats.json` — 关系统计
- 跑完后脚本会**自动删除** `Graph_Adjacency_checkpoint.npz`，只保留最终结果。

---

## 四、若仍担心断网/关机会丢进度

- 断点每 **1 万条** 自动保存一次，最多损失 1 万条内的进度。
- 只要不删 `Graph_Adjacency_checkpoint.npz` 和 `Graph_Adjacency_news_sampled.csv`，随时可以再次用上面的 nohup 命令续跑。
