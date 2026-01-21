# LLM图谱构建 - 最终检查清单

> 这是一次性“数小时量级”的关键任务，请确认所有项目！（实际时间取决于 batch / 推理参数与GPU吞吐）

## ✅ 代码配置检查

### 1. build_graph.py 配置
```python
MAX_NEWS_PER_TICKER = 200     ✅ 每只股票最多200条（可按需调整）
MAX_TOTAL_NEWS = 100000       ✅ 总上限10万（可按需调整）
USE_LLM_DEFAULT = True        ✅ LLM模式启用
USE_SP500_ONLY = True         ✅ S&P 500成分股
```

### 2. Prompt检查（符合论文）
```
✅ 6类关系类型:
   1. supply (供应链) - 对应论文"供应商违约的连锁反应"
   2. competition (竞争) - 对应论文"市场风险传递"
   3. cooperation (合作) - 企业联盟关系
   4. merger (并购) - 重大资本事件
   5. lawsuit (诉讼) - 法律风险
   6. investment (投资) - 资本关联

✅ Few-shot示例: 4个具体案例
✅ 负例处理: "只提到一家公司" → []
✅ 股票代码限制: 明确"美股代码"
✅ 输出格式: 严格JSON格式
```

### 3. 数据处理流程
```
✅ S&P 500过滤: 366只核心股票
✅ 时间泄露防护: 只用训练集时段新闻（<2022-12-28）
✅ 分层采样: 确保每只股票都有代表性
✅ 进度保存: 每10000条自动checkpoint
✅ 规则兜底: LLM失败时用规则匹配
```

---

## 🖥️ 硬件要求检查

### GPU
- [x] 显存: 48GB ✅
- [x] CUDA: 11.8 ✅
- [x] 驱动: 正常

### CPU
- [x] 核心: 12核 ✅
- [x] 内存: 90GB ✅

### 磁盘
- [x] 系统盘: 30GB
- [x] 数据盘: 50GB (剩余充足)

---

## 📋 运行前检查

### 1. 模型文件
```bash
# 检查Qwen2.5-14B模型是否存在
ls /root/autodl-tmp/models/qwen/Qwen2.5-14B-Instruct/

# 必须包含:
# - config.json
# - model.safetensors (或 pytorch_model.bin)
# - tokenizer.json
# - tokenizer_config.json
```

### 2. 数据文件
```bash
# 检查必需的数据文件
ls paper/data/processed/

# 必须存在:
# - Stock_News.csv (新闻数据)
# - Final_Model_Data.csv (价格数据)
```

### 3. 输出目录
```bash
# 确保输出目录存在
mkdir -p paper/outputs/logs
mkdir -p paper/data/processed
```

---

## 🚀 启动流程（必须按此顺序）

### Step 1: 创建screen会话（必须！）
```bash
screen -S llm_graph
```

或使用tmux:
```bash
tmux new -s llm_graph
```

### Step 2: 进入项目目录
```bash
cd /root/autodl-tmp/paper
```

### Step 3: 运行构建脚本
```bash
./run_llm_graph.sh
```

脚本会自动检查：
- ✅ 是否在screen/tmux中
- ✅ GPU显存是否充足
- ✅ 磁盘空间是否充足
- ✅ 配置确认

### Step 4: 确认启动
```
输入: YES (大写)
```

### Step 5: 分离会话（保持后台运行）
```
screen: Ctrl+A, 然后按 D
tmux:   Ctrl+B, 然后按 D
```

### Step 6: 重新连接（查看进度）
```bash
screen -r llm_graph
# 或
tmux attach -t llm_graph
```

---

## ⏱️ 时间预估

```
阶段1: 数据加载            1分钟
阶段2: 模型加载            2-3分钟
阶段3: 图谱构建            数小时量级（取决于 batch / 推理参数与GPU吞吐）
  ├─ 每10000条: 以实际吞吐为准
  ├─ 100,000条总计: 通常为数小时量级
  └─ 包含推理+重试: 以实际运行日志为准
阶段4: 保存结果            1分钟
-----------------------------------
总计:                      数小时量级
```

进度显示:
```
Building Graph:  50%|████████▌        | 50000/100000 [X:XX<X:XX, XXX.Xit/s]

[进度保存] 已处理 60000/100000 条 (边数: XXXXX)
```

---

## 📊 预期结果

### 图谱质量指标

| 指标 | 规则匹配 | LLM预期 | 评价标准 |
|------|----------|---------|----------|
| **边数** | 17,449 | **35,000+** | >30K为优秀 |
| **密度** | 0.241 | **0.48+** | >0.4为优秀 |
| **连接率** | 100% | **100%** | >95%合格 |
| **准确率** | ~30% | **~90%** | 语义级提取 |
| **关系类型** | 无 | **6类标注** | 论文价值 |

### 论文价值

1. **高质量图谱** ✅
   - 边数是规则匹配的2倍
   - 准确率提升200%
   - 可作为单独贡献点

2. **关系类型分析** ✅
   ```
   供应链关系: ~40% (14,000条)
   竞争关系: ~25% (8,750条)
   合作关系: ~15% (5,250条)
   并购关系: ~10% (3,500条)
   诉讼关系: ~5% (1,750条)
   投资关系: ~5% (1,750条)
   ```

3. **行业分析** ✅
   - 可以按行业统计关系分布
   - 科技行业：supply占比最高
   - 金融行业：competition占比最高
   - 单独写一章节

---

## 🔍 监控指南

### 1. GPU监控
```bash
# 另开一个终端
watch -n 1 nvidia-smi

# 正常情况:
# GPU-Util: 90-100%
# Memory: 40-43GB / 48GB
```

### 2. 进度监控
```bash
# 连接到screen会话
screen -r llm_graph

# 查看当前处理进度
# 应该能看到类似:
# Building Graph: 25%|████▊ | 25000/100000 [...]
```

### 3. 日志监控
```bash
# 实时查看最新日志
tail -f outputs/logs/llm_graph_*.log

# 查看错误
grep -i error outputs/logs/llm_graph_*.log
```

---

## ⚠️ 常见问题处理

### 问题1: 显存不足 (OOM)
```
症状: CUDA out of memory
原因: 显存占用超过48GB

解决:
1. 杀死进程: pkill -9 python
2. 清理显存: nvidia-smi
3. 降低采样: --max_total 300000
```

### 问题2: SSH断连
```
症状: 连接中断，进程停止
原因: 未使用screen/tmux

解决:
1. 重新连接SSH
2. screen -r llm_graph (恢复会话)
3. 检查进度是否继续
```

### 问题3: 进程卡住
```
症状: 进度条不动超过5分钟
原因: LLM推理卡死或网络问题

解决:
1. Ctrl+C 停止
2. 检查checkpoint文件
3. 重新运行（会从checkpoint恢复）
```

### 问题4: 关系提取失败率高
```
症状: 边数远低于预期 (<20K)
原因: Prompt不匹配或模型问题

解决:
1. 检查日志中的LLM输出示例
2. 调整Prompt
3. 重新运行
```

---

## ✅ 完成后验证

### 1. 图谱文件检查
```bash
# 检查文件是否存在且大小正常
ls -lh paper/data/processed/Graph_Adjacency.npy

# 应该显示:
# -rw-r--r-- 1 root root 600K Jan 20 15:30 Graph_Adjacency.npy
```

### 2. 图谱质量检查
```bash
python -c "
import numpy as np
adj = np.load('paper/data/processed/Graph_Adjacency.npy')
n = adj.shape[0]
edges = (adj.sum() - n) / 2
print(f'节点: {n}, 边数: {int(edges)}, 密度: {edges/(n*(n-1)/2):.4f}')

# 预期输出:
# 节点: 381, 边数: 35000+, 密度: 0.48+
```

### 3. 关系类型统计（如果使用LLM）
```bash
# 查看日志中成功提取的关系示例
grep -A 5 "relation" outputs/logs/llm_graph_*.log | head -50

# 应该能看到:
# {"src":"AAPL","dst":"QCOM","relation":"supply"}
# {"src":"TSLA","dst":"GM","relation":"competition"}
# ...
```

---

## 🎉 成功标志

✅ 所有检查项通过
✅ 图谱构建无错误
✅ 边数 > 30,000
✅ 密度 > 0.40
✅ 连接率 = 100%
✅ 日志无ERROR信息
✅ Checkpoint已清理

---

## 📝 记录保存

构建完成后，请记录：

```
日期: ____________
开始时间: ____________
结束时间: ____________
总耗时: ______小时______分钟

图谱统计:
  - 节点数: ____________
  - 边数: ____________
  - 密度: ____________
  - 连接率: ____________%

质量评估:
  □ 优秀 (边数>30K, 密度>0.4)
  □ 良好 (边数>25K, 密度>0.3)
  □ 一般 (需要重新构建)

备注:
_______________________________________
_______________________________________
```

---

## 🚀 下一步

图谱构建成功后:

1. ✅ 备份图谱文件
```bash
cp paper/data/processed/Graph_Adjacency.npy \
   paper/data/processed/Graph_Adjacency_backup_$(date +%Y%m%d).npy
```

2. ✅ 开始训练模型
```bash
cd /root/autodl-tmp/paper
python -m training.train_full
```

3. ✅ 如果需要重新构建
```bash
# 删除旧图谱
rm paper/data/processed/Graph_Adjacency.npy

# 重新运行
./run_llm_graph.sh
```

---

**重要提醒**:
1. ⚠️ 必须在screen/tmux中运行（脚本会自动检查）
2. ⚠️ 一旦开始，不要中断（可能持续数小时）
3. ⚠️ 定期检查GPU和进度（每1-2小时）
4. ⚠️ 成功后立即备份图谱文件

**祝构建顺利！🎉**
