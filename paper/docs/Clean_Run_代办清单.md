# Clean Run 实验方案 — 代办清单（待你确认后执行）

基于你与 Gemini 讨论的 **4:1:1 抗周期切分 + 统一脚本** 方案，下面是具体代办与设计说明。**请先审阅，确认无误后再让我执行。**

---

## 一、数据切分与图谱（必须先做）

### 1.1 新切分约定（Gold Standard Split）

| 集合 | 时间范围 | 说明 |
|------|----------|------|
| **Train** | 2018-01-01 ~ 2021-12-31 | 4 年，含 2018 震荡、2019 慢牛、2020 熔断反转、2021 牛市 |
| **Val**   | 2022-01-01 ~ 2022-12-31 | 1 年，2022 熊市，用于早停与模型选择 |
| **Test**  | 2023-01-01 ~ 2023-12-31 | 1 年，OOD 测试，论文报告区间 |

图谱截断日 **split_date = 2021-12-31**（与训练集结束日一致，防泄露）。

### 1.2 需要改动的代码位置

| 文件 | 当前值 | 目标值 |
|------|--------|--------|
| `training/train_full.py` 中 `PAPER_CONFIG` | `train_end`: 2020-06-30<br>`val_start/val_end`: 2020-07-01~2020-12-31<br>`test_start/test_end`: 2021-01-01~2023-12-31<br>`graph_split_date`: 2020-06-30 | `train_end`: 2021-12-31<br>`val_start/val_end`: 2022-01-01~2022-12-31<br>`test_start/test_end`: 2023-01-01~2023-12-31<br>`graph_split_date`: 2021-12-31 |
| `2_build_graph.py` | `--split_date` 默认已是 **2021-12-31** | 无需改，只需**重新运行**建图 |

### 1.3 你需要执行的命令（按顺序）

1. **重新建图**（因为 split_date 与训练集结束日已统一为 2021-12-31，需用新切分重新生成图）  
   ```bash
   cd paper && python 2_build_graph.py --split_date 2021-12-31
   ```
2. 建图完成后，再跑训练/实验（见下文）。

---

## 二、统一实验脚本设计

### 2.1 新建 `run_experiment.py`（论文 v1 专用）

**目标**：一个脚本、一套配置、四种实验，输出到 `outputs/final_run/<exp_name>/`，并生成汇总 CSV。

**设计要点**：

- **统一配置**（FINAL_CONFIG）  
  - 日期：train 2018-01-01~2021-12-31，val 2022 全年，test 2023 全年。  
  - 超参：与当前 SOTA 一致（如 `seq_len=60`, `n_embd=64`, `n_layers=2`, `lr=1e-3`, `dropout=0.3`, `rank_loss_weight=1.0`, `epochs=50`, `early_stop=10`）。  
  - 特征：沿用现有 `feature_columns.json`（不写死 26 个，避免与当前数据不一致）。

- **四种实验**（与 Gemini 方案一致）  
  1. `rwkv_with_graph` — 我们的模型（SOTA）  
  2. `rwkv_no_graph` — 消融：无图  
  3. `lstm_with_graph` — 对比基线：LSTM + 图  
  4. `gru_with_graph` — 对比基线：GRU + 图  

- **实现方式**  
  - 在 `run_experiment.py` 里用 `argparse`：`--model {rwkv,lstm,gru}`，`--use_graph`。  
  - 内部直接 override `train_full.CONFIG` 的日期、模型类型、是否用图等，然后调用 `train_full.main()`，避免再维护两套训练逻辑。  
  - 每个实验的 `experiment_name` 设为上述 4 个名字之一，`output_dir` 设为 `outputs/final_run`，checkpoint/artifacts 写在 `outputs/final_run/<exp_name>/` 下。

- **结果汇总**  
  - 每个实验跑完后，从该次运行的 result JSON（或 train_full 里已有的 summary）中读取 **Val RankIC**、**Test RankIC**（若有）。  
  - 四个实验都跑完后，写一张表到 `outputs/final_run/summary.csv`（列：experiment_name, Val_RankIC, Test_RankIC, 可选其他指标），方便直接贴论文 Table 1。

### 2.2 不合并 `3_train.py` 与 `3_train_ablation.py`

- 保留 `3_train.py` / `3_train_ablation.py` 用于旧版或额外消融。  
- **论文 v1 封板** 只依赖：**新日期配置 + `run_experiment.py` + 下面的一键脚本**。

---

## 三、执行队列（一键跑完 4 个实验）

### 3.1 新建 `run_all_final.sh`

```bash
#!/bin/bash
# 论文 v1 — Clean Run：4:1:1 切分 + 4 个实验（后台可挂 nohup）
set -e
cd "$(dirname "$0")"

echo "========== 1. Graph-RWKV (Ours, SOTA) =========="
python run_experiment.py --model rwkv --use_graph

echo "========== 2. RWKV w/o Graph (Ablation) =========="
python run_experiment.py --model rwkv

echo "========== 3. Graph-LSTM (Comparison) =========="
python run_experiment.py --model lstm --use_graph

echo "========== 4. Graph-GRU (Comparison) =========="
python run_experiment.py --model gru --use_graph

echo "========== 完成：结果见 outputs/final_run/summary.csv =========="
```

- 你可按需在前面加 `nohup ... &` 或 `tmux`/`screen` 在后台跑。

### 3.2 执行顺序建议

1. 改好 `train_full.py` 的日期与 `graph_split_date`。  
2. 运行 `2_build_graph.py --split_date 2021-12-31`，生成新图。  
3. 创建并运行 `run_experiment.py`（单次测试可先跑一个，例如 `python run_experiment.py --model rwkv --use_graph`）。  
4. 确认无报错后，用 `run_all_final.sh` 一次性跑完 4 个实验。  
5. 检查 `outputs/final_run/summary.csv` 和 `outputs/final_run/<exp_name>/` 下的曲线与 checkpoint。

---

## 四、预期产物（论文 v1）

- `outputs/final_run/rwkv_with_graph/` — 最佳模型、训练曲线、artifacts。  
- `outputs/final_run/rwkv_no_graph/`  
- `outputs/final_run/lstm_with_graph/`  
- `outputs/final_run/gru_with_graph/`  
- `outputs/final_run/summary.csv` — 汇总表（Val RankIC, Test RankIC 等），可直接用于论文 Table 1。

---

## 五、代办小结（你确认后我执行的顺序）

| 序号 | 内容 | 说明 |
|------|------|------|
| 1 | 修改 `training/train_full.py` | 将 `train_end`/`val_*`/`test_*`/`graph_split_date` 改为上述 4:1:1 与 2021-12-31 |
| 2 | 新建 `run_experiment.py` | 统一 FINAL_CONFIG、--model/--use_graph、调用 train_full、写 summary.csv |
| 3 | 新建 `run_all_final.sh` | 依次跑 4 个实验，可后台运行 |
| 4 | （你本机执行）重新建图 | `python 2_build_graph.py --split_date 2021-12-31` |
| 5 | （你本机执行）跑实验 | 先单跑一个验证，再 `bash run_all_final.sh` |

如果你同意以上设计和代办顺序，回复「可以执行」或指出要改的地方，我按该顺序从第 1 项开始改代码（第 4、5 步留给你本机执行）。
