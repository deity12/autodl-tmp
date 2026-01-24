项目汇总 - 快速阅读指南

概述
- 本仓库实现了基于图神经网络的股票/新闻联合建模与训练流水线（数据处理 -> 构图 -> 训练 -> 评估）。主要代码分散在 `paper/`（流程脚本）、`dataProcessed/`（ETL 与构图）、`models/`（模型实现）、`training/`（训练流程）、`evaluation/`（评估脚本）和 `data/`（数据）。

快速开始（建议顺序）
1. 阅读根级快速说明：[README.md](README.md) 与 [docs/快速开始.md](docs/%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B.md)。
2. 确认依赖并配置环境：参见 [paper/check_environment.py](paper/check_environment.py) 与 [paper/requirements.txt](paper/requirements.txt)。
3. 复现全流程（示例）：运行 `paper/run_all.py`（会按序调用 `1_preprocess_data.py`、`2_build_graph.py`、`3_train.py`、`4_evaluate.py`）。

主要目录与关键文件（一览）
- 根
  - `README.md`：总体说明与快速开始。
  - `docs/`：详细文档（模块说明、数据流程、训练/评估说明等）。优先阅读 [docs/模块说明.md](docs/%E6%A8%A1%E5%9D%97%E8%AF%B4%E6%98%8E.md) 与 [docs/数据处理流程.md](docs/%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86%E6%B5%81%E7%A8%8B.md)。
- paper/
  - `1_preprocess_data.py`：预处理入口，调用 ETL 与对齐逻辑。
  - `2_build_graph.py`：基于预处理结果构建图邻接与特征。
  - `3_train.py` / `3_train_ablation.py`：训练主脚本与消融实验脚本。
  - `4_evaluate.py`：评估脚本。
  - `run_all.py`：按步骤串联的复现脚本（推荐初学者使用）。
  - `requirements.txt`：本流程依赖（建议合并至仓库根）。
- dataProcessed/（建议重命名为 `data_processed/`）
  - `etl.py`：数据清洗、对齐、基本特征生成。
  - `build_graph.py`：生成图结构（静态/语义/关系等邻接）。
  - `dataset.py`：Dataset / DataLoader 封装。
  - `feature_engineering.py`：特征扩展与处理逻辑。
- data/
  - `raw/`：原始数据（含大量历史 CSV，应避免纳入版本库）。
  - `processed/`：处理后用于训练的 CSV/npz 等（小文件可跟踪，大文件建议外部存储）。
- models/
  - `base_model.py`：公共模型基类（训练/评估接口）。
  - `gnn_model.py`：图神经网络实现。
- training/
  - `train_full.py`、`train_ablation.py`、`date_batch_sampler.py`：训练循环与数据采样器。
- evaluation/
  - `evaluate_all.py`：批量评估流程与指标计算。
- utils/
  - `logging_utils.py`：日志工具等。
- outputs/
  - 用于保存训练检查点、日志和图像；应加入 `.gitignore`。

数据说明（要点）
- `data/raw/.../full_history/` 包含大量按股票划分的历史 CSV，体积大，不推荐提交到 Git。
- `data/processed/Final_Model_Data.csv` 是汇总过的训练数据表（用于训练/评估）。
- 建议：提供 `scripts/download_data.py` 或在 `paper/` 中写清楚数据准备步骤，方便新环境下自动下载/生成数据。

执行流程（简要命令）
- 安装依赖（示例）：

```bash
pip install -r paper/requirements.txt
```

- 运行全流程：

```bash
python paper/run_all.py
```

（或按步骤运行 `python paper/1_preprocess_data.py` → `python paper/2_build_graph.py` → `python paper/3_train.py` → `python paper/4_evaluate.py`）

常见查阅路径（按需求）
- 想看数据处理细节：`paper/1_preprocess_data.py` → `dataProcessed/etl.py`、`dataProcessed/build_graph.py`。
- 想看模型实现：`models/gnn_model.py` → `models/base_model.py`。
- 想修改训练流程：`training/train_full.py` 与 `paper/3_train.py`。
- 想运行评估：`evaluation/evaluate_all.py` 或 `paper/4_evaluate.py`。

改进建议（摘要，按优先级）
- 高优先级：
  - 添加根级 `requirements.txt` 或 `pyproject.toml`，并在 `README.md` 提示安装步骤。
  - 新增根级 `.gitignore`（忽略 `outputs/`、`__pycache__/`、大数据路径如 `data/raw/**/full_history/`）。
  - 将原始大数据移出仓库或提供下载脚本。
  - 忽略训练生成的 checkpoint/logs/figures 文件夹。
- 中优先级：
  - 统一目录命名（`dataProcessed` → `data_processed`）并修正导入路径。
  - 把 `paper/` 中的按序脚本模块化为可测试的函数，并提供 `run_all.py` 作为流程入口（已存在，可进一步改进）。
  - 引入 `black`/`ruff`/`isort` 与 `pre-commit`。
- 低优先级：
  - 增加 `tests/` 并在 CI（GitHub Actions）中运行 lint 和测试。

下一步建议（可选动作）
- 我可以直接为你：
  1) 在仓库根创建推荐的 `.gitignore`；
  2) 生成根级 `requirements.txt`（合并 `paper/requirements.txt`）；
  3) 列出并替换所有对 `dataProcessed` 的导入调用以评估重命名影响；
  4) 将 `paper/` 的脚本改为模块化入口样例 `run_pipeline.py`。

请选择要我执行的下一步编号（1/2/3/4），或告诉我改动偏好。
