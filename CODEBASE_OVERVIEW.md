# 项目代码概览

本仓库主线代码与文档均位于 `paper/`，已统一为论文实验工程布局。
完整运行说明请优先阅读 `paper/README.md`。

## 目录结构（主线）
- `paper/`：论文实验工程主目录
  - `paper/dataProcessed/`：数据处理与特征工程（ETL、对齐、建图、Dataset）
  - `paper/models/`：模型定义（Graph-RWKV + GNN）
  - `paper/training/`：训练与采样器
  - `paper/evaluation/`：评估与回测
  - `paper/docs/`：文档说明与流程细节
- `new.md`：论文文本草稿/最终版（你的研究文档）

## 核心入口脚本
- `paper/1_preprocess_data.py`：数据预处理入口（ETL + 对齐，可选过滤）
- `paper/2_build_graph.py`：构建动态图谱
- `paper/3_train.py`：全量模型训练
- `paper/3_train_ablation.py`：消融实验训练
- `paper/4_evaluate.py`：评估与回测
- `paper/check_environment.py`：环境检测
- `paper/run_all.py`：一键执行全流程

## 运行入口
请直接参考 `paper/README.md` 中的“快速开始”和命令示例。
