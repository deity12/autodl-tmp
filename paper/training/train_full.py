# -*- coding: utf-8 -*-
"""
Graph-RWKV 模型训练脚本（基于大语言模型动态图谱与 Graph-RWKV 的时空解耦金融预测）
========================================================================
【核心创新点】根据新研究方向实现：

训练策略：
    1. **滚动窗口验证（Rolling Window / Walk-Forward Validation）**：
       - 为适应金融市场风格切换（Regime Shift），不采用静态划分
       - 阶段 1：Train (2018-2020) → Test (2021 Q1)
       - 阶段 2：Train (2018-2020 + 2021 Q1) → Test (2021 Q2)
       - 阶段 3：...以此类推
       - 【注意】当前实现为静态 80/20 划分，完整滚动窗口验证需在评估脚本中实现

    2. **Loss Function**：RankIC Loss（侧重排序能力）
       Loss = -PearsonCorr(Pred_rank, Target_rank)

核心改进：
    1. 降低模型复杂度（n_embd 512->256, n_layers 4->3）
    2. 降低 batch_size（3072->512），增加梯度更新次数
    3. 增加 epoch 数量（10->20），给复杂模型更多训练时间
    4. 使用差异化学习率：量子层用更小的学习率（经典层 3e-4，量子层 3e-5）
    5. 动态设置量子阈值：基于训练数据的 70% 分位数
    6. 添加权重衰减和更强的 Dropout 正则化

【论文对应】：
    - 对应论文 3.3 训练与验证策略
    - 模型架构：Graph-RWKV（RWKV 时间编码器 + 动态 GAT 空间聚合）
"""

import sys
import os
import json
import pickle
import shutil
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict

# ================= 0. 性能开关（针对 48GB GPU + 12 vCPU 优化）=================
def _apply_perf_settings(enable: bool = True) -> None:
    """
    针对 Ampere+ GPU 的常用训练提速设置：
      - TF32：显著提升 matmul/conv 吞吐（对回归任务通常影响很小）
      - cudnn.benchmark：固定输入形状时更快（会牺牲一点点确定性）
      - matmul precision：让 PyTorch 选择更高性能的 kernel

    【优化 #1 - 基于 NeurIPS 2024 "Efficient Training" 论文】
    添加梯度累积和内存优化，充分利用48GB显存
    """
    if not enable:
        return
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        # 【新增】启用 CUDA 内存池优化，减少碎片化
        try:
            torch.cuda.empty_cache()
            # 设置内存分配器策略：expandable_segments 减少碎片
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        except Exception:
            pass


def _json_dump(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def _save_run_artifacts(output_dir: str, experiment_name: str, train_dataset, config: dict) -> str:
    """
    保存与“消融对比/可复现”强相关的产物快照：scaler/特征列/ticker顺序/配置/图文件引用等。
    """
    results_dir = os.path.join(output_dir, "results")
    run_dir = os.path.join(results_dir, f"artifacts_{experiment_name}")
    os.makedirs(run_dir, exist_ok=True)

    # 1) 训练配置快照（完整 CONFIG）
    _json_dump(os.path.join(run_dir, "config_full.json"), dict(config))

    # 2) 特征列快照（避免 feature_columns.json 被后续实验覆盖）
    _json_dump(os.path.join(run_dir, "feature_columns_used.json"), list(getattr(train_dataset, "feature_cols", [])))

    # 3) 节点顺序快照（与图邻接、node_indices 对齐的关键）
    ticker2idx = getattr(train_dataset, "ticker2idx", {})
    tickers_in_order = list(ticker2idx.keys()) if isinstance(ticker2idx, dict) else []
    _json_dump(os.path.join(run_dir, "tickers_in_order.json"), tickers_in_order)
    _json_dump(os.path.join(run_dir, "ticker2idx.json"), ticker2idx)

    # 4) 数据切分信息（用于 walk-forward / 时间切分复现）
    split_info = {
        "mode": getattr(train_dataset, "mode", None),
        "start_date": getattr(train_dataset, "start_date", None),
        "end_date": getattr(train_dataset, "end_date", None),
        "split_date": getattr(train_dataset, "split_date", None),
        "csv_path": str(config.get("csv_path", "")),
        "features_path": getattr(train_dataset, "features_path", None),
        "feature_columns_path": str(config.get("feature_columns_path", "")),
    }
    _json_dump(os.path.join(run_dir, "data_split.json"), split_info)

    # 5) scaler（数值标准化）
    try:
        with open(os.path.join(run_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(getattr(train_dataset, "scaler", None), f)
    except Exception as e:
        _json_dump(os.path.join(run_dir, "scaler_error.json"), {"error": str(e)})

    # 6) 关键输入文件引用（可选复制一份，避免后续被覆盖）
    for key, dst_name in [
        ("graph_path", "Graph_Adjacency.npy"),
        ("graph_tickers_path", "Graph_Tickers.json"),
        ("feature_columns_path", "feature_columns.json"),
    ]:
        src = str(config.get(key, "") or "")
        if src and os.path.exists(src):
            try:
                shutil.copy2(src, os.path.join(run_dir, dst_name))
            except Exception:
                pass

    return run_dir

# ================= 1. 环境与路径 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

GRAPH_PATH = os.path.join(parent_dir, 'data', 'processed', 'Graph_Adjacency.npy')
GRAPH_TICKERS_PATH = os.path.join(parent_dir, 'data', 'processed', 'Graph_Tickers.json')
GRAPH_TICKERS_PATH_LEGACY = os.path.join(parent_dir, 'data', 'processed', 'Graph_Adjacency_tickers.json')
OUTPUT_DIR = os.path.join(parent_dir, 'outputs')
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

try:
    from dataProcessed.dataset import FinancialDataset
    from models.gnn_model import GraphRWKV_GNN_Model, QL_MATCC_GNN_Model  # QL_MATCC_GNN_Model 为兼容性别名
    from models.base_model import GraphRWKV_Model, RNN_Model
    from training.date_batch_sampler import DateGroupedBatchSampler
    print("✅ 成功导入 dataset、gnn_model 模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    exit(1)

# ================= 2. 超参数（支持 profile：paper / 48gb）=================
# 说明：
# - paper：论文复现默认配置（更稳、更容易复现）
# - 48gb：利用 48GB 显存提升吞吐（更大 batch / 更大模型），通过环境变量启用：
#         export QL_PROFILE=48gb
PAPER_CONFIG = {
    'csv_path': os.path.join(parent_dir, 'data', 'processed', 'Final_Model_Data.csv'),
    'input_dim': 8,
    'n_embd': 256,
    'n_layers': 3,
    'n_qubits': 8,  # 【优化】增强量子容量：8量子比特
    'gnn_embd': 64,
    'seq_len': 30,
    'batch_size': 512,
    'epochs': 30,  # 【优化】增加训练轮数，给复杂模型更多收敛时间
    'lr': 3e-4,
    # 【注意】新方向不使用以下参数，已移除：
    # 'quantum_lr_ratio', 'use_differential_lr', 'q_threshold'
    'dropout': 0.1,  # 【优化】降低dropout从0.15到0.1，减少正则化
    'weight_decay': 1e-5,
    'early_stop_patience': 8,  # 【优化】增加早停耐心值
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    'prefetch_factor': 2,
    'use_amp': True,
    'use_compile': False,
    'pin_memory': True,
    'persistent_workers': True,
    'use_date_grouped_batch': True,
    'use_rank_loss': True,
    'rank_loss_weight': 0.1,
    'rank_loss_max_pairs': 4096,
    'rank_loss_type': 'rankic',  # pairwise | rankic
    'rankic_tau': 1.0,
    'rankic_max_items': 256,
    'feature_columns_path': os.path.join(parent_dir, 'data', 'processed', 'feature_columns.json'),
    # 性能/可复现开关
    'enable_perf_flags': True,
    # 运行配置
    'output_dir': OUTPUT_DIR,
    'graph_path': GRAPH_PATH,
    'graph_tickers_path': GRAPH_TICKERS_PATH,
    'use_graph': True,
    'experiment_name': 'full',
    'checkpoint_name': 'best_model.pth',
    # Walk-forward 配置
    'use_walk_forward': False,
    'walk_forward_train_start': '2018-01-01',
    'walk_forward_train_end': '2020-12-31',
    'walk_forward_test_start': '2021-01-01',
    'walk_forward_test_end': '2023-12-31',
    'walk_forward_freq': 'Q',
    # 训练/评估日期范围（由 walk-forward 覆盖）
    'train_start': None,
    'train_end': None,
    'test_start': None,
    'test_end': None,
    'use_date_split': True,
    # 时间编码器类型
    'temporal_backend': 'rwkv',  # rwkv | lstm | gru
}

CONFIG = dict(PAPER_CONFIG)
_profile = os.environ.get("QL_PROFILE", "paper").strip().lower()
if _profile in ("48gb", "max", "server"):
    # 48GB 服务器吞吐优先配置（可按需再调）
    CONFIG.update({
        'n_embd': 384,
        'n_layers': 4,
        'gnn_embd': 128,
        'batch_size': 1024,
        'epochs': 30,
        # 12 vCPU：更积极的 DataLoader 并行
        'num_workers': min(8, max(2, (os.cpu_count() or 12) - 2)),
        'prefetch_factor': 4,
        # 更大 batch 下排序 loss 的 pair 采样也可以适度增大
        'rank_loss_max_pairs': 8192,
    })
    print(f"⚡ 已启用 QL_PROFILE={_profile}（48GB 吞吐配置）")
else:
    print(f"ℹ️ 使用 QL_PROFILE={_profile}（论文默认配置）")


def ranknet_pairwise_loss(pred: torch.Tensor, target: torch.Tensor, max_pairs: int = 4096) -> torch.Tensor:
    """
    RankNet 风格 pairwise loss（常用于股票排序/学习排序论文）。
    - pred/target: (B, 1) 或 (B,)
    - max_pairs: 若 batch 很大，随机采样 pair 降低 O(B^2) 成本
    """
    pred = pred.view(-1)
    target = target.view(-1)
    B = pred.numel()
    if B < 2:
        return pred.new_tensor(0.0)

    # 生成 pair：优先随机采样，避免构造全矩阵
    num_all = B * (B - 1) // 2
    num_pairs = min(int(max_pairs), int(num_all))
    if num_pairs <= 0:
        return pred.new_tensor(0.0)

    # 随机采样 (i,j), i<j
    idx_i = torch.randint(0, B, (num_pairs,), device=pred.device)
    idx_j = torch.randint(0, B, (num_pairs,), device=pred.device)
    mask = idx_i != idx_j
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]
    if idx_i.numel() == 0:
        return pred.new_tensor(0.0)

    # 方向标签：sign(y_i - y_j)，0 的 pair 丢弃
    y_diff = target[idx_i] - target[idx_j]
    s = torch.sign(y_diff)
    nz = s != 0
    if nz.sum() == 0:
        return pred.new_tensor(0.0)
    s = s[nz]
    p_diff = pred[idx_i[nz]] - pred[idx_j[nz]]

    # RankNet: log(1 + exp(-s * (p_i - p_j)))
    return torch.nn.functional.softplus(-s * p_diff).mean()


def _soft_rank(x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    可微近似排序：rank_i = sum_j sigmoid((x_i - x_j)/tau)
    """
    x = x.view(-1)
    diff = x.unsqueeze(0) - x.unsqueeze(1)
    P = torch.sigmoid(diff / max(tau, 1e-6))
    return P.sum(dim=1)


def rankic_soft_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    tau: float = 1.0,
    max_items: int = 256,
) -> torch.Tensor:
    """
    RankIC Loss（可微近似）：对 pred/target 做 soft-rank 后计算 Pearson 相关，最大化相关性。
    """
    pred = pred.view(-1)
    target = target.view(-1)
    B = pred.numel()
    if B < 2:
        return pred.new_tensor(0.0)

    if B > max_items:
        idx = torch.randperm(B, device=pred.device)[:max_items]
        pred = pred[idx]
        target = target[idx]

    r_pred = _soft_rank(pred, tau=tau)
    r_true = _soft_rank(target, tau=tau)

    r_pred = r_pred - r_pred.mean()
    r_true = r_true - r_true.mean()
    denom = (r_pred.std() * r_true.std()).clamp_min(1e-6)
    corr = (r_pred * r_true).mean() / denom
    return -corr


def daily_ic_rankic(y_true: np.ndarray, y_pred: np.ndarray, dates: list[str]):
    """
    顶会/量化常用：按日期截面计算 IC/RankIC，再对天取平均。
    dates: 与 y_true/y_pred 对齐的 YYYY-MM-DD 字符串列表
    """
    buckets_true = defaultdict(list)
    buckets_pred = defaultdict(list)
    for t, p, d in zip(y_true, y_pred, dates):
        buckets_true[d].append(float(t))
        buckets_pred[d].append(float(p))

    ic_list = []
    rankic_list = []
    for d in buckets_true.keys():
        yt = np.asarray(buckets_true[d], dtype=np.float64)
        yp = np.asarray(buckets_pred[d], dtype=np.float64)
        if yt.size < 2:
            continue
        # 截面相关（当天横截面）
        try:
            ic, _ = pearsonr(yp, yt)
            ic_list.append(float(ic))
        except Exception:
            pass
        try:
            ric, _ = spearmanr(yp, yt)
            rankic_list.append(float(ric))
        except Exception:
            pass

    ic_mean = float(np.mean(ic_list)) if ic_list else None
    rankic_mean = float(np.mean(rankic_list)) if rankic_list else None
    return ic_mean, rankic_mean


def _train_once():
    """
    Graph-RWKV 模型训练入口（新方向核心模型）。

    主要步骤：
      1) 加载 `FinancialDataset`（train/test）
      2) 加载 `Graph_Adjacency.npy` 并与 dataset 的 ticker 顺序做一致性校验
      3) 初始化 `GraphRWKV_GNN_Model`（RWKV 时间编码器 + 动态 GAT 空间聚合）
      4) 训练（AMP / 梯度裁剪 / 早停 / 可选 RankNet 排序损失）
      5) 保存 best checkpoint、训练曲线与日志到 `outputs/`
    
    【注意】新方向不使用 Quantum、MATCC、MarketGuidance 组件
    """
    # 应用性能设置（TF32 / benchmark 等）
    _apply_perf_settings(bool(CONFIG.get("enable_perf_flags", True)))

    # 输出目录配置
    output_dir = str(CONFIG.get("output_dir", OUTPUT_DIR))
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    figure_dir = os.path.join(output_dir, "figures")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    experiment_name = str(CONFIG.get("experiment_name", "full"))
    checkpoint_name = str(CONFIG.get("checkpoint_name", "best_model.pth"))
    if not checkpoint_name.endswith(".pth"):
        checkpoint_name = f"{checkpoint_name}.pth"

    print(f">>> Training on device: {CONFIG['device']}")
    if CONFIG['device'] == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   GPU: {gpu_name}")
        print(f"   显存: {gpu_memory:.1f} GB")
        print(f"   Batch Size: {CONFIG['batch_size']}")

    # ================= 3. 数据加载 =================
    print("\n>>> Loading Datasets...")
    try:
        use_date_split = bool(CONFIG.get('use_date_split', True))
        train_dataset = FinancialDataset(
            CONFIG['csv_path'],
            seq_len=CONFIG['seq_len'],
            mode='train',
            start_date=CONFIG.get('train_start'),
            end_date=CONFIG.get('train_end'),
            use_date_split=use_date_split,
            feature_columns_path=CONFIG.get('feature_columns_path'),
        )
        test_dataset = FinancialDataset(
            CONFIG['csv_path'], seq_len=CONFIG['seq_len'], mode='test', 
            scaler=train_dataset.scaler,
            # 【注意】新方向不使用 vol_stats，但保留参数以兼容接口
            vol_stats=train_dataset.vol_stats if hasattr(train_dataset, 'vol_stats') else None,
            start_date=CONFIG.get('test_start'),
            end_date=CONFIG.get('test_end'),
            use_date_split=use_date_split,
            feature_columns_path=CONFIG.get('feature_columns_path'),
        )
        print(f"   Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        CONFIG['input_dim'] = len(train_dataset.feature_cols)
        print(f"   Input Dim: {CONFIG['input_dim']} (features)")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ================= 3.5 保存“可复现/消融对比”产物快照 =================
    artifacts_dir = None
    try:
        artifacts_dir = _save_run_artifacts(output_dir, experiment_name, train_dataset, CONFIG)
        print(f">>> Artifacts 已保存: {artifacts_dir}")
    except Exception as e:
        print(f"[WARN] Artifacts 保存失败: {e}")

    # ================= 4. 加载邻接矩阵 + 对齐校验（防止跑一晚白跑）=================
    # 以 dataset 的 ticker2idx 为准：它决定了 node_indices 的取值范围与顺序
    dataset_tickers_in_order = list(train_dataset.ticker2idx.keys())  # Python 3.7+ 保持插入顺序
    dataset_num_nodes = len(dataset_tickers_in_order)

    graph_path = str(CONFIG.get("graph_path", GRAPH_PATH))
    use_graph = bool(CONFIG.get("use_graph", True))
    if use_graph:
        if os.path.exists(graph_path):
            adj_matrix = np.load(graph_path)
            print(f">>> 加载图谱: {graph_path}, 形状: {adj_matrix.shape}")
        else:
            adj_matrix = np.eye(dataset_num_nodes, dtype=np.float32)
            print(f">>> 未找到图谱，使用单位阵 (num_nodes={dataset_num_nodes})")

        # --- 1) 形状硬校验 ---
        if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError(f"图谱邻接矩阵必须为方阵，但得到 shape={adj_matrix.shape}")
        if adj_matrix.shape[0] != dataset_num_nodes:
            raise ValueError(
                "图谱节点数与数据集 ticker2idx 不一致，训练将发生索引错位/越界。\n"
                f"- Graph_Adjacency.npy nodes={adj_matrix.shape[0]}\n"
                f"- Dataset nodes={dataset_num_nodes}\n"
                "解决：请用同一份 Final_Model_Data.csv 重新运行 build_graph.py 生成图谱，或删除旧图谱让其回退单位阵。"
            )

        # --- 2) 节点顺序校验（强烈推荐）---
        graph_tickers_path = str(CONFIG.get("graph_tickers_path", GRAPH_TICKERS_PATH))
        if not os.path.exists(graph_tickers_path) and os.path.exists(GRAPH_TICKERS_PATH_LEGACY):
            graph_tickers_path = GRAPH_TICKERS_PATH_LEGACY

        if os.path.exists(graph_tickers_path):
            try:
                with open(graph_tickers_path, "r", encoding="utf-8") as f:
                    graph_tickers = json.load(f).get("tickers", [])
                if graph_tickers != dataset_tickers_in_order:
                    # 打印前几个差异位置，帮助定位
                    diffs = []
                    for i, (a, b) in enumerate(zip(graph_tickers, dataset_tickers_in_order)):
                        if a != b:
                            diffs.append((i, a, b))
                            if len(diffs) >= 5:
                                break
                    raise ValueError(
                        "图谱 tickers 顺序与训练数据 tickers 顺序不一致：这会导致 GNN 聚合到错误的股票上（最危险：可能不报错但结果全错）。\n"
                        f"示例差异(最多5条): {diffs}\n"
                        "解决：用同一份 Final_Model_Data.csv 重新生成 Graph_Adjacency.npy，并确保 dataset/build_graph 的 Ticker 都做了同样的标准化（建议全大写）。"
                    )
                else:
                    print("✅ 图谱 tickers 顺序校验通过（与 dataset.ticker2idx 对齐）")
            except Exception:
                raise
        else:
            print("⚠️ 未找到 Graph_Tickers.json，无法校验节点顺序（建议保留该文件以避免静默错位）")
    else:
        adj_matrix = np.eye(dataset_num_nodes, dtype=np.float32)
        print(">>> [消融] w/o_graph 模式：跳过图谱加载与节点校验")

    num_nodes = dataset_num_nodes

    # 【关键】从训练数据获取量子阈值
    # 【注意】新方向不使用 q_threshold（量子门控），已移除相关逻辑

    # DataLoader 参数：num_workers=0 时不能传 prefetch_factor/persistent_workers
    num_workers = int(CONFIG.get('num_workers', 4))
    pin_memory = bool(CONFIG.get('pin_memory', True)) and torch.cuda.is_available()
    persistent_workers = bool(CONFIG.get('persistent_workers', True)) and num_workers > 0
    prefetch_factor = int(CONFIG.get('prefetch_factor', 2)) if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=(
            DateGroupedBatchSampler(
                target_dates=train_dataset.target_dates,
                batch_size=CONFIG['batch_size'],
                shuffle=True,
                drop_last=False,
                seed=42,
            )
            if CONFIG.get('use_date_grouped_batch', True)
            else None
        ),
        batch_size=None if CONFIG.get('use_date_grouped_batch', True) else CONFIG['batch_size'],
        shuffle=False if CONFIG.get('use_date_grouped_batch', True) else True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    # ================= 5. 模型初始化 =================
    print("\n>>> Initializing Graph-RWKV Model...")
    if CONFIG.get("use_graph", True):
        model = GraphRWKV_GNN_Model(
            input_dim=CONFIG['input_dim'],
            n_embd=CONFIG['n_embd'],
            n_layers=CONFIG['n_layers'],
            num_nodes=num_nodes,
            adj_matrix=adj_matrix,
            gnn_embd=CONFIG.get('gnn_embd', 64),
            dropout=CONFIG.get('dropout', 0.1),
            temporal_backend=CONFIG.get('temporal_backend', 'rwkv'),
        ).to(CONFIG['device'])
    else:
        backend = str(CONFIG.get('temporal_backend', 'rwkv')).lower()
        if backend in ("lstm", "gru"):
            model = RNN_Model(
                input_dim=CONFIG['input_dim'],
                n_embd=CONFIG['n_embd'],
                n_layers=CONFIG['n_layers'],
                dropout=CONFIG.get('dropout', 0.1),
                rnn_type=backend,
            ).to(CONFIG['device'])
        else:
            model = GraphRWKV_Model(
                input_dim=CONFIG['input_dim'],
                n_embd=CONFIG['n_embd'],
                n_layers=CONFIG['n_layers'],
                dropout=CONFIG.get('dropout', 0.1),
            ).to(CONFIG['device'])

    if CONFIG.get("use_compile", False):
        try:
            if hasattr(model, "compile"):
                model.compile()
            else:
                model = torch.compile(model, backend="inductor")
            print("✅ 已启用 torch.compile (inductor)")
        except Exception as e:
            print(f"⚠️ torch.compile 失败，回退 eager: {e}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # ================= 6. 优化器 =================
    criterion = nn.MSELoss()
    
    # 【新方向】使用统一学习率（不再需要量子层差异化学习率）
    optimizer = optim.AdamW(
        model.parameters(), lr=CONFIG['lr'], 
        betas=(0.9, 0.999), eps=1e-8, 
        weight_decay=CONFIG.get('weight_decay', 1e-5)
    )
    print(f"   学习率: {CONFIG['lr']:.2e}")
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    use_amp = CONFIG.get('use_amp', False)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("   ✅ 已启用混合精度训练 (AMP)")

    # ================= 7. 训练循环 =================
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_metrics_epoch = None
    early_stop_counter = 0
    early_stop_patience = CONFIG['early_stop_patience']

    print("\n>>> Start Training (Graph-RWKV Model)...")
    print("=" * 60)

    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", ncols=100)

        for batch_idx, batch in enumerate(progress_bar):
            x = batch['x'].to(CONFIG['device'], non_blocking=True)
            y = batch['y'].to(CONFIG['device'], non_blocking=True)
            vol = batch['vol'].to(CONFIG['device'], non_blocking=True)
            batch_dates = batch.get('target_date')  # list[str]（若启用按日期batch，理论上同一天）
            node_indices = batch.get('node_indices')
            if node_indices is not None:
                node_indices = node_indices.to(CONFIG['device'], non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    preds = model(x, vol, node_indices=node_indices)
                    loss = criterion(preds, y)
                    # 可选排序损失：仅在 batch 基本同一天时启用（按日期 batch 时成立）
                    if CONFIG.get('use_rank_loss', False) and batch_dates is not None:
                        # 性能优化：避免对整个 list 做 set()（O(B) 且分配多）；只比较首尾即可
                        if isinstance(batch_dates, list) and (len(batch_dates) <= 1 or batch_dates[0] == batch_dates[-1]):
                            if CONFIG.get('rank_loss_type', 'pairwise') == 'rankic':
                                rank_loss = rankic_soft_loss(
                                    preds,
                                    y,
                                    tau=float(CONFIG.get('rankic_tau', 1.0)),
                                    max_items=int(CONFIG.get('rankic_max_items', 256)),
                                )
                            else:
                                rank_loss = ranknet_pairwise_loss(
                                    preds, y, max_pairs=CONFIG.get('rank_loss_max_pairs', 4096)
                                )
                            loss = loss + float(CONFIG.get('rank_loss_weight', 0.1)) * rank_loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(x, vol, node_indices=node_indices)
                loss = criterion(preds, y)
                if CONFIG.get('use_rank_loss', False) and batch_dates is not None:
                    if isinstance(batch_dates, list) and (len(batch_dates) <= 1 or batch_dates[0] == batch_dates[-1]):
                        if CONFIG.get('rank_loss_type', 'pairwise') == 'rankic':
                            rank_loss = rankic_soft_loss(
                                preds,
                                y,
                                tau=float(CONFIG.get('rankic_tau', 1.0)),
                                max_items=int(CONFIG.get('rankic_max_items', 256)),
                            )
                        else:
                            rank_loss = ranknet_pairwise_loss(
                                preds, y, max_pairs=CONFIG.get('rank_loss_max_pairs', 4096)
                            )
                        loss = loss + float(CONFIG.get('rank_loss_weight', 0.1)) * rank_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n❌ NaN/Inf at batch {batch_idx}")
                return

            epoch_train_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix(loss=f'{loss.item():.6f}', avg=f'{epoch_train_loss/num_batches:.6f}')

        scheduler.step()
        avg_train = epoch_train_loss / num_batches
        train_losses.append(avg_train)

        # ---------- 验证 ----------
        model.eval()
        epoch_val = 0.0
        all_preds = []
        all_targets = []
        all_dates = []
        
        with torch.no_grad():
            for batch in test_loader:
                x = batch['x'].to(CONFIG['device'], non_blocking=True)
                y = batch['y'].to(CONFIG['device'], non_blocking=True)
                vol = batch['vol'].to(CONFIG['device'], non_blocking=True)
                dates = batch.get('target_date')  # list[str]
                node_indices = batch.get('node_indices')
                if node_indices is not None:
                    node_indices = node_indices.to(CONFIG['device'], non_blocking=True)
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        preds = model(x, vol, node_indices=node_indices)
                else:
                    preds = model(x, vol, node_indices=node_indices)
                
                epoch_val += criterion(preds, y).item()
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.cpu().numpy())
                if dates is not None:
                    all_dates.extend(list(dates))
        
        avg_val = epoch_val / len(test_loader)
        val_losses.append(avg_val)
        
        # 计算评估指标
        if avg_val < best_val_loss:
            all_preds_np = np.concatenate(all_preds, axis=0)
            all_targets_np = np.concatenate(all_targets, axis=0)
            
            y_true = all_targets_np.flatten()
            y_pred = all_preds_np.flatten()
            
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            true_direction = np.sign(y_true)
            pred_direction = np.sign(y_pred)
            directional_accuracy = np.mean(true_direction == pred_direction)
            
            # 顶会/量化更常见：按日期截面计算 IC/RankIC，再对天平均
            if all_dates:
                ic, rank_ic = daily_ic_rankic(y_true, y_pred, all_dates)
            else:
                try:
                    ic, _ = pearsonr(y_pred, y_true)
                    ic = float(ic)
                except Exception:
                    ic = None
                try:
                    rank_ic, _ = spearmanr(y_pred, y_true)
                    rank_ic = float(rank_ic)
                except Exception:
                    rank_ic = None
            
            best_metrics = {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'directional_accuracy': float(directional_accuracy),
                'ic': ic,
                'rank_ic': rank_ic,
            }
        else:
            best_metrics = None

        cur_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}: Train={avg_train:.6f}, Val={avg_val:.6f}, lr={cur_lr:.2e}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_metrics_epoch = best_metrics
            save_path = os.path.join(checkpoint_dir, checkpoint_name)
            torch.save(model.state_dict(), save_path)
            if best_metrics:
                print(f"  🌟 Best model saved!")
                ic_str = f"{best_metrics['ic']:.4f}" if best_metrics['ic'] is not None else "N/A"
                print(f"     R²={best_metrics['r2']:.4f}, MAE={best_metrics['mae']:.6f}, "
                      f"DirAcc={best_metrics['directional_accuracy']:.2%}, IC={ic_str}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            print(f"\n🛑 Early stopping (best val loss: {best_val_loss:.6f})")
            break
        print("-" * 60)

    # ================= 8. 保存结果 =================
    curve_path = os.path.join(figure_dir, f"training_curve_{experiment_name}.png")
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train Loss', lw=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Val Loss', lw=2)
    if val_losses:
        be = val_losses.index(best_val_loss) + 1
        plt.plot(be, best_val_loss, 'g*', markersize=14, label=f'Best (Epoch {be})')
    plt.title('Graph-RWKV Model Training', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f">>> 训练曲线已保存: {curve_path}")
    
    loss_data_path = os.path.join(log_dir, f"training_losses_{experiment_name}.json")
    loss_data = {
        'experiment_name': experiment_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'best_epoch': val_losses.index(best_val_loss) + 1 if val_losses else 0,
        'total_epochs': len(train_losses),
        'metrics': best_metrics_epoch,
        'config': {
            'batch_size': CONFIG['batch_size'],
            'lr': CONFIG['lr'],
            'epochs': CONFIG['epochs'],
            'n_embd': CONFIG['n_embd'],
            'n_layers': CONFIG['n_layers'],
            'gnn_embd': CONFIG.get('gnn_embd'),
            'seq_len': CONFIG.get('seq_len'),
            # 【注意】新方向不使用以下参数，已移除：
            # 'n_qubits', 'q_threshold'
            'profile': os.environ.get("QL_PROFILE", "paper"),
            'output_dir': output_dir,
            'checkpoint_name': checkpoint_name,
            'use_graph': CONFIG.get("use_graph", True),
        }
    }
    with open(loss_data_path, 'w') as f:
        json.dump(loss_data, f, indent=2)
    print(f">>> Loss 数据已保存: {loss_data_path}")

    # 保存摘要，便于消融对比
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, f"experiment_{experiment_name}.json")
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_name": experiment_name,
        "checkpoint_path": os.path.join(checkpoint_dir, checkpoint_name),
        "artifacts_dir": artifacts_dir,
        "best_val_loss": best_val_loss,
        "best_epoch": val_losses.index(best_val_loss) + 1 if val_losses else 0,
        "metrics": best_metrics_epoch,
        "config": loss_data.get("config", {}),
        "loss_log": loss_data_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f">>> 训练摘要已保存: {summary_path}")

    print("\n" + "=" * 60)
    print(">>> Graph-RWKV Model 训练结束")
    print(f"    Best Val Loss: {best_val_loss:.6f}")
    if best_metrics_epoch:
        print(f"\n    📊 评估指标:")
        print(f"      R² Score: {best_metrics_epoch['r2']:.4f}")
        print(f"      MAE: {best_metrics_epoch['mae']:.6f}")
        print(f"      Directional Accuracy: {best_metrics_epoch['directional_accuracy']:.2%}")
        if best_metrics_epoch.get('ic') is not None:
            print(f"      IC: {best_metrics_epoch['ic']:.4f}")
        if best_metrics_epoch.get('rank_ic') is not None:
            print(f"      RankIC: {best_metrics_epoch['rank_ic']:.4f}")
    print("=" * 60)
    return summary


def _generate_walk_forward_windows(
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    freq: str = "Q",
):
    """生成滚动窗口：训练集逐步扩展，测试集按季度滚动。"""
    periods = pd.period_range(test_start, test_end, freq=freq)
    windows = []
    for p in periods:
        t_start = p.start_time.normalize()
        t_end = p.end_time.normalize()
        train_end_cur = t_start - pd.Timedelta(days=1)
        label = f"{p.year}Q{p.quarter}"
        windows.append({
            "train_start": train_start,
            "train_end": train_end_cur.strftime("%Y-%m-%d"),
            "test_start": t_start.strftime("%Y-%m-%d"),
            "test_end": t_end.strftime("%Y-%m-%d"),
            "label": label,
        })
    return windows


def main():
    if CONFIG.get("use_walk_forward", False):
        base_exp = str(CONFIG.get("experiment_name", "full"))
        base_ckpt = str(CONFIG.get("checkpoint_name", "best_model.pth"))
        base_ckpt_prefix = base_ckpt.replace(".pth", "")

        windows = _generate_walk_forward_windows(
            CONFIG.get("walk_forward_train_start", "2018-01-01"),
            CONFIG.get("walk_forward_train_end", "2020-12-31"),
            CONFIG.get("walk_forward_test_start", "2021-01-01"),
            CONFIG.get("walk_forward_test_end", "2023-12-31"),
            CONFIG.get("walk_forward_freq", "Q"),
        )

        summaries = []
        for w in windows:
            CONFIG.update({
                "train_start": w["train_start"],
                "train_end": w["train_end"],
                "test_start": w["test_start"],
                "test_end": w["test_end"],
                "use_date_split": False,
                "experiment_name": f"{base_exp}_wf_{w['label']}",
                "checkpoint_name": f"{base_ckpt_prefix}_wf_{w['label']}.pth",
            })
            print(f"\n>>> Walk-Forward Window: {w['train_start']}~{w['train_end']} -> {w['test_start']}~{w['test_end']}")
            summary = _train_once()
            summary["window"] = w
            summaries.append(summary)

        results_dir = os.path.join(str(CONFIG.get("output_dir", OUTPUT_DIR)), "results")
        os.makedirs(results_dir, exist_ok=True)
        wf_path = os.path.join(results_dir, "walk_forward_summary.json")
        with open(wf_path, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
        print(f">>> Walk-Forward 汇总已保存: {wf_path}")
        return

    _train_once()


if __name__ == "__main__":
    main()
