# -*- coding: utf-8 -*-
"""
统一评估脚本：对比 Full Model 与所有消融模型的表现
========================================================================
功能：
    1. 加载 Full Model（由 train_gnn.py 训练）和所有消融模型（由 run_ablation.py 训练）
    2. 在测试集上计算统一指标（MSE、R²、方向准确率、IC、RankIC 等）
    3. 按波动率分组评估（低/中/高波动），验证论文核心论点
    4. 生成对比表格和图表

使用方法：
    python evaluate_all.py

前置条件：
    1. 先运行 train_gnn.py 训练 Full Model（生成 best_model_gnn.pth）
    2. 再运行 run_ablation.py 训练消融模型（生成 ablation/best_model_*.pth）
"""

import torch
import numpy as np
import pandas as pd
import os
import sys
import json
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from datetime import datetime

# ================= 1. 环境与路径配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)  # 添加项目根目录到路径

# 路径配置
GRAPH_PATH = os.path.join(parent_dir, 'data', 'processed', 'Graph_Adjacency.npy')
CSV_PATH = os.path.join(parent_dir, 'data', 'processed', 'Final_Model_Data.csv')
OUTPUT_DIR = os.path.join(parent_dir, 'outputs')
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 检查数据文件
if not os.path.exists(CSV_PATH):
    print(f"❌ 错误: 找不到 {CSV_PATH}")
    exit(1)

try:
    from dataProcessed.dataset import FinancialDataset
    from models.gnn_model import QL_MATCC_GNN_Model
    print("✅ 成功导入基础模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    exit(1)

# 配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def _load_train_config(parent_dir_: str) -> dict:
    """
    尝试读取训练日志中的配置（避免评估时模型维度不匹配）。

    优先级：
      1) outputs/logs/training_losses_full.json（train_full.py 输出）
      2) 环境变量 QL_PROFILE（paper / 48gb）
      3) 兜底默认（256/3/64）
    """
    cfg = {}
    path = os.path.join(parent_dir_, "outputs", "logs", "training_losses_full.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # train_full.py 的结构：loss_data["config"] 是一个 dict
            cfg = data.get("config", {}) if isinstance(data, dict) else {}
        except Exception:
            cfg = {}

    profile = os.environ.get("QL_PROFILE", (cfg.get("profile") if isinstance(cfg, dict) else None) or "paper").strip().lower()
    # 兜底：profile 推断
    if not isinstance(cfg, dict):
        cfg = {}

    if profile in ("48gb", "max", "server"):
        defaults = dict(n_embd=384, n_layers=4, gnn_embd=128, n_qubits=4, seq_len=30)
    else:
        defaults = dict(n_embd=256, n_layers=3, gnn_embd=64, n_qubits=4, seq_len=30)

    # 合并：log > profile 默认
    merged = dict(defaults)
    for k in ("n_embd", "n_layers", "gnn_embd", "n_qubits", "seq_len"):
        if k in cfg and cfg[k] is not None:
            merged[k] = int(cfg[k]) if k != "seq_len" else int(cfg[k])
    merged["profile"] = profile
    return merged


def _eval_batch_size(profile: str) -> int:
    """评估阶段 batch_size：可用环境变量 EVAL_BATCH_SIZE 覆盖。"""
    env = os.environ.get("EVAL_BATCH_SIZE")
    if env:
        try:
            return int(env)
        except Exception:
            pass
    # 评估只做前向，通常 batch 可以大一些；但大模型仍可能 OOM，后续会自动降级
    return 4096 if profile in ("48gb", "max", "server") else 2048


TRAIN_CFG = _load_train_config(parent_dir)

# 模型配置（尽量与训练时保持一致）
MODEL_CONFIG = {
    'input_dim': 8,
    'n_embd': int(TRAIN_CFG['n_embd']),
    'n_layers': int(TRAIN_CFG['n_layers']),
    'n_qubits': int(TRAIN_CFG['n_qubits']),
    'gnn_embd': int(TRAIN_CFG['gnn_embd']),
    'seq_len': int(TRAIN_CFG['seq_len']),
    'batch_size': _eval_batch_size(TRAIN_CFG['profile']),
}

# 所有模型的配置
MODELS_TO_EVALUATE = [
    {
        'name': 'Full Model',
        'exp_name': 'full_model',
        'path': os.path.join(CHECKPOINT_DIR, 'best_model_full.pth'),  # train_full.py 的输出
        'use_quantum': True,
        'use_graph': True,
        'use_matcc': True,
        'use_market_guidance': True,
    },
    {
        'name': 'w/o Quantum',
        'exp_name': 'no_quantum',
        'path': os.path.join(CHECKPOINT_DIR, 'best_model_no_quantum.pth'),
        'use_quantum': False,
        'use_graph': True,
        'use_matcc': True,
        'use_market_guidance': True,
    },
    {
        'name': 'w/o Graph',
        'exp_name': 'no_graph',
        'path': os.path.join(CHECKPOINT_DIR, 'best_model_no_graph.pth'),
        'use_quantum': True,
        'use_graph': False,
        'use_matcc': True,
        'use_market_guidance': True,
    },
    {
        'name': 'w/o MATCC',
        'exp_name': 'no_matcc',
        'path': os.path.join(CHECKPOINT_DIR, 'best_model_no_matcc.pth'),
        'use_quantum': True,
        'use_graph': True,
        'use_matcc': False,
        'use_market_guidance': True,
    },
    {
        'name': 'w/o Market',
        'exp_name': 'no_market_guidance',
        'path': os.path.join(CHECKPOINT_DIR, 'best_model_no_market_guidance.pth'),
        'use_quantum': True,
        'use_graph': True,
        'use_matcc': True,
        'use_market_guidance': False,
    },
]


# ================= 2. 辅助函数 =================
def calculate_metrics(y_true, y_pred):
    """
    计算回归任务的常用评估指标（用于整体对比与分组对比）。

    指标：
      - mse / mae / rmse / r2：回归误差与拟合优度
      - dir_acc：方向准确率（sign(pred) 与 sign(true) 一致的比例）
      - ic / rank_ic：信息系数（Pearson）与秩信息系数（Spearman）
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # 方向准确率
    true_dir = np.sign(y_true)
    pred_dir = np.sign(y_pred)
    dir_acc = np.mean(true_dir == pred_dir)
    
    # IC & RankIC
    try:
        ic, _ = pearsonr(y_pred, y_true)
        ic = float(ic)
    except:
        ic = None
    
    try:
        rank_ic, _ = spearmanr(y_pred, y_true)
        rank_ic = float(rank_ic)
    except:
        rank_ic = None
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'dir_acc': dir_acc,
        'ic': ic,
        'rank_ic': rank_ic,
        'n_samples': len(y_true),
    }


def load_model_and_predict(model_config, test_loader, adj_matrix, num_nodes):
    """加载模型并获取预测结果"""
    model_path = model_config['path']
    
    if not os.path.exists(model_path):
        print(f"   ⚠️ 模型文件不存在: {model_path}")
        return None, None
    
    # 处理 w/o Graph 的情况
    if not model_config['use_graph']:
        adj = np.eye(num_nodes, dtype=np.float32)
    else:
        adj = adj_matrix
    
    # 初始化模型（必须与保存权重的维度一致，否则 load_state_dict 会报 size mismatch）
    model = QL_MATCC_GNN_Model(
        input_dim=MODEL_CONFIG['input_dim'],
        n_embd=MODEL_CONFIG['n_embd'],
        n_layers=MODEL_CONFIG['n_layers'],
        n_qubits=MODEL_CONFIG['n_qubits'],
        num_nodes=num_nodes,
        adj_matrix=adj,
        gnn_embd=MODEL_CONFIG['gnn_embd'],
        use_quantum=model_config['use_quantum'],
        use_matcc=model_config['use_matcc'],
        use_market_guidance=model_config['use_market_guidance'],
    ).to(DEVICE)
    
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 预测
    all_preds = []
    all_labels = []
    all_vols = []
    
    with torch.inference_mode():
        for batch in test_loader:
            x = batch['x'].to(DEVICE, non_blocking=True)
            y = batch['y'].to(DEVICE, non_blocking=True)
            vol = batch['vol'].to(DEVICE, non_blocking=True)
            node_idx = batch.get('node_indices')
            if node_idx is not None:
                node_idx = node_idx.to(DEVICE, non_blocking=True)
            
            preds = model(x, vol, node_indices=node_idx)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_vols.append(vol.cpu().numpy())
    
    preds = np.concatenate(all_preds, axis=0).flatten()
    labels = np.concatenate(all_labels, axis=0).flatten()
    vols = np.concatenate(all_vols, axis=0).flatten()
    
    return preds, labels, vols


# ================= 3. 主程序 =================
def main():
    """
    统一评估入口：
      1) 加载测试集
      2) 依次加载可用模型并预测
      3) 计算整体指标 + 按波动率分组指标
      4) 保存 CSV/PNG/JSON 到 outputs/results 与 outputs/figures
    """
    print("="*70)
    print("📊 统一评估：Full Model vs 消融模型")
    print("="*70)
    print(f"📁 工作目录: {current_dir}")
    print(f"📊 数据文件: {CSV_PATH}")
    print(f"💻 设备: {DEVICE}")
    print()
    
    # 检查模型文件
    print(">>> 检查模型文件...")
    available_models = []
    for m in MODELS_TO_EVALUATE:
        if os.path.exists(m['path']):
            print(f"   ✅ {m['name']}: {m['path']}")
            available_models.append(m)
        else:
            print(f"   ❌ {m['name']}: 文件不存在 ({m['path']})")
    
    if len(available_models) == 0:
        print("\n❌ 没有可用的模型文件，请先运行训练脚本")
        return
    
    print(f">>> 评估将使用模型配置: n_embd={MODEL_CONFIG['n_embd']}, n_layers={MODEL_CONFIG['n_layers']}, gnn_embd={MODEL_CONFIG['gnn_embd']}, batch={MODEL_CONFIG['batch_size']} (profile={TRAIN_CFG['profile']})")

    # 加载数据
    print("\n>>> 加载测试数据...")
    train_dataset = FinancialDataset(CSV_PATH, seq_len=MODEL_CONFIG['seq_len'], mode='train')
    test_dataset = FinancialDataset(
        CSV_PATH,
        seq_len=MODEL_CONFIG['seq_len'],
        mode='test',
        scaler=train_dataset.scaler,
        vol_stats=train_dataset.vol_stats,
    )
    # 更贴近服务器：用更多 worker + pin_memory（若 CUDA）
    num_workers = min(8, max(2, (os.cpu_count() or 12) - 2))
    pin_memory = torch.cuda.is_available()

    # 评估 batch 可能 OOM：发生时自动减半重试
    bs = int(MODEL_CONFIG['batch_size'])
    while True:
        try:
            test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None)
            break
        except Exception:
            # DataLoader 构建失败较少见，继续兜底
            bs = max(256, bs // 2)
            if bs <= 256:
                test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=0)
                break
    MODEL_CONFIG['batch_size'] = bs
    print(f"   测试集样本数: {len(test_dataset)}")
    
    # 加载图
    if GRAPH_PATH and os.path.exists(GRAPH_PATH):
        adj_matrix = np.load(GRAPH_PATH)
        print(f"   图谱: {GRAPH_PATH}, 形状: {adj_matrix.shape}")
    else:
        df_t = pd.read_csv(CSV_PATH, usecols=['Ticker'])
        num_nodes = int(df_t['Ticker'].nunique())
        adj_matrix = np.eye(num_nodes, dtype=np.float32)
        print(f"   未找到图谱，使用单位阵，形状: {adj_matrix.shape}")
    
    num_nodes = adj_matrix.shape[0]
    
    # 评估所有模型
    print("\n>>> 评估所有模型...")
    all_results = []
    predictions = {}  # 存储所有模型的预测结果
    
    y_true = None  # 真实值（所有模型相同）
    vols = None    # 波动率
    
    for model_config in available_models:
        print(f"\n   📌 正在评估: {model_config['name']}...")
        preds, labels, vol_data = load_model_and_predict(model_config, test_loader, adj_matrix, num_nodes)
        
        if preds is None:
            continue
        
        if y_true is None:
            y_true = labels
            vols = vol_data
        
        predictions[model_config['exp_name']] = preds
        
        # 计算全量指标
        metrics = calculate_metrics(labels, preds)
        metrics['name'] = model_config['name']
        metrics['exp_name'] = model_config['exp_name']
        all_results.append(metrics)
        
        print(f"      MSE: {metrics['mse']:.6f}, R²: {metrics['r2']:.4f}, "
              f"DirAcc: {metrics['dir_acc']:.4f}, IC: {metrics['ic']:.4f if metrics['ic'] else 'N/A'}")
    
    # ================= 全量评估结果 =================
    print("\n" + "="*70)
    print("📊 全量评估结果对比")
    print("="*70)
    
    df_results = pd.DataFrame(all_results)
    cols_order = ['name', 'mse', 'mae', 'rmse', 'r2', 'dir_acc', 'ic', 'rank_ic', 'n_samples']
    cols_order = [c for c in cols_order if c in df_results.columns]
    df_results = df_results[cols_order]
    
    print(df_results.to_string(index=False))
    
    # 保存全量结果
    df_results.to_csv(os.path.join(RESULTS_DIR, 'evaluation_overall.csv'), index=False, float_format='%.6f')
    
    # ================= 分组评估（按波动率）=================
    if 'full_model' in predictions and len(predictions) > 1:
        print("\n" + "="*70)
        print("📊 分组评估（按波动率分位数）")
        print("="*70)
        
        # 按波动率分组
        quantiles = np.quantile(vols, [0.4, 0.8])
        groups = {
            'Low Vol (0-40%)': vols <= quantiles[0],
            'Mid Vol (40-80%)': (vols > quantiles[0]) & (vols <= quantiles[1]),
            'High Vol (80-100%)': vols > quantiles[1],
        }
        
        group_results = []
        
        for group_name, mask in groups.items():
            y_group = y_true[mask]
            
            for exp_name, preds in predictions.items():
                preds_group = preds[mask]
                metrics = calculate_metrics(y_group, preds_group)
                
                # 找到对应的模型名称
                model_name = next((m['name'] for m in available_models if m['exp_name'] == exp_name), exp_name)
                
                group_results.append({
                    'group': group_name,
                    'model': model_name,
                    'exp_name': exp_name,
                    'mse': metrics['mse'],
                    'r2': metrics['r2'],
                    'dir_acc': metrics['dir_acc'],
                    'ic': metrics['ic'],
                    'rank_ic': metrics['rank_ic'],
                    'n_samples': metrics['n_samples'],
                })
        
        df_group = pd.DataFrame(group_results)
        
        # 打印分组结果
        for group_name in groups.keys():
            print(f"\n🔹 {group_name}")
            df_sub = df_group[df_group['group'] == group_name]
            print(df_sub[['model', 'mse', 'r2', 'dir_acc', 'ic', 'rank_ic']].to_string(index=False))
        
        # 保存分组结果
        df_group.to_csv(os.path.join(RESULTS_DIR, 'evaluation_by_volatility.csv'), index=False, float_format='%.6f')
        
        # ================= 关键结论：高波动组对比 =================
        print("\n" + "="*70)
        print("🎯 关键结论：高波动组（Top 20%）Full Model vs w/o Quantum 对比")
        print("="*70)
        
        high_vol_full = df_group[(df_group['group'] == 'High Vol (80-100%)') & (df_group['exp_name'] == 'full_model')]
        high_vol_noq = df_group[(df_group['group'] == 'High Vol (80-100%)') & (df_group['exp_name'] == 'no_quantum')]
        
        if not high_vol_full.empty and not high_vol_noq.empty:
            full_mse = high_vol_full['mse'].values[0]
            noq_mse = high_vol_noq['mse'].values[0]
            full_ic = high_vol_full['ic'].values[0]
            noq_ic = high_vol_noq['ic'].values[0]
            full_rank_ic = high_vol_full['rank_ic'].values[0]
            noq_rank_ic = high_vol_noq['rank_ic'].values[0]
            
            print(f"\n指标               | Full Model     | w/o Quantum    | Winner")
            print("-" * 60)
            winner_mse = "✅ Full" if full_mse < noq_mse else "❌ NoQ"
            winner_ic = "✅ Full" if (full_ic or 0) > (noq_ic or 0) else "❌ NoQ"
            winner_rank_ic = "✅ Full" if (full_rank_ic or 0) > (noq_rank_ic or 0) else "❌ NoQ"
            print(f"MSE               | {full_mse:<14.6f} | {noq_mse:<14.6f} | {winner_mse}")
            print(f"IC                | {full_ic if full_ic else 'N/A':<14.4f} | {noq_ic if noq_ic else 'N/A':<14.4f} | {winner_ic}")
            print(f"RankIC            | {full_rank_ic if full_rank_ic else 'N/A':<14.4f} | {noq_rank_ic if noq_rank_ic else 'N/A':<14.4f} | {winner_rank_ic}")
            
            # 结论
            if full_mse < noq_mse:
                print("\n✅ 好消息！Full Model 在高波动样本上 MSE 更优！")
                print("   论文核心论点得到验证：量子模块擅长捕捉高波动/混沌特征。")
            elif (full_ic or 0) > (noq_ic or 0) or (full_rank_ic or 0) > (noq_rank_ic or 0):
                print("\n⚠️ MSE 未领先，但 IC/RankIC 更优")
                print("   可以从相关性角度阐述量子模块的优势。")
            else:
                print("\n⚠️ 高波动组 Full Model 未能领先 w/o Quantum")
                print("   建议：使用分层学习率或调整量子层参数后重新训练")
    
    # ================= 生成对比图表 =================
    print("\n>>> 生成对比图表...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：全量 MSE 对比
    df_sorted = df_results.sort_values('mse')
    colors = ['#FFD700' if n == 'Full Model' else '#1f77b4' for n in df_sorted['name']]
    axes[0].barh(df_sorted['name'], df_sorted['mse'], color=colors)
    axes[0].set_xlabel('MSE (Lower is Better)')
    axes[0].set_title('Overall MSE Comparison', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(df_sorted['mse']):
        axes[0].text(v, i, f' {v:.6f}', va='center', fontsize=9)
    
    # 右图：R² 对比
    df_sorted_r2 = df_results.sort_values('r2', ascending=False)
    colors_r2 = ['#FFD700' if n == 'Full Model' else '#2ca02c' for n in df_sorted_r2['name']]
    axes[1].barh(df_sorted_r2['name'], df_sorted_r2['r2'], color=colors_r2)
    axes[1].set_xlabel('R² (Higher is Better)')
    axes[1].set_title('Overall R² Comparison', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(df_sorted_r2['r2']):
        axes[1].text(v, i, f' {v:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, 'evaluation_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 图表已保存: {fig_path}")
    
    # ================= 保存汇总报告 =================
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': DEVICE,
        'n_samples': len(test_dataset),
        'models_evaluated': [m['name'] for m in available_models],
        'overall_results': all_results,
    }
    
    with open(os.path.join(RESULTS_DIR, 'evaluation_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("\n" + "="*70)
    print("✅ 评估完成！")
    print(f"\n📁 结果保存在: {RESULTS_DIR}")
    print("   - evaluation_overall.csv (全量指标)")
    print("   - evaluation_by_volatility.csv (分组指标)")
    print("   - evaluation_comparison.png (对比图表)")
    print("   - evaluation_report.json (完整报告)")
    print("="*70)


if __name__ == "__main__":
    main()
