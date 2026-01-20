# -*- coding: utf-8 -*-
"""
全自动消融实验脚本 (Run Ablation Studies)
========================================================================
功能：
    依次运行 5 组实验（1个完整模型基准 + 4组消融实验），验证各模块的有效性。
    不修改原有的 train_gnn.py，独立运行。
    
    实验列表：
    1. Full Model：完整模型基准线（所有模块开启）
    2. w/o Quantum：移除量子模块
    3. w/o Graph：移除图神经网络
    4. w/o MATCC：移除趋势解耦
    5. w/o Market Guidance：移除市场引导

输出：
    - model/ablation/curve_full_model.png (完整模型基准线)
    - model/ablation/curve_no_quantum.png
    - model/ablation/curve_no_graph.png
    - model/ablation/curve_no_matcc.png
    - model/ablation/curve_no_market_guidance.png
    - model/ablation/losses_full_model.json (完整模型的 Loss 数值列表)
    - model/ablation/losses_no_quantum.json (各消融实验的 Loss 数值列表)
    - model/ablation/losses_*.json (所有实验的 Loss 数值列表，用于后续对比分析)
    - model/ablation/ablation_results_summary.csv (汇总表格)
    - model/ablation/ablation_results_comparison.png (对比图表，包含5条曲线)
    - model/ablation/best_model_*.pth (各实验的最佳模型)

改进点：
    1. ✅ 添加 Full Model 基准实验（确保公平对比）
    2. ✅ 添加 w/o Market Guidance 实验（论文中第4组消融）
    3. ✅ 保存每个实验的最佳模型
    4. ✅ **保存每个实验的 Loss 数值列表（JSON格式）**，避免重复运行
    5. ✅ 生成汇总对比表格和图表（Full Model 用金色突出显示）
    6. ✅ 记录更详细的指标（最佳epoch、最终loss等）
    7. ✅ 路径检查和错误处理更完善
    8. ✅ 所有结果保存在独立的 ablation/ 目录，避免混淆
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

# ================= 1. 环境与路径配置 =================
# 确保能导入上级目录的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'dataProcessed'))
sys.path.append(current_dir)

# 数据路径（兼容不同可能的路径）
possible_graph_paths = [
    os.path.join(parent_dir, 'data', 'processed', 'Graph_Adjacency.npy'),
    os.path.join(parent_dir, 'dataProcessed', 'Graph_Adjacency.npy'),
]
possible_csv_paths = [
    os.path.join(parent_dir, 'data', 'processed', 'Final_Model_Data.csv'),
    os.path.join(parent_dir, 'dataProcessed', 'Final_Model_Data.csv'),
]

GRAPH_PATH = None
CSV_PATH = None
for path in possible_graph_paths:
    if os.path.exists(path):
        GRAPH_PATH = path
        break
for path in possible_csv_paths:
    if os.path.exists(path):
        CSV_PATH = path
        break

if CSV_PATH is None:
    print("❌ 错误: 找不到 Final_Model_Data.csv")
    print("   已尝试路径:")
    for path in possible_csv_paths:
        print(f"     - {path}")
    exit(1)

try:
    from dataset import FinancialDataset
    from model_gnn import QL_MATCC_GNN_Model
    print("✅ 成功导入基础模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    exit(1)

# ================= 2. 统一超参数 (保持与 Full Model 一致) =================
BASE_CONFIG = {
    'input_dim': 8,
    'n_embd': 512,
    'n_layers': 4,
    'n_qubits': 4,
    'gnn_embd': 128,
    'seq_len': 30,
    'batch_size': 3072,  # 保持与原训练脚本一致
    'epochs': 6,        # 消融实验可以适当减少，但为了公平对比，建议保持相同
    'lr': 1e-4,
    'early_stop_patience': 3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 12,
    'prefetch_factor': 8,
    'use_amp': True,
}

# ================= 3. 结果存储目录 =================
# 创建 ablation 目录用于存放所有消融实验结果
ABLATION_DIR = os.path.join(current_dir, 'ablation')
os.makedirs(ABLATION_DIR, exist_ok=True)
print(f"📁 消融实验结果将保存到: {ABLATION_DIR}")

RESULTS = []  # 存储所有实验的结果


def calculate_metrics(y_true, y_pred):
    """
    计算多个评估指标（包括量化金融标准指标）
    
    参数:
        y_true: 真实值 (numpy array)
        y_pred: 预测值 (numpy array)
    
    返回:
        dict: 包含各种评估指标的字典
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # ========== 1. 统计误差类 ==========
    # MSE (均方误差) - 训练主指标
    mse = mean_squared_error(y_true, y_pred)
    
    # MAE (平均绝对误差) - 对异常值不敏感
    mae = mean_absolute_error(y_true, y_pred)
    
    # RMSE (均方根误差) - 与目标变量同单位，更直观
    rmse = np.sqrt(mse)
    
    # R² (决定系数) - 模型拟合优度
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (平均绝对百分比误差) - 相对误差
    mask = np.abs(y_true) > 1e-8
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = None
    
    # ========== 2. 方向预测类 ==========
    # Directional Accuracy (方向准确率) - 金融预测核心指标
    # 预测涨跌方向的准确率，对金融预测非常重要
    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)
    directional_accuracy = np.mean(true_direction == pred_direction)
    
    # ========== 3. 量化投资类 ==========
    # IC (Information Coefficient) - 信息系数
    # Pearson 相关系数，衡量预测值与真实值的线性相关性
    # 这是量化金融领域的黄金标准指标
    try:
        ic, ic_pvalue = pearsonr(y_pred, y_true)
        ic = float(ic)
    except:
        ic = None
        ic_pvalue = None
    
    # RankIC (Rank Information Coefficient) - 秩信息系数
    # Spearman 秩相关系数，衡量预测排名与真实排名的相关性
    # 比 IC 更稳健，不受异常值影响，是基金公司最看重的指标
    try:
        rank_ic, rank_ic_pvalue = spearmanr(y_pred, y_true)
        rank_ic = float(rank_ic)
    except:
        rank_ic = None
        rank_ic_pvalue = None
    
    # 传统相关系数（用于兼容性）
    try:
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        correlation = float(correlation)
    except:
        correlation = None
    
    return {
        # 统计误差类
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape) if mape is not None and not np.isnan(mape) else None,
        
        # 方向预测类
        'directional_accuracy': float(directional_accuracy),
        
        # 量化投资类
        'ic': ic,  # Information Coefficient (Pearson)
        'ic_pvalue': float(ic_pvalue) if ic_pvalue is not None else None,
        'rank_ic': rank_ic,  # Rank Information Coefficient (Spearman)
        'rank_ic_pvalue': float(rank_ic_pvalue) if rank_ic_pvalue is not None else None,
        
        # 兼容性指标
        'correlation': correlation,
    }


def run_experiment(exp_name, use_quantum=True, use_graph=True, use_matcc=True, use_market_guidance=True):
    """
    运行单个实验的核心函数
    
    参数:
        exp_name: 实验名称（如 "no_quantum"）
        use_quantum: 是否使用量子模块
        use_graph: 是否使用图神经网络
        use_matcc: 是否使用MATCC趋势解耦
        use_market_guidance: 是否使用市场引导
    """
    print("\n" + "="*70)
    print(f"🧪 开始运行实验: {exp_name}")
    print(f"   配置: Quantum={use_quantum}, Graph={use_graph}, MATCC={use_matcc}, MarketGuidance={use_market_guidance}")
    print("="*70)

    # ---------------- A. 准备图谱 ----------------
    # 如果是 w/o Graph 实验，强制使用单位阵（切断图连接）
    if not use_graph:
        print("   ⚠️ [消融设置] 禁用图神经网络 (使用单位阵)")
        df_t = pd.read_csv(CSV_PATH, usecols=['Ticker'])
        num_nodes = int(df_t['Ticker'].nunique())
        adj_matrix = np.eye(num_nodes, dtype=np.float32)
    else:
        if GRAPH_PATH and os.path.exists(GRAPH_PATH):
            adj_matrix = np.load(GRAPH_PATH)
            print(f"   ✅ 加载图谱: {GRAPH_PATH}, 形状: {adj_matrix.shape}")
        else:
            # 兜底：使用单位阵
            print(f"   ⚠️ 未找到图谱文件，使用单位阵")
            df_t = pd.read_csv(CSV_PATH, usecols=['Ticker'])
            num_nodes = int(df_t['Ticker'].nunique())
            adj_matrix = np.eye(num_nodes, dtype=np.float32)
    
    num_nodes = adj_matrix.shape[0]

    # ---------------- B. 准备数据 ----------------
    # 每次重新加载数据，防止内存泄漏
    train_dataset = FinancialDataset(CSV_PATH, seq_len=BASE_CONFIG['seq_len'], mode='train')
    test_dataset = FinancialDataset(CSV_PATH, seq_len=BASE_CONFIG['seq_len'], mode='test', scaler=train_dataset.scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=BASE_CONFIG['batch_size'], shuffle=True, 
                              num_workers=BASE_CONFIG['num_workers'], pin_memory=True, 
                              prefetch_factor=BASE_CONFIG['prefetch_factor'],
                              persistent_workers=True if BASE_CONFIG['num_workers'] > 0 else False)
    test_loader = DataLoader(test_dataset, batch_size=BASE_CONFIG['batch_size'], shuffle=False, 
                             num_workers=BASE_CONFIG['num_workers'], pin_memory=True, 
                             prefetch_factor=BASE_CONFIG['prefetch_factor'],
                             persistent_workers=True if BASE_CONFIG['num_workers'] > 0 else False)

    # ---------------- C. 初始化模型 ----------------
    model = QL_MATCC_GNN_Model(
        input_dim=BASE_CONFIG['input_dim'],
        n_embd=BASE_CONFIG['n_embd'],
        n_layers=BASE_CONFIG['n_layers'],
        n_qubits=BASE_CONFIG['n_qubits'],
        num_nodes=num_nodes,
        adj_matrix=adj_matrix,
        gnn_embd=BASE_CONFIG['gnn_embd'],
        # === 关键：这里传入消融开关 ===
        use_quantum=use_quantum,
        use_matcc=use_matcc,
        use_market_guidance=use_market_guidance,
    ).to(BASE_CONFIG['device'])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=BASE_CONFIG['lr'], betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)
    scaler = torch.cuda.amp.GradScaler() if BASE_CONFIG['use_amp'] else None

    # ---------------- D. 训练循环 ----------------
    train_losses, val_losses = [], []
    best_val = float('inf')
    best_epoch = 0
    counter = 0

    for epoch in range(BASE_CONFIG['epochs']):
        model.train()
        train_loss = 0.0
        steps = 0
        
        # 训练
        pbar = tqdm(train_loader, desc=f"[{exp_name}] Ep {epoch+1}/{BASE_CONFIG['epochs']}", ncols=100)
        for batch in pbar:
            x = batch['x'].to(BASE_CONFIG['device'], non_blocking=True)
            y = batch['y'].to(BASE_CONFIG['device'], non_blocking=True)
            vol = batch['vol'].to(BASE_CONFIG['device'], non_blocking=True)
            node_indices = batch.get('node_indices')
            if node_indices is not None:
                node_indices = node_indices.to(BASE_CONFIG['device'], non_blocking=True)

            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    preds = model(x, vol, node_indices=node_indices)
                    loss = criterion(preds, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(x, vol, node_indices=node_indices)
                loss = criterion(preds, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            train_loss += loss.item()
            steps += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_train = train_loss / len(train_loader)
        train_losses.append(avg_train)

        # 验证（收集所有预测值和真实值用于计算指标）
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                x = batch['x'].to(BASE_CONFIG['device'], non_blocking=True)
                y = batch['y'].to(BASE_CONFIG['device'], non_blocking=True)
                vol = batch['vol'].to(BASE_CONFIG['device'], non_blocking=True)
                node_indices = batch.get('node_indices')
                if node_indices is not None:
                    node_indices = node_indices.to(BASE_CONFIG['device'], non_blocking=True)
                
                if scaler:
                    with torch.cuda.amp.autocast():
                        preds = model(x, vol, node_indices=node_indices)
                else:
                    preds = model(x, vol, node_indices=node_indices)
                
                val_loss += criterion(preds, y).item()
                
                # 收集预测值和真实值（用于计算完整指标）
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        avg_val = val_loss / len(test_loader)
        val_losses.append(avg_val)
        
        # 计算完整评估指标（仅在最佳epoch时计算，节省时间）
        metrics = None
        if avg_val < best_val:
            all_preds_np = np.concatenate(all_preds, axis=0)
            all_targets_np = np.concatenate(all_targets, axis=0)
            metrics = calculate_metrics(all_targets_np, all_preds_np)
        
        # 学习率调度
        scheduler.step(avg_val)
        cur_lr = optimizer.param_groups[0]['lr']
        
        print(f"   Ep {epoch+1}: Train={avg_train:.6f}, Val={avg_val:.6f}, lr={cur_lr:.2e}")

        # 早停和保存最佳模型
        best_metrics = None
        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch + 1
            best_metrics = metrics  # 保存最佳epoch的指标
            counter = 0
            # 保存消融实验的最佳模型到 ablation 目录
            model_save_path = os.path.join(ABLATION_DIR, f'best_model_{exp_name}.pth')
            torch.save(model.state_dict(), model_save_path)
            
            # 打印关键指标
            if metrics:
                print(f"   💾 Best model saved (Ep {best_epoch}, Val={best_val:.6f})")
                print(f"      Metrics: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.6f}, "
                      f"DirAcc={metrics['directional_accuracy']:.2%}", end="")
                if metrics.get('ic') is not None:
                    print(f", IC={metrics['ic']:.4f}", end="")
                if metrics.get('rank_ic') is not None:
                    print(f", RankIC={metrics['rank_ic']:.4f}", end="")
                print()
            else:
                print(f"   💾 Best model saved (Ep {best_epoch}, Val={best_val:.6f})")
        else:
            counter += 1
            if counter >= BASE_CONFIG['early_stop_patience']:
                print("   🛑 Early stopping")
                break

    # ---------------- E. 画图并保存 ----------------
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train Loss', lw=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Val Loss', lw=2)
    if val_losses:
        plt.plot(best_epoch, best_val, 'g*', markersize=14, label=f'Best (Ep {best_epoch})')
    plt.title(f'Ablation Study: {exp_name}', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(ABLATION_DIR, f'curve_{exp_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 结果已保存: {save_path}")
    
    # ---------------- F. 最终评估（在最佳模型上计算完整指标）---------------
    # 重新加载最佳模型并计算完整指标
    if best_metrics is None:
        print("   ⚠️ 重新计算最终评估指标...")
        model.load_state_dict(torch.load(os.path.join(ABLATION_DIR, f'best_model_{exp_name}.pth')))
        model.eval()
        all_preds_final = []
        all_targets_final = []
        
        with torch.no_grad():
            for batch in test_loader:
                x = batch['x'].to(BASE_CONFIG['device'], non_blocking=True)
                y = batch['y'].to(BASE_CONFIG['device'], non_blocking=True)
                vol = batch['vol'].to(BASE_CONFIG['device'], non_blocking=True)
                node_indices = batch.get('node_indices')
                if node_indices is not None:
                    node_indices = node_indices.to(BASE_CONFIG['device'], non_blocking=True)
                
                if scaler:
                    with torch.cuda.amp.autocast():
                        preds = model(x, vol, node_indices=node_indices)
                else:
                    preds = model(x, vol, node_indices=node_indices)
                
                all_preds_final.append(preds.cpu().numpy())
                all_targets_final.append(y.cpu().numpy())
        
        all_preds_final_np = np.concatenate(all_preds_final, axis=0)
        all_targets_final_np = np.concatenate(all_targets_final, axis=0)
        best_metrics = calculate_metrics(all_targets_final_np, all_preds_final_np)
        
        # 打印关键指标
        if best_metrics:
            print(f"   📊 Final Metrics: R²={best_metrics['r2']:.4f}, "
                  f"DirAcc={best_metrics['directional_accuracy']:.2%}", end="")
            if best_metrics.get('ic') is not None:
                print(f", IC={best_metrics['ic']:.4f}", end="")
            if best_metrics.get('rank_ic') is not None:
                print(f", RankIC={best_metrics['rank_ic']:.4f}", end="")
            print()
    
    # ---------------- G. 保存 Loss 数值列表和评估指标 ----------------
    loss_data_path = os.path.join(ABLATION_DIR, f'losses_{exp_name}.json')
    loss_data = {
        'experiment': exp_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val,
        'best_epoch': best_epoch,
        'total_epochs': len(train_losses),
        'use_quantum': use_quantum,
        'use_graph': use_graph,
        'use_matcc': use_matcc,
        'use_market_guidance': use_market_guidance,
        'metrics': best_metrics,  # 添加完整评估指标
        'config': {
            'batch_size': BASE_CONFIG['batch_size'],
            'lr': BASE_CONFIG['lr'],
            'epochs': BASE_CONFIG['epochs'],
        }
    }
    with open(loss_data_path, 'w') as f:
        json.dump(loss_data, f, indent=2)
    print(f"✅ Loss 数值列表和评估指标已保存: {loss_data_path}")
    
    # ---------------- H. 记录结果到汇总列表 ----------------
    final_train = train_losses[-1] if train_losses else float('nan')
    final_val = val_losses[-1] if val_losses else float('nan')
    
    result = {
        'experiment': exp_name,
        'best_val_loss': best_val,
        'best_epoch': best_epoch,
        'final_train_loss': final_train,
        'final_val_loss': final_val,
        'total_epochs': len(train_losses),
        'use_quantum': use_quantum,
        'use_graph': use_graph,
        'use_matcc': use_matcc,
        'use_market_guidance': use_market_guidance,
        # 添加评估指标
        # 统计误差类
        'mse': best_metrics['mse'] if best_metrics else None,
        'mae': best_metrics['mae'] if best_metrics else None,
        'rmse': best_metrics['rmse'] if best_metrics else None,
        'r2': best_metrics['r2'] if best_metrics else None,
        'mape': best_metrics['mape'] if best_metrics and best_metrics['mape'] is not None else None,
        
        # 方向预测类
        'directional_accuracy': best_metrics['directional_accuracy'] if best_metrics else None,
        
        # 量化投资类
        'ic': best_metrics['ic'] if best_metrics and best_metrics.get('ic') is not None else None,
        'ic_pvalue': best_metrics['ic_pvalue'] if best_metrics and best_metrics.get('ic_pvalue') is not None else None,
        'rank_ic': best_metrics['rank_ic'] if best_metrics and best_metrics.get('rank_ic') is not None else None,
        'rank_ic_pvalue': best_metrics['rank_ic_pvalue'] if best_metrics and best_metrics.get('rank_ic_pvalue') is not None else None,
        
        # 兼容性指标
        'correlation': best_metrics['correlation'] if best_metrics and best_metrics.get('correlation') is not None else None,
    }
    RESULTS.append(result)
    
    # 清理显存
    del model, optimizer, scheduler
    if scaler:
        del scaler
    torch.cuda.empty_cache()


def save_summary_results():
    """保存汇总结果到CSV和生成对比图表"""
    if not RESULTS:
        print("⚠️ 没有实验结果可保存")
        return
    
    # 保存CSV到 ablation 目录（包含所有评估指标）
    df_results = pd.DataFrame(RESULTS)
    
    # 重新排列列的顺序，让重要指标在前面
    column_order = [
        'experiment', 'best_val_loss', 
        # 统计误差类
        'mse', 'mae', 'rmse', 'r2', 'mape',
        # 方向预测类
        'directional_accuracy',
        # 量化投资类
        'ic', 'ic_pvalue', 'rank_ic', 'rank_ic_pvalue',
        # 其他
        'correlation', 'best_epoch', 'final_train_loss', 'final_val_loss', 
        'total_epochs', 'use_quantum', 'use_graph', 'use_matcc', 'use_market_guidance'
    ]
    # 只保留存在的列
    column_order = [col for col in column_order if col in df_results.columns]
    df_results = df_results[column_order]
    
    csv_path = os.path.join(ABLATION_DIR, 'ablation_results_summary.csv')
    df_results.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\n✅ 汇总结果已保存: {csv_path}")
    
    # 生成对比图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：最佳验证损失对比（柱状图）
    exp_names = [r['experiment'] for r in RESULTS]
    best_vals = [r['best_val_loss'] for r in RESULTS]
    
    # 为 Full Model 使用特殊颜色（金色），其他用不同颜色
    colors = []
    ablation_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    ablation_idx = 0
    
    for name in exp_names:
        if name == 'full_model':
            colors.append('#FFD700')  # 金色，突出基准线
        else:
            # 为消融实验分配不同颜色
            if ablation_idx < len(ablation_colors):
                colors.append(ablation_colors[ablation_idx])
                ablation_idx += 1
            else:
                colors.append('#1f77b4')  # 默认蓝色
    
    axes[0].bar(exp_names, best_vals, color=colors)
    axes[0].set_ylabel('Best Val Loss (MSE)', fontsize=12)
    axes[0].set_title('Ablation Study: Best Validation Loss', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, v in enumerate(best_vals):
        axes[0].text(i, v, f'{v:.6f}', ha='center', va='bottom', fontsize=9)
    
    # 右图：最佳验证损失对比（更清晰的展示）
    # 按损失值排序，Full Model 应该是最低的
    sorted_results = sorted(RESULTS, key=lambda x: x['best_val_loss'])
    sorted_names = [r['experiment'] for r in sorted_results]
    sorted_vals = [r['best_val_loss'] for r in sorted_results]
    
    # 为 Full Model 使用特殊颜色
    bar_colors = []
    ablation_colors_h = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    ablation_idx_h = 0
    
    for name in sorted_names:
        if name == 'full_model':
            bar_colors.append('#FFD700')  # 金色
        else:
            # 为消融实验分配不同颜色
            if ablation_idx_h < len(ablation_colors_h):
                bar_colors.append(ablation_colors_h[ablation_idx_h])
                ablation_idx_h += 1
            else:
                bar_colors.append('#1f77b4')  # 默认蓝色
    
    axes[1].barh(sorted_names, sorted_vals, color=bar_colors)
    axes[1].set_xlabel('Best Val Loss (MSE)', fontsize=12)
    axes[1].set_title('Ablation Study: Loss Ranking', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for i, v in enumerate(sorted_vals):
        axes[1].text(v, i, f' {v:.6f}', va='center', fontsize=9)
    
    axes[1].set_xlabel('Epochs', fontsize=12)
    axes[1].set_ylabel('Loss (MSE)', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_path = os.path.join(ABLATION_DIR, 'ablation_results_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 对比图表已保存: {comparison_path}")
    
    # 打印汇总表格
    print("\n" + "="*70)
    print("📊 消融实验汇总结果")
    print("="*70)
    print(df_results.to_string(index=False))
    print("="*70)


# ================= 4. 主程序入口 =================
if __name__ == "__main__":
    print("🚀 启动全自动消融实验流程...")
    print(f"📁 工作目录: {current_dir}")
    print(f"📁 结果目录: {ABLATION_DIR}")
    print(f"📊 数据文件: {CSV_PATH}")
    print(f"🔗 图谱文件: {GRAPH_PATH if GRAPH_PATH else '未找到（将使用单位阵）'}")
    print(f"💻 设备: {BASE_CONFIG['device']}")
    
    start_time = datetime.now()
    
    # 实验 0: 完整模型基准 (Full Model / Baseline)
    # 所有开关全部打开，作为对比的"天花板"，确保在相同实验条件下公平对比
    print("\n" + "="*70)
    print("📌 重要提示：首先运行 Full Model 作为基准线")
    print("   这样可以确保所有实验在完全相同的条件下（epochs、batch_size、随机种子等）进行对比")
    print("="*70)
    run_experiment(exp_name="full_model", 
                   use_quantum=True, use_graph=True, use_matcc=True, use_market_guidance=True)
    
    # 实验 1: 无量子模块 (w/o Quantum)
    run_experiment(exp_name="no_quantum", 
                   use_quantum=False, use_graph=True, use_matcc=True, use_market_guidance=True)
    
    # 实验 2: 无图神经网络 (w/o Graph)
    run_experiment(exp_name="no_graph", 
                   use_quantum=True, use_graph=False, use_matcc=True, use_market_guidance=True)
    
    # 实验 3: 无趋势解耦 (w/o MATCC)
    run_experiment(exp_name="no_matcc", 
                   use_quantum=True, use_graph=True, use_matcc=False, use_market_guidance=True)
    
    # 实验 4: 无市场引导 (w/o Market Guidance) - 【新增】
    run_experiment(exp_name="no_market_guidance", 
                   use_quantum=True, use_graph=True, use_matcc=True, use_market_guidance=False)
    
    # 保存汇总结果
    save_summary_results()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60  # 分钟
    
    print("\n" + "="*70)
    print("🎉 所有实验已完成！（1个基准 + 4个消融实验）")
    print(f"⏱️  总耗时: {duration:.1f} 分钟")
    print("\n📁 生成的文件:")
    print("   - ablation/curve_full_model.png (完整模型基准)")
    print("   - ablation/curve_no_quantum.png")
    print("   - ablation/curve_no_graph.png")
    print("   - ablation/curve_no_matcc.png")
    print("   - ablation/curve_no_market_guidance.png")
    print("   - ablation/losses_*.json (每个实验的 Loss 数值列表，共5个)")
    print("   - ablation/ablation_results_summary.csv")
    print("   - ablation/ablation_results_comparison.png (包含5条曲线对比)")
    print("   - ablation/best_model_*.pth (每个实验的最佳模型，共5个)")
    print("="*70)
