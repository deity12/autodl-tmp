# -*- coding: utf-8 -*-
"""
QL-MATCC-GNN 模型训练脚本（对应论文完整架构：LLM 图谱 + Quantum-RWKV + MATCC + GNN）
========================================================================
相对 train_autodl.py 的修改：
    1. 使用 QL_MATCC_GNN_Model（融合 GAT 图神经网络）
    2. 加载邻接矩阵 Graph_Adjacency.npy（由 dataProcessed/build_graph.py 生成）
    3. 前向时传入 node_indices=None（退化模式：batch 内无图结构时 GNN 仅自环）
    4. 模型与曲线分别保存为 best_model_gnn.pth、training_curve_gnn.png，避免覆盖原版

运行顺序建议：
    1. 先运行 dataProcessed/build_graph.py 生成 Graph_Adjacency.npy
    2. 再运行本脚本：python model/train_gnn.py
"""

import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

# ================= 1. 环境与路径 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'dataProcessed'))
sys.path.append(current_dir)

# 邻接矩阵路径（与 build_graph 输出一致）
GRAPH_PATH = os.path.join(parent_dir, 'data', 'processed', 'Graph_Adjacency.npy')

try:
    from dataset import FinancialDataset
    from model_gnn import QL_MATCC_GNN_Model
    print("✅ 成功导入 dataset、model_gnn 模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    exit(1)

# ================= 2. 超参数（针对 RTX 3090 48GB 优化）=================
CONFIG = {
    'csv_path': os.path.join(parent_dir, 'data', 'processed', 'Final_Model_Data.csv'),
    'input_dim': 8,
    # 【显存优化】模型维度：48GB 显存允许更大的嵌入维度，提升模型表达能力
    'n_embd': 512,  # 可尝试 768 或 1024，但需配合 batch_size 调整
    'n_layers': 4,
    'n_qubits': 4,
    # 【显存优化】GNN：图嵌入维度（GAT 的 N×N×2*gnn_embd）
    # 48GB 显存允许适当增大，提升图特征表达能力
    'gnn_embd': 128,  # 从 64 提升到 128，充分利用显存
    'seq_len': 30,
    # GNN 的 GAT 层在 batch 内做邻居聚合，理论上同一 batch 中会混合不同时刻的样本。
    # 本项目在工程上采用“较大 batch size”（例如 512）的折中策略：
    #   1）利用大显存，将 batch 拉大，使得每个 batch 中来自同一时间段、同一行业的样本占比较高；
    #   2）在实践中，这是时空图神经网络常用的 Sampled Batch 近似训练方式，可在可接受的时间错位下获得稳定收益。
    # 这一点会在论文实验设计部分进行说明。
    # 【关键优化】Batch Size：RTX 3090 48GB 显存可支持更大的 batch
    # 大 batch 的优势：
    #   1）提高训练稳定性，梯度估计更准确
    #   2）充分利用 GPU 并行计算能力，加速训练
    #   3）对于 GNN，大 batch 中来自同一时间段、相关股票的样本占比更高，减少时间错位噪声
    #   4）这是时空图神经网络常用的 Sampled Batch 近似训练方式
    'batch_size': 3072,  # 从 512 提升到 1024，充分利用 48GB 显存
    'epochs': 10,
    'lr': 1e-4,
    'early_stop_patience': 3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # 【数据加载优化】num_workers：匹配 CPU 核心数（12核），最大化数据加载并行度
    # 更多 worker 可以提前准备好数据，避免 GPU 等待 CPU 数据预处理
    'num_workers': 12,  # 从 8 提升到 12，匹配 CPU 核心数
    # 【数据预取优化】prefetch_factor：每个 worker 预取的 batch 数量
    # 增大此值可以进一步减少 GPU 等待时间，但会占用更多内存
    'prefetch_factor': 8,  # 从 4 提升到 8，更激进的数据预取
    # 【混合精度训练】启用 AMP 可以：
    #   1）减少约 50% 的显存占用（经典模块用 FP16）
    #   2）加速训练（RTX 3090 的 Tensor Core 对 FP16 有硬件加速）
    #   3）量子模块保持 FP32 以确保数值稳定性
    'use_amp': True,  # 启用混合精度训练
    # 【编译优化】启用 torch.compile 可以进一步加速（PyTorch 2.0+）
    # 但首次运行需要编译时间，且可能与某些自定义操作不兼容
    'use_compile': False,  # 可选：启用 JIT 编译加速（需 PyTorch 2.0+）
}


def main():
    print(f">>> Training on device: {CONFIG['device']}")
    if CONFIG['device'] == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        print(f"   GPU: {gpu_name}")
        print(f"   显存: {gpu_memory:.1f} GB")
        print(f"   Batch Size: {CONFIG['batch_size']} (充分利用 {gpu_memory:.1f}GB 显存)")
        print(f"   DataLoader Workers: {CONFIG['num_workers']} (匹配 CPU 核心数)")

    # ================= 3. 加载邻接矩阵 =================
    if os.path.exists(GRAPH_PATH):
        adj_matrix = np.load(GRAPH_PATH)
        print(f">>> 加载图谱: {GRAPH_PATH}, 形状: {adj_matrix.shape}")
    else:
        # 若未生成图谱，用单位阵兜底（GNN 退化为仅自环）
        df_t = pd.read_csv(CONFIG['csv_path'], usecols=['Ticker'])
        num_nodes = int(df_t['Ticker'].nunique())
        adj_matrix = np.eye(num_nodes, dtype=np.float32)
        print(f">>> 未找到 {GRAPH_PATH}，使用单位阵 (num_nodes={num_nodes})。建议先运行: python dataProcessed/build_graph.py")

    num_nodes = adj_matrix.shape[0]

    # ================= 4. 数据加载 =================
    print("\n>>> Loading Datasets...")
    try:
        train_dataset = FinancialDataset(CONFIG['csv_path'], seq_len=CONFIG['seq_len'], mode='train')
        test_dataset = FinancialDataset(CONFIG['csv_path'], seq_len=CONFIG['seq_len'], mode='test', scaler=train_dataset.scaler)
        print(f"   Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG.get('num_workers', 8),
        pin_memory=True,
        prefetch_factor=CONFIG.get('prefetch_factor', 4),
        persistent_workers=True if CONFIG.get('num_workers', 0) > 0 else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG.get('num_workers', 8),
        pin_memory=True,
        prefetch_factor=CONFIG.get('prefetch_factor', 4),
        persistent_workers=True if CONFIG.get('num_workers', 0) > 0 else False,
    )

    # ================= 5. 模型初始化（QL-MATCC-GNN）=================
    print("\n>>> Initializing QL_MATCC_GNN_Model...")
    model = QL_MATCC_GNN_Model(
        input_dim=CONFIG['input_dim'],
        n_embd=CONFIG['n_embd'],
        n_layers=CONFIG['n_layers'],
        n_qubits=CONFIG['n_qubits'],
        num_nodes=num_nodes,
        adj_matrix=adj_matrix,
        gnn_embd=CONFIG.get('gnn_embd', 64),
    ).to(CONFIG['device'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)
    
    # 【混合精度训练】初始化 GradScaler（用于 FP16 训练的梯度缩放）
    # AMP 会自动将经典模块转换为 FP16，量子模块保持 FP32
    use_amp = CONFIG.get('use_amp', False)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("   ✅ 已启用混合精度训练 (AMP)：经典模块使用 FP16，量子模块保持 FP32")
    
    # 【JIT 编译优化】可选：使用 torch.compile 加速模型（PyTorch 2.0+）
    # 注意：首次运行需要编译时间，且可能与某些自定义操作（如量子线路）不兼容
    if CONFIG.get('use_compile', False) and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("   ✅ 已启用 torch.compile 加速")
        except Exception as e:
            print(f"   ⚠️ torch.compile 启用失败: {e}，继续使用未编译版本")

    # ================= 6. 训练循环 =================
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_metrics_epoch = None
    early_stop_counter = 0
    early_stop_patience = CONFIG['early_stop_patience']

    print("\n>>> Start Training (QL-MATCC-GNN)...")
    print(">>> Dataset 已提供 node_indices，将用其从邻接矩阵切片 batch 内子图做 GNN 聚合。")
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
            # Dataset 已返回 node_indices (B,)，用于从全图邻接矩阵切片 batch 内子图
            node_indices = batch.get('node_indices')
            if node_indices is not None:
                node_indices = node_indices.to(CONFIG['device'], non_blocking=True)

            optimizer.zero_grad()
            
            # 【混合精度训练】使用 autocast 上下文管理器
            # 在 autocast 内的操作会自动选择 FP16/FP32（量子模块会保持 FP32）
            if use_amp:
                with torch.cuda.amp.autocast():
                    preds = model(x, vol, node_indices=node_indices)
                    loss = criterion(preds, y)
                # 使用 scaler 进行反向传播和梯度缩放（防止 FP16 下梯度下溢）
                scaler.scale(loss).backward()
                # 梯度裁剪（在 scaler 缩放后的梯度上进行）
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()  # 更新 scaler 的缩放因子
            else:
                # 标准 FP32 训练流程
                preds = model(x, vol, node_indices=node_indices)
                loss = criterion(preds, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n❌ NaN/Inf at batch {batch_idx}")
                return

            epoch_train_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix(loss=f'{loss.item():.6f}', avg=f'{epoch_train_loss/num_batches:.6f}')

        avg_train = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train)

        # ---------- 验证（收集预测值和真实值用于计算指标）----------
        model.eval()
        epoch_val = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                x = batch['x'].to(CONFIG['device'], non_blocking=True)
                y = batch['y'].to(CONFIG['device'], non_blocking=True)
                vol = batch['vol'].to(CONFIG['device'], non_blocking=True)
                node_indices = batch.get('node_indices')
                if node_indices is not None:
                    node_indices = node_indices.to(CONFIG['device'], non_blocking=True)
                
                # 【混合精度验证】验证时也使用 autocast 以保持一致性
                if use_amp:
                    with torch.cuda.amp.autocast():
                        preds = model(x, vol, node_indices=node_indices)
                else:
                    preds = model(x, vol, node_indices=node_indices)
                
                epoch_val += criterion(preds, y).item()
                
                # 收集预测值和真实值（用于计算完整指标）
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        avg_val = epoch_val / len(test_loader)
        val_losses.append(avg_val)
        
        # 计算完整评估指标（仅在最佳epoch时计算，节省时间）
        if avg_val < best_val_loss:
            all_preds_np = np.concatenate(all_preds, axis=0)
            all_targets_np = np.concatenate(all_targets, axis=0)
            
            # 计算指标
            y_true = all_targets_np.flatten()
            y_pred = all_preds_np.flatten()
            
            # ========== 1. 统计误差类 ==========
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            # MAPE
            mask = np.abs(y_true) > 1e-8
            if np.sum(mask) > 0:
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                mape = None
            
            # ========== 2. 方向预测类 ==========
            true_direction = np.sign(y_true)
            pred_direction = np.sign(y_pred)
            directional_accuracy = np.mean(true_direction == pred_direction)
            
            # ========== 3. 量化投资类 ==========
            # IC (Information Coefficient) - Pearson 相关系数
            try:
                ic, ic_pvalue = pearsonr(y_pred, y_true)
                ic = float(ic)
            except:
                ic = None
                ic_pvalue = None
            
            # RankIC (Rank Information Coefficient) - Spearman 秩相关系数
            try:
                rank_ic, rank_ic_pvalue = spearmanr(y_pred, y_true)
                rank_ic = float(rank_ic)
            except:
                rank_ic = None
                rank_ic_pvalue = None
            
            # 传统相关系数（兼容性）
            try:
                correlation = np.corrcoef(y_true, y_pred)[0, 1]
                correlation = float(correlation)
            except:
                correlation = None
            
            best_metrics = {
                # 统计误差类
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'mape': float(mape) if mape is not None else None,
                
                # 方向预测类
                'directional_accuracy': float(directional_accuracy),
                
                # 量化投资类
                'ic': ic,
                'ic_pvalue': float(ic_pvalue) if ic_pvalue is not None else None,
                'rank_ic': rank_ic,
                'rank_ic_pvalue': float(rank_ic_pvalue) if rank_ic_pvalue is not None else None,
                
                # 兼容性指标
                'correlation': correlation,
            }
        else:
            best_metrics = None

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val)
        cur_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}: Train={avg_train:.6f}, Val={avg_val:.6f}, lr={cur_lr:.2e}" + (" ⬇️" if cur_lr != old_lr else ""))

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_metrics_epoch = best_metrics  # 保存最佳epoch的指标
            save_path = os.path.join(current_dir, 'best_model_gnn.pth')
            torch.save(model.state_dict(), save_path)
            if best_metrics:
                print(f"  🌟 Best model saved: {save_path}")
                print(f"     Metrics: R²={best_metrics['r2']:.4f}, MAE={best_metrics['mae']:.6f}, "
                      f"DirAcc={best_metrics['directional_accuracy']:.2%}, Corr={best_metrics['correlation']:.4f}")
            else:
                print(f"  🌟 Best model saved: {save_path}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            print(f"\n🛑 Early stopping (best val loss: {best_val_loss:.6f})")
            break
        print("-" * 60)

    # ================= 7. 曲线与总结 =================
    curve_path = os.path.join(current_dir, 'training_curve_gnn.png')
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train Loss', lw=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Val Loss', lw=2)
    if val_losses:
        be = val_losses.index(best_val_loss) + 1
        plt.plot(be, best_val_loss, 'g*', markersize=14, label=f'Best (Epoch {be})')
    plt.title('QL-MATCC-GNN Training', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f">>> 训练曲线已保存: {curve_path}")
    
    # ================= 8. 最终评估（在最佳模型上计算完整指标）=================
    # 如果最佳epoch时没有计算指标，现在重新计算
    if best_metrics_epoch is None:
        print(">>> 重新计算最终评估指标...")
        model.load_state_dict(torch.load(os.path.join(current_dir, 'best_model_gnn.pth')))
        model.eval()
        all_preds_final = []
        all_targets_final = []
        
        with torch.no_grad():
            for batch in test_loader:
                x = batch['x'].to(CONFIG['device'], non_blocking=True)
                y = batch['y'].to(CONFIG['device'], non_blocking=True)
                vol = batch['vol'].to(CONFIG['device'], non_blocking=True)
                node_indices = batch.get('node_indices')
                if node_indices is not None:
                    node_indices = node_indices.to(CONFIG['device'], non_blocking=True)
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        preds = model(x, vol, node_indices=node_indices)
                else:
                    preds = model(x, vol, node_indices=node_indices)
                
                all_preds_final.append(preds.cpu().numpy())
                all_targets_final.append(y.cpu().numpy())
        
        all_preds_final_np = np.concatenate(all_preds_final, axis=0)
        all_targets_final_np = np.concatenate(all_targets_final, axis=0)
        
        y_true = all_targets_final_np.flatten()
        y_pred = all_preds_final_np.flatten()
        
        # ========== 1. 统计误差类 ==========
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        mask = np.abs(y_true) > 1e-8
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = None
        
        # ========== 2. 方向预测类 ==========
        true_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)
        directional_accuracy = np.mean(true_direction == pred_direction)
        
        # ========== 3. 量化投资类 ==========
        # IC (Information Coefficient)
        try:
            ic, ic_pvalue = pearsonr(y_pred, y_true)
            ic = float(ic)
        except:
            ic = None
            ic_pvalue = None
        
        # RankIC (Rank Information Coefficient)
        try:
            rank_ic, rank_ic_pvalue = spearmanr(y_pred, y_true)
            rank_ic = float(rank_ic)
        except:
            rank_ic = None
            rank_ic_pvalue = None
        
        # 传统相关系数（兼容性）
        try:
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            correlation = float(correlation)
        except:
            correlation = None
        
        best_metrics_epoch = {
            # 统计误差类
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape) if mape is not None else None,
            
            # 方向预测类
            'directional_accuracy': float(directional_accuracy),
            
            # 量化投资类
            'ic': ic,
            'ic_pvalue': float(ic_pvalue) if ic_pvalue is not None else None,
            'rank_ic': rank_ic,
            'rank_ic_pvalue': float(rank_ic_pvalue) if rank_ic_pvalue is not None else None,
            
            # 兼容性指标
            'correlation': correlation,
        }
    
    # ================= 9. 保存 Loss 数值列表和评估指标 =================
    loss_data_path = os.path.join(current_dir, 'training_losses_gnn.json')
    loss_data = {
        'experiment_name': 'full_model_gnn',
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'best_epoch': val_losses.index(best_val_loss) + 1 if val_losses else 0,
        'total_epochs': len(train_losses),
        'metrics': best_metrics_epoch,  # 添加完整评估指标
        'config': {
            'batch_size': CONFIG['batch_size'],
            'lr': CONFIG['lr'],
            'epochs': CONFIG['epochs'],
            'n_embd': CONFIG['n_embd'],
            'n_layers': CONFIG['n_layers'],
        }
    }
    with open(loss_data_path, 'w') as f:
        json.dump(loss_data, f, indent=2)
    print(f">>> Loss 数值列表和评估指标已保存: {loss_data_path}")

    print("\n" + "=" * 60)
    print(">>> QL-MATCC-GNN 训练结束")
    print(f"    Best Val Loss: {best_val_loss:.6f}")
    if best_metrics_epoch:
        print("\n    📊 评估指标:")
        print(f"    【统计误差类】")
        print(f"      R² Score: {best_metrics_epoch['r2']:.4f}")
        print(f"      MAE: {best_metrics_epoch['mae']:.6f}")
        print(f"      RMSE: {best_metrics_epoch['rmse']:.6f}")
        print(f"    【方向预测类】")
        print(f"      Directional Accuracy: {best_metrics_epoch['directional_accuracy']:.2%}")
        print(f"    【量化投资类】")
        if best_metrics_epoch.get('ic') is not None:
            print(f"      IC (Information Coefficient): {best_metrics_epoch['ic']:.4f}")
        if best_metrics_epoch.get('rank_ic') is not None:
            print(f"      RankIC (Rank Information Coefficient): {best_metrics_epoch['rank_ic']:.4f}")
    print(f"\n    Model: best_model_gnn.pth")
    print(f"    Loss Data: training_losses_gnn.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
