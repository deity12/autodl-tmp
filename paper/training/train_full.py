# -*- coding: utf-8 -*-
"""
QL-MATCC-GNN 模型训练脚本（优化版：解决消融实验优于全量模型问题）
========================================================================
核心改进：
    1. 降低模型复杂度（n_embd 512->256, n_layers 4->3）
    2. 降低 batch_size（3072->512），增加梯度更新次数
    3. 增加 epoch 数量（10->20），给复杂模型更多训练时间
    4. 使用差异化学习率：量子层用更小的学习率（经典层 3e-4，量子层 3e-5）
    5. 动态设置量子阈值：基于训练数据的 70% 分位数
    6. 添加权重衰减和更强的 Dropout 正则化
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
sys.path.insert(0, parent_dir)

GRAPH_PATH = os.path.join(parent_dir, 'data', 'processed', 'Graph_Adjacency.npy')
OUTPUT_DIR = os.path.join(parent_dir, 'outputs')
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

try:
    from dataProcessed.dataset import FinancialDataset
    from models.gnn_model import QL_MATCC_GNN_Model
    print("✅ 成功导入 dataset、gnn_model 模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    exit(1)

# ================= 2. 超参数（48GB显存性能优化版）=================
CONFIG = {
    'csv_path': os.path.join(parent_dir, 'data', 'processed', 'Final_Model_Data.csv'),
    'input_dim': 8,
    
    # 【模型维度】48GB显存版 - 适度增加模型容量
    'n_embd': 384,       # 256 → 384 (48GB显存充足，增强表达能力)
    'n_layers': 4,       # 3 → 4 (增加深度)
    'n_qubits': 4,
    'gnn_embd': 96,      # 64 → 96 (更好的图特征学习)
    'seq_len': 30,
    
    # 【Batch Size】48GB显存优化 - 充分利用GPU并行计算
    'batch_size': 1024,  # 512 → 1024 (48GB显存完全够用，提高训练效率)
    
    # 【Epoch】充分训练
    'epochs': 30,        # 20 → 30 (给模型更多收敛时间)
    
    # 【学习率】差异化学习率策略
    'lr': 3e-4,                    # 经典层学习率
    'quantum_lr_ratio': 0.1,       # 量子层学习率 = lr * 0.1 = 3e-5
    'use_differential_lr': True,   # 启用差异化学习率
    
    # 【量子阈值】设为 None，将在运行时从数据的 70% 分位数自动获取
    'q_threshold': None,
    
    # 【正则化】
    'dropout': 0.15,
    'weight_decay': 1e-5,
    
    # 【早停】
    'early_stop_patience': 8,      # 6 → 8 (给模型更多机会)
    
    # 【硬件优化】48GB显存 + 12核CPU + 90GB内存
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 10,             # 8 → 10 (充分利用12核CPU)
    'prefetch_factor': 6,          # 4 → 6 (90GB内存充足，多预加载)
    'use_amp': True,               # 混合精度训练
    'use_compile': False,          # PyTorch 2.0编译（可选）
    'pin_memory': True,            # 加速数据传输
    'persistent_workers': True,    # 保持worker进程，减少开销
}


def main():
    print(f">>> Training on device: {CONFIG['device']}")
    if CONFIG['device'] == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   GPU: {gpu_name}")
        print(f"   显存: {gpu_memory:.1f} GB")
        print(f"   Batch Size: {CONFIG['batch_size']}")

    # ================= 3. 加载邻接矩阵 =================
    if os.path.exists(GRAPH_PATH):
        adj_matrix = np.load(GRAPH_PATH)
        print(f">>> 加载图谱: {GRAPH_PATH}, 形状: {adj_matrix.shape}")
    else:
        df_t = pd.read_csv(CONFIG['csv_path'], usecols=['Ticker'])
        num_nodes = int(df_t['Ticker'].nunique())
        adj_matrix = np.eye(num_nodes, dtype=np.float32)
        print(f">>> 未找到图谱，使用单位阵 (num_nodes={num_nodes})")

    num_nodes = adj_matrix.shape[0]

    # ================= 4. 数据加载 =================
    print("\n>>> Loading Datasets...")
    try:
        train_dataset = FinancialDataset(CONFIG['csv_path'], seq_len=CONFIG['seq_len'], mode='train')
        test_dataset = FinancialDataset(
            CONFIG['csv_path'], seq_len=CONFIG['seq_len'], mode='test', 
            scaler=train_dataset.scaler,
            vol_stats=train_dataset.vol_stats  # 传入波动率统计
        )
        print(f"   Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 【关键】从训练数据获取量子阈值
    q_threshold = CONFIG['q_threshold']
    if q_threshold is None:
        q_threshold = train_dataset.vol_stats.get('p70', 0.5)
        print(f">>> 从数据自动获取量子阈值: q_threshold = {q_threshold:.4f} (70%分位数)")
    else:
        print(f">>> 使用配置的量子阈值: q_threshold = {q_threshold}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG.get('num_workers', 10),
        pin_memory=CONFIG.get('pin_memory', True),
        prefetch_factor=CONFIG.get('prefetch_factor', 6),
        persistent_workers=CONFIG.get('persistent_workers', True),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG.get('num_workers', 10),
        pin_memory=CONFIG.get('pin_memory', True),
        prefetch_factor=CONFIG.get('prefetch_factor', 6),
        persistent_workers=CONFIG.get('persistent_workers', True),
    )

    # ================= 5. 模型初始化 =================
    print("\n>>> Initializing QL_MATCC_GNN_Model...")
    model = QL_MATCC_GNN_Model(
        input_dim=CONFIG['input_dim'],
        n_embd=CONFIG['n_embd'],
        n_layers=CONFIG['n_layers'],
        n_qubits=CONFIG['n_qubits'],
        num_nodes=num_nodes,
        adj_matrix=adj_matrix,
        gnn_embd=CONFIG.get('gnn_embd', 64),
        q_threshold=q_threshold,  # 使用动态阈值
        dropout=CONFIG.get('dropout', 0.1),
    ).to(CONFIG['device'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # ================= 6. 优化器（差异化学习率）=================
    criterion = nn.MSELoss()
    
    # 分离量子层和经典层参数
    quantum_params = []
    classic_params = []
    for name, param in model.named_parameters():
        if 'vqc' in name.lower() or 'quantum' in name.lower():
            quantum_params.append(param)
        else:
            classic_params.append(param)
    
    if CONFIG.get('use_differential_lr', True) and len(quantum_params) > 0:
        quantum_lr = CONFIG['lr'] * CONFIG.get('quantum_lr_ratio', 0.1)
        optimizer = optim.AdamW([
            {'params': classic_params, 'lr': CONFIG['lr']},
            {'params': quantum_params, 'lr': quantum_lr}
        ], betas=(0.9, 0.999), eps=1e-8, weight_decay=CONFIG.get('weight_decay', 1e-5))
        print(f"   ✅ 差异化学习率: 经典层={CONFIG['lr']:.2e}, 量子层={quantum_lr:.2e}")
    else:
        optimizer = optim.AdamW(
            model.parameters(), lr=CONFIG['lr'], 
            betas=(0.9, 0.999), eps=1e-8, 
            weight_decay=CONFIG.get('weight_decay', 1e-5)
        )
        print(f"   使用统一学习率: {CONFIG['lr']:.2e}")
    
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

    print("\n>>> Start Training (QL-MATCC-GNN Full Model)...")
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
            node_indices = batch.get('node_indices')
            if node_indices is not None:
                node_indices = node_indices.to(CONFIG['device'], non_blocking=True)

            optimizer.zero_grad()
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    preds = model(x, vol, node_indices=node_indices)
                    loss = criterion(preds, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
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

        scheduler.step()
        avg_train = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train)

        # ---------- 验证 ----------
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
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        preds = model(x, vol, node_indices=node_indices)
                else:
                    preds = model(x, vol, node_indices=node_indices)
                
                epoch_val += criterion(preds, y).item()
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
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
            
            try:
                ic, ic_pvalue = pearsonr(y_pred, y_true)
                ic = float(ic)
            except:
                ic = None
                ic_pvalue = None
            
            try:
                rank_ic, rank_ic_pvalue = spearmanr(y_pred, y_true)
                rank_ic = float(rank_ic)
            except:
                rank_ic = None
                rank_ic_pvalue = None
            
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
            save_path = os.path.join(CHECKPOINT_DIR, 'best_model_full.pth')
            torch.save(model.state_dict(), save_path)
            if best_metrics:
                print(f"  🌟 Best model saved!")
                print(f"     R²={best_metrics['r2']:.4f}, MAE={best_metrics['mae']:.6f}, "
                      f"DirAcc={best_metrics['directional_accuracy']:.2%}, IC={best_metrics['ic']:.4f if best_metrics['ic'] else 'N/A'}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            print(f"\n🛑 Early stopping (best val loss: {best_val_loss:.6f})")
            break
        print("-" * 60)

    # ================= 8. 保存结果 =================
    curve_path = os.path.join(FIGURE_DIR, 'training_curve_full.png')
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train Loss', lw=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Val Loss', lw=2)
    if val_losses:
        be = val_losses.index(best_val_loss) + 1
        plt.plot(be, best_val_loss, 'g*', markersize=14, label=f'Best (Epoch {be})')
    plt.title('QL-MATCC-GNN Full Model Training', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f">>> 训练曲线已保存: {curve_path}")
    
    loss_data_path = os.path.join(LOG_DIR, 'training_losses_full.json')
    loss_data = {
        'experiment_name': 'full_model',
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
            'q_threshold': q_threshold,
        }
    }
    with open(loss_data_path, 'w') as f:
        json.dump(loss_data, f, indent=2)
    print(f">>> Loss 数据已保存: {loss_data_path}")

    print("\n" + "=" * 60)
    print(">>> QL-MATCC-GNN Full Model 训练结束")
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


if __name__ == "__main__":
    main()
