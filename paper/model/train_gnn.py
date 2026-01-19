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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# ================= 2. 超参数 =================
CONFIG = {
    'csv_path': os.path.join(parent_dir, 'data', 'processed', 'Final_Model_Data.csv'),
    'input_dim': 8,
    'n_embd': 512,
    'n_layers': 4,
    'n_qubits': 4,
    # GNN：图嵌入用较小维度可减轻显存（GAT 的 N×N×2*gnn_embd）
    'gnn_embd': 64,
    'seq_len': 30,
    # GNN 的 GAT 层有 O(B^2) 显存，较大 n_embd+gnn 时建议适当减小 batch
    'batch_size': 512,
    'epochs': 10,
    'lr': 1e-4,
    'early_stop_patience': 3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 8,
    'prefetch_factor': 4,
}


def main():
    print(f">>> Training on device: {CONFIG['device']}")
    if CONFIG['device'] == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

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
    use_amp = False

    # ================= 6. 训练循环 =================
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
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

        # ---------- 验证 ----------
        model.eval()
        epoch_val = 0.0
        with torch.no_grad():
            for batch in test_loader:
                x = batch['x'].to(CONFIG['device'], non_blocking=True)
                y = batch['y'].to(CONFIG['device'], non_blocking=True)
                vol = batch['vol'].to(CONFIG['device'], non_blocking=True)
                node_indices = batch.get('node_indices')
                if node_indices is not None:
                    node_indices = node_indices.to(CONFIG['device'], non_blocking=True)
                preds = model(x, vol, node_indices=node_indices)
                epoch_val += criterion(preds, y).item()
        avg_val = epoch_val / len(test_loader)
        val_losses.append(avg_val)

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val)
        cur_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}: Train={avg_train:.6f}, Val={avg_val:.6f}, lr={cur_lr:.2e}" + (" ⬇️" if cur_lr != old_lr else ""))

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_path = os.path.join(current_dir, 'best_model_gnn.pth')
            torch.save(model.state_dict(), save_path)
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

    print("\n" + "=" * 60)
    print(">>> QL-MATCC-GNN 训练结束")
    print(f"    Best Val Loss: {best_val_loss:.6f}")
    print(f"    Model: best_model_gnn.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()
