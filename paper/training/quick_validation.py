# -*- coding: utf-8 -*-
"""
快速验证脚本：测试全量模型是否优于消融模型
========================================================================
目的：在投入大量计算资源前，快速验证模型设计的有效性

配置：
- 5 epochs（快速收敛测试）
- 小 batch size（512）
- 只使用部分数据（前20%）
- 测试 4 个配置：Full, w/o Quantum, w/o Decomposition, w/o Graph

预期结果：
- Full Model 应该在 IC/RankIC 上最优
- 如果不是，需要调整超参数或模型设计
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict
from tqdm import tqdm
import json

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from dataProcessed.dataset import FinancialDataset
from models.gnn_model import QL_MATCC_GNN_Model

# 验证级别配置
VALIDATION_LEVELS = {
    'ultra_fast': {
        'data_fraction': 0.1,
        'epochs': 3,
        'description': '超快速验证（10%数据，3轮）- 仅用于快速排查明显问题'
    },
    'fast': {
        'data_fraction': 0.2,
        'epochs': 5,
        'description': '快速验证（20%数据，5轮）- 初步验证模型设计'
    },
    'medium': {
        'data_fraction': 0.5,
        'epochs': 10,
        'description': '中等验证（50%数据，10轮）- 较可靠的验证（推荐）'
    },
    'thorough': {
        'data_fraction': 1.0,
        'epochs': 15,
        'description': '完整验证（100%数据，15轮）- 最可靠但耗时较长'
    }
}

# 基础配置
BASE_CONFIG = {
    'csv_path': os.path.join(parent_dir, 'data', 'processed', 'Final_Model_Data.csv'),
    'graph_path': os.path.join(parent_dir, 'data', 'processed', 'Graph_Adjacency.npy'),
    'input_dim': 8,
    'n_embd': 256,
    'n_layers': 3,
    'n_qubits': 8,
    'gnn_embd': 64,
    'dropout': 0.2,
    'seq_len': 30,
    'pred_len': 1,
    'batch_size': 512,
    'lr': 3e-4,
}

def quick_train_and_eval(model, train_loader, test_loader, config, device):
    """快速训练并评估"""
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)

    # 训练
    model.train()
    for epoch in range(config['epochs']):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            node_indices = batch['node_indices'].to(device)

            optimizer.zero_grad()
            pred = model(x, node_indices)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1} Loss: {avg_loss:.6f}")

    # 评估
    model.eval()
    all_preds = []
    all_targets = []
    all_dates = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            node_indices = batch['node_indices'].to(device)
            dates = batch.get('target_date', [None] * len(y))

            pred = model(x, node_indices)

            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())
            all_dates.extend(dates)

    # 计算指标
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 整体 IC/RankIC
    ic, _ = pearsonr(all_preds, all_targets)
    rank_ic, _ = spearmanr(all_preds, all_targets)

    # 按日期计算 IC/RankIC（更准确）
    date_ics = []
    date_rank_ics = []

    df = pd.DataFrame({
        'pred': all_preds,
        'target': all_targets,
        'date': all_dates
    })

    for date, group in df.groupby('date'):
        if len(group) > 5:  # 至少5个样本
            try:
                ic_d, _ = pearsonr(group['pred'], group['target'])
                rank_ic_d, _ = spearmanr(group['pred'], group['target'])
                if not np.isnan(ic_d):
                    date_ics.append(ic_d)
                if not np.isnan(rank_ic_d):
                    date_rank_ics.append(rank_ic_d)
            except:
                pass

    avg_ic = np.mean(date_ics) if date_ics else ic
    avg_rank_ic = np.mean(date_rank_ics) if date_rank_ics else rank_ic

    return {
        'IC': avg_ic,
        'RankIC': avg_rank_ic,
        'IC_overall': ic,
        'RankIC_overall': rank_ic,
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description='快速验证：全量模型 vs 消融模型')
    parser.add_argument('--level', type=str, default='medium',
                       choices=['ultra_fast', 'fast', 'medium', 'thorough'],
                       help='验证级别 (默认: medium)')
    args = parser.parse_args()

    # 选择验证级别
    level_config = VALIDATION_LEVELS[args.level]
    config = {**BASE_CONFIG, **level_config}

    print("=" * 70)
    print("快速验证：全量模型 vs 消融模型")
    print("=" * 70)
    print(f"\n验证级别: {args.level}")
    print(f"说明: {level_config['description']}")
    print(f"数据比例: {level_config['data_fraction']*100:.0f}%")
    print(f"训练轮数: {level_config['epochs']}")

    if args.level in ['ultra_fast', 'fast']:
        print("\n⚠️  注意: 当前级别使用较少数据，结果仅供参考")
        print("   复杂模型可能需要更多数据才能显示优势")
        print("   建议使用 --level medium 获得更可靠的结果")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 加载数据
    print("\n[1/5] 加载数据...")
    train_dataset = FinancialDataset(
        config['csv_path'],
        seq_len=config['seq_len'],
        pred_len=config['pred_len'],
        mode='train'
    )
    test_dataset = FinancialDataset(
        config['csv_path'],
        seq_len=config['seq_len'],
        pred_len=config['pred_len'],
        mode='test',
        scaler=train_dataset.scaler,
        vol_stats=train_dataset.vol_stats
    )

    # 只使用部分数据
    train_size = int(len(train_dataset) * config['data_fraction'])
    test_size = int(len(test_dataset) * config['data_fraction'])

    train_subset = Subset(train_dataset, range(train_size))
    test_subset = Subset(test_dataset, range(test_size))

    print(f"  训练样本: {len(train_subset)} (原始: {len(train_dataset)})")
    print(f"  测试样本: {len(test_subset)} (原始: {len(test_dataset)})")

    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=config['batch_size'], shuffle=False)

    # 加载图谱
    adj_matrix = np.load(config['graph_path'])
    adj_tensor = torch.FloatTensor(adj_matrix).to(device)

    # 测试配置
    configs = [
        {'name': 'Full Model', 'use_quantum': True, 'use_decomposition': True, 'use_graph': True},
        {'name': 'w/o Quantum', 'use_quantum': False, 'use_decomposition': True, 'use_graph': True},
        {'name': 'w/o Decomposition', 'use_quantum': True, 'use_decomposition': False, 'use_graph': True},
        {'name': 'w/o Graph', 'use_quantum': True, 'use_decomposition': True, 'use_graph': False},
    ]

    results = {}

    for i, cfg in enumerate(configs):
        print(f"\n[{i+2}/5] 测试配置: {cfg['name']}")
        print("-" * 70)

        # 创建模型
        model = QL_MATCC_GNN_Model(
            input_dim=config['input_dim'],
            n_embd=config['n_embd'],
            n_layers=config['n_layers'],
            n_qubits=config['n_qubits'],
            dropout=config['dropout'],
            adj_matrix=adj_tensor,
            gnn_embd=config['gnn_embd'],
            use_quantum=cfg['use_quantum'],
            use_decomposition=cfg['use_decomposition'],
            use_graph=cfg['use_graph'],
            vol_threshold=train_dataset.vol_stats['p70']
        ).to(device)

        # 训练并评估
        metrics = quick_train_and_eval(model, train_loader, test_loader, config, device)
        results[cfg['name']] = metrics

        print(f"\n  结果:")
        print(f"    IC (按日期): {metrics['IC']:.4f}")
        print(f"    RankIC (按日期): {metrics['RankIC']:.4f}")
        print(f"    IC (整体): {metrics['IC_overall']:.4f}")
        print(f"    RankIC (整体): {metrics['RankIC_overall']:.4f}")

    # 总结
    print("\n" + "=" * 70)
    print("验证结果总结")
    print("=" * 70)

    df_results = pd.DataFrame(results).T
    df_results = df_results.sort_values('RankIC', ascending=False)

    print("\n按 RankIC 排序:")
    print(df_results.to_string())

    # 判断
    best_model = df_results.index[0]
    full_rank = df_results.index.tolist().index('Full Model') + 1

    print(f"\n最佳模型: {best_model}")
    print(f"Full Model 排名: {full_rank}/4")

    if best_model == 'Full Model':
        print("\n✅ 验证通过！Full Model 表现最优，可以进行完整实验。")
    else:
        print(f"\n⚠️ 警告！{best_model} 优于 Full Model")
        print("建议:")
        print("1. 调整 Full Model 的超参数（学习率、dropout、模型大小）")
        print("2. 增加训练轮数（当前只有5轮）")
        print("3. 检查模块之间是否存在冲突")
        print("4. 考虑移除表现不佳的模块")

    # 保存结果
    output_path = os.path.join(parent_dir, 'outputs', 'quick_validation_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'config': QUICK_CONFIG,
            'results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
            'best_model': best_model,
            'full_model_rank': full_rank
        }, f, indent=2)

    print(f"\n结果已保存至: {output_path}")

if __name__ == "__main__":
    main()
