# -*- coding: utf-8 -*-
"""
消融实验对比图表生成工具
========================================================================
功能：
    读取已保存的 Loss 数值列表（JSON 文件），生成对比图表。
    无需重新运行实验，直接读取数据即可生成论文用的对比图。

使用场景：
    1. 消融实验已运行完成，想要重新生成对比图表
    2. 想要对比不同时间运行的实验结果
    3. 想要生成不同风格的对比图表用于论文

用法：
    python plot_ablation_comparison.py
    或指定目录：
    python plot_ablation_comparison.py --ablation_dir paper/model/ablation
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_loss_data(ablation_dir):
    """
    从 ablation 目录加载所有实验的 Loss 数据
    """
    loss_files = list(Path(ablation_dir).glob('losses_*.json'))
    if not loss_files:
        print(f"❌ 在 {ablation_dir} 中未找到 losses_*.json 文件")
        return []
    
    results = []
    for loss_file in sorted(loss_files):
        try:
            with open(loss_file, 'r') as f:
                data = json.load(f)
            results.append(data)
            print(f"✅ 加载: {loss_file.name}")
        except Exception as e:
            print(f"⚠️ 加载失败 {loss_file.name}: {e}")
    
    return results


def plot_training_curves(results, save_path):
    """
    绘制所有实验的训练曲线对比图（叠加在一起）
    """
    plt.figure(figsize=(12, 6))
    
    colors = {
        'full_model': '#FFD700',  # 金色
        'no_quantum': '#ff7f0e',  # 橙色
        'no_graph': '#2ca02c',    # 绿色
        'no_matcc': '#d62728',    # 红色
        'no_market_guidance': '#9467bd',  # 紫色
    }
    
    # 按实验名称排序，确保 full_model 在前面
    sorted_results = sorted(results, key=lambda x: (
        0 if x['experiment'] == 'full_model' else 1,
        x['experiment']
    ))
    
    for result in sorted_results:
        exp_name = result['experiment']
        train_losses = result['train_losses']
        val_losses = result['val_losses']
        epochs = range(1, len(train_losses) + 1)
        
        color = colors.get(exp_name, '#1f77b4')
        label_prefix = 'Full Model' if exp_name == 'full_model' else f"w/o {exp_name.replace('no_', '').replace('_', ' ').title()}"
        
        # 训练损失（虚线）
        plt.plot(epochs, train_losses, '--', color=color, alpha=0.6, 
                linewidth=1.5, label=f'{label_prefix} (Train)')
        # 验证损失（实线）
        plt.plot(epochs, val_losses, '-', color=color, alpha=0.9,
                linewidth=2, label=f'{label_prefix} (Val)')
        
        # 标记最佳点
        best_epoch = result['best_epoch']
        best_val = result['best_val_loss']
        plt.plot(best_epoch, best_val, '*', color=color, 
                markersize=12, markeredgecolor='black', markeredgewidth=0.5)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Ablation Study: Training Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, loc='best', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 训练曲线对比图已保存: {save_path}")


def plot_bar_comparison(results, save_path):
    """
    绘制柱状图对比（最佳验证损失）
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 提取数据
    exp_names = [r['experiment'] for r in results]
    best_vals = [r['best_val_loss'] for r in results]
    
    # 为 Full Model 使用特殊颜色
    colors = []
    ablation_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    ablation_idx = 0
    
    display_names = []
    for name in exp_names:
        if name == 'full_model':
            colors.append('#FFD700')  # 金色
            display_names.append('Full Model')
        else:
            if ablation_idx < len(ablation_colors):
                colors.append(ablation_colors[ablation_idx])
                ablation_idx += 1
            else:
                colors.append('#1f77b4')
            display_names.append(f"w/o {name.replace('no_', '').replace('_', ' ').title()}")
    
    # 左图：按原始顺序
    axes[0].bar(display_names, best_vals, color=colors)
    axes[0].set_ylabel('Best Val Loss (MSE)', fontsize=12)
    axes[0].set_title('Ablation Study: Best Validation Loss', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, v in enumerate(best_vals):
        axes[0].text(i, v, f'{v:.6f}', ha='center', va='bottom', fontsize=9)
    
    # 右图：按损失值排序
    sorted_results = sorted(zip(display_names, best_vals, colors), key=lambda x: x[1])
    sorted_names = [x[0] for x in sorted_results]
    sorted_vals = [x[1] for x in sorted_results]
    sorted_colors = [x[2] for x in sorted_results]
    
    axes[1].barh(sorted_names, sorted_vals, color=sorted_colors)
    axes[1].set_xlabel('Best Val Loss (MSE)', fontsize=12)
    axes[1].set_title('Ablation Study: Loss Ranking', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for i, v in enumerate(sorted_vals):
        axes[1].text(v, i, f' {v:.6f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 柱状图对比已保存: {save_path}")


def generate_summary_table(results, save_path):
    """
    生成汇总表格（包含所有评估指标）
    """
    data = []
    for r in results:
        exp_name = 'Full Model' if r['experiment'] == 'full_model' else f"w/o {r['experiment'].replace('no_', '').replace('_', ' ').title()}"
        
        row = {
            'Experiment': exp_name,
            'Best Val Loss (MSE)': f"{r['best_val_loss']:.6f}",
            'Best Epoch': r['best_epoch'],
        }
        
        # 添加评估指标（如果存在）
        if 'metrics' in r and r['metrics']:
            metrics = r['metrics']
            row.update({
                # 统计误差类
                'MSE': f"{metrics.get('mse', 'N/A'):.6f}" if metrics.get('mse') is not None else 'N/A',
                'MAE': f"{metrics.get('mae', 'N/A'):.6f}" if metrics.get('mae') is not None else 'N/A',
                'RMSE': f"{metrics.get('rmse', 'N/A'):.6f}" if metrics.get('rmse') is not None else 'N/A',
                'R²': f"{metrics.get('r2', 'N/A'):.4f}" if metrics.get('r2') is not None else 'N/A',
                'MAPE (%)': f"{metrics.get('mape', 'N/A'):.2f}" if metrics.get('mape') is not None else 'N/A',
                # 方向预测类
                'Dir. Accuracy': f"{metrics.get('directional_accuracy', 'N/A'):.2%}" if metrics.get('directional_accuracy') is not None else 'N/A',
                # 量化投资类
                'IC': f"{metrics.get('ic', 'N/A'):.4f}" if metrics.get('ic') is not None else 'N/A',
                'RankIC': f"{metrics.get('rank_ic', 'N/A'):.4f}" if metrics.get('rank_ic') is not None else 'N/A',
                # 兼容性指标
                'Correlation': f"{metrics.get('correlation', 'N/A'):.4f}" if metrics.get('correlation') is not None else 'N/A',
            })
        else:
            # 如果没有metrics，尝试从顶层获取（兼容旧格式）
            row.update({
                # 统计误差类
                'MSE': f"{r.get('mse', 'N/A'):.6f}" if r.get('mse') is not None else 'N/A',
                'MAE': f"{r.get('mae', 'N/A'):.6f}" if r.get('mae') is not None else 'N/A',
                'RMSE': f"{r.get('rmse', 'N/A'):.6f}" if r.get('rmse') is not None else 'N/A',
                'R²': f"{r.get('r2', 'N/A'):.4f}" if r.get('r2') is not None else 'N/A',
                'MAPE (%)': f"{r.get('mape', 'N/A'):.2f}" if r.get('mape') is not None else 'N/A',
                # 方向预测类
                'Dir. Accuracy': f"{r.get('directional_accuracy', 'N/A'):.2%}" if r.get('directional_accuracy') is not None else 'N/A',
                # 量化投资类
                'IC': f"{r.get('ic', 'N/A'):.4f}" if r.get('ic') is not None else 'N/A',
                'RankIC': f"{r.get('rank_ic', 'N/A'):.4f}" if r.get('rank_ic') is not None else 'N/A',
                # 兼容性指标
                'Correlation': f"{r.get('correlation', 'N/A'):.4f}" if r.get('correlation') is not None else 'N/A',
            })
        
        row.update({
            'Final Train Loss': f"{r['train_losses'][-1]:.6f}" if r['train_losses'] else 'N/A',
            'Final Val Loss': f"{r['val_losses'][-1]:.6f}" if r['val_losses'] else 'N/A',
            'Total Epochs': r['total_epochs'],
        })
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"✅ 汇总表格已保存: {save_path}")
    print("\n" + "="*70)
    print("📊 汇总结果（包含所有评估指标）:")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)


def calculate_improvements(results):
    """
    计算各消融实验相比 Full Model 的性能下降（多指标对比）
    """
    full_model_result = next((r for r in results if r['experiment'] == 'full_model'), None)
    if not full_model_result:
        print("⚠️ 未找到 Full Model 结果，无法计算性能下降")
        return
    
    full_metrics = full_model_result.get('metrics', {})
    if not full_metrics:
        # 尝试从顶层获取
        full_metrics = {k: v for k in ['mse', 'mae', 'rmse', 'r2', 'correlation', 'directional_accuracy', 'ic', 'rank_ic'] 
                       if (k := k) in full_model_result and full_model_result[k] is not None}
    
    print("\n" + "="*70)
    print("📉 各模块贡献分析（相比 Full Model）:")
    print("="*70)
    
    for r in results:
        if r['experiment'] != 'full_model':
            exp_name = f"w/o {r['experiment'].replace('no_', '').replace('_', ' ').title()}"
            ablation_metrics = r.get('metrics', {})
            if not ablation_metrics:
                ablation_metrics = {k: r.get(k) for k in ['mse', 'mae', 'rmse', 'r2', 'correlation', 'directional_accuracy', 'ic', 'rank_ic'] 
                                   if k in r and r[k] is not None}
            
            print(f"\n{exp_name}:")
            
            # MSE对比
            if 'mse' in full_metrics and 'mse' in ablation_metrics:
                increase = ablation_metrics['mse'] - full_metrics['mse']
                increase_pct = (increase / full_metrics['mse']) * 100 if full_metrics['mse'] > 0 else 0
                print(f"  MSE: 增加 {increase:.6f} ({increase_pct:+.2f}%)")
            
            # R²对比
            if 'r2' in full_metrics and 'r2' in ablation_metrics:
                decrease = full_metrics['r2'] - ablation_metrics['r2']
                print(f"  R²: 下降 {decrease:.4f} (从 {full_metrics['r2']:.4f} 到 {ablation_metrics['r2']:.4f})")
            
            # 方向准确率对比
            if 'directional_accuracy' in full_metrics and 'directional_accuracy' in ablation_metrics:
                decrease = full_metrics['directional_accuracy'] - ablation_metrics['directional_accuracy']
                print(f"  方向准确率: 下降 {decrease:.2%} (从 {full_metrics['directional_accuracy']:.2%} 到 {ablation_metrics['directional_accuracy']:.2%})")
            
            # IC 对比（量化投资核心指标）
            if 'ic' in full_metrics and 'ic' in ablation_metrics:
                decrease = full_metrics['ic'] - ablation_metrics['ic']
                print(f"  IC (信息系数): 下降 {decrease:.4f} (从 {full_metrics['ic']:.4f} 到 {ablation_metrics['ic']:.4f})")
            
            # RankIC 对比（量化投资核心指标）
            if 'rank_ic' in full_metrics and 'rank_ic' in ablation_metrics:
                decrease = full_metrics['rank_ic'] - ablation_metrics['rank_ic']
                print(f"  RankIC (秩信息系数): 下降 {decrease:.4f} (从 {full_metrics['rank_ic']:.4f} 到 {ablation_metrics['rank_ic']:.4f})")
            
            # 相关系数对比（兼容性）
            if 'correlation' in full_metrics and 'correlation' in ablation_metrics:
                decrease = full_metrics['correlation'] - ablation_metrics['correlation']
                print(f"  相关系数: 下降 {decrease:.4f} (从 {full_metrics['correlation']:.4f} 到 {ablation_metrics['correlation']:.4f})")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='生成消融实验对比图表')
    parser.add_argument('--ablation_dir', type=str, 
                       default=os.path.join(os.path.dirname(__file__), 'ablation'),
                       help='消融实验目录路径（默认: paper/model/ablation）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（默认与 ablation_dir 相同）')
    
    args = parser.parse_args()
    
    ablation_dir = args.ablation_dir
    output_dir = args.output_dir or ablation_dir
    
    print("="*70)
    print("📊 消融实验对比图表生成工具")
    print("="*70)
    print(f"📁 输入目录: {ablation_dir}")
    print(f"📁 输出目录: {output_dir}")
    
    # 加载数据
    results = load_loss_data(ablation_dir)
    if not results:
        print("\n❌ 未找到任何实验结果数据，请先运行消融实验。")
        return
    
    print(f"\n✅ 成功加载 {len(results)} 个实验结果")
    
    # 生成图表
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 训练曲线对比
    curve_path = os.path.join(output_dir, 'ablation_curves_comparison.png')
    plot_training_curves(results, curve_path)
    
    # 2. 柱状图对比
    bar_path = os.path.join(output_dir, 'ablation_bar_comparison.png')
    plot_bar_comparison(results, bar_path)
    
    # 3. 生成汇总表格
    summary_path = os.path.join(output_dir, 'ablation_summary_regenerated.csv')
    generate_summary_table(results, summary_path)
    
    # 4. 计算性能下降
    calculate_improvements(results)
    
    print("\n" + "="*70)
    print("🎉 所有对比图表已生成完成！")
    print("="*70)


if __name__ == "__main__":
    main()
