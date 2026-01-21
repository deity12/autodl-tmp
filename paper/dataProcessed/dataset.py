"""
PyTorch Dataset：将 `Final_Model_Data.csv` 转换为可训练样本（Step 4）
====================================================

本模块提供 `FinancialDataset`，用于：
  - 数据清洗（裁剪极端值、处理 NaN/Inf）
  - 按时间 80/20 切分 train/test（避免未来信息泄露）
  - 特征标准化（训练集 fit，测试集 transform）
  - 以“同一股票”为单位构造滑动窗口序列样本
  - 计算训练集波动率分位数 `vol_stats`（常用 p70 作为量子门控阈值）
  - 构建 `ticker2idx`，并在 `__getitem__` 返回 `node_indices` 供 GNN 使用

输入：
  - `data/processed/Final_Model_Data.csv`（来自 `dataProcessed/align.py`）

输出（每条样本，dict）：
  - `x`: (seq_len, input_dim) 过去若干天特征
  - `y`: (1,) 目标日对数收益率
  - `vol`: (1,) 波动率（最后一日）
  - `node_indices`: (,) 股票节点索引（用于图聚合）
  - `target_date`: str 目标日期（用于按日期截面 IC/RankIC 或按日分组 batch）
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
import os

class FinancialDataset(Dataset):
    """
    金融数据集类，改进点：
    1. 使用 RobustScaler 作为可选项，对异常值更鲁棒
    2. 计算并存储波动率分位数，用于动态设置量子阈值
    3. 改进的数据清洗流程
    4. 添加数据增强选项（可选）
    """
    def __init__(self, csv_path, seq_len=30, pred_len=1, mode='train', scaler=None, 
                 vol_stats=None, use_robust_scaler=False):
        """
        参数说明:
            csv_path: 清洗后的数据文件 Final_Model_Data.csv 的路径
            seq_len: 输入序列长度（例如：过去30天的数据）
            pred_len: 预测时长（例如：预测未来1天）
            mode: 'train'（训练）或 'test'（测试）
            scaler: 已拟合的标准化器（测试模式下必须提供）
            vol_stats: 波动率统计信息（测试模式下必须提供）
            use_robust_scaler: 是否使用 RobustScaler（对异常值更鲁棒）
        """
        print(f"正在加载 {mode} 数据，来源：{csv_path}...")
        
        # [错误处理] 检查文件是否存在
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"数据文件未找到: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mode = mode
        self.use_robust_scaler = use_robust_scaler

        # 【关键对齐】统一股票代码格式为大写，确保与 build_graph.py 输出的 Graph_Adjacency_tickers.json 一致
        # 避免出现因大小写差异导致的“图谱索引错位”（最危险：不一定报错，但会让训练结果失真）
        if 'Ticker' in self.df.columns:
            self.df['Ticker'] = self.df['Ticker'].astype(str).str.upper()
            # 清理异常 ticker（极少数情况下会出现 NaN -> "NAN"）
            self.df = self.df[self.df['Ticker'] != 'NAN'].copy()
        
        # 定义特征列和目标列
        self.feature_cols = ['Open', 'Close', 'High', 'Low', 'Volume', 'Market_Close', 'Market_Vol', 'Volatility_20d']
        self.target_col = 'Log_Ret'
        
        # =======================================================
        # 🛡️ 【改进】鲁棒性数据清洗防火墙
        # =======================================================
        
        # 1. 裁剪（Clipping）：使用更合理的范围
        # 日收益率 ±50% 已经是极端情况（股票涨停/跌停）
        if self.target_col in self.df.columns:
            self.df[self.target_col] = self.df[self.target_col].clip(-0.5, 0.5)
            
        # 2. 处理无穷大值：将 Inf 和 -Inf 替换为 NaN
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        
        # 3. 填充/删除缺失值
        numeric_cols = self.feature_cols + [self.target_col]
        
        # 使用前向填充修复缺失的价格数据
        self.df[numeric_cols] = self.df[numeric_cols].ffill()
        
        # 删除仍有缺失值的行
        before_len = len(self.df)
        self.df = self.df.dropna(subset=numeric_cols)
        after_len = len(self.df)
        
        if before_len != after_len:
            print(f"⚠️ 已清理并删除 {before_len - after_len} 行包含无效数据（NaN）的记录")

        # 4. 波动率修正：使用更合理的范围
        if 'Volatility_20d' in self.df.columns:
             self.df['Volatility_20d'] = self.df['Volatility_20d'].fillna(0).clip(0, 2.0)

        print("✅ 数据清洗完成：无无穷值、无缺失值、极端值已裁剪。")
        # =======================================================
        
        # 1. 按日期排序
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

        # 【GNN 支持】在划分 train/test 前，用全量 Ticker 构建 ticker->idx 映射
        # 顺序与 build_graph.py、Final_Model_Data 的 sorted(unique) 一致，便于邻接矩阵对齐
        all_tickers = sorted(self.df['Ticker'].unique())
        self.ticker2idx = {t: i for i, t in enumerate(all_tickers)}
        
        # 2. 划分训练集与测试集
        dates = sorted(self.df['Date'].unique())
        if len(dates) < 2:
            raise ValueError(f"数据量不足：仅找到 {len(dates)} 个日期。")
            
        split_idx = int(len(dates) * 0.8)
        split_idx = min(split_idx, len(dates) - 1)
        split_date = dates[split_idx]
        
        if mode == 'train':
            self.df = self.df[self.df['Date'] < split_date].copy()
        else:
            self.df = self.df[self.df['Date'] >= split_date].copy()
            
        # 【关键修复】重置索引！避免后续滑动窗口出错
        self.df = self.df.reset_index(drop=True)
            
        # 3. 标准化
        if mode == 'train':
            if use_robust_scaler:
                # RobustScaler 对异常值更鲁棒
                self.scaler = RobustScaler(quantile_range=(10, 90))
            else:
                self.scaler = StandardScaler()
            feature_array = self.df[self.feature_cols].values
            self.df[self.feature_cols] = self.scaler.fit_transform(feature_array)
        else:
            if scaler is None:
                raise ValueError("测试模式下必须提供已拟合的标准化器（scaler）。")
            self.scaler = scaler
            feature_array = self.df[self.feature_cols].values
            self.df[self.feature_cols] = self.scaler.transform(feature_array)
        
        # 【性能优化】转换为 Numpy float32 类型（节省内存，加速训练）
        self.data_x = self.df[self.feature_cols].values.astype(np.float32)
        self.data_y = self.df[self.target_col].values.astype(np.float32)
        self.data_vol = self.df['Volatility_20d'].values.astype(np.float32)
        
        # =======================================================
        # 【新增】计算波动率分位数，用于动态设置量子阈值
        # =======================================================
        if mode == 'train':
            # 在标准化后的波动率上计算分位数
            vol_col_idx = self.feature_cols.index('Volatility_20d')
            vol_standardized = self.data_x[:, vol_col_idx]  # 标准化后的波动率
            self.vol_stats = {
                'mean': float(np.mean(vol_standardized)),
                'std': float(np.std(vol_standardized)),
                'p50': float(np.percentile(vol_standardized, 50)),  # 中位数
                'p60': float(np.percentile(vol_standardized, 60)),
                'p70': float(np.percentile(vol_standardized, 70)),  # 推荐阈值
                'p80': float(np.percentile(vol_standardized, 80)),
                'p90': float(np.percentile(vol_standardized, 90)),
                'min': float(np.min(vol_standardized)),
                'max': float(np.max(vol_standardized)),
            }
            print(f"📊 波动率统计（标准化后）:")
            print(f"   mean={self.vol_stats['mean']:.3f}, std={self.vol_stats['std']:.3f}")
            print(f"   p50={self.vol_stats['p50']:.3f}, p70={self.vol_stats['p70']:.3f}, p90={self.vol_stats['p90']:.3f}")
            print(f"   ⭐ 推荐量子阈值 q_threshold: {self.vol_stats['p70']:.3f} (70%分位数)")
        else:
            if vol_stats is None:
                # 如果测试时没提供 vol_stats，使用默认值
                self.vol_stats = {'p70': 0.5}
                print("⚠️ 测试模式未提供 vol_stats，使用默认阈值 0.5")
            else:
                self.vol_stats = vol_stats
        
        # 4. 构建滑动窗口索引（确保不跨股票拼接序列）
        print("正在构建滑动窗口索引...")
        self.indices = []
        self.target_dates = []  # 用于“按日期分组”的rank/ranking训练（顶会常见做法）
        
        # 按股票代码（Ticker）分组，保证每个序列只来自同一只股票
        groups = self.df.groupby('Ticker')
        for _, group in groups:
            group_len = len(group)
            if group_len > seq_len + pred_len:
                start_row = group.index[0]
                # 高效地添加所有有效的起始位置索引
                for i in range(group_len - seq_len - pred_len + 1):
                    s = start_row + i
                    end_row = s + self.seq_len
                    target_row = end_row + self.pred_len - 1
                    # 边界检查：确保 target_row 不超出 group 范围
                    if target_row >= start_row + group_len:
                        continue
                    self.indices.append(s)
                    # 目标对齐：y 来自 target_row，因此用于排序/RankIC 的"截面日期"应以 target_date 分组
                    self.target_dates.append(self.df['Date'].iloc[target_row])
                    
        print(f"{mode} 数据集共生成样本数: {len(self.indices)}")
        # 统一为字符串，便于 sampler/groupby（DataLoader 对 Timestamp 的默认 collate 行为不稳定）
        self.target_dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in self.target_dates]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_row = self.indices[idx]
        end_row = start_row + self.seq_len
        target_row = end_row + self.pred_len - 1
        
        # 直接使用 NumPy 切片（速度最快）
        x = self.data_x[start_row : end_row]       # 形状: (序列长度, 特征数)
        y = self.data_y[target_row]                # 标量（单日对数收益率）
        vol = self.data_vol[end_row - 1]           # 标量（最后一日的波动率）
        # 【GNN 支持】当前样本对应的股票在图中的节点索引，供 QL_MATCC_GNN_Model 做图聚合
        ticker = self.df['Ticker'].iloc[start_row]
        node_idx = self.ticker2idx.get(ticker, 0)
        
        return {
            'x': torch.from_numpy(x),                   # 转换为 PyTorch 张量
            'y': torch.tensor([y], dtype=torch.float32),
            'vol': torch.tensor([vol], dtype=torch.float32),
            'node_indices': torch.tensor(node_idx, dtype=torch.long),
            'target_date': self.target_dates[idx],      # 用于按日期做截面排序/RankIC loss
        }

# ================= 测试代码 =================
if __name__ == "__main__":
    CSV_PATH = './data/processed/Final_Model_Data.csv'
    
    # 1. 测试训练数据加载器
    print(">>> 正在初始化训练数据集...")
    train_dataset = FinancialDataset(CSV_PATH, seq_len=30, mode='train')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    print("\n>>> 检查训练批次数据...")
    for batch in train_loader:
        print("输入张量形状:", batch['x'].shape)
        print("目标张量形状:", batch['y'].shape)
        print("波动率张量形状:", batch['vol'].shape)
        print("节点索引形状:", batch['node_indices'].shape)
        print("样本数据加载成功！")
        
        if torch.cuda.is_available():
            print(f"数据所在设备: {batch['x'].device}（GPU 可用但当前未使用）")
        break

    # 2. 测试测试数据加载器（传入 vol_stats）
    print("\n>>> 正在初始化测试数据集...")
    test_dataset = FinancialDataset(
        CSV_PATH, seq_len=30, mode='test', 
        scaler=train_dataset.scaler,
        vol_stats=train_dataset.vol_stats  # 传入训练集的波动率统计
    )
    
    # 【统计信息输出】
    print(f"\n>>> 数据集统计信息:")
    print(f"  - 训练样本数量: {len(train_dataset)}")
    print(f"  - 测试样本数量: {len(test_dataset)}")
    print(f"  - 输入序列长度: {train_dataset.seq_len}")
    print(f"  - 预测步长: {train_dataset.pred_len}")
    print(f"  - 特征维度数: {len(train_dataset.feature_cols)}")
    print(f"  - 波动率分位数 (p70): {train_dataset.vol_stats['p70']:.4f}")
    print(f"  - 推荐量子阈值: {train_dataset.vol_stats['p70']:.4f}")