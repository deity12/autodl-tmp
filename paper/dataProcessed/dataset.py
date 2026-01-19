import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os

class FinancialDataset(Dataset):
    def __init__(self, csv_path, seq_len=30, pred_len=1, mode='train', scaler=None):
        """
        参数说明:
            csv_path: 清洗后的数据文件 Final_Model_Data.csv 的路径
            seq_len: 输入序列长度（例如：过去30天的数据）
            pred_len: 预测时长（例如：预测未来1天）
            mode: 'train'（训练）或 'test'（测试）
            scaler: 已拟合的标准化器（测试模式下必须提供）
        """
        print(f"正在加载 {mode} 数据，来源：{csv_path}...")
        
        # [错误处理] 检查文件是否存在
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"数据文件未找到: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mode = mode
        
        # 定义特征列和目标列
        self.feature_cols = ['Open', 'Close', 'High', 'Low', 'Volume', 'Market_Close', 'Market_Vol', 'Volatility_20d']
        self.target_col = 'Log_Ret'
        
        # =======================================================
        # 🛡️ 【新增】鲁棒性数据清洗防火墙（防止异常值破坏模型）
        # =======================================================
        
        # 1. 裁剪（Clipping）：解决极端异常值问题（例如 Log_Ret = 14.92）
        # 强制将收益率限制在 -100% (-1.0) 到 +100% (1.0) 之间
        if self.target_col in self.df.columns:
            self.df[self.target_col] = self.df[self.target_col].clip(-1.0, 1.0)
            
        # 2. 处理无穷大值：将 Inf 和 -Inf 替换为 NaN
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        
        # 3. 填充/删除缺失值：
        # 定义需要检查的数值型列
        numeric_cols = self.feature_cols + [self.target_col]
        
        # 使用前向填充（forward fill）修复缺失的价格数据（例如之前发现的20行缺失）
        self.df[numeric_cols] = self.df[numeric_cols].ffill()
        
        # 如果仍有缺失值（例如在数据最开始的位置），则直接删除这些行
        before_len = len(self.df)
        self.df = self.df.dropna(subset=numeric_cols)
        after_len = len(self.df)
        
        if before_len != after_len:
            print(f"⚠️ 已清理并删除 {before_len - after_len} 行包含无效数据（NaN）的记录")

        # 4. 波动率修正：防止量子层计算溢出
        if 'Volatility_20d' in self.df.columns:
             self.df['Volatility_20d'] = self.df['Volatility_20d'].fillna(0).clip(0, 5.0)

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
            
        # 3. 标准化（使用 StandardScaler）
        # 使用 NumPy 数组进行标准化，避免 DataFrame 的额外开销
        if mode == 'train':
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
        
        # 4. 构建滑动窗口索引（确保不跨股票拼接序列）
        print("正在构建滑动窗口索引...")
        self.indices = []
        
        # 按股票代码（Ticker）分组，保证每个序列只来自同一只股票
        groups = self.df.groupby('Ticker')
        for _, group in groups:
            group_len = len(group)
            if group_len > seq_len + pred_len:
                start_row = group.index[0]
                # 高效地添加所有有效的起始位置索引
                for i in range(group_len - seq_len - pred_len + 1):
                    self.indices.append(start_row + i)
                    
        print(f"{mode} 数据集共生成样本数: {len(self.indices)}")

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
        # 【性能优化】DataLoader 设置
        num_workers=0,  # Windows 系统建议设为 0，避免多进程冲突
        pin_memory=True if torch.cuda.is_available() else False,
        # prefetch_factor 在 num_workers=0 时不可用，故省略
    )
    
    print("\n>>> 检查训练批次数据...")
    for batch in train_loader:
        print("输入张量形状:", batch['x'].shape)
        print("目标张量形状:", batch['y'].shape)
        print("波动率张量形状:", batch['vol'].shape)
        print("样本数据加载成功！")
        
        if torch.cuda.is_available():
            print(f"数据所在设备: {batch['x'].device}（GPU 可用但当前未使用）")
        break

    # 2. 测试测试数据加载器
    print("\n>>> 正在初始化测试数据集...")
    test_dataset = FinancialDataset(CSV_PATH, seq_len=30, mode='test', scaler=train_dataset.scaler)
    
    # 【统计信息输出】
    print(f"\n>>> 数据集统计信息:")
    print(f"  - 训练样本数量: {len(train_dataset)}")
    print(f"  - 测试样本数量: {len(test_dataset)}")
    print(f"  - 输入序列长度: {train_dataset.seq_len}")
    print(f"  - 预测步长: {train_dataset.pred_len}")
    print(f"  - 特征维度数: {len(train_dataset.feature_cols)}")