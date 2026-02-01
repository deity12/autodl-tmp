"""
PyTorch Dataset：将 `Final_Model_Data.csv` 转换为可训练样本（Step 4）
====================================================

【核心创新点】根据新研究方向，本模块支持 Graph-RWKV 模型的数据加载：

本模块提供 `FinancialDataset`，用于：
  - 数据清洗（裁剪极端值、处理 NaN/Inf）
  - 按时间 80/20 切分 train/test（避免未来信息泄露）
     【注意】完整滚动窗口验证需在评估脚本中实现（见论文 3.3）
  - 特征标准化（训练集 fit，测试集 transform）
  - 以“同一股票”为单位构造滑动窗口序列样本
  - 计算训练集波动率分位数 `vol_stats`（已注释：新方向不使用量子门控）
  - 构建 `ticker2idx`，并在 `__getitem__` 返回 `node_indices` 供 GNN 使用
     【关键】确保与 Graph_Tickers.json 中的节点顺序一致，避免索引错位

输入：
  - `paper/data/processed/Final_Model_Data.csv`（来自 `dataProcessed/align.py`）
  - `paper/data/processed/Graph_Tickers.json`（来自 `dataProcessed/build_graph.py`，用于节点对齐）

输出（每条样本，dict）：
  - `x`: (seq_len, input_dim) 过去若干天特征（输入 RWKV 时间编码器）
  - `y`: (1,) 目标日对数收益率（预测目标）
  - `vol`: (1,) 波动率（最后一日，保留以兼容接口，但新方向中不使用）
  - `node_indices`: (,) 股票节点索引（用于 GAT 空间聚合）
  - `target_date`: str 目标日期（用于按日期截面 IC/RankIC 或按日分组 batch）

【论文对应】：
    - 对应论文 3.1 数据集准备
    - 支持 RWKV 时间序列编码器的输入格式
    - 支持动态图注意力网络的节点索引对齐
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
    def __init__(
        self,
        csv_path,
        features_path=None,
        seq_len=30,
        pred_len=1,
        mode='train',
        scaler=None,
        vol_stats=None,
        use_robust_scaler=False,
        start_date=None,
        end_date=None,
        use_date_split=True,
        feature_cols=None,
        feature_columns_path=None,
    ):
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
        self.features_path = features_path
        self.split_date = None
        self.start_date = start_date
        self.end_date = end_date

        # 【关键对齐】统一股票代码格式为大写，确保与 build_graph.py 输出的 Graph_Tickers.json 一致
        # 避免出现因大小写差异导致的“图谱索引错位”（最危险：不一定报错，但会让训练结果失真）
        if 'Ticker' in self.df.columns:
            # 与 build_graph / filter_sp500 统一：大小写归一 + '-'/' .' 归一（如 BRK-B -> BRK.B）
            self.df['Ticker'] = (
                self.df['Ticker']
                .astype(str)
                .str.upper()
                .str.replace("-", ".", regex=False)
            )
            # 清理异常 ticker（极少数情况下会出现 NaN -> "NAN"）
            self.df = self.df[self.df['Ticker'] != 'NAN'].copy()
        
        # 定义特征列和目标列
        default_feature_cols = [
            'Open', 'Close', 'High', 'Low', 'Volume',
            'Market_Close', 'Market_Vol', 'Volatility_20d',
        ]
        if feature_cols is not None:
            self.feature_cols = list(feature_cols)
        else:
            if feature_columns_path is None:
                feature_columns_path = os.path.join(os.path.dirname(csv_path), 'feature_columns.json')
            if os.path.exists(feature_columns_path):
                try:
                    import json
                    with open(feature_columns_path, 'r', encoding='utf-8') as f:
                        self.feature_cols = json.load(f)
                    print(f"    [特征] 从 {feature_columns_path} 读取特征列，共 {len(self.feature_cols)} 维")
                except Exception as e:
                    print(f"    [WARN] 读取特征列失败: {e}，回退默认 8 维特征")
                    self.feature_cols = default_feature_cols
            else:
                self.feature_cols = default_feature_cols
        # [DEBUG] 强制锁定为基础特征，排除 Alpha158 干扰
        self.feature_cols = ['Open', 'Close', 'High', 'Low', 'Volume']
        print(f"⚠️ [DEBUG] 底线测试：已强制锁定特征为 {self.feature_cols}")
        self.target_col = 'Log_Ret'

        # 如果特征列不在主 CSV 中，则尝试从外部特征文件（Parquet）合并进来。
        # 这与“Alpha158-like 因子单独落盘为 Parquet”流程对齐。
        missing_cols = [c for c in self.feature_cols if c not in self.df.columns]
        if missing_cols:
            # 默认在同目录查找：sp500_alpha158_features.parquet
            feat_path = features_path
            if feat_path is None:
                candidate = os.path.join(os.path.dirname(csv_path), "sp500_alpha158_features.parquet")
                if os.path.exists(candidate):
                    feat_path = candidate

            if feat_path and os.path.exists(feat_path):
                try:
                    df_feat = pd.read_parquet(feat_path)
                    # 【修复】确保两边的 Date 列类型一致
                    self.df["Date"] = pd.to_datetime(self.df["Date"], errors="coerce")
                    df_feat["Date"] = pd.to_datetime(df_feat["Date"], errors="coerce")
                    df_feat["Ticker"] = (
                        df_feat["Ticker"]
                        .astype(str)
                        .str.upper()
                        .str.replace("-", ".", regex=False)
                    )
                    # 左连接：保留主数据的交易日与样本定义
                    self.df = self.df.merge(df_feat, on=["Date", "Ticker"], how="left")
                except Exception as e:
                    raise ValueError(f"特征列缺失且外部特征文件读取/合并失败: {feat_path}, err={e}") from e

                missing_cols = [c for c in self.feature_cols if c not in self.df.columns]

            if missing_cols:
                raise ValueError(f"特征列不存在: {missing_cols}")
        
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

        # 【GNN 支持 V4】读取图谱节点列表以确保索引对齐
        # 如果 Graph_Tickers.json 存在，使用其中的节点顺序；否则使用数据中的全量 ticker
        import json
        graph_tickers_path = os.path.join(os.path.dirname(csv_path), 'Graph_Tickers.json')
        if os.path.exists(graph_tickers_path):
            try:
                with open(graph_tickers_path, 'r') as f:
                    graph_data = json.load(f)
                    graph_tickers = graph_data.get('tickers', [])
                print(f"    [V4 对齐] 从 Graph_Tickers.json 读取 {len(graph_tickers)} 个图节点")
                self.ticker2idx = {t: i for i, t in enumerate(graph_tickers)}
                # 过滤数据：只保留图中存在的股票
                self.df = self.df[self.df['Ticker'].isin(set(graph_tickers))].copy()
                print(f"    [V4 对齐] 过滤后数据包含 {self.df['Ticker'].nunique()} 只股票")
            except Exception as e:
                print(f"    [WARN] 读取 Graph_Tickers.json 失败: {e}，使用数据中的全量 ticker")
                all_tickers = sorted(self.df['Ticker'].unique())
                self.ticker2idx = {t: i for i, t in enumerate(all_tickers)}
        else:
            print(f"    [INFO] Graph_Tickers.json 不存在，使用数据中的全量 ticker")
            all_tickers = sorted(self.df['Ticker'].unique())
            self.ticker2idx = {t: i for i, t in enumerate(all_tickers)}
        
        # 2. 日期过滤与切分
        if start_date or end_date:
            if start_date:
                self.df = self.df[self.df['Date'] >= pd.to_datetime(start_date)].copy()
            if end_date:
                self.df = self.df[self.df['Date'] <= pd.to_datetime(end_date)].copy()
        elif use_date_split:
            dates = sorted(self.df['Date'].unique())
            if len(dates) < 2:
                raise ValueError(f"数据量不足：仅找到 {len(dates)} 个日期。")
                
            split_idx = int(len(dates) * 0.8)
            split_idx = min(split_idx, len(dates) - 1)
            split_date = dates[split_idx]
            self.split_date = pd.to_datetime(split_date)
            
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
        # 【注意】新方向不使用量子门控，vol_stats 计算已注释
        # =======================================================
        # 为了兼容接口，保留 vol_stats 但设为空字典
        if mode == 'train':
            # 【已注释】新方向不使用量子门控，不再需要计算波动率分位数
            # vol_col_idx = self.feature_cols.index('Volatility_20d')
            # vol_standardized = self.data_x[:, vol_col_idx]
            # self.vol_stats = {...}
            self.vol_stats = {}  # 空字典以兼容接口
        else:
            # 兼容接口，但新方向中不使用
            self.vol_stats = vol_stats if vol_stats is not None else {}
        
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
    CSV_PATH = "./paper/data/processed/Final_Model_Data.csv"
    
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
    p70 = train_dataset.vol_stats.get('p70') if hasattr(train_dataset, "vol_stats") else None
    if p70 is not None:
        print(f"  - 波动率分位数 (p70): {p70:.4f}")
        print(f"  - 推荐量子阈值: {p70:.4f}")
    else:
        print("  - 波动率分位数 (p70): N/A")
        print("  - 推荐量子阈值: N/A")
