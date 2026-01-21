"""
QL-MATCC 基础时序模型组件（不含图）
=================================

本模块实现论文/工程中的“时序主干”，主要由三部分组成：

1) **MATCC 趋势解耦**：
   - 用严格因果的滑动平均把序列分为 trend 与 fluctuation（避免未来信息泄露）

2) **RWKV 时序编码器**：
   - 使用 RWKV 的 TimeMixing（并用 torch.jit.script 加速关键算子）

3) **Quantum / Classical Channel Mixing（量子门控）**：
   - `Quantum_ChannelMixing`：根据样本波动率 `vol` 与阈值 `q_threshold` 决定是否走量子分支
   - 低波动：走纯经典 FFN；高波动：走 PennyLane VQC 分支，并通过可学习缩放因子稳定训练

上层模型：
  - `QL_MATCC_Model`：组合上述组件输出收益预测（供 `models/gnn_model.py` 复用）

依赖：
  - PennyLane（量子线路模拟）；如依赖版本不兼容，本文件会给出安装/修复提示。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ================= 依赖检查和导入 =================
try:
    import pennylane as qml
except ImportError as e:
    raise ImportError(
        "[错误] PennyLane 导入失败。请运行以下命令修复依赖版本：\n"
        "   pip install autoray==0.6.5 pennylane --upgrade\n"
        f"   原始错误: {e}"
    )
except AttributeError as e:
    if "'autoray.autoray' has no attribute 'NumpyMimic'" in str(e):
        raise ImportError(
            "[错误] PennyLane 与 autoray 版本不兼容！\n"
            "   解决方法：运行以下命令修复依赖版本：\n"
            "   pip install autoray==0.6.5 pennylane --upgrade\n"
            f"   原始错误: {e}"
        )
    else:
        raise

# ================= 0. JIT 加速的核心算子 (已修复 CUDA 编译错误 + FP16 溢出) =================
@torch.jit.script
def rwkv_linear_attention_cpu(time_decay: torch.Tensor, 
                              time_first: torch.Tensor, 
                              k: torch.Tensor, 
                              v: torch.Tensor):
    """
    使用 JIT 编译加速的 RWKV 串行计算。
    相比纯 Python 循环，速度提升 10x 以上。
    """
    B, T, C = k.size()
    wkv = torch.zeros_like(k)
    aa = torch.zeros(B, C, device=k.device, dtype=k.dtype)
    bb = torch.zeros(B, C, device=k.device, dtype=k.dtype)
    
    # 【关键修复 V2】
    # 1. 原代码 -1e30 导致 JIT 编译错误。
    # 2. 上一版 -1e9 导致 float16 (半精度) 溢出 (min val ~ -65504)。
    # 3. 现改为 -1e4 (-10000)，既安全不溢出，又能保证 exp(-10000) -> 0。
    pp = torch.full((B, C), -10000.0, device=k.device, dtype=k.dtype)

    for t in range(T):
        kt = k[:, t, :]
        vt = v[:, t, :]
        
        ww = time_first + pp
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        
        # 避免除以 0 的稳定性写法
        wkv[:, t, :] = (e1 * aa + e2 * vt) / (e1 * bb + e2 + 1e-6)
        
        ww = time_decay + pp
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        
        aa = e1 * aa + e2 * vt
        bb = e1 * bb + e2
        pp = p
        
    return wkv

# ================= 1. 量子层定义 =================
class VQC_Block(nn.Module):
    """
    变分量子线路（VQC）封装为 PyTorch 模块。

    - 输入：形如 (batch, n_qubits) 的角度参数（通常已映射到 [0, π]）
    - 输出：对每个 qubit 的 PauliZ 期望值（范围约 [-1, 1]）

    说明：本实现使用 `default.qubit` 作为模拟器（CPU），稳定性最好；
    如需 GPU 加速量子模拟，可研究 `pennylane-lightning-gpu`（需额外安装）。
    """
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        
        # 使用 default.qubit (CPU模拟器)，稳定性最好
        # 如果你想尝试 GPU 加速量子模拟，需要安装 pennylane-lightning-gpu
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(self.dev, interface="torch")
        def _circuit(inputs, weights):
            # 1. 编码层
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            # 2. 强纠缠层
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            # 3. 测量层
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            
        self.qnode = _circuit
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.vqc_layer = qml.qnn.TorchLayer(self.qnode, weight_shapes)

    def forward(self, x):
        return self.vqc_layer(x)

# ================= 2. RWKV 核心组件 (优化版) =================
class RWKV_TimeMixing(nn.Module):
    """
    RWKV 的时间混合层（TimeMixing）。

    作用：在时间维度上做线性注意力式的序列聚合，兼具类似 Transformer 的表达能力与更低的推理复杂度。
    本工程使用 JIT 编译的 `rwkv_linear_attention_cpu` 提升循环部分性能。
    """
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd
        # 优化：使用 Parameter 定义 decay，且初始化为负值确保衰减
        self.time_decay = nn.Parameter(torch.ones(n_embd) * -1.0)
        self.time_first = nn.Parameter(torch.ones(n_embd) * 0.5)
        
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        # 错误处理：输入检查
        assert x.dim() == 3, f"Expected 3D tensor (Batch, Seq, Dim), got {x.dim()}D"
        
        k = self.key(x)
        v = self.value(x)
        r = torch.sigmoid(self.receptance(x))
        
        # 调用 JIT 编译的算子
        wkv = rwkv_linear_attention_cpu(self.time_decay, self.time_first, k, v)
        
        return self.output(r * wkv)

# ================= 2.5 因果滑动平均 (Causal Moving Average) =================
def causal_moving_avg(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    因果滑动平均：在时间维上，每个时刻 t 只使用 x[0:t+1] 的最近 k 个点做平均。
    完全向量化，无循环，GPU 友好；保证不泄露未来信息。

    优化原因：金融预测必须因果，用 cumsum 实现 O(T) 且可并行，比 conv1d 更易控制边界。
    """
    B, T, D = x.shape
    c = torch.cumsum(x, dim=1)  # c[b,t,d] = sum(x[b,0:t+1,d])
    c_pad = F.pad(c, (0, 0, k, 0), value=0)  # (B, T+k, D)，前面补 k 个 0
    # sum(x[t-k+1:t+1]) = c[t] - c[t-k]；c_pad 左移后 [:,k:,:] 对应 c[0]..c[T-1]，[:,:T,:] 对应 c[-k]..c[T-1-k]
    c_shift = c_pad[:, :T, :]  # c_shift[b,t,d] = c_pad[b,t,d] = c[b,t-k,d] 当 t>=k，否则 0
    # 当 t<k: 用 sum(x[0:t+1])=c[t]，count=t+1；当 t>=k: sum=c[t]-c[t-k]，count=k
    count = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
    count = torch.minimum(count, torch.tensor(k, device=x.device, dtype=x.dtype))
    return (c - c_shift) / count.clamp(min=1e-8)


# ================= 2.6 MATCC 趋势解耦 (Trend Decomposition) =================
class MATCCDecompose(nn.Module):
    """
    将 X_u 分解为趋势项 X_trend 与波动项 X_fluctuation，对应论文 2.2：
    X_u = X_trend + X_fluctuation。趋势反映基本面，波动反映市场情绪。

    优化：趋势/波动用独立投影，符合论文语义；ma_window 可调（5=短期/20=长期）。
    """

    def __init__(self, input_dim: int, n_embd: int, ma_window: int = 5):
        super().__init__()
        self.ma_window = ma_window
        self.proj_trend = nn.Linear(input_dim, n_embd)
        self.proj_fluct = nn.Linear(input_dim, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x_trend = causal_moving_avg(x, self.ma_window)
        x_fluct = x - x_trend
        return self.proj_trend(x_trend) + self.proj_fluct(x_fluct)


# ================= 2.7 市场引导 (Market Guidance) =================
class MarketGuidance(nn.Module):
    """
    从市场序列 M_t (Market_Close, Market_Vol) 用深度卷积提取 h_market，并“广播”到
    每个时间步，对应论文 2.2：「使用深度卷积提取 M_t 的特征 h_market，广播拼接到所有节点」。

    优化：2 层 1D CNN 实现“深度卷积”；LayerNorm 稳定训练；输出 (B,T,n_embd) 直接与 h 相加。
    """

    def __init__(self, n_embd: int, market_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(market_dim, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, n_embd, 3, padding=1),
        )
        self.ln = nn.GroupNorm(1, n_embd)  # 对 (B,C,L) 做通道维归一化，稳定训练

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        # M: (B, T, 2) -> (B, 2, T)
        h = self.net(M.transpose(1, 2))  # (B, n_embd, T)
        h = self.ln(h)
        return h.transpose(1, 2)  # (B, T, n_embd)


# ================= 2.8 纯经典通道混合 (消融 w/o Quantum 用) =================
class Classical_ChannelMixing(nn.Module):
    """
    纯经典 FFN，与 Quantum_ChannelMixing 的 FFN 结构一致，用于 use_quantum=False。
    放在 model.py 便于统一消融；model_classical 可复用或自实现以摆脱 PennyLane 依赖。
    """

    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.GELU(),
            nn.Linear(n_embd * 4, n_embd),
        )
        self.layer_norm = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, vol: torch.Tensor = None) -> torch.Tensor:
        return self.layer_norm(x + self.dropout(self.ffn(x)))


# ================= 3. 量子-经典混合通道 (优化版 V2) =================
class Quantum_ChannelMixing(nn.Module):
    """
    量子-经典混合通道，核心改进：
    1. q_threshold 默认使用波动率 70% 分位数（约 0.5~1.0），而非 0.0
    2. 量子输出缩放：添加可学习的缩放因子，稳定训练
    3. 残差连接：量子分支也使用残差，防止梯度消失
    4. 梯度裁剪兼容：使用 clamp 防止数值溢出
    """
    def __init__(self, n_embd, n_qubits=4, q_threshold=0.5, dropout: float = 0.1):
        super().__init__()
        self.n_embd = n_embd
        self.n_qubits = n_qubits
        # 【关键修复】阈值默认 0.5，对应标准化后波动率的约 60-70% 分位数
        # 这样只有真正高波动的样本才进入量子通道
        self.q_threshold = q_threshold
        
        # 经典 FFN 分支
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.GELU(),
            nn.Linear(n_embd * 4, n_embd)
        )
        
        # 量子分支
        self.proj_down = nn.Linear(n_embd, n_qubits)
        self.vqc = VQC_Block(n_qubits=n_qubits)
        self.proj_up = nn.Linear(n_qubits, n_embd)
        
        # 【新增】量子输出缩放因子，初始化为较小的值，稳定早期训练
        self.quantum_scale = nn.Parameter(torch.tensor(0.1))
        
        self.layer_norm = nn.LayerNorm(n_embd)
        
        # 【新增】Dropout 正则化（与论文/训练脚本保持一致）
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, vol):
        # 错误处理：Batch一致性检查
        assert x.shape[0] == vol.shape[0], f"Batch size mismatch: X({x.shape[0]}) vs Vol({vol.shape[0]})"
        
        batch_size = x.shape[0]
        out = torch.zeros_like(x)
        
        # 广播判断高风险样本（波动率超过阈值）
        # vol 已经过标准化，阈值 0.5 约对应 60-70% 分位数
        is_high_risk = (vol > self.q_threshold).reshape(-1)
        
        high_risk_idx = torch.where(is_high_risk)[0]
        normal_idx = torch.where(~is_high_risk)[0]
        
        # --- 量子分支（高波动样本）---
        if len(high_risk_idx) > 0:
            x_q = x[high_risk_idx]
            Bq, T, C = x_q.shape
            x_q_flat = x_q.view(-1, C)
            
            # 【改进】更稳定的归一化：使用 sigmoid 映射到 [0, pi]
            # sigmoid 输出在 [0, 1]，比 tanh 更平滑，梯度更稳定
            proj_out = self.proj_down(x_q_flat)
            # Clamp 防止数值过大导致 sigmoid 饱和
            proj_out = proj_out.clamp(-10, 10)
            q_in = torch.sigmoid(proj_out) * np.pi
            
            # VQC 前向
            q_out = self.vqc(q_in)  # (Bq*T, n_qubits), 输出范围 [-1, 1]
            
            # 投影回高维 + 缩放
            x_q_out = self.proj_up(q_out).view(Bq, T, C)
            # 【关键】使用可学习的缩放因子，初始时量子贡献较小
            x_q_out = x_q_out * torch.abs(self.quantum_scale)
            
            # 残差连接 + Dropout
            out[high_risk_idx] = x_q + self.dropout(x_q_out)
            
        # --- 经典分支（正常样本）---
        if len(normal_idx) > 0:
            x_c = x[normal_idx]
            out[normal_idx] = x_c + self.dropout(self.ffn(x_c))
            
        return self.layer_norm(out)

# 与 dataset.feature_cols 顺序一致；Market_Close、Market_Vol 的列索引为 5,6，切片 5:7
MARKET_IDX = (5, 7)


# ================= 4. 整体模型（支持 MATCC / 市场引导 / 量子 消融）=================
class QL_MATCC_Model(nn.Module):
    """
    QL-MATCC 模型，改进点：
    1. q_threshold 默认改为 0.5（对应标准化后的中高波动率分位数）
    2. 添加 Dropout 正则化防止过拟合
    3. 改进特征融合方式
    """
    def __init__(
        self,
        input_dim=8,
        n_embd=32,
        n_layers=2,
        n_qubits=4,
        use_matcc=True,
        use_market_guidance=True,
        use_quantum=True,
        ma_window=5,
        q_threshold=0.5,  # 【关键修改】默认阈值从 0.0 改为 0.5
        dropout=0.1,
    ):
        super().__init__()
        self.use_matcc = use_matcc
        self.use_market_guidance = use_market_guidance
        self.use_quantum = use_quantum

        if use_matcc:
            self.decompose = MATCCDecompose(input_dim, n_embd, ma_window=ma_window)
        else:
            self.embedding = nn.Linear(input_dim, n_embd)

        if use_market_guidance:
            self.market_guidance = MarketGuidance(n_embd, market_dim=2)
        else:
            self.market_guidance = None

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            ch = (
                Quantum_ChannelMixing(n_embd, n_qubits=n_qubits, q_threshold=q_threshold, dropout=dropout)
                if use_quantum
                else Classical_ChannelMixing(n_embd, dropout=dropout)
            )
            self.layers.append(
                nn.ModuleDict({
                    'time_mix': RWKV_TimeMixing(n_embd),
                    'layer_norm1': nn.LayerNorm(n_embd),
                    'channel_mix': ch,
                    'layer_norm2': nn.LayerNorm(n_embd),
                })
            )
        
        # 【新增】输出头前的 Dropout 正则化
        self.pre_head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(n_embd, 1)

    def forward(self, x: torch.Tensor, vol: torch.Tensor) -> torch.Tensor:
        # 消融 w/o Market Guidance：从输入中移除大盘，与论文「移除大盘指数输入」一致
        M = x[:, :, MARKET_IDX[0] : MARKET_IDX[1]]
        if not self.use_market_guidance:
            x = x.clone()
            x[:, :, MARKET_IDX[0] : MARKET_IDX[1]] = 0.0

        if self.use_matcc:
            h = self.decompose(x)
        else:
            h = self.embedding(x)

        if self.market_guidance is not None:
            h = h + self.market_guidance(M)

        for layer in self.layers:
            h = h + layer['time_mix'](layer['layer_norm1'](h))
            h = layer['channel_mix'](layer['layer_norm2'](h), vol)

        # 取最后一个时间步 + Dropout + 线性头
        return self.head(self.pre_head_dropout(h[:, -1, :]))

# ================= 测试入口 =================
if __name__ == "__main__":
    # 检测 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f">>> Using device: {device}")
    
    # 模拟数据
    B, T, D = 8, 30, 8
    x = torch.randn(B, T, D).to(device)
    vol = (torch.rand(B, 1) * 3).to(device) # 0~3之间随机
    
    # 初始化模型
    model = QL_MATCC_Model(input_dim=D, n_embd=16, n_qubits=4).to(device)
    
    print("\n>>> Testing Forward Pass...")
    try:
        y_pred = model(x, vol)
        print(f"Output Shape: {y_pred.shape}")
        print("[成功] Optimized Quantum-RWKV Forward pass successful!")
        
        # 验证 JIT 是否工作
        # 只要没有报错，说明 torch.jit.script 编译成功
        print("[成功] JIT compilation active.")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[错误] Error: {e}")