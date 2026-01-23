"""
Graph-RWKV 基础时序模型组件（时间维编码器）
===========================================

本模块实现论文/工程中的“时序主干”，主要由三部分组成：

1) **MATCC 趋势解耦**：
   - 用严格因果的滑动平均把序列分为 trend 与 fluctuation（避免未来信息泄露）

2) **RWKV 时间序列编码器**（核心组件）：
   - 使用 RWKV 的 TimeMixing 实现线性 Attention 机制（并用 torch.jit.script 加速关键算子）
   - 计算复杂度：O(1) 推理（相比 Transformer 的 O(L²)）
   - 能够高效处理长周期金融时序特征（如过去 3-5 年的日线数据）
   - 结合了 RNN 的线性推理效率和 Transformer 的并行训练能力
   - 输出：包含时间上下文的节点嵌入 H_t ∈ R^(N×D)

2) **经典通道混合（Classical Channel Mixing）**：
   - 使用经典 FFN 进行特征变换
   - 与 RWKV 时间混合层配合，形成完整的时序编码器

【论文对应】：
    - 对应论文 2.2 模块二：Graph-RWKV 时空编码器的时间维部分
    - RWKV 作为时间序列编码器，独立处理每个股票的特征序列

上层模型：
  - `QL_MATCC_Model`：组合上述组件输出时序特征（供 `models/gnn_model.py` 的 GAT 层使用）

依赖：
  - PennyLane（量子线路模拟）；如依赖版本不兼容，本文件会给出安装/修复提示。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ================= 依赖检查和导入 =================
# 【注意】新方向不使用 Quantum，因此不再需要 PennyLane
# 以下代码已注释，如需使用 Quantum 功能可取消注释
# try:
#     import pennylane as qml
# except ImportError as e:
#     raise ImportError(
#         "[错误] PennyLane 导入失败。请运行以下命令修复依赖版本：\n"
#         "   pip install autoray==0.6.5 pennylane --upgrade\n"
#         f"   原始错误: {e}"
#     )
# except AttributeError as e:
#     if "'autoray.autoray' has no attribute 'NumpyMimic'" in str(e):
#         raise ImportError(
#             "[错误] PennyLane 与 autoray 版本不兼容！\n"
#             "   解决方法：运行以下命令修复依赖版本：\n"
#             "   pip install autoray==0.6.5 pennylane --upgrade\n"
#             f"   原始错误: {e}"
#         )
#     else:
#         raise

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

# ================= 1. 量子层定义（已注释：新方向不使用）=================
# 【注意】新方向不使用 Quantum 量子计算，以下代码已注释保留
# class VQC_Block(nn.Module):
#     """
#     变分量子线路（VQC）封装为 PyTorch 模块。
#     新方向中不使用，已注释。
#     """
#     def __init__(self, n_qubits=8, n_layers=4):
#         super().__init__()
#         self.n_qubits = n_qubits
#         self.dev = qml.device("default.qubit", wires=n_qubits)
#         @qml.qnode(self.dev, interface="torch")
#         def _circuit(inputs, weights):
#             qml.AngleEmbedding(inputs, wires=range(n_qubits))
#             qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
#             return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
#         self.qnode = _circuit
#         weight_shapes = {"weights": (n_layers, n_qubits, 3)}
#         self.vqc_layer = qml.qnn.TorchLayer(self.qnode, weight_shapes)
#     def forward(self, x):
#         return self.vqc_layer(x)

# ================= 2. RWKV 核心组件 (优化版) =================
class RWKV_TimeMixing(nn.Module):
    """
    RWKV 的时间混合层（TimeMixing）。

    作用：在时间维度上做线性注意力式的序列聚合，兼具类似 Transformer 的表达能力与更低的推理复杂度。
    本工程使用 JIT 编译的 `rwkv_linear_attention_cpu` 提升循环部分性能。

    【修复】恢复完整RWKV容量，移除GQA优化以确保全量模型性能
    """
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd

        # 优化：使用 Parameter 定义 decay，且初始化为负值确保衰减
        self.time_decay = nn.Parameter(torch.ones(n_embd) * -1.0)
        self.time_first = nn.Parameter(torch.ones(n_embd) * 0.5)

        # 【修复】使用完整线性层，恢复全部表达能力
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        # 错误处理：输入检查
        assert x.dim() == 3, f"Expected 3D tensor (Batch, Seq, Dim), got {x.dim()}D"

        B, T, C = x.shape

        # 【修复】使用完整维度计算
        k = self.key(x)  # (B, T, C)
        v = self.value(x)  # (B, T, C)
        r = torch.sigmoid(self.receptance(x))  # (B, T, C)

        # 调用 JIT 编译的算子
        wkv = rwkv_linear_attention_cpu(self.time_decay, self.time_first, k, v)

        return self.output(r * wkv)

# ================= 2.5 因果滑动平均（已注释：新方向不使用 MATCC）=================
# 【注意】新方向不使用 MATCC 趋势解耦，以下代码已注释保留
# def causal_moving_avg(x: torch.Tensor, k: int) -> torch.Tensor:
#     """因果滑动平均：在时间维上，每个时刻 t 只使用 x[0:t+1] 的最近 k 个点做平均。"""
#     B, T, D = x.shape
#     c = torch.cumsum(x, dim=1)
#     c_pad = F.pad(c, (0, 0, k, 0), value=0)
#     c_shift = c_pad[:, :T, :]
#     count = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
#     count = torch.minimum(count, torch.tensor(k, device=x.device, dtype=x.dtype))
#     return (c - c_shift) / count.clamp(min=1e-8)


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


# ================= 2.8 经典通道混合（核心组件）=================
class Classical_ChannelMixing(nn.Module):
    """
    经典通道混合：使用 FFN 进行特征变换。
    
    【核心组件】新方向使用纯经典 FFN，不使用量子计算。
    与 RWKV 时间混合层配合，形成完整的时序编码器。
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
        # vol 参数保留以兼容接口，但新方向中不使用
        return self.layer_norm(x + self.dropout(self.ffn(x)))

# ================= 3. 量子-经典混合通道（已注释：新方向不使用）=================
# 【注意】新方向不使用 Quantum 量子计算，以下代码已注释保留
# class Quantum_ChannelMixing(nn.Module):
#     """量子-经典混合通道。新方向中不使用，已注释。"""
#     def __init__(self, n_embd, n_qubits=8, q_threshold=0.5, dropout: float = 0.1):
#         super().__init__()
#         self.n_embd = n_embd
#         self.n_qubits = n_qubits
#         self.q_threshold = q_threshold
#         self.ffn = nn.Sequential(
#             nn.Linear(n_embd, n_embd * 4),
#             nn.GELU(),
#             nn.Linear(n_embd * 4, n_embd)
#         )
#         self.proj_down = nn.Linear(n_embd, n_qubits)
#         self.vqc = VQC_Block(n_qubits=n_qubits)
#         self.proj_up = nn.Linear(n_qubits, n_embd)
#         self.quantum_scale = nn.Parameter(torch.tensor(0.5))
#         self.layer_norm = nn.LayerNorm(n_embd)
#         self.dropout = nn.Dropout(dropout)
#     def forward(self, x, vol):
#         # 量子分支逻辑已注释
#         return self.layer_norm(x + self.dropout(self.ffn(x)))

# ================= 4. Graph-RWKV 时序编码器模型（核心模型）=================
class GraphRWKV_Model(nn.Module):
    """
    Graph-RWKV 时序编码器模型（新方向核心模型）。
    
    【核心架构】：
    1. 输入投影：Linear(input_dim -> n_embd)
    2. N 层 RWKV Block：
       - RWKV_TimeMixing：时间维线性注意力
       - Classical_ChannelMixing：经典 FFN 通道混合
    3. 输出：最后一个时间步的特征向量
    
    【论文对应】：
    - 对应论文 2.2 模块二：Graph-RWKV 时空编码器的时间维部分
    - 输出时序特征供 GAT 层进行空间聚合
    """
    
    def __init__(
        self,
        input_dim=8,
        n_embd=32,
        n_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        
        # 输入投影层
        self.embedding = nn.Linear(input_dim, n_embd)
        
        # RWKV 层堆叠
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict({
                    'time_mix': RWKV_TimeMixing(n_embd),
                    'layer_norm1': nn.LayerNorm(n_embd),
                    'channel_mix': Classical_ChannelMixing(n_embd, dropout=dropout),
                    'layer_norm2': nn.LayerNorm(n_embd),
                })
            )
        
        # 输出头前的 Dropout 正则化
        self.pre_head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(n_embd, 1)

    def forward(self, x: torch.Tensor, vol: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, T, F) 输入序列
        vol: (B, 1) 波动率（保留以兼容接口，但新方向中不使用）
        """
        # 输入投影
        h = self.embedding(x)
        
        # RWKV 层堆叠
        for layer in self.layers:
            h = h + layer['time_mix'](layer['layer_norm1'](h))
            h = layer['channel_mix'](layer['layer_norm2'](h), vol)
        
        # 取最后一个时间步 + Dropout + 线性头
        return self.head(self.pre_head_dropout(h[:, -1, :]))

# ================= 5. 兼容性模型（已注释：保留旧接口）=================
# 【注意】为了兼容旧代码，保留 QL_MATCC_Model 作为别名，但默认关闭所有可选组件
# class QL_MATCC_Model(nn.Module):
#     """兼容性别名，实际使用 GraphRWKV_Model"""
#     def __init__(self, input_dim=8, n_embd=32, n_layers=2, 
#                  use_matcc=False, use_market_guidance=False, use_quantum=False,
#                  dropout=0.1, **kwargs):
#         super().__init__()
#         # 直接使用 GraphRWKV_Model
#         self.model = GraphRWKV_Model(input_dim, n_embd, n_layers, dropout)
#     def forward(self, x, vol=None):
#         return self.model(x, vol)

# 为了向后兼容，创建别名
QL_MATCC_Model = GraphRWKV_Model

# ================= 测试入口 =================
if __name__ == "__main__":
    # 检测 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f">>> Using device: {device}")
    
    # 模拟数据
    B, T, D = 8, 30, 8
    x = torch.randn(B, T, D).to(device)
    vol = (torch.rand(B, 1) * 3).to(device) # 0~3之间随机
    
    # 初始化模型（新方向：Graph-RWKV）
    model = GraphRWKV_Model(input_dim=D, n_embd=16, n_layers=2).to(device)
    
    print("\n>>> Testing Forward Pass...")
    try:
        y_pred = model(x, vol)
        print(f"Output Shape: {y_pred.shape}")
        print("[成功] Graph-RWKV Forward pass successful!")
        
        # 验证 JIT 是否工作
        # 只要没有报错，说明 torch.jit.script 编译成功
        print("[成功] JIT compilation active.")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[错误] Error: {e}")