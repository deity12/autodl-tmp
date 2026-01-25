# -*- coding: utf-8 -*-
"""
Graph-RWKV 模型：基于大语言模型动态图谱与 Graph-RWKV 的时空解耦金融预测
========================================================================
【核心创新点】根据新研究方向实现：

架构：
    1. **RWKV 时间序列编码器**（时间维）：
       - 利用 RWKV 的线性 Attention 机制（O(1) 推理复杂度）捕捉长程依赖
       - 每个股票的特征序列独立进入 RWKV 模块
       - 输出：包含时间上下文的节点嵌入 H_t ∈ R^(N×D)
    
    2. **动态图注意力网络 GAT**（空间维）：
       - 基于 LLM 增强的情感加权混合图进行空间聚合
       - 混合图包含：语义图（LLM提取的关系+情感极性）+ 统计图（收益率相关性）
       - 使预测考虑到供应链上下游及竞争对手的实时状态
       - 输出：空间聚合后的节点嵌入 H'_t = GAT(H_t, A_t^final)
    
    3. **时空特征融合**：
       - Concat(H_temporal, H_graph) -> Linear -> 收益率预测
       - 实现时空解耦建模，同时捕捉时间趋势和空间依赖

【论文对应】：
    - 时间维：RWKV-TimeSeries Encoder（对应论文 2.2 模块二）
    - 空间维：动态图注意力（对应论文 2.1 模块一）
    - 混合图：A_t^final = Norm(A_t^semantic + λ · A_t^stat)（对应论文 2.1）

依赖：
    - base_model：GraphRWKV_Model（新方向核心时序编码器）, RWKV_TimeMixing
    - 邻接矩阵：./data/processed/Graph_Adjacency.npy（由 build_graph.py 生成，包含情感加权混合图）
    
【注意】以下组件在新方向中不使用，已注释：
    - Quantum 量子计算相关代码
    - MATCC 趋势解耦相关代码
    - Market Guidance 市场引导相关代码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

try:
    from torch_geometric.nn import GATv2Conv
    _HAS_PYG = True
except Exception:
    _HAS_PYG = False

# 从同目录的 base_model.py 复用：GraphRWKV_Model（新方向核心模型）、RWKV_TimeMixing
from .base_model import GraphRWKV_Model, QL_MATCC_Model, RWKV_TimeMixing, RNN_TemporalEncoder
# 【注意】Quantum_ChannelMixing、MATCCDecompose 在新方向中不使用，已注释


# ================= 1. 图注意力层 GAT (优化版) =================
class GraphAttentionLayer(nn.Module):
    """
    广播优化版 GAT：使用拆分注意力向量（a_src/a_dst）并通过广播计算注意力得分，
    避免构造 (N, N, 2*D) 的中间大张量，从而明显降低内存峰值。
    注意：仍会构造 (heads, N, N) 的 attention 矩阵，对于非常大尺寸图应考虑 edge-wise 实现。
    """

    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True, num_heads=2):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.num_heads = num_heads

        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        self.head_dim = out_features // num_heads

        # 多头线性变换 (num_heads, in_features, head_dim)
        self.W = nn.Parameter(torch.empty(size=(num_heads, in_features, self.head_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # 拆分注意力向量以利用广播：a^T [h_i || h_j] = a_src^T h_i + a_dst^T h_j
        self.a_src = nn.Parameter(torch.empty(size=(num_heads, self.head_dim, 1)))
        self.a_dst = nn.Parameter(torch.empty(size=(num_heads, self.head_dim, 1)))
        nn.init.xavier_uniform_(self.a_src.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        h: (N, in_features)
        adj: (N, N) 或 None
        返回: (N, out_features)
        """
        N = h.size(0)

        # (num_heads, N, head_dim)
        Wh = torch.einsum('ni,hid->hnd', h, self.W)

        # attn_src: (heads, N, 1)
        attn_src = torch.matmul(Wh, self.a_src)
        # attn_dst: (heads, N, 1)
        attn_dst = torch.matmul(Wh, self.a_dst)

        # 广播相加 -> (heads, N, N)
        attn_logits = attn_src + attn_dst.transpose(-1, -2)
        e = self.leakyrelu(attn_logits)

        if adj is not None:
            if adj.dim() == 2:
                adj = adj.unsqueeze(0)
            # 数值稳定性：防止 log(0)
            adj_safe = adj.clamp(min=1e-6)
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e + torch.log(adj_safe), zero_vec)
        else:
            attention = e

        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # (heads, N, dim)
        h_prime = torch.matmul(attention, Wh)

        # (N, heads*dim)
        h_prime = h_prime.permute(1, 0, 2).contiguous().view(N, -1)

        return F.elu(h_prime) if self.concat else h_prime


# ================= 2. Graph-RWKV 完整模型（新方向核心模型）=================
class GraphRWKV_GNN_Model(nn.Module):
    """
    【新方向核心模型】Graph-RWKV：基于大语言模型动态图谱与 Graph-RWKV 的时空解耦金融预测
    
    【核心架构】：
    1. RWKV 时间序列编码器（时间维）：
       - 使用 GraphRWKV_Model 提取时序特征
       - 输出：包含时间上下文的节点嵌入 H_t ∈ R^(N×D)
    
    2. 动态图注意力网络 GAT（空间维）：
       - 基于 LLM 增强的情感加权混合图进行空间聚合
       - 混合图包含：语义图（LLM提取的关系+情感极性）+ 统计图（收益率相关性）
       - 输出：空间聚合后的节点嵌入 H'_t = GAT(H_t, A_t^final)
    
    3. 时空特征融合：
       - Concat(H_temporal, H_graph) -> Linear -> 收益率预测
    """

    def __init__(
        self,
        input_dim=8,
        n_embd=32,
        n_layers=2,
        num_nodes=500,
        adj_matrix=None,
        gnn_embd=None,
        dropout=0.1,
        max_neighbors: int = 128,
        use_pyg: Optional[bool] = None,
        temporal_backend: str = "rwkv",
    ):
        super().__init__()
        gnn_embd = gnn_embd or min(n_embd, 64)
        self.max_neighbors = int(max_neighbors)
        self.use_pyg = bool(_HAS_PYG if use_pyg is None else use_pyg)

        # ----- 1. 时序编码器：支持 RWKV/LSTM/GRU -----
        temporal_backend = str(temporal_backend).lower()
        if temporal_backend in ("lstm", "gru"):
            self.temporal_encoder = RNN_TemporalEncoder(
                input_dim=input_dim,
                n_embd=n_embd,
                n_layers=n_layers,
                dropout=dropout,
                rnn_type=temporal_backend,
            )
        else:
            self.temporal_encoder = GraphRWKV_Model(
                input_dim=input_dim,
                n_embd=n_embd,
                n_layers=n_layers,
                dropout=dropout,
            )
            # 去掉 head，只做特征提取（供 GAT 使用）
            self.temporal_encoder.head = nn.Identity()
            self.temporal_encoder.pre_head_dropout = nn.Identity()

        # 若 GAT 使用较小维度以省显存，则先做投影
        self.gnn_proj = nn.Linear(n_embd, gnn_embd) if gnn_embd != n_embd else nn.Identity()

        # ----- 2. 图注意力层 GAT -----
        if self.use_pyg:
            try:
                self.gnn = GATv2Conv(
                    gnn_embd,
                    gnn_embd,
                    heads=2,
                    concat=False,
                    dropout=dropout,
                    edge_dim=1,
                    add_self_loops=False,
                )
            except Exception:
                self.use_pyg = False
                self.gnn = GraphAttentionLayer(
                    in_features=gnn_embd,
                    out_features=gnn_embd,
                    dropout=dropout,
                    alpha=0.2,
                )
        else:
            self.gnn = GraphAttentionLayer(
                in_features=gnn_embd,
                out_features=gnn_embd,
                dropout=dropout,
                alpha=0.2,
            )
        
        # 【新增】GAT 后的 LayerNorm，稳定图特征
        self.gnn_ln = nn.LayerNorm(gnn_embd)

        # 许多顶会工作在 mini-batch 场景会给"未出现在 batch 的邻居节点"提供一个可学习的静态表示，
        # 避免图消息传递完全依赖 batch 采样（否则邻居不在 batch 时无法聚合）。
        self.node_embedding = nn.Embedding(num_nodes, gnn_embd)
        nn.init.normal_(self.node_embedding.weight, mean=0.0, std=0.02)

        # ----- 3. 融合头：时序 + 图 -> 预测 -----
        fusion_dim = n_embd + gnn_embd
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1),
        )

        # ----- 5. 邻接矩阵：不参与梯度，随模型保存；未提供时退化为单位阵（仅自环）-----
        if adj_matrix is None:
            adj_matrix = np.eye(num_nodes, dtype=np.float32)
        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
        self.register_buffer("adj", adj_matrix)

        # ----- 6. 邻居缓存（性能优化）-----
        # 说明：
        # - 旧实现：每次 forward 都对 dense 邻接做 row.nonzero()，并且 self.adj.to(device) 会产生隐形拷贝
        # - 新实现：初始化时预计算每个节点的 Top-K 邻居索引（不足用 -1 padding），forward 只做索引与去重
        # - 对于大量“孤立节点”（度=0）的场景，缓存能显著降低开销
        self.register_buffer(
            "neighbor_index",
            self._build_neighbor_index(self.adj, max_k=self.max_neighbors),
        )

        self.n_embd = n_embd
        self.gnn_embd = gnn_embd

    @staticmethod
    def _build_neighbor_index(adj: torch.Tensor, max_k: int) -> torch.Tensor:
        """
        预计算每个节点的邻居列表（Top-K），用于 forward 的快速邻居扩展。

        Args:
            adj: (N, N) 邻接矩阵（float/bool 均可，>0 视为有边；通常包含自环）
            max_k: 每个节点最多保留多少个邻居（不含自环）。<=0 则返回空矩阵。

        Returns:
            neighbor_index: (N, K) 的 long Tensor，缺省位置用 -1 填充。
        """
        if adj is None:
            return torch.empty((0, 0), dtype=torch.long)
        N = int(adj.shape[0])
        K = int(max(0, max_k))
        if K == 0 or N == 0:
            return torch.empty((N, 0), dtype=torch.long)

        # 在 CPU 上构建更稳（一次性成本），随后作为 buffer 随模型迁移到 GPU
        adj_cpu = adj.detach().cpu()
        if adj_cpu.is_leaf:
            adj_cpu = adj_cpu.clone()
        # 移除自环
        try:
            adj_cpu.fill_diagonal_(0)
        except Exception:
            pass

        # 使用 topk 获取权重最大的 K 个邻居 (对于二值图，相当于前 K 个非零)
        # values: (N, K), indices: (N, K)
        values, indices = torch.topk(adj_cpu, k=K, dim=1)

        # 掩码处理：若权重 <= eps 则视为不存在
        mask = values > 1e-6

        neigh = torch.full((N, K), -1, dtype=torch.long)
        if mask.any():
            neigh[mask] = indices[mask]

        return neigh

    def forward(self, x, vol=None, node_indices=None):
        """
        x: (B, T, F) 输入序列
        vol: (B, 1) 波动率（保留以兼容接口，但新方向中不使用）
        node_indices: (B,) 可选，当前 batch 各样本在整图中的节点索引；
                      若为 None，则用单位阵作为 batch 邻接（GNN 退化为仅自环，等价于 MLP）
        """
        # Step 1: 时序特征 (B, n_embd) - 使用 RWKV 时间编码器
        h_temporal = self.temporal_encoder(x, vol)

        # Step 2: 投影到 GAT 维度（若 gnn_embd != n_embd）
        h_gnn_in = self.gnn_proj(h_temporal)  # (B, gnn_embd)

        # Step 3: 图聚合（mini-batch + 邻居扩展）
        # 关键修复：不要把 batch 的每条样本都当作“不同节点”直接做 GAT，
        # 否则当 batch 里出现同一 ticker 的多个样本时，会被当成多个节点，语义错误。
        if node_indices is None:
            if self.adj is None:
                h_graph = h_gnn_in
            else:
                adj = self.adj
                if self.use_pyg:
                    edge_index = (adj > 0).nonzero(as_tuple=False).t().contiguous()
                    edge_attr = adj[edge_index[0], edge_index[1]].view(-1, 1)
                    h_graph = self.gnn(h_gnn_in, edge_index, edge_attr)
                else:
                    h_graph = self.gnn(h_gnn_in, adj)
                h_graph = self.gnn_ln(h_graph)
        elif self.adj is None:
            h_graph = h_gnn_in
        else:
            device = x.device
            node_indices = node_indices.view(-1)

            # 3.1 去重：unique tickers in batch
            uniq_nodes, inv = torch.unique(node_indices, sorted=False, return_inverse=True)

            # 3.2 聚合重复 ticker 的特征（mean pooling）
            uniq_feats = torch.zeros((uniq_nodes.numel(), h_gnn_in.size(1)), device=device, dtype=h_gnn_in.dtype)
            uniq_counts = torch.zeros((uniq_nodes.numel(), 1), device=device, dtype=h_gnn_in.dtype)
            uniq_feats.index_add_(0, inv, h_gnn_in)
            uniq_counts.index_add_(0, inv, torch.ones((h_gnn_in.size(0), 1), device=device, dtype=h_gnn_in.dtype))
            uniq_feats = uniq_feats / uniq_counts.clamp(min=1.0)

            # 3.3 邻居扩展：使用预计算 neighbor_index（避免 forward 里反复 nonzero 扫描邻接矩阵）
            K = int(max(0, self.max_neighbors))
            adj = self.adj  # buffer 会随 model.to(device) 自动迁移，避免每步 .to() 的隐形拷贝
            neigh_nodes = torch.empty(0, dtype=torch.long, device=device)
            if K > 0 and getattr(self, "neighbor_index", None) is not None and self.neighbor_index.numel() > 0:
                nb = self.neighbor_index[uniq_nodes]  # (U, K)
                nb = nb.reshape(-1)
                nb = nb[nb >= 0]
                if nb.numel() > 0:
                    neigh_nodes = torch.unique(nb)

            # torch.unique 默认会排序输出（sorted=True），便于后续 searchsorted 做稳定映射
            sub_nodes = torch.unique(torch.cat([uniq_nodes, neigh_nodes], dim=0))

            # 3.4 子图初始节点特征：默认用可学习 node embedding；batch 节点用时序特征覆盖
            sub_feats = self.node_embedding(sub_nodes)  # (Ns, gnn_embd)

            # 将 uniq_nodes 映射到 sub_nodes 的位置（sub_nodes 已排序，映射稳定且更快）
            uniq_pos_in_sub = torch.searchsorted(sub_nodes, uniq_nodes)
            sub_feats[uniq_pos_in_sub] = uniq_feats

            # 注意：这里构造的是诱导子图的 dense 邻接；当图节点规模在几百（S&P500）时非常快
            sub_adj = adj.index_select(0, sub_nodes).index_select(1, sub_nodes)
            if self.use_pyg:
                edge_index = (sub_adj > 0).nonzero(as_tuple=False).t().contiguous()
                edge_attr = sub_adj[edge_index[0], edge_index[1]].view(-1, 1)
                sub_out = self.gnn(sub_feats, edge_index, edge_attr)
            else:
                sub_out = self.gnn(sub_feats, sub_adj)
            sub_out = self.gnn_ln(sub_out)

            # 取回 batch 节点的图特征，并映射回每条样本
            uniq_graph = sub_out[uniq_pos_in_sub]  # (U, gnn_embd) 顺序与 uniq_nodes 对齐
            h_graph = uniq_graph[inv]              # (B, gnn_embd) 映射回每条样本

        # Step 4: 融合头预测
        h_combined = torch.cat([h_temporal, h_graph], dim=1)
        out = self.fusion_head(h_combined)
        return out


# ================= 测试入口 =================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, D = 4, 30, 8
    x = torch.randn(B, T, D).to(device)
    vol = (torch.rand(B, 1) * 2).to(device)

    # 随机邻接或单位阵
    adj = np.eye(10)
    adj[0, 1] = adj[1, 0] = 1

    model = GraphRWKV_GNN_Model(
        input_dim=D,
        n_embd=32,
        n_layers=2,
        num_nodes=10,
        adj_matrix=adj,
        gnn_embd=32,
    ).to(device)

    with torch.no_grad():
        y = model(x, vol, node_indices=None)
    print("GraphRWKV_GNN_Model 测试通过, 输出形状:", y.shape)

# 为了向后兼容，创建别名
QL_MATCC_GNN_Model = GraphRWKV_GNN_Model
