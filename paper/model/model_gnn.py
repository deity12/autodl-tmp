# -*- coding: utf-8 -*-
"""
QL-MATCC-GNN 模型：在 Quantum-RWKV + MATCC 基础上融合 GNN（对应论文 2.4 动态图演化预测）
========================================================================
架构：
    1. MATCC：趋势/波动解耦
    2. Quantum-RWKV：时序特征 H_temporal
    3. GAT：基于 LLM 图谱的空间聚合 H_graph
    4. 融合：Concat(H_temporal, H_graph) -> Linear -> 收益率预测

依赖：
    - model.model：QL_MATCC_Model, Quantum_ChannelMixing, RWKV_TimeMixing, MATCCDecompose
    - 邻接矩阵：./data/processed/Graph_Adjacency.npy（由 build_graph.py 生成）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 从同目录的 model.py 复用：QL_MATCC_Model、量子/经典通道、RWKV、MATCC 等
# 注意：需保证 model 目录在 sys.path 中，使 "model" 解析为 model.py
from model import QL_MATCC_Model, Quantum_ChannelMixing, RWKV_TimeMixing, MATCCDecompose


# ================= 1. 图注意力层 GAT =================
class GraphAttentionLayer(nn.Module):
    """
    简化的 GAT 层：利用注意力机制聚合邻居特征。
    对应论文：基于 LLM 构建的图谱进行空间特征聚合。

    公式：h' = ELU( softmax( LeakyReLU( a^T [Wh_i || Wh_j] ) ) @ (W @ h) )
    其中邻接矩阵 adj 决定哪些 (i,j) 可参与注意力；非边位置掩码为 -inf，softmax 后为 0。
    """

    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # 线性变换 W: in_features -> out_features
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # 注意力系数 a: [Wh_i || Wh_j] -> 标量
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        h: (N, in_features) 节点特征，N=当前 batch 的节点数（或全图节点数）
        adj: (N, N) 邻接矩阵，>0 表示有边；用于掩码注意力，只对有边的节点对计算注意力
        """
        Wh = torch.mm(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # 无边的位置置为 -inf，softmax 后为 0，不参与聚合
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        """ 构造 (i,j) 对的 [Wh_i || Wh_j]，输出 (N, N, 2*out_features)。 """
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)


# ================= 2. 完整模型 QL_MATCC_GNN_Model =================
class QL_MATCC_GNN_Model(nn.Module):
    """
    【论文最终模型】QL-MATCC-GNN
    1. MATCC：趋势/波动解耦
    2. Quantum-RWKV：时序特征 H_temporal（取最后一时刻）
    3. GAT：基于邻接矩阵的空间聚合 H_graph
    4. 融合：Concat(H_temporal, H_graph) -> Linear -> 1（收益率）
    """

    def __init__(
        self,
        input_dim=8,
        n_embd=32,
        n_layers=2,
        n_qubits=4,
        num_nodes=500,
        adj_matrix=None,
        gnn_embd=None,
        use_matcc=True,
        use_market_guidance=True,
        use_quantum=True,
        ma_window=5,
        q_threshold=0.0,
    ):
        super().__init__()
        gnn_embd = gnn_embd or min(n_embd, 64)

        # ----- 1. 时序编码器：复用 QL_MATCC_Model，去掉 head，只做特征提取 -----
        self.temporal_encoder = QL_MATCC_Model(
            input_dim=input_dim,
            n_embd=n_embd,
            n_layers=n_layers,
            n_qubits=n_qubits,
            use_matcc=use_matcc,
            use_market_guidance=use_market_guidance,
            use_quantum=use_quantum,
            ma_window=ma_window,
            q_threshold=q_threshold,
        )
        self.temporal_encoder.head = nn.Identity()

        # 若 GAT 使用较小维度以省显存，则先做投影
        self.gnn_proj = nn.Linear(n_embd, gnn_embd) if gnn_embd != n_embd else nn.Identity()

        # ----- 2. 图注意力层 GAT -----
        self.gnn = GraphAttentionLayer(
            in_features=gnn_embd,
            out_features=gnn_embd,
            dropout=0.1,
            alpha=0.2,
        )

        # ----- 3. 融合头：时序 + 图 -> 预测 -----
        self.fusion_head = nn.Linear(n_embd + gnn_embd, 1)

        # ----- 4. 邻接矩阵：不参与梯度，随模型保存；未提供时退化为单位阵（仅自环）-----
        if adj_matrix is None:
            adj_matrix = np.eye(num_nodes, dtype=np.float32)
        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
        self.register_buffer("adj", adj_matrix)

        self.n_embd = n_embd
        self.gnn_embd = gnn_embd

    def forward(self, x, vol, node_indices=None):
        """
        x: (B, T, F) 输入序列
        vol: (B, 1) 波动率，供 Quantum 分支使用
        node_indices: (B,) 可选，当前 batch 各样本在整图中的节点索引；
                      若为 None，则用单位阵作为 batch 邻接（GNN 退化为仅自环，等价于 MLP）
        """
        # Step 1: 时序特征 (B, n_embd)
        h_temporal = self.temporal_encoder(x, vol)

        # Step 2: 投影到 GAT 维度（若 gnn_embd != n_embd）
        h_gnn_in = self.gnn_proj(h_temporal)

        # Step 3: 构造当前 batch 的邻接子矩阵
        if node_indices is not None and self.adj is not None:
            batch_adj = self.adj[node_indices][:, node_indices]
        else:
            # 退化：无图信息时仅自环，GAT 输出 ≈ 恒等
            B = x.size(0)
            batch_adj = torch.eye(B, device=x.device, dtype=x.dtype)

        h_graph = self.gnn(h_gnn_in, batch_adj)

        # Step 4: 融合并预测
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

    model = QL_MATCC_GNN_Model(
        input_dim=D,
        n_embd=32,
        n_layers=2,
        n_qubits=4,
        num_nodes=10,
        adj_matrix=adj,
        gnn_embd=32,
    ).to(device)

    with torch.no_grad():
        y = model(x, vol, node_indices=None)
    print("QL_MATCC_GNN_Model 测试通过, 输出形状:", y.shape)
