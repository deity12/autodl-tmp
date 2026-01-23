# -*- coding: utf-8 -*-
"""
模型定义模块（新方向：Graph-RWKV）
- base_model: GraphRWKV_Model（RWKV 时间序列编码器）
- gnn_model: GraphRWKV_GNN_Model（RWKV 时间编码器 + 动态 GAT 空间聚合）

【注意】以下组件在新方向中不使用，已注释：
- Quantum_ChannelMixing, VQC_Block（量子计算）
- MATCCDecompose（趋势解耦）
"""

from .base_model import (
    GraphRWKV_Model,
    QL_MATCC_Model,  # 兼容性别名
    Classical_ChannelMixing,
    RWKV_TimeMixing,
    MATCCDecompose,  # 保留以兼容，但会抛出 NotImplementedError
)

from .gnn_model import (
    GraphRWKV_GNN_Model,
    QL_MATCC_GNN_Model,  # 兼容性别名
    GraphAttentionLayer,
)

__all__ = [
    'GraphRWKV_Model',
    'GraphRWKV_GNN_Model',
    'QL_MATCC_Model',  # 兼容性别名
    'QL_MATCC_GNN_Model',  # 兼容性别名
    'Classical_ChannelMixing',
    'RWKV_TimeMixing',
    'MATCCDecompose',  # 保留以兼容
    'GraphAttentionLayer',
]
