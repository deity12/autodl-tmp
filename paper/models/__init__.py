# -*- coding: utf-8 -*-
"""
模型定义模块
- base_model: 基础模型 QL_MATCC_Model（量子-RWKV + MATCC）
- gnn_model: GNN模型 QL_MATCC_GNN_Model（加入图注意力网络）
"""

from models.base_model import (
    QL_MATCC_Model,
    Quantum_ChannelMixing,
    Classical_ChannelMixing,
    RWKV_TimeMixing,
    MATCCDecompose,
    VQC_Block,
)

from models.gnn_model import (
    QL_MATCC_GNN_Model,
    GraphAttentionLayer,
)

__all__ = [
    'QL_MATCC_Model',
    'QL_MATCC_GNN_Model',
    'Quantum_ChannelMixing',
    'Classical_ChannelMixing',
    'RWKV_TimeMixing',
    'MATCCDecompose',
    'VQC_Block',
    'GraphAttentionLayer',
]
