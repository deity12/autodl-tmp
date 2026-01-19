#!/bin/bash

# ==========================================
# 师兄特调：金融风控项目环境配置脚本
# 适配：AutoDL 基础镜像 / PyTorch 2.1.2 / CUDA 11.8
# ==========================================

echo ">>> [1/5] 开启 AutoDL 学术加速 (下载速度起飞)..."
source /etc/network_turbo

echo ">>> [2/5] 升级 pip 并安装基础科学库..."
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn tqdm jupyterlab

echo ">>> [3/5] 安装量子计算 (PennyLane)..."
pip install pennylane
# 你的 CUDA 是 11.8，安装 gpu 加速插件可能会有版本对齐问题
# 这里先只装基础版，保证代码能跑通。如果慢，后续再单独处理加速。

echo ">>> [4/5] 安装 LLM 大模型必备库..."
# bitsandbytes 必须要有，不然大模型跑不起来
# transformers, accelerate 是 HuggingFace 的核心
pip install transformers accelerate bitsandbytes sentencepiece protobuf modelscope

echo ">>> [5/5] 安装图神经网络 (PyG) [关键步骤]..."
# !!! 注意：这里我专门为你改成了 cu118 (匹配你的 CUDA 11.8) !!!
# 如果装错版本，这一步会直接报错
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

echo ">>> [6/6] 安装其他项目依赖..."
pip install einops ninja

echo "=========================================="
echo "环境配置完毕！CUDA 11.8 适配成功！"
echo "请记得把数据存放在 /root/autodl-tmp/ 目录下！"
echo "=========================================="