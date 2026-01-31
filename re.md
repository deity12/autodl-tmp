------------------------------------------------------------
Epoch 5/30 [Train]: 100%|██████████| 570/570 [01:34<00:00,  6.05it/s, avg=-0.020623, loss=-0.013229]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 5/30: Train=-0.020623, Val=0.001603, lr=1.00e-03
------------------------------------------------------------
Epoch 6/30 [Train]: 100%|██████████| 570/570 [01:33<00:00,  6.09it/s, avg=-0.016834, loss=-0.010167]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 6/30: Train=-0.016834, Val=0.002372, lr=9.76e-04
------------------------------------------------------------
Epoch 7/30 [Train]: 100%|██████████| 570/570 [01:33<00:00,  6.07it/s, avg=-0.012100, loss=-0.324822]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 7/30: Train=-0.012100, Val=0.000922, lr=9.05e-04

🛑 Early stopping (best val loss: 0.000695)

>>> 加载最佳模型并在测试集上评估...

📊 测试集最终评估指标（2021-01-01 ~ 2023-12-31）:
   MSE: 0.000469, RMSE: 0.021662
   R²: -0.1663, MAE: 0.015402
   方向准确率: 49.93%
   IC: 0.0031, ICIR: 0.0294
   RankIC: 0.0045, RankICIR: 0.0358
>>> 训练曲线已保存: ./outputs/figures/training_curve_ablation_w_o_semantic.png
>>> Loss 数据已保存: ./outputs/logs/training_losses_ablation_w_o_semantic.json
>>> 训练摘要已保存: ./outputs/results/experiment_ablation_w_o_semantic.json

============================================================
>>> Graph-RWKV Model 训练结束
    Best Val Loss: 0.000695

    📊 测试集指标（论文报告此区间）:
      IC: 0.0031
      RankIC: 0.0045
      ICIR: 0.0294
      RankICIR: 0.0358
      R² Score: -0.1663
      Directional Accuracy: 49.93%
============================================================
2026-02-01 00:30:20 - INFO - 开始消融训练: w/o_statistical
2026-02-01 00:30:20 - INFO - 训练配置覆盖: {'csv_path': './paper/data/processed/Final_Model_Data.csv', 'n_embd': 64, 'n_layers': 2, 'gnn_embd': 32, 'dropout': 0.3, 'seq_len': 60, 'weight_decay': 0.001, 'early_stop_patience': 5, 'rank_loss_weight': 1.0, 'batch_size': 1024, 'epochs': 30, 'lr': 0.001, 'num_workers': 10, 'prefetch_factor': 4, 'pin_memory': True, 'persistent_workers': True, 'use_amp': True, 'use_compile': False, 'output_dir': './outputs', 'graph_path': './paper/data/processed/Graph_Adjacency_semantic.npy', 'graph_tickers_path': './paper/data/processed/Graph_Tickers.json', 'experiment_name': 'ablation_w_o_statistical', 'checkpoint_name': 'best_model_ablation_w_o_statistical.pth', 'use_graph': True, 'temporal_backend': 'rwkv', 'use_rank_loss': True}
>>> [Reproducibility] Random seed set to 42
>>> Training on device: cuda
   GPU: NVIDIA GeForce RTX 3090
   显存: 47.4 GB
   Batch Size: 1024

>>> Loading Datasets...
正在加载 train 数据，来源：./paper/data/processed/Final_Model_Data.csv...
    [特征] 从 /root/autodl-tmp/paper/data/processed/feature_columns.json 读取特征列，共 158 维
✅ 数据清洗完成：无无穷值、无缺失值、极端值已裁剪。
    [V4 对齐] 从 Graph_Tickers.json 读取 375 个图节点
    [V4 对齐] 过滤后数据包含 375 只股票
正在构建滑动窗口索引...
train 数据集共生成样本数: 209216
正在加载 test 数据，来源：./paper/data/processed/Final_Model_Data.csv...
    [特征] 从 /root/autodl-tmp/paper/data/processed/feature_columns.json 读取特征列，共 158 维
✅ 数据清洗完成：无无穷值、无缺失值、极端值已裁剪。
    [V4 对齐] 从 Graph_Tickers.json 读取 375 个图节点
    [V4 对齐] 过滤后数据包含 375 只股票
正在构建滑动窗口索引...
test 数据集共生成样本数: 19652
正在加载 test 数据，来源：./paper/data/processed/Final_Model_Data.csv...
    [特征] 从 /root/autodl-tmp/paper/data/processed/feature_columns.json 读取特征列，共 158 维
✅ 数据清洗完成：无无穷值、无缺失值、极端值已裁剪。
    [V4 对齐] 从 Graph_Tickers.json 读取 375 个图节点
    [V4 对齐] 过滤后数据包含 375 只股票
正在构建滑动窗口索引...
test 数据集共生成样本数: 199988
   Train: 209216, Valid: 19652, Test: 199988
   Input Dim: 158 (features)
>>> Artifacts 已保存: ./outputs/results/artifacts_ablation_w_o_statistical
>>> 加载图谱: ./paper/data/processed/Graph_Adjacency_semantic.npy, 形状: (375, 375)
✅ 图谱 tickers 顺序校验通过（与 dataset.ticker2idx 对齐）

>>> Initializing Graph-RWKV Model...
   Total parameters: 133,569
   学习率: 1.00e-03
   ✅ 已启用混合精度训练 (AMP)

>>> Start Training (Graph-RWKV Model)...
============================================================
Epoch 1/30 [Train]: 100%|███████████| 570/570 [01:36<00:00,  5.89it/s, avg=0.015982, loss=-0.046603]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 1/30: Train=0.015982, Val=0.004675, lr=9.05e-04
  🌟 Best model saved!
     R²=-7.2331, MAE=0.063068, DirAcc=45.70%, IC=-0.0133
------------------------------------------------------------
Epoch 2/30 [Train]: 100%|███████████| 570/570 [01:34<00:00,  6.04it/s, avg=0.002361, loss=-0.006948]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 2/30: Train=0.002361, Val=0.000822, lr=6.55e-04
  🌟 Best model saved!
     R²=-0.4327, MAE=0.021374, DirAcc=52.44%, IC=0.0002
------------------------------------------------------------
Epoch 3/30 [Train]: 100%|███████████| 570/570 [01:35<00:00,  5.98it/s, avg=-0.005154, loss=0.017276]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 3/30: Train=-0.005154, Val=0.000790, lr=3.46e-04
  🌟 Best model saved!
     R²=-0.3717, MAE=0.019650, DirAcc=48.15%, IC=0.0092
------------------------------------------------------------
Epoch 4/30 [Train]: 100%|██████████| 570/570 [01:32<00:00,  6.15it/s, avg=-0.004227, loss=-0.023179]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 4/30: Train=-0.004227, Val=0.001311, lr=9.64e-05
------------------------------------------------------------
Epoch 5/30 [Train]: 100%|███████████| 570/570 [01:33<00:00,  6.12it/s, avg=-0.013653, loss=0.000382]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 5/30: Train=-0.013653, Val=0.001580, lr=1.00e-03
------------------------------------------------------------
Epoch 6/30 [Train]: 100%|██████████| 570/570 [01:33<00:00,  6.12it/s, avg=-0.006826, loss=-0.116150]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 6/30: Train=-0.006826, Val=0.005057, lr=9.76e-04
------------------------------------------------------------
Epoch 7/30 [Train]: 100%|██████████| 570/570 [01:38<00:00,  5.79it/s, avg=-0.005869, loss=-0.143975]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 7/30: Train=-0.005869, Val=0.002065, lr=9.05e-04
------------------------------------------------------------
Epoch 8/30 [Train]: 100%|██████████| 570/570 [01:32<00:00,  6.16it/s, avg=-0.004115, loss=-0.060197]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 8/30: Train=-0.004115, Val=0.001484, lr=7.94e-04

🛑 Early stopping (best val loss: 0.000790)

>>> 加载最佳模型并在测试集上评估...

📊 测试集最终评估指标（2021-01-01 ~ 2023-12-31）:
   MSE: 0.000564, RMSE: 0.023745
   R²: -0.4013, MAE: 0.017914
   方向准确率: 48.99%
   IC: 0.0046, ICIR: 0.0269
   RankIC: 0.0076, RankICIR: 0.0417
>>> 训练曲线已保存: ./outputs/figures/training_curve_ablation_w_o_statistical.png
>>> Loss 数据已保存: ./outputs/logs/training_losses_ablation_w_o_statistical.json
>>> 训练摘要已保存: ./outputs/results/experiment_ablation_w_o_statistical.json

============================================================
>>> Graph-RWKV Model 训练结束
    Best Val Loss: 0.000790

    📊 测试集指标（论文报告此区间）:
      IC: 0.0046
      RankIC: 0.0076
      ICIR: 0.0269
      RankICIR: 0.0417
      R² Score: -0.4013
      Directional Accuracy: 48.99%
============================================================
2026-02-01 00:44:23 - INFO - 开始消融训练: w/o_sentiment
2026-02-01 00:44:23 - INFO - 训练配置覆盖: {'csv_path': './paper/data/processed/Final_Model_Data.csv', 'n_embd': 64, 'n_layers': 2, 'gnn_embd': 32, 'dropout': 0.3, 'seq_len': 60, 'weight_decay': 0.001, 'early_stop_patience': 5, 'rank_loss_weight': 1.0, 'batch_size': 1024, 'epochs': 30, 'lr': 0.001, 'num_workers': 10, 'prefetch_factor': 4, 'pin_memory': True, 'persistent_workers': True, 'use_amp': True, 'use_compile': False, 'output_dir': './outputs', 'graph_path': './paper/data/processed/Graph_Adjacency_semantic_nosent.npy', 'graph_tickers_path': './paper/data/processed/Graph_Tickers.json', 'experiment_name': 'ablation_w_o_sentiment', 'checkpoint_name': 'best_model_ablation_w_o_sentiment.pth', 'use_graph': True, 'temporal_backend': 'rwkv', 'use_rank_loss': True}
>>> [Reproducibility] Random seed set to 42
>>> Training on device: cuda
   GPU: NVIDIA GeForce RTX 3090
   显存: 47.4 GB
   Batch Size: 1024

>>> Loading Datasets...
正在加载 train 数据，来源：./paper/data/processed/Final_Model_Data.csv...
    [特征] 从 /root/autodl-tmp/paper/data/processed/feature_columns.json 读取特征列，共 158 维
✅ 数据清洗完成：无无穷值、无缺失值、极端值已裁剪。
    [V4 对齐] 从 Graph_Tickers.json 读取 375 个图节点
    [V4 对齐] 过滤后数据包含 375 只股票
正在构建滑动窗口索引...
train 数据集共生成样本数: 209216
正在加载 test 数据，来源：./paper/data/processed/Final_Model_Data.csv...
    [特征] 从 /root/autodl-tmp/paper/data/processed/feature_columns.json 读取特征列，共 158 维
✅ 数据清洗完成：无无穷值、无缺失值、极端值已裁剪。
    [V4 对齐] 从 Graph_Tickers.json 读取 375 个图节点
    [V4 对齐] 过滤后数据包含 375 只股票
正在构建滑动窗口索引...
test 数据集共生成样本数: 19652
正在加载 test 数据，来源：./paper/data/processed/Final_Model_Data.csv...
    [特征] 从 /root/autodl-tmp/paper/data/processed/feature_columns.json 读取特征列，共 158 维
✅ 数据清洗完成：无无穷值、无缺失值、极端值已裁剪。
    [V4 对齐] 从 Graph_Tickers.json 读取 375 个图节点
    [V4 对齐] 过滤后数据包含 375 只股票
正在构建滑动窗口索引...
test 数据集共生成样本数: 199988
   Train: 209216, Valid: 19652, Test: 199988
   Input Dim: 158 (features)
>>> Artifacts 已保存: ./outputs/results/artifacts_ablation_w_o_sentiment
>>> 加载图谱: ./paper/data/processed/Graph_Adjacency_semantic_nosent.npy, 形状: (375, 375)
✅ 图谱 tickers 顺序校验通过（与 dataset.ticker2idx 对齐）

>>> Initializing Graph-RWKV Model...
   Total parameters: 133,569
   学习率: 1.00e-03
   ✅ 已启用混合精度训练 (AMP)

>>> Start Training (Graph-RWKV Model)...
============================================================
Epoch 1/30 [Train]: 100%|████████████| 570/570 [01:35<00:00,  5.96it/s, avg=0.014722, loss=0.000984]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 1/30: Train=0.014722, Val=0.003439, lr=9.05e-04
  🌟 Best model saved!
     R²=-5.2890, MAE=0.026017, DirAcc=53.18%, IC=0.0063
------------------------------------------------------------
Epoch 2/30 [Train]: 100%|███████████| 570/570 [01:34<00:00,  6.00it/s, avg=0.001995, loss=-0.002066]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 2/30: Train=0.001995, Val=0.000813, lr=6.55e-04
  🌟 Best model saved!
     R²=-0.4129, MAE=0.020930, DirAcc=50.76%, IC=0.0048
------------------------------------------------------------
Epoch 3/30 [Train]: 100%|██████████| 570/570 [01:29<00:00,  6.38it/s, avg=-0.004761, loss=-0.076896]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 3/30: Train=-0.004761, Val=0.002338, lr=3.46e-04
------------------------------------------------------------
Epoch 4/30 [Train]: 100%|██████████| 570/570 [01:21<00:00,  6.96it/s, avg=-0.005881, loss=-0.025908]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 4/30: Train=-0.005881, Val=0.002504, lr=9.64e-05
------------------------------------------------------------
Epoch 5/30 [Train]: 100%|██████████| 570/570 [01:21<00:00,  6.99it/s, avg=-0.007526, loss=-0.027862]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 5/30: Train=-0.007526, Val=0.001868, lr=1.00e-03
------------------------------------------------------------
Epoch 6/30 [Train]: 100%|███████████| 570/570 [01:22<00:00,  6.91it/s, avg=0.000004, loss=-0.090478]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 6/30: Train=0.000004, Val=0.000918, lr=9.76e-04
------------------------------------------------------------
Epoch 7/30 [Train]: 100%|██████████| 570/570 [01:27<00:00,  6.52it/s, avg=-0.003160, loss=-0.059722]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 7/30: Train=-0.003160, Val=0.002244, lr=9.05e-04

🛑 Early stopping (best val loss: 0.000813)

>>> 加载最佳模型并在测试集上评估...

📊 测试集最终评估指标（2021-01-01 ~ 2023-12-31）:
   MSE: 0.000631, RMSE: 0.025122
   R²: -0.5686, MAE: 0.019483
   方向准确率: 50.15%
   IC: 0.0052, ICIR: 0.0383
   RankIC: 0.0055, RankICIR: 0.0393
>>> 训练曲线已保存: ./outputs/figures/training_curve_ablation_w_o_sentiment.png
>>> Loss 数据已保存: ./outputs/logs/training_losses_ablation_w_o_sentiment.json
>>> 训练摘要已保存: ./outputs/results/experiment_ablation_w_o_sentiment.json

============================================================
>>> Graph-RWKV Model 训练结束
    Best Val Loss: 0.000813

    📊 测试集指标（论文报告此区间）:
      IC: 0.0052
      RankIC: 0.0055
      ICIR: 0.0383
      RankICIR: 0.0393
      R² Score: -0.5686
      Directional Accuracy: 50.15%
============================================================
2026-02-01 00:56:03 - INFO - 开始消融训练: w/o_rankic
2026-02-01 00:56:03 - INFO - 训练配置覆盖: {'csv_path': './paper/data/processed/Final_Model_Data.csv', 'n_embd': 64, 'n_layers': 2, 'gnn_embd': 32, 'dropout': 0.3, 'seq_len': 60, 'weight_decay': 0.001, 'early_stop_patience': 5, 'rank_loss_weight': 1.0, 'batch_size': 1024, 'epochs': 30, 'lr': 0.001, 'num_workers': 10, 'prefetch_factor': 4, 'pin_memory': True, 'persistent_workers': True, 'use_amp': True, 'use_compile': False, 'output_dir': './outputs', 'graph_path': './paper/data/processed/Graph_Adjacency.npy', 'graph_tickers_path': './paper/data/processed/Graph_Tickers.json', 'experiment_name': 'ablation_w_o_rankic', 'checkpoint_name': 'best_model_ablation_w_o_rankic.pth', 'use_graph': True, 'temporal_backend': 'rwkv', 'use_rank_loss': False}
>>> [Reproducibility] Random seed set to 42
>>> Training on device: cuda
   GPU: NVIDIA GeForce RTX 3090
   显存: 47.4 GB
   Batch Size: 1024

>>> Loading Datasets...
正在加载 train 数据，来源：./paper/data/processed/Final_Model_Data.csv...
    [特征] 从 /root/autodl-tmp/paper/data/processed/feature_columns.json 读取特征列，共 158 维
✅ 数据清洗完成：无无穷值、无缺失值、极端值已裁剪。
    [V4 对齐] 从 Graph_Tickers.json 读取 375 个图节点
    [V4 对齐] 过滤后数据包含 375 只股票
正在构建滑动窗口索引...
train 数据集共生成样本数: 209216
正在加载 test 数据，来源：./paper/data/processed/Final_Model_Data.csv...
    [特征] 从 /root/autodl-tmp/paper/data/processed/feature_columns.json 读取特征列，共 158 维
✅ 数据清洗完成：无无穷值、无缺失值、极端值已裁剪。
    [V4 对齐] 从 Graph_Tickers.json 读取 375 个图节点
    [V4 对齐] 过滤后数据包含 375 只股票
正在构建滑动窗口索引...
test 数据集共生成样本数: 19652
正在加载 test 数据，来源：./paper/data/processed/Final_Model_Data.csv...
    [特征] 从 /root/autodl-tmp/paper/data/processed/feature_columns.json 读取特征列，共 158 维
✅ 数据清洗完成：无无穷值、无缺失值、极端值已裁剪。
    [V4 对齐] 从 Graph_Tickers.json 读取 375 个图节点
    [V4 对齐] 过滤后数据包含 375 只股票
正在构建滑动窗口索引...
test 数据集共生成样本数: 199988
   Train: 209216, Valid: 19652, Test: 199988
   Input Dim: 158 (features)
>>> Artifacts 已保存: ./outputs/results/artifacts_ablation_w_o_rankic
>>> 加载图谱: ./paper/data/processed/Graph_Adjacency.npy, 形状: (375, 375)
✅ 图谱 tickers 顺序校验通过（与 dataset.ticker2idx 对齐）

>>> Initializing Graph-RWKV Model...
   Total parameters: 133,569
   学习率: 1.00e-03
   ✅ 已启用混合精度训练 (AMP)

>>> Start Training (Graph-RWKV Model)...
============================================================
Epoch 1/30 [Train]: 100%|████████████| 570/570 [01:39<00:00,  5.72it/s, avg=0.003395, loss=0.000346]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 1/30: Train=0.003395, Val=0.000618, lr=9.05e-04
  🌟 Best model saved!
     R²=-0.0670, MAE=0.017235, DirAcc=53.71%, IC=-0.0164
------------------------------------------------------------
Epoch 2/30 [Train]: 100%|████████████| 570/570 [01:21<00:00,  7.01it/s, avg=0.000788, loss=0.000164]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 2/30: Train=0.000788, Val=0.000601, lr=6.55e-04
  🌟 Best model saved!
     R²=-0.0377, MAE=0.016890, DirAcc=53.91%, IC=0.0236
------------------------------------------------------------
Epoch 3/30 [Train]: 100%|████████████| 570/570 [01:23<00:00,  6.81it/s, avg=0.000709, loss=0.000245]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 3/30: Train=0.000709, Val=0.000594, lr=3.46e-04
  🌟 Best model saved!
     R²=-0.0245, MAE=0.016315, DirAcc=46.27%, IC=0.0231
------------------------------------------------------------
Epoch 4/30 [Train]: 100%|████████████| 570/570 [01:29<00:00,  6.39it/s, avg=0.000691, loss=0.000212]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 4/30: Train=0.000691, Val=0.000587, lr=9.64e-05
  🌟 Best model saved!
     R²=-0.0122, MAE=0.016202, DirAcc=50.71%, IC=0.0228
------------------------------------------------------------
Epoch 5/30 [Train]: 100%|████████████| 570/570 [01:35<00:00,  5.95it/s, avg=0.000678, loss=0.000156]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 5/30: Train=0.000678, Val=0.000583, lr=1.00e-03
  🌟 Best model saved!
     R²=-0.0059, MAE=0.016155, DirAcc=53.03%, IC=0.0215
------------------------------------------------------------
Epoch 6/30 [Train]: 100%|████████████| 570/570 [01:29<00:00,  6.38it/s, avg=0.000696, loss=0.000564]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 6/30: Train=0.000696, Val=0.000588, lr=9.76e-04
------------------------------------------------------------
Epoch 7/30 [Train]: 100%|████████████| 570/570 [01:30<00:00,  6.32it/s, avg=0.000690, loss=0.000265]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 7/30: Train=0.000690, Val=0.000579, lr=9.05e-04
  🌟 Best model saved!
     R²=0.0014, MAE=0.016162, DirAcc=53.91%, IC=0.0185
------------------------------------------------------------
Epoch 8/30 [Train]: 100%|████████████| 570/570 [01:37<00:00,  5.86it/s, avg=0.000674, loss=0.000080]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 8/30: Train=0.000674, Val=0.000581, lr=7.94e-04
------------------------------------------------------------
Epoch 9/30 [Train]: 100%|████████████| 570/570 [01:30<00:00,  6.28it/s, avg=0.000679, loss=0.001182]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 9/30: Train=0.000679, Val=0.000580, lr=6.55e-04
------------------------------------------------------------
Epoch 10/30 [Train]: 100%|███████████| 570/570 [01:26<00:00,  6.59it/s, avg=0.000675, loss=0.000154]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 10/30: Train=0.000675, Val=0.000580, lr=5.01e-04
------------------------------------------------------------
Epoch 11/30 [Train]: 100%|███████████| 570/570 [01:32<00:00,  6.18it/s, avg=0.000668, loss=0.000243]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 11/30: Train=0.000668, Val=0.000584, lr=3.46e-04
------------------------------------------------------------
Epoch 12/30 [Train]: 100%|███████████| 570/570 [01:29<00:00,  6.36it/s, avg=0.000672, loss=0.000224]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 12/30: Train=0.000672, Val=0.000595, lr=2.07e-04

🛑 Early stopping (best val loss: 0.000579)

>>> 加载最佳模型并在测试集上评估...

📊 测试集最终评估指标（2021-01-01 ~ 2023-12-31）:
   MSE: 0.000423, RMSE: 0.020572
   R²: -0.0518, MAE: 0.014542
   方向准确率: 51.12%
   IC: 0.0020, ICIR: 0.0101
   RankIC: 0.0030, RankICIR: 0.0133
>>> 训练曲线已保存: ./outputs/figures/training_curve_ablation_w_o_rankic.png
>>> Loss 数据已保存: ./outputs/logs/training_losses_ablation_w_o_rankic.json
>>> 训练摘要已保存: ./outputs/results/experiment_ablation_w_o_rankic.json

============================================================
>>> Graph-RWKV Model 训练结束
    Best Val Loss: 0.000579

    📊 测试集指标（论文报告此区间）:
      IC: 0.0020
      RankIC: 0.0030
      ICIR: 0.0101
      RankICIR: 0.0133
      R² Score: -0.0518
      Directional Accuracy: 51.12%
============================================================
2026-02-01 01:15:42 - INFO - 开始消融训练: lstm_backbone
2026-02-01 01:15:42 - INFO - 训练配置覆盖: {'csv_path': './paper/data/processed/Final_Model_Data.csv', 'n_embd': 64, 'n_layers': 2, 'gnn_embd': 32, 'dropout': 0.3, 'seq_len': 60, 'weight_decay': 0.001, 'early_stop_patience': 5, 'rank_loss_weight': 1.0, 'batch_size': 1024, 'epochs': 30, 'lr': 0.001, 'num_workers': 10, 'prefetch_factor': 4, 'pin_memory': True, 'persistent_workers': True, 'use_amp': True, 'use_compile': False, 'output_dir': './outputs', 'graph_path': './paper/data/processed/Graph_Adjacency.npy', 'graph_tickers_path': './paper/data/processed/Graph_Tickers.json', 'experiment_name': 'ablation_lstm_backbone', 'checkpoint_name': 'best_model_ablation_lstm_backbone.pth', 'use_graph': True, 'temporal_backend': 'lstm', 'use_rank_loss': True}
>>> [Reproducibility] Random seed set to 42
>>> Training on device: cuda
   GPU: NVIDIA GeForce RTX 3090
   显存: 47.4 GB
   Batch Size: 1024

>>> Loading Datasets...
正在加载 train 数据，来源：./paper/data/processed/Final_Model_Data.csv...
    [特征] 从 /root/autodl-tmp/paper/data/processed/feature_columns.json 读取特征列，共 158 维
✅ 数据清洗完成：无无穷值、无缺失值、极端值已裁剪。
    [V4 对齐] 从 Graph_Tickers.json 读取 375 个图节点
    [V4 对齐] 过滤后数据包含 375 只股票
正在构建滑动窗口索引...
train 数据集共生成样本数: 209216
正在加载 test 数据，来源：./paper/data/processed/Final_Model_Data.csv...
    [特征] 从 /root/autodl-tmp/paper/data/processed/feature_columns.json 读取特征列，共 158 维
✅ 数据清洗完成：无无穷值、无缺失值、极端值已裁剪。
    [V4 对齐] 从 Graph_Tickers.json 读取 375 个图节点
    [V4 对齐] 过滤后数据包含 375 只股票
正在构建滑动窗口索引...
test 数据集共生成样本数: 19652
正在加载 test 数据，来源：./paper/data/processed/Final_Model_Data.csv...
    [特征] 从 /root/autodl-tmp/paper/data/processed/feature_columns.json 读取特征列，共 158 维
✅ 数据清洗完成：无无穷值、无缺失值、极端值已裁剪。
    [V4 对齐] 从 Graph_Tickers.json 读取 375 个图节点
    [V4 对齐] 过滤后数据包含 375 只股票
正在构建滑动窗口索引...
test 数据集共生成样本数: 199988
   Train: 209216, Valid: 19652, Test: 199988
   Input Dim: 158 (features)
>>> Artifacts 已保存: ./outputs/results/artifacts_ablation_lstm_backbone
>>> 加载图谱: ./paper/data/processed/Graph_Adjacency.npy, 形状: (375, 375)
✅ 图谱 tickers 顺序校验通过（与 dataset.ticker2idx 对齐）

>>> Initializing Graph-RWKV Model...
   Total parameters: 100,161
   学习率: 1.00e-03
   ✅ 已启用混合精度训练 (AMP)

>>> Start Training (Graph-RWKV Model)...
============================================================
Epoch 1/30 [Train]: 100%|███████████| 570/570 [00:08<00:00, 67.83it/s, avg=0.007728, loss=-0.002428]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 1/30: Train=0.007728, Val=0.001901, lr=9.05e-04
  🌟 Best model saved!
     R²=-2.3800, MAE=0.036161, DirAcc=45.71%, IC=-0.0012
------------------------------------------------------------
Epoch 2/30 [Train]: 100%|████████████| 570/570 [00:06<00:00, 85.35it/s, avg=0.002649, loss=0.018475]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 2/30: Train=0.002649, Val=0.002995, lr=6.55e-04
------------------------------------------------------------
Epoch 3/30 [Train]: 100%|███████████| 570/570 [00:06<00:00, 88.33it/s, avg=-0.003270, loss=0.070752]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 3/30: Train=-0.003270, Val=0.000956, lr=3.46e-04
  🌟 Best model saved!
     R²=-0.6643, MAE=0.022702, DirAcc=47.98%, IC=-0.0028
------------------------------------------------------------
Epoch 4/30 [Train]: 100%|██████████| 570/570 [00:06<00:00, 84.08it/s, avg=-0.007634, loss=-0.015352]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 4/30: Train=-0.007634, Val=0.001445, lr=9.64e-05
------------------------------------------------------------
Epoch 5/30 [Train]: 100%|███████████| 570/570 [00:07<00:00, 74.33it/s, avg=-0.010492, loss=0.030896]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 5/30: Train=-0.010492, Val=0.001775, lr=1.00e-03
------------------------------------------------------------
Epoch 6/30 [Train]: 100%|███████████| 570/570 [00:06<00:00, 82.08it/s, avg=-0.005170, loss=0.019213]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 6/30: Train=-0.005170, Val=0.001482, lr=9.76e-04
------------------------------------------------------------
Epoch 7/30 [Train]: 100%|██████████| 570/570 [00:07<00:00, 78.22it/s, avg=-0.003737, loss=-0.032427]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 7/30: Train=-0.003737, Val=0.001395, lr=9.05e-04
------------------------------------------------------------
Epoch 8/30 [Train]: 100%|████████████| 570/570 [00:07<00:00, 75.84it/s, avg=0.001836, loss=0.032732]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 8/30: Train=0.001836, Val=0.000997, lr=7.94e-04

🛑 Early stopping (best val loss: 0.000956)

>>> 加载最佳模型并在测试集上评估...

📊 测试集最终评估指标（2021-01-01 ~ 2023-12-31）:
   MSE: 0.000696, RMSE: 0.026384
   R²: -0.7301, MAE: 0.020869
   方向准确率: 48.99%
   IC: 0.0061, ICIR: 0.0366
   RankIC: 0.0050, RankICIR: 0.0303
>>> 训练曲线已保存: ./outputs/figures/training_curve_ablation_lstm_backbone.png
>>> Loss 数据已保存: ./outputs/logs/training_losses_ablation_lstm_backbone.json
>>> 训练摘要已保存: ./outputs/results/experiment_ablation_lstm_backbone.json

============================================================
>>> Graph-RWKV Model 训练结束
    Best Val Loss: 0.000956

    📊 测试集指标（论文报告此区间）:
      IC: 0.0061
      RankIC: 0.0050
      ICIR: 0.0366
      RankICIR: 0.0303
      R² Score: -0.7301
      Directional Accuracy: 48.99%
============================================================
2026-02-01 01:18:08 - INFO - 开始消融训练: gru_backbone
2026-02-01 01:18:08 - INFO - 训练配置覆盖: {'csv_path': './paper/data/processed/Final_Model_Data.csv', 'n_embd': 64, 'n_layers': 2, 'gnn_embd': 32, 'dropout': 0.3, 'seq_len': 60, 'weight_decay': 0.001, 'early_stop_patience': 5, 'rank_loss_weight': 1.0, 'batch_size': 1024, 'epochs': 30, 'lr': 0.001, 'num_workers': 10, 'prefetch_factor': 4, 'pin_memory': True, 'persistent_workers': True, 'use_amp': True, 'use_compile': False, 'output_dir': './outputs', 'graph_path': './paper/data/processed/Graph_Adjacency.npy', 'graph_tickers_path': './paper/data/processed/Graph_Tickers.json', 'experiment_name': 'ablation_gru_backbone', 'checkpoint_name': 'best_model_ablation_gru_backbone.pth', 'use_graph': True, 'temporal_backend': 'gru', 'use_rank_loss': True}
>>> [Reproducibility] Random seed set to 42
>>> Training on device: cuda
   GPU: NVIDIA GeForce RTX 3090
   显存: 47.4 GB
   Batch Size: 1024

>>> Loading Datasets...
正在加载 train 数据，来源：./paper/data/processed/Final_Model_Data.csv...
    [特征] 从 /root/autodl-tmp/paper/data/processed/feature_columns.json 读取特征列，共 158 维
✅ 数据清洗完成：无无穷值、无缺失值、极端值已裁剪。
    [V4 对齐] 从 Graph_Tickers.json 读取 375 个图节点
    [V4 对齐] 过滤后数据包含 375 只股票
正在构建滑动窗口索引...
train 数据集共生成样本数: 209216
正在加载 test 数据，来源：./paper/data/processed/Final_Model_Data.csv...
    [特征] 从 /root/autodl-tmp/paper/data/processed/feature_columns.json 读取特征列，共 158 维
✅ 数据清洗完成：无无穷值、无缺失值、极端值已裁剪。
    [V4 对齐] 从 Graph_Tickers.json 读取 375 个图节点
    [V4 对齐] 过滤后数据包含 375 只股票
正在构建滑动窗口索引...
test 数据集共生成样本数: 19652
正在加载 test 数据，来源：./paper/data/processed/Final_Model_Data.csv...
    [特征] 从 /root/autodl-tmp/paper/data/processed/feature_columns.json 读取特征列，共 158 维
✅ 数据清洗完成：无无穷值、无缺失值、极端值已裁剪。
    [V4 对齐] 从 Graph_Tickers.json 读取 375 个图节点
    [V4 对齐] 过滤后数据包含 375 只股票
正在构建滑动窗口索引...
test 数据集共生成样本数: 199988
   Train: 209216, Valid: 19652, Test: 199988
   Input Dim: 158 (features)
>>> Artifacts 已保存: ./outputs/results/artifacts_ablation_gru_backbone
>>> 加载图谱: ./paper/data/processed/Graph_Adjacency.npy, 形状: (375, 375)
✅ 图谱 tickers 顺序校验通过（与 dataset.ticker2idx 对齐）

>>> Initializing Graph-RWKV Model...
   Total parameters: 83,521
   学习率: 1.00e-03
   ✅ 已启用混合精度训练 (AMP)

>>> Start Training (Graph-RWKV Model)...
============================================================
Epoch 1/30 [Train]: 100%|███████████| 570/570 [00:08<00:00, 68.78it/s, avg=0.016543, loss=-0.101523]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 1/30: Train=0.016543, Val=0.003064, lr=9.05e-04
  🌟 Best model saved!
     R²=-4.5266, MAE=0.042307, DirAcc=47.91%, IC=-0.0041
------------------------------------------------------------
Epoch 2/30 [Train]: 100%|███████████| 570/570 [00:07<00:00, 80.58it/s, avg=-0.003413, loss=0.020479]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 2/30: Train=-0.003413, Val=0.002004, lr=6.55e-04
  🌟 Best model saved!
     R²=-2.5420, MAE=0.034650, DirAcc=49.31%, IC=0.0023
------------------------------------------------------------
Epoch 3/30 [Train]: 100%|███████████| 570/570 [00:06<00:00, 82.39it/s, avg=-0.008941, loss=0.027567]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 3/30: Train=-0.008941, Val=0.001634, lr=3.46e-04
  🌟 Best model saved!
     R²=-1.8503, MAE=0.031145, DirAcc=51.22%, IC=-0.0021
------------------------------------------------------------
Epoch 4/30 [Train]: 100%|██████████| 570/570 [00:07<00:00, 73.39it/s, avg=-0.018451, loss=-0.037578]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 4/30: Train=-0.018451, Val=0.000909, lr=9.64e-05
  🌟 Best model saved!
     R²=-0.5861, MAE=0.021786, DirAcc=48.94%, IC=0.0069
------------------------------------------------------------
Epoch 5/30 [Train]: 100%|██████████| 570/570 [00:07<00:00, 76.38it/s, avg=-0.020661, loss=-0.033517]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 5/30: Train=-0.020661, Val=0.001162, lr=1.00e-03
------------------------------------------------------------
Epoch 6/30 [Train]: 100%|██████████| 570/570 [00:07<00:00, 74.76it/s, avg=-0.021821, loss=-0.014885]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 6/30: Train=-0.021821, Val=0.001340, lr=9.76e-04
------------------------------------------------------------
Epoch 7/30 [Train]: 100%|███████████| 570/570 [00:07<00:00, 73.58it/s, avg=-0.003415, loss=0.066119]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 7/30: Train=-0.003415, Val=0.001030, lr=9.05e-04
------------------------------------------------------------
Epoch 8/30 [Train]: 100%|██████████| 570/570 [00:08<00:00, 66.66it/s, avg=-0.002395, loss=-0.177689]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 8/30: Train=-0.002395, Val=0.001515, lr=7.94e-04
------------------------------------------------------------
Epoch 9/30 [Train]: 100%|███████████| 570/570 [00:07<00:00, 73.62it/s, avg=-0.011273, loss=0.290408]

>>> Validating (20 batches, first batch may take 10–30s)...
  [Val] First batch loaded, running forward...
  [Val] First batch done.
  [Val] batch 5/20
  [Val] batch 10/20
  [Val] batch 15/20
  [Val] batch 20/20

Epoch 9/30: Train=-0.011273, Val=0.001419, lr=6.55e-04

🛑 Early stopping (best val loss: 0.000909)

>>> 加载最佳模型并在测试集上评估...

📊 测试集最终评估指标（2021-01-01 ~ 2023-12-31）:
   MSE: 0.000661, RMSE: 0.025715
   R²: -0.6435, MAE: 0.019621
   方向准确率: 49.69%
   IC: 0.0062, ICIR: 0.0363
   RankIC: 0.0067, RankICIR: 0.0385
>>> 训练曲线已保存: ./outputs/figures/training_curve_ablation_gru_backbone.png
>>> Loss 数据已保存: ./outputs/logs/training_losses_ablation_gru_backbone.json
>>> 训练摘要已保存: ./outputs/results/experiment_ablation_gru_backbone.json

============================================================
>>> Graph-RWKV Model 训练结束
    Best Val Loss: 0.000909

    📊 测试集指标（论文报告此区间）:
      IC: 0.0062
      RankIC: 0.0067
      ICIR: 0.0363
      RankICIR: 0.0385
      R² Score: -0.6435
      Directional Accuracy: 49.69%
============================================================
[W201 01:20:50.035286370 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())
root@autodl-container-azgnjchvp3-7d016b8e:~/autodl-tmp# 