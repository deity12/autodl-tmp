(base) root@autodl-container-azgnjchvp3-7d016b8e:~/autodl-tmp# cd /root/autodl-tmp/paper
(base) root@autodl-container-azgnjchvp3-7d016b8e:~/autodl-tmp/paper# nohup /root/miniconda3/bin/python dataProcessed/build_graph.py --split_date 2020-12-31 >> build_graph.log 2>&1 &
[1] 282607
(base) root@autodl-container-azgnjchvp3-7d016b8e:~/autodl-tmp/paper# ps aux | grep build_graph
root     194422  0.0  0.0   3244  1044 pts/1    S+   00:59   0:00 tail -f build_graph.log
root     282607  103  0.2 55896412 2017416 pts/4 Rl  11:02   0:56 /root/miniconda3/bin/python dataProcessed/build_graph.py --split_date 2020-12-31
root     283047  0.0  0.0   4028  2088 pts/4    S+   11:03   0:00 grep --color=auto build_graph
(base) root@autodl-container-azgnjchvp3-7d016b8e:~/autodl-tmp/paper# tail -f /root/autodl-tmp/paper/build_graph.log
    设备: cuda
Loading checkpoint shards: 100%|██████████| 8/8 [00:13<00:00,  1.74s/it]
[OK] 模型加载成功！

>>> [Step 2] 开始建图 (共 85878 条新闻)...
======================================================================
[Resume] 检测到已保存的采样新闻: /root/autodl-tmp/paper/data/processed/Graph_Adjacency_news_sampled.csv (n=85878)
[Resume] 从 checkpoint 恢复：next_pos=80000, batch_size=64
[批处理模式] batch=64, max_input_tokens=1536, max_new_tokens=96, do_sample=False
Building Graph: 100%|██████████| 85878/85878 [1:03:28<00:00,  1.54it/s]
[Resume] (规则模式) 从 checkpoint 恢复：next_pos=80000
Building Graph: 100%|██████████| 85878/85878 [00:01<00:00, 4993.32it/s]

>>> [Step 2.5] 应用时间衰减累积（语义图）...
    ✅ 时间衰减累积完成

>>> [Step 3] 构建统计相关性图（隐式层）...

>>> [统计图构建] 计算过去 30 天收益率的皮尔逊相关系数...
    [防泄露] 统计图数据截断: 486138 -> 268521 行（仅保留 < 2020-12-31 的数据）
    正在计算 375 只股票的相关系数矩阵...
    ✅ 统计图构建完成：2760 条边（|ρ| > 0.6），有效股票数: 289/375
    平均相关系数（强相关边）: 0.6885

>>> [Step 4] 时间衰减累积与混合图构建...
    混合图参数：λ = 1.0（统计图权重）
    语义图边数: 19
    统计图边数: 2760
    混合图边数: 1389
    ✅ 混合图构建完成

>>> [Step 5] 保存最终结果...
[清理] 已删除临时checkpoint文件

======================================================================
>>> 图谱统计信息
======================================================================
    节点数 (股票数): 375
    边数 (股票关系): 19
    图密度: 0.000285
    有连接的股票数: 335 / 375 (89.3%)
    平均度: 0.11
    最大度: 1
    孤立节点数: 40

[OK] 已保存至 /root/autodl-tmp/paper/data/processed/Graph_Adjacency.npy，形状: (375, 375)
======================================================================