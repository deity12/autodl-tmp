root@autodl-container-azgnjchvp3-7d016b8e:~/autodl-tmp/paper/data/processed# cd /root/autodl-tmp/paper
root@autodl-container-azgnjchvp3-7d016b8e:~/autodl-tmp/paper# tail -f /root/autodl-tmp/paper/build_graph.log
[加载中] 正在加载本地模型: /root/autodl-tmp/models/qwen/Qwen2.5-14B-Instruct ...
    设备: cuda
Loading checkpoint shards: 100%|██████████| 8/8 [00:10<00:00,  1.30s/it]
[OK] 模型加载成功！

>>> [Step 2] 开始建图 (共 85443 条新闻)...
======================================================================
[OK] 已保存采样新闻（用于断点续跑/复现）: /root/autodl-tmp/paper/data/processed/Graph_Adjacency_news_sampled.csv
[批处理模式] batch=64, max_input_tokens=1536, max_new_tokens=512, do_sample=False
Building Graph: 100%|██████████| 85443/85443 [14:18:17<00:00,  1.66it/s]  

>>> [Step 2.5] 应用时间衰减累积（语义图）...
    ✅ 时间衰减累积完成

>>> [Step 3] 构建统计相关性图（隐式层）...

>>> [统计图构建] 计算过去 30 天收益率的皮尔逊相关系数...
    [防泄露] 统计图数据截断: 486138 -> 231376 行（仅保留 < 2020-06-30 的数据）
    正在计算 375 只股票的相关系数矩阵...
    ✅ 统计图构建完成：28102 条边（|ρ| > 0.6），有效股票数: 359/375
    平均相关系数（强相关边）: 0.7308

>>> [Step 4] 时间衰减累积与混合图构建...
    混合图参数：λ = 1.0（统计图权重）
    语义图边数: 1300
    统计图边数: 28102
    混合图边数: 28712
    ✅ 混合图构建完成

>>> [Step 5] 保存最终结果...
[清理] 已删除临时checkpoint文件

======================================================================
>>> 图谱统计信息
======================================================================
    节点数 (股票数): 375
    边数 (股票关系): 1
    图密度: 0.000018
    有连接的股票数: 337 / 375 (89.9%)
    平均度: 0.01
    最大度: 0
    孤立节点数: 38

[OK] 已保存至 /root/autodl-tmp/paper/data/processed/Graph_Adjacency.npy，形状: (375, 375)
======================================================================