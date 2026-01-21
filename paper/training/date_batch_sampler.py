"""
按日期截面分组的 BatchSampler（顶会股票排序/RankIC 常用做法）
-------------------------------------------------------
目的：
  - 让每个 batch 尽量来自同一天的截面（cross-section），便于：
    1) 训练时使用 ranking loss（如 RankNet）优化“排序能力”
    2) 评估时计算按日 IC / RankIC（而不是把所有日期拼在一起算相关）

依赖：
  - dataset 返回字段 target_date（字符串 YYYY-MM-DD）
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator, List, Sequence
import random


class DateGroupedBatchSampler:
    """
    按 `target_date` 分组采样 batch。

    直观理解：
      - 把样本索引按日期分桶（同一天的样本放在一起）
      - 每个桶内再切分成 batch_size 大小的小批次

    适用场景：
      - RankIC/排序类训练：需要同一日期的横截面（cross-section）样本来构造排序约束
      - 评估：按日期计算 IC/RankIC 更符合量化研究口径
    """
    def __init__(
        self,
        target_dates: Sequence[str],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.target_dates = list(target_dates)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)

        if self.batch_size <= 0:
            raise ValueError(f"batch_size 必须 > 0，但得到 {self.batch_size}")
        if len(self.target_dates) == 0:
            raise ValueError("target_dates 为空，无法按日期分组。")

        date2idx = defaultdict(list)
        for i, d in enumerate(self.target_dates):
            date2idx[str(d)].append(i)
        self._date2idx = dict(date2idx)
        self._dates = list(self._date2idx.keys())

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)
        dates = list(self._dates)
        if self.shuffle:
            rng.shuffle(dates)

        for d in dates:
            idxs = list(self._date2idx[d])
            if self.shuffle:
                rng.shuffle(idxs)

            # 一个日期可能覆盖 > batch_size 个股票，切成多个 batch
            for s in range(0, len(idxs), self.batch_size):
                batch = idxs[s : s + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        n = 0
        for d in self._dates:
            m = len(self._date2idx[d])
            if self.drop_last:
                n += m // self.batch_size
            else:
                n += (m + self.batch_size - 1) // self.batch_size
        return n

