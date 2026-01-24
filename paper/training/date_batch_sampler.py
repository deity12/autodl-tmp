#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按日期分组的 Batch Sampler：
用于金融任务的截面训练（同一天内股票一起训练/排序）。
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterable, Iterator, List, Sequence

from torch.utils.data import Sampler


class DateGroupedBatchSampler(Sampler[List[int]]):
    """按 target_date 分组的 batch sampler。"""

    def __init__(
        self,
        target_dates: Sequence[str],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)

        date_to_indices = defaultdict(list)
        for idx, d in enumerate(target_dates):
            date_to_indices[str(d)].append(idx)

        self._groups = list(date_to_indices.values())
        self._num_samples = len(target_dates)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)
        groups = list(self._groups)

        if self.shuffle:
            rng.shuffle(groups)
            for g in groups:
                rng.shuffle(g)

        for indices in groups:
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch

        if self.shuffle:
            self.seed += 1

    def __len__(self) -> int:
        if self.batch_size <= 0:
            return 0
        total = 0
        for indices in self._groups:
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += math.ceil(len(indices) / self.batch_size)
        return total
