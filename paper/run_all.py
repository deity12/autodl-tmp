#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键运行完整流程（ETL -> 对齐 -> 建图 -> 训练 -> 消融 -> 评估）。"""

from __future__ import annotations

import argparse
import subprocess
import sys


def _run(cmd: list[str], desc: str) -> None:
    print("\n" + "=" * 70)
    print(f"执行: {desc}")
    print("命令:", " ".join(cmd))
    print("=" * 70)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"❌ 失败: {desc}")
        sys.exit(result.returncode)
    print(f"✅ 完成: {desc}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键运行 Graph-RWKV 全流程")
    parser.add_argument("--skip_ablation", action="store_true", help="跳过消融实验")
    parser.add_argument("--skip_eval", action="store_true", help="跳过评估")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    _run(["python", "1_preprocess_data.py", "--run_align"], "数据预处理 + 对齐")
    _run(["python", "2_build_graph.py", "--llm"], "构建图谱")
    _run(["python", "3_train.py"], "训练全量模型")

    if not args.skip_ablation:
        _run(["python", "3_train_ablation.py", "--ablation", "all"], "消融实验")

    if not args.skip_eval:
        _run(
            [
                "python",
                "4_evaluate.py",
                "--checkpoint",
                "./outputs/checkpoints/best_model.pth",
                "--test_data",
                "./data/processed/Final_Model_Data.csv",
            ],
            "模型评估",
        )

    print("\n✅ 全流程执行完成")


if __name__ == "__main__":
    main()
