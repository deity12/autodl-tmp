#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
summarize_final_run.py - 汇总 Clean Run v1 结果到 summary.csv

输入：paper/outputs/final_run/<exp>/results/experiment_<exp_name>.json
输出：paper/outputs/final_run/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, Optional


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _pick(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize final_run experiments to summary.csv")
    p.add_argument("--run_root", type=str, default=None, help="final_run 根目录（默认 paper/outputs/final_run）")
    args = p.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    paper_dir = os.path.dirname(script_dir)
    run_root = args.run_root or os.path.join(paper_dir, "outputs", "final_run")

    rows = []
    if not os.path.exists(run_root):
        raise FileNotFoundError(f"run_root 不存在: {run_root}")

    for name in sorted(os.listdir(run_root)):
        exp_dir = os.path.join(run_root, name)
        if not os.path.isdir(exp_dir):
            continue
        # train_full.py 会写 results/experiment_<experiment_name>.json
        summary_path = os.path.join(exp_dir, "results", f"experiment_{name}.json")
        data = _read_json(summary_path)
        if not data:
            continue

        val_rankic = _pick(data, "val_metrics.rank_ic")
        test_rankic = _pick(data, "test_metrics.rank_ic")
        test_ic = _pick(data, "test_metrics.ic")
        test_icir = _pick(data, "test_metrics.icir")
        test_rankicir = _pick(data, "test_metrics.rankicir")

        cfg = data.get("config", {}) if isinstance(data.get("config"), dict) else {}
        rows.append(
            {
                "exp_name": name,
                "use_graph": cfg.get("use_graph"),
                "temporal_backend": cfg.get("temporal_backend"),
                "input_dim": cfg.get("input_dim"),
                "seq_len": cfg.get("seq_len"),
                "val_rankic": val_rankic,
                "test_rankic": test_rankic,
                "test_ic": test_ic,
                "test_icir": test_icir,
                "test_rankicir": test_rankicir,
            }
        )

    out_path = os.path.join(run_root, "summary.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "exp_name",
                "use_graph",
                "temporal_backend",
                "input_dim",
                "seq_len",
                "val_rankic",
                "test_rankic",
                "test_ic",
                "test_icir",
                "test_rankicir",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f">>> summary.csv 已生成: {out_path} (rows={len(rows)})")


if __name__ == "__main__":
    main()

