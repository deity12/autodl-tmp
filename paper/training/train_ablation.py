#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""兼容入口：转发到顶层 3_train_ablation.py。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    script_path = Path(__file__).resolve().parents[1] / "3_train_ablation.py"
    cmd = [sys.executable, str(script_path)] + sys.argv[1:]
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
