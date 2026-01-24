#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速检查环境与依赖。"""

from __future__ import annotations

import sys


def _check_python() -> bool:
    ok = sys.version_info >= (3, 9)
    print(f"Python 版本: {sys.version}")
    if not ok:
        print("❌ 需要 Python >= 3.9")
        return False
    if sys.version_info < (3, 10):
        print("⚠️ 建议 Python >= 3.10（pandas-ta 等部分依赖在 Py3.9 上可能无法安装，将自动回退纯 pandas 特征工程）")
    return ok


def _check_torch() -> bool:
    try:
        import torch  # noqa: F401
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        return True
    except Exception as e:
        print(f"❌ PyTorch 导入失败: {e}")
        return False


def _check_deps() -> bool:
    # parquet/特征工程相关依赖按“推荐/可选”提示，不强制中断
    deps = ["pandas", "numpy", "sklearn", "tqdm"]
    ok = True
    for dep in deps:
        try:
            __import__(dep)
            print(f"✅ {dep} 已安装")
        except Exception:
            print(f"❌ {dep} 未安装")
            ok = False

    # 推荐依赖：缺失时给出明确提示
    for dep, tip in [
        ("pyarrow", "缺失将无法读写 .parquet（建议: pip install pyarrow）"),
        ("pandas_ta", "缺失将回退纯 pandas 特征工程（可选: pip install pandas-ta）"),
    ]:
        try:
            __import__(dep)
            print(f"✅ {dep} 已安装")
        except Exception:
            print(f"⚠️ {dep} 未安装：{tip}")
    return ok


def main() -> None:
    print(">>> 环境检查开始")
    ok = True
    ok &= _check_python()
    ok &= _check_torch()
    ok &= _check_deps()
    if ok:
        print("\n✅ 环境检查通过")
    else:
        print("\n❌ 环境检查失败，请安装缺失依赖")
        sys.exit(1)


if __name__ == "__main__":
    main()
