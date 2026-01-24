#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日志工具：为脚本提供统一的日志与打印捕获能力。

设计目标：
1) 同时输出到控制台与文件
2) 日志文件名包含时间戳
3) 对 print 输出进行记录（写入日志文件）
"""

from __future__ import annotations

import builtins
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional


@dataclass(frozen=True)
class LogContext:
    """日志上下文信息。"""

    logger: logging.Logger
    print_logger: logging.Logger
    log_file: Path
    restore_print: Callable[[], None]


def setup_logging(output_dir: str, script_name: str) -> LogContext:
    """
    创建日志系统，并捕获 print 输出写入日志文件。

    Args:
        output_dir: 输出目录（会在其中创建 logs/ 子目录）
        script_name: 脚本名称，用于日志文件命名

    Returns:
        LogContext: 日志对象与恢复 print 的函数
    """
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{script_name}_{timestamp}.log"

    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(console_handler)

    print_logger = logging.getLogger(f"{script_name}.print")
    print_logger.setLevel(logging.INFO)
    print_logger.propagate = False
    print_logger.handlers.clear()
    print_logger.addHandler(file_handler)

    original_print = builtins.print

    def wrapped_print(*args, **kwargs) -> None:
        file = kwargs.get("file")
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")

        original_print(*args, **kwargs)

        if file not in (None, sys.stdout):
            return
        if "\r" in str(end):
            return

        msg = sep.join(str(a) for a in args)
        msg = msg + ("" if msg.endswith("\n") else "")
        if msg.strip():
            print_logger.info(msg)

    def restore_print() -> None:
        builtins.print = original_print

    builtins.print = wrapped_print

    logger.info("日志文件已创建: %s", log_file)
    return LogContext(
        logger=logger,
        print_logger=print_logger,
        log_file=log_file,
        restore_print=restore_print,
    )
