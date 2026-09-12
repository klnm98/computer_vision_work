"""依赖安装与环境准备。

包含针对受限文件系统环境（例如某些沙箱只允许在工作区内写入、且拒绝往
``tempfile.mkdtemp`` 创建的 0700 目录写文件）的兼容处理：
把 pip 的临时目录指向工作区，并让 ``os.mkdir`` 以可写权限创建临时目录。
"""
from __future__ import annotations

import os
import sys

from . import paths

TEMP_DIR = os.path.join(paths.ROOT, ".tmp")

TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
TORCH_PKGS = ["torch==2.9.1", "torchvision==0.24.1"]
EXTRA_PKGS = ["ultralytics>=8.3.0"]


def _prepare_environment() -> None:
    """把临时目录放到工作区内，并放宽 mkdtemp 目录权限。"""
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.environ["TEMP"] = TEMP_DIR
    os.environ["TMP"] = TEMP_DIR
    os.environ["TMPDIR"] = TEMP_DIR

    real_mkdir = os.mkdir

    def _mkdir(path, mode=0o777, *, dir_fd=None):
        return real_mkdir(path, 0o777, dir_fd=dir_fd)

    os.mkdir = _mkdir  # type: ignore[assignment]


def pip_install(argv: list[str]) -> int:
    """在进程内调用 pip。"""
    _prepare_environment()
    from pip._internal.cli.main import main as pip_main

    return pip_main(argv)


def has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def install_requirements(cuda: bool = True) -> int:
    """安装运行本系统所需的全部依赖。"""
    print(f"[环境] 临时目录 -> {TEMP_DIR}")
    rc = 0
    if not has_module("torch"):
        index = TORCH_INDEX if cuda else None
        args = ["install", *TORCH_PKGS]
        if index:
            args += ["--index-url", index]
        print(f"[环境] 安装 PyTorch: {' '.join(args)}")
        rc |= pip_install(args)
    else:
        import torch

        print(f"[环境] PyTorch 已安装: {torch.__version__} (cuda={torch.cuda.is_available()})")

    if not has_module("ultralytics"):
        print("[环境] 安装 Ultralytics YOLO ...")
        rc |= pip_install(["install", *EXTRA_PKGS])
    else:
        import ultralytics

        print(f"[环境] Ultralytics 已安装: {ultralytics.__version__}")
    return rc


if __name__ == "__main__":
    raise SystemExit(install_requirements())
