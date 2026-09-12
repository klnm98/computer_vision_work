"""木块识别系统（wooden block detection）。

基于公开真实数据集（YCB-Video / BOP YCB-V）训练 YOLO 目标检测模型，
实现木块的实时准确识别与框选。
"""
from __future__ import annotations

import os

# Ultralytics 默认把配置写到 %APPDATA%\\Ultralytics；在受限环境（例如只能写工作区
# 的沙箱）下会失败，因此统一重定向到项目根目录下的 Ultralytics/。
# 必须在 import ultralytics 之前设置。
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.dirname(_HERE))

__all__ = ["paths", "sources", "remote_zip", "dataset", "train", "evaluate", "detector",
           "image_utils", "env_setup"]
__version__ = "2.0.0"
