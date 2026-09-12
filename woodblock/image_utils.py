"""图像小工具：去除手机截图的黑边、绘制结果等。"""
from __future__ import annotations

import numpy as np


def crop_letterbox(img: np.ndarray, thresh: int = 25, min_ratio: float = 0.02) -> np.ndarray:
    """裁掉四周接近纯黑的边框（手机截图/视频播放器的黑边）。

    找到最大的非黑矩形：逐行/逐列统计亮度大于 thresh 的像素比例，
    把比例过低的边缘行/列去掉。
    """
    gray = img.mean(axis=2) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    bright = gray > thresh
    rows = bright.mean(axis=1)
    cols = bright.mean(axis=0)

    top = 0
    while top < h and rows[top] < min_ratio:
        top += 1
    bottom = h
    while bottom > top and rows[bottom - 1] < min_ratio:
        bottom -= 1
    left = 0
    while left < w and cols[left] < min_ratio:
        left += 1
    right = w
    while right > left and cols[right - 1] < min_ratio:
        right -= 1
    if right - left < w * 0.2 or bottom - top < h * 0.2:
        return img  # 裁剪过头了，放弃
    return img[top:bottom, left:right].copy()


def resize_long_side(img: np.ndarray, long_side: int = 1280) -> np.ndarray:
    """按长边等比缩放（超大截图加速推理）。"""
    import cv2

    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
