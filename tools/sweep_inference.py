"""比较不同推理设置（分辨率 / TTA）在 block.jpg 上的框质量与耗时。

参考框由人工目视估计（仅用于比较松紧程度，不是真值标注）。
"""
from __future__ import annotations

import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from woodblock.detector import WoodBlockDetector  # noqa: E402
from woodblock.image_utils import crop_letterbox, resize_long_side  # noqa: E402

# 目视估计的木块范围（在 625x1280 的裁剪图上）
REF = (85, 365, 490, 790)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


img = resize_long_side(crop_letterbox(cv2.imread("block.jpg")), 1280)
print(f"输入 {img.shape[1]}x{img.shape[0]}  参考框 {REF} (面积 {((REF[2]-REF[0])*(REF[3]-REF[1]))})")
print(f"{'imgsz':>6} {'tta':>5} {'conf':>6} {'框':>26} {'面积':>8} {'与参考IoU':>10} {'耗时ms':>8}")
for imgsz in (512, 640, 800, 960, 1280):
    for tta in (False, True):
        det = WoodBlockDetector(weights="models/wood_block_yolo11s.pt", conf=0.3,
                                imgsz=imgsz, device="0", tta=tta, verbose=False)
        t0 = time.time()
        dets = det.detect(img)
        dt = (time.time() - t0) * 1000
        if not dets:
            print(f"{imgsz:>6} {str(tta):>5} {'-':>6} {'未检测到':>26} {'-':>8} {'-':>10} {dt:>8.0f}")
            continue
        d = dets[0]
        b = d.as_int_box()
        print(f"{imgsz:>6} {str(tta):>5} {d.conf:>6.3f} {str(b):>26} {int(d.area):>8} "
              f"{iou(b, REF):>10.3f} {dt:>8.0f}  n={len(dets)}")
