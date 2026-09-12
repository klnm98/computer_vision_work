"""在"摄像头视角"（横构图）下测试定位紧凑度。

block.jpg 是竖屏手机截图（1080x2376），而摄像头画面通常是横构图。
这里从原图裁出包含木块的横构图窗口，比较检测框的相对大小。
"""
from __future__ import annotations

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from woodblock.detector import WoodBlockDetector  # noqa: E402
from woodblock.image_utils import crop_letterbox  # noqa: E402

img = crop_letterbox(cv2.imread("block.jpg"))
h, w = img.shape[:2]
print("去黑边后:", f"{w}x{h}")
# 木块在去黑边图中的大致范围（按比例估计）
bx1, by1, bx2, by2 = int(w * 0.15), int(h * 0.28), int(w * 0.83), int(h * 0.60)
print("目视估计木块框:", (bx1, by1, bx2, by2), "面积", (bx2 - bx1) * (by2 - by1))

det = WoodBlockDetector(conf=0.35, device="0", verbose=False)

variants = {}
# 横构图窗口（1080x720 左右），与摄像头画面接近
cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
win_w, win_h = min(w, 1280), 720
x0 = max(0, min(cx - win_w // 2, w - win_w))
y0 = max(0, min(cy - win_h // 2, h - win_h))
variants["landscape_720p"] = img[y0:y0 + win_h, x0:x0 + win_w]
variants["block_crop"] = img[max(0, by1 - 120):min(h, by2 + 120), max(0, bx1 - 120):min(w, bx2 + 120)]
variants["portrait_full"] = img

for tag, v in variants.items():
    vh, vw = v.shape[:2]
    dets = det.detect(v)
    print(f"\n{tag}: {vw}x{vh}")
    for d in dets[:3]:
        x1, y1, x2, y2 = d.as_int_box()
        # 把木块估计框映射到该窗口坐标系
        gx1, gy1 = bx1 - (x0 if tag == "landscape_720p" else (max(0, bx1 - 120) if tag == "block_crop" else 0)), \
                   by1 - (y0 if tag == "landscape_720p" else (max(0, by1 - 120) if tag == "block_crop" else 0))
        gx2 = gx1 + (bx2 - bx1)
        gy2 = gy1 + (by2 - by1)
        inter = max(0, min(x2, gx2) - max(x1, gx1)) * max(0, min(y2, gy2) - max(y1, gy1))
        union = d.area + (gx2 - gx1) * (gy2 - gy1) - inter
        iou = inter / union if union > 0 else 0.0
        print(f"  conf={d.conf:.3f} box=({x1},{y1},{x2},{y2}) area={int(d.area)} "
              f"| 目视框=({gx1},{gy1},{gx2},{gy2}) IoU={iou:.3f} 面积比={d.area/max(1,(gx2-gx1)*(gy2-gy1)):.2f}")
    out = f"reports/block_variant_{tag}.jpg"
    cv2.imwrite(out, v)
    print("  saved:", out)
