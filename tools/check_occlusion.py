"""可视化检查遮挡增强（_occ）效果：左原图，右增强后。"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from woodblock import paths  # noqa: E402

img_dir = os.path.join(paths.DATASET_DIR, "images", "train")
lbl_dir = os.path.join(paths.DATASET_DIR, "labels", "train")
occs = sorted(f for f in os.listdir(img_dir) if "_occ" in f)[:4]
if not occs:
    raise SystemExit("没有遮挡增强样本")

tiles = []
for name in occs:
    stem = name.replace("_occ.png", "")
    orig = cv2.imread(os.path.join(img_dir, stem + ".png"))
    occ = cv2.imread(os.path.join(img_dir, name))
    line = open(os.path.join(lbl_dir, name.rsplit(".", 1)[0] + ".txt"), encoding="utf8").read().split()
    h, w = occ.shape[:2]
    _, cx, cy, bw, bh = (float(v) for v in line[:5])
    x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
    x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
    pad = 40
    bx1, by1 = max(0, x1 - pad), max(0, y1 - pad)
    bx2, by2 = min(w, x2 + pad), min(h, y2 + pad)
    a = orig[by1:by2, bx1:bx2].copy()
    b = occ[by1:by2, bx1:bx2].copy()
    box = (x1 - bx1, y1 - by1, x2 - bx1, y2 - by1)
    for t in (a, b):
        cv2.rectangle(t, box[:2], box[2:], (0, 220, 0), 1)
    tiles.append(np.hstack([cv2.resize(a, (300, 300)), cv2.resize(b, (300, 300))]))

canvas = np.vstack(tiles)
p = os.path.join(paths.REPORTS_DIR, "occlusion_aug_check.jpg")
cv2.imwrite(p, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("saved:", p)
