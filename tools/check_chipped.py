"""可视化检查缺角增强效果。"""
from __future__ import annotations

import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from woodblock import dataset as ds  # noqa: E402
from woodblock import paths  # noqa: E402

img_dir = os.path.join(paths.DATASET_DIR, "images", "train")
lbl_dir = os.path.join(paths.DATASET_DIR, "labels", "train")
names = [f for f in sorted(os.listdir(img_dir)) if f.endswith(".png") and not f.startswith("neg_")
         and "_chip" not in f]
if not names:
    raise SystemExit("训练集还没有图片")

rng = random.Random(1)
tiles = []
for name in names[:4]:
    img = cv2.imread(os.path.join(img_dir, name))
    h, w = img.shape[:2]
    line = open(os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt"), encoding="utf8").read().split()
    _, cx, cy, bw, bh = (float(v) for v in line[:5])
    box = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
    x1, y1, x2, y2 = (int(v) for v in box)

    # 裁剪出木块周围区域放大看（左：原图；右：缺角增强后）
    pad = 40
    crop = img[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)].copy()
    out = ds.make_chipped(img, box, rng)
    if out is None:
        continue
    crop2 = out[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)].copy()
    cb = (x1 - max(0, x1 - pad), y1 - max(0, y1 - pad), x2 - max(0, x1 - pad), y2 - max(0, y1 - pad))
    cv2.rectangle(crop, cb[:2], cb[2:], (0, 220, 0), 1)
    cv2.rectangle(crop2, cb[:2], cb[2:], (0, 220, 0), 1)
    both = np.hstack([cv2.resize(crop, (300, 300)), cv2.resize(crop2, (300, 300))])
    tiles.append(both)

if tiles:
    canvas = np.vstack(tiles)
    p = os.path.join(paths.REPORTS_DIR, "chipped_aug_check.jpg")
    cv2.imwrite(p, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("saved:", p, canvas.shape)
else:
    print("没有生成任何缺角样本")
