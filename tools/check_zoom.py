"""可视化检查"近景放大"增强样本：左原图，右放大后。"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from woodblock import paths  # noqa: E402

img_dir = os.path.join(paths.DATASET_DIR, "images", "train")
lbl_dir = os.path.join(paths.DATASET_DIR, "labels", "train")
zooms = sorted(f for f in os.listdir(img_dir) if "_zoom" in f)[:4]
if not zooms:
    raise SystemExit("没有近景增强样本")


def draw(img, stem):
    h, w = img.shape[:2]
    line = open(os.path.join(lbl_dir, stem + ".txt"), encoding="utf8").read().split()
    _, cx, cy, bw, bh = (float(v) for v in line[:5])
    x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
    x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 1)
    return img


tiles = []
for name in zooms:
    stem = name.rsplit(".", 1)[0]
    orig_stem = stem.replace("_zoom", "")
    a = draw(cv2.imread(os.path.join(img_dir, orig_stem + ".png")), orig_stem)
    b = draw(cv2.imread(os.path.join(img_dir, name)), stem)
    tiles.append(np.hstack([cv2.resize(a, (320, 240)), cv2.resize(b, (320, 240))]))
p = os.path.join(paths.REPORTS_DIR, "zoom_aug_check.jpg")
cv2.imwrite(p, np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
print("saved:", p)
