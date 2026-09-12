"""为不同的多尺度配置测量"误检率 / 召回率 / block.jpg 近景命中"，用于选择默认配置。"""
from __future__ import annotations

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from woodblock import paths  # noqa: E402
from woodblock.detector import WoodBlockDetector  # noqa: E402
from woodblock.evaluate import _iou, load_bop_test  # noqa: E402
from woodblock.image_utils import crop_letterbox, resize_long_side  # noqa: E402

CONF = paths.CONF_THRESHOLD
CONFIGS = [(1.0,), (1.0, 0.75), (1.0, 0.7), (1.0, 0.7, 0.45)]

items = load_bop_test()
negs = [it for it in items if not it["pos"]][:825]
poss = [it for it in items if it["pos"]]

img = crop_letterbox(cv2.imread("block.jpg"))
h, w = img.shape[:2]
bx1, by1, bx2, by2 = int(w * 0.15), int(h * 0.28), int(w * 0.83), int(h * 0.60)
cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
win_w, win_h = min(w, 1280), 720
x0 = max(0, min(cx - win_w // 2, w - win_w))
y0 = max(0, min(cy - win_h // 2, h - win_h))
variants = {
    "以原图": resize_long_side(img, 1280),
    "横构图近景": img[y0:y0 + win_h, x0:x0 + win_w],
}

print(f"{'scales':>22} {'误检率':>8} {'误检框':>7} {'召回率':>8} | " + " | ".join(variants))
for scales in CONFIGS:
    det = WoodBlockDetector(conf=CONF, device="0", scales=scales, verbose=False)
    fp = fp_boxes = 0
    for it in negs:
        im = cv2.imread(it["path"])
        if im is None:
            continue
        ds = det.detect(im)
        if ds:
            fp += 1
            fp_boxes += len(ds)
    tp = 0
    for it in poss:
        im = cv2.imread(it["path"])
        ds = det.detect(im)
        for g in it["wood"]:
            if any(_iou([d.x1, d.y1, d.x2, d.y2], g) >= 0.5 for d in ds):
                tp += 1
    marks = []
    for tag, v in variants.items():
        ds = det.detect(v)
        marks.append(f"{ds[0].conf:.2f}" if ds else "miss")
    print(f"{str(scales):>22} {fp/len(negs)*100:7.2f}% {fp_boxes:7d} {tp/75*100:7.1f}% | " + " | ".join(marks))
