"""在验证集上可视化最终模型的预测（绿）与真值框（红），检查定位是否紧凑。"""
from __future__ import annotations

import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from woodblock import paths  # noqa: E402
from woodblock.detector import WoodBlockDetector  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35
img_dir = os.path.join(paths.DATASET_DIR, "images", "val")
lbl_dir = os.path.join(paths.DATASET_DIR, "labels", "val")
names = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
random.Random(3).shuffle(names)
names = names[:n]

det = WoodBlockDetector(conf=conf, device="0", verbose=False)
ious = []
tiles = []
for name in names:
    img = cv2.imread(os.path.join(img_dir, name))
    h, w = img.shape[:2]
    line = open(os.path.join(lbl_dir, name.rsplit(".", 1)[0] + ".txt"), encoding="utf8").read().split()
    _, cx, cy, bw, bh = (float(v) for v in line[:5])
    gx1, gy1 = (cx - bw / 2) * w, (cy - bh / 2) * h
    gx2, gy2 = (cx + bw / 2) * w, (cy + bh / 2) * h
    cv2.rectangle(img, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0, 0, 255), 1)
    dets = det.detect(img)
    if dets:
        d = dets[0]
        cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 220, 0), 1)
        inter = max(0, min(d.x2, gx2) - max(d.x1, gx1)) * max(0, min(d.y2, gy2) - max(d.y1, gy1))
        union = d.area + (gx2 - gx1) * (gy2 - gy1) - inter
        iou_v = inter / union if union > 0 else 0
        ious.append(iou_v)
        cv2.putText(img, f"{d.conf:.2f} IoU={iou_v:.2f}", (int(gx1), max(14, int(gy1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    tiles.append(cv2.resize(img, (320, 240)))

cols = 4
rows = (len(tiles) + cols - 1) // cols
canvas = np.zeros((rows * 240, cols * 320, 3), np.uint8)
for i, t in enumerate(tiles):
    r, c = divmod(i, cols)
    canvas[r * 240:(r + 1) * 240, c * 320:(c + 1) * 320] = t
out = os.path.join(paths.REPORTS_DIR, "val_predictions.jpg")
cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
print("saved:", out)
if ious:
    print(f"平均 IoU = {np.mean(ious):.3f}  最小 {np.min(ious):.3f}  最大 {np.max(ious):.3f}  "
          f"(n={len(ious)})")
