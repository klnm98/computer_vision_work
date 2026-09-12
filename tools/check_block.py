"""在 block.jpg 上可视化检测结果（可指定权重与阈值）。"""
from __future__ import annotations

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from woodblock.detector import WoodBlockDetector  # noqa: E402
from woodblock.image_utils import crop_letterbox, resize_long_side  # noqa: E402

weights = sys.argv[1] if len(sys.argv) > 1 else "models/interim.pt"
conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
device = sys.argv[3] if len(sys.argv) > 3 else "cpu"

img = cv2.imread("block.jpg")
variants = {
    "full": resize_long_side(img, 1280),
    "crop": resize_long_side(crop_letterbox(img), 1280),
}
det = WoodBlockDetector(weights=weights, conf=conf, device=device, verbose=False)
tiles = []
for tag, im in variants.items():
    vis = im.copy()
    dets = det.detect(im)
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = d.as_int_box()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0) if i == 0 else (0, 0, 255), 3)
        cv2.putText(vis, f"#{i} {d.conf:.2f}", (x1 + 4, max(24, y1 + 26)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0) if i == 0 else (0, 0, 255), 2)
    print(f"{tag}: {len(dets)} dets")
    for d in dets:
        print("   ", round(d.conf, 3), d.as_int_box())
    scale = 700 / vis.shape[0]
    tiles.append(cv2.resize(vis, (int(vis.shape[1] * scale), 700)))
h = max(t.shape[0] for t in tiles)
padded = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20)) for t in tiles]
out = os.path.join("reports", "block_interim_check.jpg")
os.makedirs("reports", exist_ok=True)
cv2.imwrite(out, cv2.hconcat(padded), [cv2.IMWRITE_JPEG_QUALITY, 90])
print("saved:", out)
