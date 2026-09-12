"""把 YOLO 标签画回图片，生成拼图用于人工核对标注是否正确。"""
from __future__ import annotations

import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from woodblock import paths  # noqa: E402


def montage(split: str = "train", n: int = 12, cols: int = 4, tile: int = 320,
            out: str | None = None, positives: bool = True, seed: int = 0) -> str:
    img_dir = os.path.join(paths.DATASET_DIR, "images", split)
    lbl_dir = os.path.join(paths.DATASET_DIR, "labels", split)
    names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg")))
    if positives:
        names = [f for f in names if os.path.getsize(os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")) > 0]
    random.Random(seed).shuffle(names)
    names = names[:n]
    if not names:
        raise SystemExit(f"{img_dir} 下没有可用图片")

    rows = (len(names) + cols - 1) // cols
    canvas = np.zeros((rows * tile, cols * tile, 3), np.uint8)
    for i, name in enumerate(names):
        img = cv2.imread(os.path.join(img_dir, name))
        if img is None:
            continue
        h, w = img.shape[:2]
        lp = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")
        for line in open(lp, encoding="utf8"):
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = (float(v) for v in parts[:5])
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
        small = cv2.resize(img, (tile, tile))
        cv2.putText(small, name[:26], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        r, c = divmod(i, cols)
        canvas[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = small

    out = out or os.path.join(paths.REPORTS_DIR, f"label_check_{split}.jpg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("saved:", out)
    return out


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    montage(split=split, n=n)
