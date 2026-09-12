"""比较两个模型的性能：BOP 测试集指标 + block.jpg 在"竖屏截图/横构图近景"下的检测。

用法: python tools/compare_models.py A.pt B.pt [--max-negative 300]
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from woodblock import evaluate as ev  # noqa: E402
from woodblock.detector import WoodBlockDetector  # noqa: E402
from woodblock.image_utils import crop_letterbox, resize_long_side  # noqa: E402


def block_variants(path: str = "block.jpg"):
    img = crop_letterbox(cv2.imread(path))
    h, w = img.shape[:2]
    bx1, by1, bx2, by2 = int(w * 0.15), int(h * 0.28), int(w * 0.83), int(h * 0.60)
    cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
    win_w, win_h = min(w, 1280), 720
    x0 = max(0, min(cx - win_w // 2, w - win_w))
    y0 = max(0, min(cy - win_h // 2, h - win_h))
    return {
        "portrait_full": resize_long_side(img, 1280),
        "landscape_720p": img[y0:y0 + win_h, x0:x0 + win_w],
        "tight_crop": img[max(0, by1 - 100):min(h, by2 + 100), max(0, bx1 - 100):min(w, bx2 + 100)],
    }, (bx1, by1, bx2, by2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="+")
    ap.add_argument("--max-negative", type=int, default=300)
    ap.add_argument("--max-positive", type=int, default=75)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    variants, ref = block_variants()
    for w in args.weights:
        print("=" * 70)
        print("模型:", w)
        rep = ev.evaluate(weights=w, conf=args.conf, skip_val=True,
                          max_negative=args.max_negative, max_positive=args.max_positive,
                          device=args.device)
        n, p = rep["negative"], rep["positive"]
        print(f"  BOP 负样本: {n['images']} 张, 误检率 {n['fp_image_rate']*100:.2f}%, "
              f"框到其他物体 {n['boxes_on_other_objects']} 个")
        print(f"  BOP 木块召回: {p['recall']*100:.1f}%  (IoU_mean={p['mean_iou']:.3f})")
        det = WoodBlockDetector(weights=w, conf=args.conf, device=args.device, verbose=False)
        for tag, img in variants.items():
            dets = det.detect(img)
            if not dets:
                print(f"  block.jpg[{tag}] {img.shape[1]}x{img.shape[0]}: 未检测到")
                continue
            d = dets[0]
            x1, y1, x2, y2 = d.as_int_box()
            print(f"  block.jpg[{tag}] {img.shape[1]}x{img.shape[0]}: n={len(dets)} "
                  f"conf={d.conf:.3f} box=({x1},{y1},{x2},{y2}) 面积比={d.area/ (img.shape[0]*img.shape[1]):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
