"""诊断：在 BOP 测试场景上对比真值框与模型预测，找出漏检/误检原因。"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from woodblock import paths  # noqa: E402
from woodblock.detector import WoodBlockDetector  # noqa: E402
from woodblock.evaluate import load_bop_test  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="000055")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default=os.path.join("reports", "bop_diag"))
    args = ap.parse_args()

    items = [it for it in load_bop_test() if it["scene"] == args.scene]
    if not items:
        raise SystemExit(f"测试集中没有场景 {args.scene}")
    det = WoodBlockDetector(weights=args.weights, conf=args.conf, device=args.device, verbose=False)

    os.makedirs(args.out, exist_ok=True)
    print(f"场景 {args.scene}: {len(items)} 帧, 含木块 {sum(i['pos'] for i in items)} 帧")
    print(f"{'frame':>7} {'GT框':>28} {'GT尺寸':>10}  预测")
    tiles = []
    stats = {"gt": 0, "hit": 0, "miss": 0, "fp": 0}
    for it in items:
        img = cv2.imread(it["path"])
        h, w = img.shape[:2]
        preds = det.detect(img)
        gt = it["wood"][0] if it["wood"] else None
        gt_str = "[" + ",".join(f"{v:.0f}" for v in gt) + "]" if gt else "-"
        gt_size = f"{gt[2]-gt[0]:.0f}x{gt[3]-gt[1]:.0f}" if gt else "-"
        pred_str = ", ".join(f"{d.conf:.2f}({d.width:.0f}x{d.height:.0f})" for d in preds[:3]) or "-"
        print(f"{it['frame']:>7} {gt_str:>28} {gt_size:>10}  {pred_str}")

        hit = False
        if gt:
            stats["gt"] += 1
            for d in preds:
                ix1, iy1 = max(d.x1, gt[0]), max(d.y1, gt[1])
                ix2, iy2 = min(d.x2, gt[2]), min(d.y2, gt[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = d.area + (gt[2] - gt[0]) * (gt[3] - gt[1]) - inter
                if union > 0 and inter / union >= 0.5:
                    hit = True
            stats["hit" if hit else "miss"] += 1
        stats["fp"] += max(0, len(preds) - (1 if hit else 0))

        if len(tiles) < args.n:
            vis = img.copy()
            if gt:
                cv2.rectangle(vis, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])), (0, 0, 255), 2)
                cv2.putText(vis, "GT", (int(gt[0]), max(14, int(gt[1]) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            for d in preds:
                x1, y1, x2, y2 = d.as_int_box()
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(vis, f"{d.conf:.2f}", (x1, max(14, y2 + 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
            scale = 400 / max(h, w)
            tiles.append(cv2.resize(vis, (int(w * scale), int(h * scale))))

    if tiles:
        cols = min(4, len(tiles))
        rows = (len(tiles) + cols - 1) // cols
        th = max(t.shape[0] for t in tiles)
        tw = max(t.shape[1] for t in tiles)
        canvas = np.zeros((rows * th, cols * tw, 3), np.uint8)
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            canvas[r * th:r * th + t.shape[0], c * tw:c * tw + t.shape[1]] = t
        p = os.path.join(args.out, f"scene_{args.scene}_conf{args.conf}.jpg")
        cv2.imwrite(p, canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("拼图:", p)
    print(f"\n统计: GT {stats['gt']}, 命中 {stats['hit']}, 漏检 {stats['miss']}, 多余框 {stats['fp']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
