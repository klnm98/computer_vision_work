"""测试多尺度推理：把画面按不同比例缩放后再检测，合并结果。

动机：模型在训练集里见过的木块尺度有限。摄像头贴近拍摄时木块可能占满画面，
单一尺度会漏检；在多个尺度上推理即可覆盖"木块很大 / 很小"两种情况。
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from woodblock.detector import Detection, WoodBlockDetector  # noqa: E402
from woodblock.image_utils import crop_letterbox  # noqa: E402


def nms(dets: list[Detection], iou_thr: float = 0.5) -> list[Detection]:
    dets = sorted(dets, key=lambda d: -d.conf)
    keep: list[Detection] = []
    for d in dets:
        if all(_iou(d, k) < iou_thr for k in keep):
            keep.append(d)
    return keep


def _iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def detect_multiscale(det: WoodBlockDetector, img, scales=(1.0, 0.6, 0.45),
                      iou_thr: float = 0.5) -> list[Detection]:
    h, w = img.shape[:2]
    out: list[Detection] = []
    for s in scales:
        if s == 1.0:
            small, sx, sy = img, 1.0, 1.0
        else:
            sx = sy = s
            small = cv2.resize(img, (max(64, int(w * s)), max(64, int(h * s))),
                               interpolation=cv2.INTER_AREA)
        for d in det.detect(small):
            out.append(Detection(d.x1 / sx, d.y1 / sy, d.x2 / sx, d.y2 / sy, d.conf))
    return nms(out, iou_thr)


def main() -> int:
    weights = sys.argv[1] if len(sys.argv) > 1 else "models/wood_block_v2_backup.pt"
    conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35
    img = crop_letterbox(cv2.imread("block.jpg"))
    h, w = img.shape[:2]
    bx1, by1, bx2, by2 = int(w * 0.15), int(h * 0.28), int(w * 0.83), int(h * 0.60)
    cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
    win_w, win_h = min(w, 1280), 720
    x0 = max(0, min(cx - win_w // 2, w - win_w))
    y0 = max(0, min(cy - win_h // 2, h - win_h))
    variants = {
        "portrait_full": cv2.resize(img, (int(w * 1280 / h), 1280)) if h > 1280 else img,
        "landscape_720p": img[y0:y0 + win_h, x0:x0 + win_w],
        "tight_crop": img[max(0, by1 - 100):min(h, by2 + 100), max(0, bx1 - 100):min(w, bx2 + 100)],
    }
    det = WoodBlockDetector(weights=weights, conf=conf, device="0", verbose=False)
    scale_sets = {
        "1.0": (1.0,),
        "1.0+0.6+0.45": (1.0, 0.6, 0.45),
        "1.0+0.65+0.4+0.25": (1.0, 0.65, 0.4, 0.25),
        "1.0+0.7+0.45+0.3+0.2": (1.0, 0.7, 0.45, 0.3, 0.2),
    }
    for tag, v in variants.items():
        print(f"{tag} {v.shape[1]}x{v.shape[0]}")
        for name, scales in scale_sets.items():
            multi = detect_multiscale(det, v, scales=scales)
            fmt = ", ".join(f"{d.conf:.2f}({d.as_int_box()})" for d in multi[:3]) or "无"
            print(f"   scales={name:22s}: {fmt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
