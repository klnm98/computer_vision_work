"""取 ONNX(fp32) 对 block.jpg 原图的精确输出，作为 Android 端后处理单测的基准值。

注意：这里刻意不做任何裁剪/缩放，与 Android 端「直接喂原图」的调用方式一致。
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.export_onnx import _decode, _preprocess  # noqa: E402

MODEL = os.path.join("android", "app", "src", "main", "assets", "wood_block_yolo11s.onnx")
IMG = "block.jpg"

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
name = sess.get_inputs()[0].name
img = cv2.imread(IMG)
print("输入图片:", IMG, img.shape)

x, s, left, top = _preprocess(img, 640)
print(f"letterbox: scale={s:.6f} padX={left} padY={top}")
out = sess.run(None, {name: x})[0]
print("模型输出形状:", out.shape)

for conf in (0.35, 0.25, 0.15):
    boxes = _decode(out, s, left, top, conf_th=conf)
    print(f"conf>={conf}: {len(boxes)} 个框")
    for b, c in boxes:
        print(f"   conf={c:.4f} box=({b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f})")

boxes = _decode(out, s, left, top, conf_th=0.35)
ref = {
    "image": IMG, "width": img.shape[1], "height": img.shape[0],
    "letterbox_scale": s, "pad_x": left, "pad_y": top,
    "conf_threshold": 0.35,
    "detections": [{"conf": float(c), "box": [float(v) for v in b]} for b, c in boxes],
}
p = os.path.join("android", "app", "src", "test", "resources", "onnx_reference.json")
os.makedirs(os.path.dirname(p), exist_ok=True)
json.dump(ref, open(p, "w", encoding="utf8"), ensure_ascii=False, indent=1)
print("基准值已写入", p)
