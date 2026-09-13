"""为 Android 单元测试生成测试资源。

Android 的 JVM 单测环境没有 java.awt/javax.imageio，无法在测试里解码图片，
因此这里用 Python 生成：
  * letterbox_input.bin  —— block.jpg 经 letterbox 后的 640x640 RGB uint8（1.2 MB）
  * letterbox_meta.json   —— 对应参数与桌面端基准检测结果
测试再把它喂给同一个 ONNX，验证解码/坐标反变换/过滤/NMS 与桌面端一致。
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.export_onnx import _decode, _preprocess  # noqa: E402

import onnxruntime as ort  # noqa: E402

RES = os.path.join("android", "app", "src", "test", "resources")
MODEL = os.path.join("android", "app", "src", "main", "assets", "wood_block_yolo11s.onnx")
IMG = "block.jpg"
SIZE = 640

os.makedirs(RES, exist_ok=True)
img = cv2.imread(IMG)
h, w = img.shape[:2]

x, scale, pad_x, pad_y = _preprocess(img, SIZE)          # (1,3,640,640) float32 RGB 0~1
rgb = (x[0].transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)  # HWC RGB uint8
assert rgb.shape == (SIZE, SIZE, 3), rgb.shape
rgb.tofile(os.path.join(RES, "letterbox_input.bin"))
print(f"letterbox_input.bin: {rgb.shape} {rgb.nbytes/1e6:.2f} MB")

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
boxes = _decode(out, scale, pad_x, pad_y, conf_th=0.35)
meta = {
    "image": IMG, "src_width": w, "src_height": h, "input_size": SIZE,
    "letterbox": {"scale": float(scale), "pad_x": int(pad_x), "pad_y": int(pad_y)},
    "conf_threshold": 0.35,
    "detections": [{"conf": float(c), "box": [float(v) for v in b]} for b, c in boxes],
    "model_output_shape": list(out.shape),
}
with open(os.path.join(RES, "letterbox_meta.json"), "w", encoding="utf8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)
print("letterbox_meta.json:", json.dumps(meta["letterbox"], ensure_ascii=False))
for d in meta["detections"]:
    print(f"  基准框 conf={d['conf']:.4f} box={[round(v,1) for v in d['box']]}")
