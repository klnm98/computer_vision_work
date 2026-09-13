"""把训练好的 YOLO 木块模型导出为 ONNX（供 Android 端使用），并校验一致性。

因为本机 Python 3.14 没有 TensorFlow wheel（无法导出 TFLite），所以移动端采用
ONNX + ONNX Runtime。导出后会：
  1. 用 onnxruntime 直接跑一遍，和 PyTorch 版的框对比（应几乎一致）；
  2. 在 block.jpg / BOP 真实图上对比两者的检测结果（数量、位置、置信度）；
  3. 可选导出 int8 量化版并对比体积与精度损失。

用法:
    python tools/export_onnx.py                 # fp32 + int8，并做一致性校验
    python tools/export_onnx.py --no-int8
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from woodblock import paths  # noqa: E402

OUT_DIR = os.path.join(paths.ROOT, "android", "app", "src", "main", "assets")
FP32_NAME = "wood_block_yolo11s.onnx"
INT8_NAME = "wood_block_yolo11s_int8.onnx"


def export_fp32(weights: str, imgsz: int = 640, opset: int = 12) -> str:
    from ultralytics import YOLO

    model = YOLO(weights)
    path = model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=False,
                        dynamic=False, half=False, nms=False)
    print(f"[导出] ONNX(fp32): {path}  ({os.path.getsize(path)/1e6:.1f} MB)")
    return path


def export_int8(fp32_path: str, calib_images: list[str], imgsz: int = 640) -> str | None:
    """静态 int8 量化（用真实图片做校准）。"""
    try:
        from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    except ImportError as e:  # pragma: no cover
        print(f"[量化] 不可用: {e}")
        return None
    import cv2

    # onnxruntime 量化会往系统临时目录写文件；受限环境下改到工作区内
    tmp_dir = os.path.join(paths.ROOT, ".tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    os.environ["TEMP"] = tmp_dir
    os.environ["TMP"] = tmp_dir
    os.environ["TMPDIR"] = tmp_dir

    class Reader(CalibrationDataReader):
        def __init__(self, files: list[str], size: int):
            self.files = files
            self.size = size
            self.i = 0

        def get_next(self):
            if self.i >= len(self.files):
                return None
            f = self.files[self.i]
            self.i += 1
            img = cv2.imread(f)
            if img is None:
                return self.get_next()
            h, w = img.shape[:2]
            s = self.size / max(h, w)
            img = cv2.resize(img, (int(round(w * s)), int(round(h * s))))
            canvas = np.full((self.size, self.size, 3), 114, np.uint8)
            canvas[:img.shape[0], :img.shape[1]] = img
            x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            return {"images": np.ascontiguousarray(x)}

    out = os.path.join(OUT_DIR, INT8_NAME)
    quantize_static(fp32_path, out, Reader(calib_images, imgsz),
                    quant_format=QuantFormat.QDQ, per_channel=False,
                    weight_type=QuantType.QUInt8, activation_type=QuantType.QUInt8)
    print(f"[量化] ONNX(int8): {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
    return out


# --------------------------------------------------------------------- 校验
def _preprocess(img, size: int = 640):
    """letterbox 到 size×size，返回 (tensor, 缩放比, 左padding, 上padding)。"""
    import cv2

    h, w = img.shape[:2]
    s = min(size / h, size / w)
    nh, nw = int(round(h * s)), int(round(w * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x), s, left, top


def _decode(out: np.ndarray, s: float, left: int, top: int, conf_th: float = 0.25,
            iou_th: float = 0.45, max_det: int = 20):
    """YOLO11 检测头输出 (1, 4+nc, N) -> 框列表（原图坐标）。"""
    pred = out[0]
    if pred.shape[0] < pred.shape[1]:
        pred = pred.transpose(1, 0)          # (N, 4+nc)
    boxes = pred[:, :4]
    scores = pred[:, 4:]
    cls = scores.argmax(1)
    conf = scores[np.arange(len(scores)), cls]
    keep = conf >= conf_th
    boxes, conf, cls = boxes[keep], conf[keep], cls[keep]
    if len(boxes) == 0:
        return []
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - left) / s
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - top) / s
    order = np.argsort(-conf)
    xyxy, conf = xyxy[order], conf[order]
    out_boxes = []
    for b, c in zip(xyxy, conf):
        if all(_iou(b, k[0]) < iou_th for k in out_boxes):
            out_boxes.append((b, float(c)))
        if len(out_boxes) >= max_det:
            break
    return out_boxes


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def verify(onnx_path: str, images: list[str], weights: str, conf: float = 0.35) -> dict:
    import cv2
    import onnxruntime as ort
    from ultralytics import YOLO

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    yolo = YOLO(weights)
    rows = []
    for path in images:
        img = cv2.imread(path)
        if img is None:
            continue
        x, s, left, top = _preprocess(img)
        out = sess.run(None, {inp_name: x})[0]
        onnx_boxes = _decode(out, s, left, top, conf_th=conf)
        r = yolo.predict(img, conf=conf, imgsz=640, verbose=False)[0]
        pt_boxes = []
        if r.boxes is not None:
            for b in r.boxes:
                pt_boxes.append(([float(v) for v in b.xyxy[0].tolist()], float(b.conf[0])))
        # 配对比较
        pairs = 0
        worst = 1.0
        for pb, pc in pt_boxes:
            if not onnx_boxes:
                continue
            j = max(range(len(onnx_boxes)), key=lambda k: _iou(pb, onnx_boxes[k][0]))
            v = _iou(pb, onnx_boxes[j][0])
            if v > 0.5:
                pairs += 1
                worst = min(worst, v)
        rows.append({"image": os.path.basename(path), "pytorch": len(pt_boxes),
                     "onnx": len(onnx_boxes), "matched": pairs,
                     "worst_iou": round(float(worst), 3) if pairs else None,
                     "onnx_conf": [round(float(c), 3) for _, c in onnx_boxes[:3]],
                     "pt_conf": [round(float(c), 3) for _, c in pt_boxes[:3]]})
    return {"rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=paths.DEFAULT_MODEL)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-int8", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="复用已导出的 fp32 ONNX，跳过导出步骤")
    ap.add_argument("--verify", type=int, default=8, help="用多少张真实图片做一致性校验")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    dst_fp32 = os.path.join(OUT_DIR, FP32_NAME)
    src_fp32 = os.path.join(paths.MODELS_DIR, FP32_NAME)
    if args.reuse and os.path.exists(dst_fp32):
        print(f"[导出] 复用已有 ONNX: {dst_fp32} ({os.path.getsize(dst_fp32)/1e6:.1f} MB)")
    else:
        fp32 = export_fp32(args.weights, args.imgsz)
        shutil.copyfile(fp32, dst_fp32)
        src_fp32 = fp32
        print(f"[导出] 复制到 {dst_fp32}")

    # 校准/校验用真实图片：BOP 测试集 + val 集
    bop = os.path.join(paths.RAW_DIR, "ycbv_test_bop19", "test")
    calib: list[str] = []
    if os.path.isdir(bop):
        for scene in sorted(os.listdir(bop))[:6]:
            rgb = os.path.join(bop, scene, "rgb")
            if os.path.isdir(rgb):
                calib += [os.path.join(rgb, f) for f in sorted(os.listdir(rgb))[:12]]
    val_dir = os.path.join(paths.DATASET_DIR, "images", "val")
    if os.path.isdir(val_dir):
        calib += [os.path.join(val_dir, f) for f in sorted(os.listdir(val_dir))[:40]]
    block = os.path.join(paths.ROOT, "block.jpg")
    if os.path.exists(block):
        calib.append(block)
    print(f"[量化] 校准图片 {len(calib)} 张")

    if not args.no_int8 and calib:
        try:
            export_int8(dst_fp32, calib[:120], args.imgsz)
        except Exception as e:  # noqa: BLE001
            print(f"[量化] 失败（不影响 fp32）: {type(e).__name__}: {e}")

    test_imgs = ([block] if os.path.exists(block) else []) + calib[:args.verify]
    print("\n[校验] ONNX(fp32) 与 PyTorch 结果对比:")
    report = verify(dst_fp32, test_imgs, args.weights)
    for row in report["rows"]:
        flag = "一致" if row["matched"] == max(row["pytorch"], row["onnx"]) else "有差异"
        print(f"  {row['image'][:34]:36s} PT={row['pytorch']} ONNX={row['onnx']} "
              f"匹配={row['matched']} 最差IoU={row['worst_iou']} -> {flag}")
    ip = os.path.join(OUT_DIR, INT8_NAME)
    if os.path.exists(ip):
        print("\n[校验] ONNX(int8) 与 PyTorch 结果对比:")
        for row in verify(ip, test_imgs, args.weights)["rows"]:
            flag = "一致" if row["matched"] == max(row["pytorch"], row["onnx"]) else "有差异"
            print(f"  {row['image'][:34]:36s} PT={row['pytorch']} INT8={row['onnx']} "
                  f"匹配={row['matched']} 最差IoU={row['worst_iou']} -> {flag}")
    with open(os.path.join(OUT_DIR, "onnx_export_report.json"), "w", encoding="utf8") as f:
        json.dump({"fp32": os.path.basename(dst_fp32),
                   "fp32_mb": round(os.path.getsize(dst_fp32) / 1e6, 1),
                   "int8": os.path.basename(ip) if os.path.exists(ip) else None,
                   "int8_mb": round(os.path.getsize(ip) / 1e6, 1) if os.path.exists(ip) else None,
                   "verify": report}, f, ensure_ascii=False, indent=1)
    print("\n[导出] 完成，报告: onnx_export_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
