"""推理封装：加载训练好的木块检测模型，输出稳定的木块框。

为满足"不可框选到其他物体"，推理侧采用三重把关：
1. 单类别模型 —— 网络只会输出 wood_block 一种类别；
2. 置信度阈值 —— 由 ``evaluate`` 在真实负样本（不含木块的场景）上标定；
3. 形态与时序过滤 —— 极端长宽比/过小框直接丢弃；摄像头模式下还需
   连续多帧命中同一位置（N-of-M）才认为目标真实存在，抑制偶发误检。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from . import paths

_DEFAULT_SCALES = (1.0, 0.7, 0.45, 0.3)  # 多尺度重采样推理的比例


def _iou(a: "Detection", b: "Detection") -> float:
    """两个框的 IoU。"""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Detection:
    """一个木块检测框（像素坐标）。"""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def as_int_box(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


def pick_device(preferred: str | None = None) -> str:
    """选择推理设备：优先 GPU，其次 CPU。"""
    if preferred:
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


class WoodBlockDetector:
    """木块检测器。

    ``scales`` 支持"多尺度重采样"推理：先把画面按不同比例缩放再检测，最后按
    NMS 合并。原因是 YOLO 的输入会被统一归一化到 ``imgsz``（例如 640），所以
    直接把图送进去时，物体在输入里的相对大小是固定的；预先缩放相当于引入
    不同强度的重采样/模糊，能覆盖清晰度差异很大的输入（例如手机截图 vs
    摄像头画面），实测可显著提升对"木块占画面很大"这类输入的召回。
    """

    def __init__(self, weights: str | None = None, conf: float = paths.CONF_THRESHOLD,
                 iou: float = paths.IOU_THRESHOLD, imgsz: int = paths.IMG_SIZE,
                 device: str | None = None, max_det: int = paths.MAX_DETECTIONS,
                 max_aspect: float = 3.5, min_area_ratio: float = 2e-4,
                 tta: bool = False, scales: tuple[float, ...] | None = None,
                 verbose: bool = True):
        from ultralytics import YOLO

        self.weights = weights or paths.DEFAULT_MODEL
        if not os.path.exists(self.weights):
            raise SystemExit(
                f"找不到模型权重 {self.weights}\n请先训练：python main.py train"
            )
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.max_aspect = max_aspect
        self.min_area_ratio = min_area_ratio
        self.tta = tta
        self.scales = tuple(scales) if scales else tuple(paths.MULTI_SCALES)
        self.device = pick_device(device)
        self.model = YOLO(self.weights)
        if verbose:
            print(f"[推理] 模型={self.weights} device={self.device} conf={self.conf} "
                  f"iou={self.iou} tta={self.tta} scales={self.scales}")

    # ------------------------------------------------------------------
    def _accept(self, det: Detection, frame_w: int, frame_h: int) -> bool:
        """形态合理性检查：剔除极端狭长框与过小框（几乎不可能是木块）。"""
        w, h = det.width, det.height
        if w <= 1 or h <= 1:
            return False
        aspect = max(w / h, h / w)
        if aspect > self.max_aspect:
            return False
        if det.area / float(frame_w * frame_h) < self.min_area_ratio:
            return False
        return True

    def _predict_raw(self, frame) -> list[Detection]:
        """单次推理，返回未过滤的框（原图坐标）。"""
        results = self.model.predict(
            frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
            device=self.device, max_det=self.max_det, verbose=False,
            augment=self.tta,
        )
        out: list[Detection] = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for b in boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                conf = float(b.conf[0]) if b.conf is not None else 0.0
                out.append(Detection(x1, y1, x2, y2, conf))
        return out

    @staticmethod
    def _nms(dets: list[Detection], iou_thr: float) -> list[Detection]:
        """按置信度贪心 NMS，合并多尺度结果。"""
        keep: list[Detection] = []
        for d in sorted(dets, key=lambda x: -x.conf):
            if all(_iou(d, k) < iou_thr for k in keep):
                keep.append(d)
        return keep

    def detect(self, frame) -> list[Detection]:
        """对单帧 BGR 图像推理，返回通过过滤的木块框（原图坐标）。"""
        h, w = frame.shape[:2]
        raw: list[Detection] = []
        if len(self.scales) == 1:
            raw = self._predict_raw(frame)
        else:
            import cv2

            for s in self.scales:
                if abs(s - 1.0) < 1e-6:
                    raw.extend(self._predict_raw(frame))
                    continue
                small = cv2.resize(frame, (max(64, int(w * s)), max(64, int(h * s))),
                                   interpolation=cv2.INTER_AREA)
                for d in self._predict_raw(small):
                    raw.append(Detection(d.x1 / s, d.y1 / s, d.x2 / s, d.y2 / s, d.conf))
            raw = self._nms(raw, self.iou)

        out = [d for d in raw if self._accept(d, w, h)]
        out.sort(key=lambda d: -d.conf)
        return out


class TemporalFilter:
    """N-of-M 时序确认 + 框平滑，用于摄像头实时检测。

    只有当同一位置在最近 ``window`` 帧中出现至少 ``min_hits`` 次时才输出，
    可显著减少瞬时误检（例如背景纹理被误判）。
    """

    def __init__(self, min_hits: int = 3, window: int = 8, iou_threshold: float = 0.3,
                 smoothing: float = 0.6):
        self.min_hits = min_hits
        self.window = window
        self.iou_threshold = iou_threshold
        self.smoothing = smoothing
        self.tracks: list[dict] = []

    @staticmethod
    def _iou(a: Detection, b: Detection) -> float:
        ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
        ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = a.area + b.area - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections: list[Detection]) -> list[Detection]:
        """输入当前帧检测，返回经时序确认后的框。"""
        matched_det: set[int] = set()
        for tr in self.tracks:
            best, best_iou = -1, 0.0
            for i, det in enumerate(detections):
                if i in matched_det:
                    continue
                iou = self._iou(tr["det"], det)
                if iou > best_iou:
                    best, best_iou = i, iou
            if best >= 0 and best_iou >= self.iou_threshold:
                matched_det.add(best)
                det = detections[best]
                a = self.smoothing
                tr["det"] = Detection(
                    a * tr["det"].x1 + (1 - a) * det.x1,
                    a * tr["det"].y1 + (1 - a) * det.y1,
                    a * tr["det"].x2 + (1 - a) * det.x2,
                    a * tr["det"].y2 + (1 - a) * det.y2,
                    max(tr["det"].conf, det.conf),
                )
                tr["hits"] += 1
                tr["age"] = 0
            else:
                tr["age"] += 1

        for i, det in enumerate(detections):
            if i not in matched_det:
                self.tracks.append({"det": det, "hits": 1, "age": 0})

        self.tracks = [t for t in self.tracks if t["age"] <= self.window]
        return [t["det"] for t in self.tracks if t["hits"] >= self.min_hits]

    def reset(self) -> None:
        self.tracks.clear()
