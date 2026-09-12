"""评估与阈值标定。

评估三件事（全部使用公开真实数据）：
1. **验证集 mAP**（按场景切分，避免同场景近重复帧造成的虚高）；
2. **负样本误检率** —— 在 BOP YCB-V 官方测试集里"不含木块"的真实图像上
   统计误检，用来验证"不可框选到其他物体"；
3. **木块召回率** —— 在 BOP YCB-V 官方测试集含木块的图像上与真值框比对。

另外提供：
* ``eval_single_image``  —— 单图检测（如 block.jpg）；
* ``eval_chipped``       —— 人为把木块一个角"敲掉"，验证缺角仍可识别；
* ``calibrate_threshold``—— 依据负样本误检分布标定置信度阈值。
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np

from . import dataset as ds
from . import paths, sources


# --------------------------------------------------------------------- 工具
def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_bop_test() -> list[dict]:
    """读取 BOP YCB-V 测试集索引。

    每张图返回：是否含木块、木块真值框，以及**其他所有物体的真值框**
    （用于判断模型是否把别的物体框了进来）。
    """
    root = os.path.join(paths.RAW_DIR, "ycbv_test_bop19", "test")
    if not os.path.isdir(root):
        raise SystemExit(
            "尚未准备 BOP 测试集，请先运行：python main.py prepare"
        )
    items = []
    for scene in sorted(os.listdir(root)):
        sdir = os.path.join(root, scene)
        gt_p = os.path.join(sdir, "scene_gt.json")
        info_p = os.path.join(sdir, "scene_gt_info.json")
        if not (os.path.exists(gt_p) and os.path.exists(info_p)):
            continue
        gt = json.load(open(gt_p, encoding="utf8"))
        info = json.load(open(info_p, encoding="utf8"))
        for frame, objs in gt.items():
            rgb = os.path.join(sdir, "rgb", f"{int(frame):06d}.png")
            if not os.path.exists(rgb):
                continue
            wood, others = [], []
            for i, o in enumerate(objs):
                bi = info[frame][i]
                bx = bi.get("bbox_visib") or bi["bbox_obj"]
                x, y, w, h = bx
                if w < 8 or h < 8:
                    continue
                box = [float(x), float(y), float(x + w), float(y + h)]
                if o["obj_id"] == paths.WOOD_BLOCK_OBJ_ID:
                    wood.append(box)
                else:
                    others.append({"name": sources.obj_name(o["obj_id"]), "box": box})
            items.append({"path": rgb, "scene": scene, "frame": int(frame),
                          "wood": wood, "others": others, "pos": bool(wood)})
    return items


# --------------------------------------------------------------------- 主评估
def evaluate(weights: str | None = None, conf: float = paths.CONF_THRESHOLD,
             data_yaml: str | None = None, max_negative: int = 400,
             imgsz: int = paths.IMG_SIZE, device: str | None = None,
             max_positive: int = 0, skip_val: bool = False,
             scales: tuple[float, ...] | None = None) -> dict:
    """完整评估并返回指标字典。"""
    from ultralytics import YOLO

    from . import compat
    from .detector import pick_device

    compat.patch_ultralytics()
    weights = weights or paths.DEFAULT_MODEL
    if not os.path.exists(weights):
        raise SystemExit(f"找不到权重 {weights}，请先训练：python main.py train")
    device = pick_device(device)
    model = YOLO(weights)
    report: dict = {"weights": weights, "conf": conf, "device": device}

    # ---- 1. 验证集 mAP
    data_yaml = data_yaml or os.path.join(paths.DATASET_DIR, "data.yaml")
    if os.path.exists(data_yaml) and not skip_val:
        print("[评估] 1/3 验证集 mAP ...")
        m = model.val(data=data_yaml, imgsz=imgsz, device=device, verbose=False, workers=compat.safe_workers(),
                      plots=False, project=paths.RUNS_DIR, name="val")
        report["val"] = {
            "mAP50": float(m.box.map50),
            "mAP50_95": float(m.box.map),
            "precision": float(m.box.mp),
            "recall": float(m.box.mr),
        }
        print(f"    mAP50={report['val']['mAP50']:.4f}  mAP50-95={report['val']['mAP50_95']:.4f}  "
              f"P={report['val']['precision']:.4f}  R={report['val']['recall']:.4f}")
    else:
        print("[评估] 跳过验证集" + ("（--skip-val）" if skip_val else "（未找到 data.yaml）"))
        report["val"] = None

    # ---- 2/3. BOP 官方测试集：一次推理，随后在不同阈值下统计误检与召回
    print("[评估] 2/3 BOP 测试集：负样本误检 + 木块召回（一次推理，多阈值统计）...")
    items = load_bop_test()
    negatives = [it for it in items if not it["pos"]][:max_negative]
    positives = [it for it in items if it["pos"]]
    if max_positive:
        positives = positives[:max_positive]

    import cv2

    from .detector import WoodBlockDetector

    low = 0.05  # 先以很低阈值推理，保留预测再离线扫描阈值
    # 用与实际部署一致的检测器（含多尺度重采样），保证报告指标等于上线表现
    det = WoodBlockDetector(weights=weights, conf=low, iou=paths.IOU_THRESHOLD,
                            imgsz=imgsz, device=device, scales=scales, verbose=False)

    def predict_boxes(img):
        return [(d.conf, [d.x1, d.y1, d.x2, d.y2]) for d in det.detect(img)]

    # 负样本：记录每张图的预测（含落点标签），用于任意阈值下的误检统计
    neg_preds: list[dict] = []
    for it in negatives:
        img = cv2.imread(it["path"])
        if img is None:
            continue
        rec = {"best": 0.0, "boxes": []}
        for c, box in predict_boxes(img):
            rec["best"] = max(rec["best"], c)
            hit = None
            for o in it["others"]:
                if _iou(box, o["box"]) >= 0.3:
                    hit = o["name"]
                    break
            rec["boxes"].append((c, hit))
        neg_preds.append(rec)

    # 正样本：记录预测与真值框
    pos_preds: list[dict] = []
    for it in positives:
        img = cv2.imread(it["path"])
        if img is None:
            continue
        pos_preds.append({"gt": it["wood"], "preds": predict_boxes(img)})

    def stats_at(th: float) -> dict:
        fp_images = fp_boxes = on_object = on_bg = 0
        per_obj: dict[str, int] = {}
        for rec in neg_preds:
            kept = [(c, h) for c, h in rec["boxes"] if c >= th]
            if kept:
                fp_images += 1
                fp_boxes += len(kept)
            for _, h in kept:
                if h:
                    on_object += 1
                    per_obj[h] = per_obj.get(h, 0) + 1
                else:
                    on_bg += 1
        tp = fn = extra = 0
        ious: list[float] = []
        for rec in pos_preds:
            preds = [b for c, b in rec["preds"] if c >= th]
            used = set()
            for g in rec["gt"]:
                best_i, best_iou = -1, 0.0
                for i, pb in enumerate(preds):
                    if i in used:
                        continue
                    v = _iou(pb, g)
                    if v > best_iou:
                        best_i, best_iou = i, v
                if best_iou >= 0.5:
                    used.add(best_i)
                    tp += 1
                    ious.append(best_iou)
                else:
                    fn += 1
            extra += len(preds) - len(used)
        n_img = max(1, len(neg_preds))
        return {
            "threshold": round(th, 3),
            "fp_images": fp_images,
            "fp_image_rate": fp_images / n_img,
            "fp_boxes": fp_boxes,
            "fp_boxes_on_other_objects": on_object,
            "fp_boxes_on_background": on_bg,
            "per_object_hits": dict(sorted(per_obj.items(), key=lambda kv: -kv[1])),
            "gt_boxes": tp + fn,
            "true_positives": tp,
            "recall": tp / max(1, tp + fn),
            "extra_detections": extra,
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
        }

    sweep = [stats_at(t) for t in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                                   0.45, 0.50, 0.60, 0.70, 0.80)]
    report["threshold_sweep"] = sweep
    at_conf = min(sweep, key=lambda s: abs(s["threshold"] - conf))

    # 兼容原有字段：以当前 conf 为基准的负样本/正样本统计
    neg_confs_arr = np.array([r["best"] for r in neg_preds]) if neg_preds else np.zeros(1)
    report["negative"] = {
        "images": len(neg_preds),
        "false_positive_images": at_conf["fp_images"],
        "fp_image_rate": at_conf["fp_image_rate"],
        "false_positive_boxes": at_conf["fp_boxes"],
        "boxes_on_other_objects": at_conf["fp_boxes_on_other_objects"],
        "boxes_on_background": at_conf["fp_boxes_on_background"],
        "per_object_hits": at_conf["per_object_hits"],
        "max_conf": float(neg_confs_arr.max()),
        "p99_conf": float(np.percentile(neg_confs_arr, 99)),
        "p999_conf": float(np.percentile(neg_confs_arr, 99.9)),
        "mean_conf": float(neg_confs_arr.mean()),
    }
    report["positive"] = {
        "images": len(pos_preds),
        "gt_boxes": at_conf["gt_boxes"],
        "true_positives": at_conf["true_positives"],
        "false_negatives": at_conf["gt_boxes"] - at_conf["true_positives"],
        "extra_detections": at_conf["extra_detections"],
        "recall": at_conf["recall"],
        "mean_iou": at_conf["mean_iou"],
    }
    n, p = report["negative"], report["positive"]
    print(f"    负样本 {n['images']} 张 @conf={at_conf['threshold']}: 误检 {n['false_positive_images']} 张 "
          f"({n['fp_image_rate']*100:.2f}%)，误检框 {n['false_positive_boxes']} 个"
          f"（框到其他物体 {n['boxes_on_other_objects']}，框到背景 {n['boxes_on_background']}）")
    print(f"    负样本置信度: max={n['max_conf']:.3f} p99={n['p99_conf']:.3f}")
    print(f"    含木块 {p['images']} 张 / 真值框 {p['gt_boxes']} 个: 召回 {p['true_positives']} = "
          f"{p['recall']*100:.1f}%，平均 IoU={p['mean_iou']:.3f}，多余框 {p['extra_detections']}")
    print("    阈值扫描（阈值 | 误检图像率 | 误检框 | 木块召回率）:")
    for s in sweep:
        print(f"      {s['threshold']:.2f} | {s['fp_image_rate']*100:6.2f}% | {s['fp_boxes']:4d} | "
              f"{s['recall']*100:6.1f}%")
    return report


# --------------------------------------------------------------------- 阈值
def calibrate_threshold(weights: str | None = None, target_fp_rate: float = 0.005,
                        imgsz: int = paths.IMG_SIZE, device: str | None = None,
                        max_negative: int = 400) -> dict:
    """用负样本置信度分布标定阈值：取满足目标误检率的分位数。"""
    import cv2
    from ultralytics import YOLO

    from . import compat
    from .detector import pick_device

    compat.patch_ultralytics()
    weights = weights or paths.DEFAULT_MODEL
    device = pick_device(device)
    model = YOLO(weights)
    items = [it for it in load_bop_test() if not it["pos"]][:max_negative]

    best_confs = []
    for it in items:
        img = cv2.imread(it["path"])
        if img is None:
            continue
        r = model.predict(img, conf=0.01, iou=paths.IOU_THRESHOLD, imgsz=imgsz,
                          device=device, max_det=paths.MAX_DETECTIONS, verbose=False)[0]
        best_confs.append(float(r.boxes.conf.max()) if r.boxes is not None and len(r.boxes) else 0.0)
    arr = np.array(best_confs) if best_confs else np.zeros(1)
    q = np.quantile(arr, 1.0 - target_fp_rate)
    suggested = float(min(0.95, max(paths.CONF_THRESHOLD, np.ceil(q * 100) / 100 + 0.01)))
    out = {"images": len(best_confs), "suggested_threshold": suggested,
           "p99": float(np.percentile(arr, 99)), "max": float(arr.max()),
           "target_fp_rate": target_fp_rate}
    print(f"[标定] 负样本 {out['images']} 张 -> 建议阈值 {suggested:.2f} "
          f"(p99={out['p99']:.3f}, max={out['max']:.3f})")
    return out


# --------------------------------------------------------------------- 单图
def eval_single_image(image_path: str, weights: str | None = None,
                      conf: float = paths.CONF_THRESHOLD, save_to: str | None = None,
                      device: str | None = None, crop: bool = True,
                      long_side: int = 1280) -> dict:
    """对单张图片检测并（可选）保存可视化结果。

    手机截图（如 block.jpg）带有黑边与播放器界面，默认先去掉黑边再按长边缩放，
    更接近摄像头实际看到的画面。
    """
    import cv2

    from .detector import WoodBlockDetector
    from .image_utils import crop_letterbox, resize_long_side

    det = WoodBlockDetector(weights=weights, conf=conf, device=device, verbose=False)
    raw = cv2.imread(image_path)
    if raw is None:
        raise SystemExit(f"无法读取图片 {image_path}")
    img = crop_letterbox(raw) if crop else raw
    img = resize_long_side(img, long_side)
    dets = det.detect(img)
    vis = img.copy()
    for d in dets:
        x1, y1, x2, y2 = d.as_int_box()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 3)
        cv2.putText(vis, f"wood_block {d.conf:.2f}", (x1, max(24, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        cv2.imwrite(save_to, vis)
    return {"image": image_path, "input_shape": list(raw.shape[:2]),
            "used_shape": list(img.shape[:2]), "cropped": crop,
            "detections": [
                {"box": [round(v, 1) for v in (d.x1, d.y1, d.x2, d.y2)], "conf": round(d.conf, 4)}
                for d in dets], "saved": save_to}


# --------------------------------------------------------------------- 缺角
def eval_chipped(image_path: str, weights: str | None = None,
                 conf: float = paths.CONF_THRESHOLD, cut_ratio: float = 0.3,
                 save_to: str | None = None, device: str | None = None,
                 crop: bool = True, long_side: int = 1280) -> dict:
    """缺角鲁棒性测试。

    先用模型找到木块，再用与训练时相同的"敲掉一个角"方式修改图像（三角形切口 +
    相邻背景纹理填补），重新检测，看是否仍能框出木块。
    """
    import cv2
    import random

    from .dataset import make_chipped
    from .detector import WoodBlockDetector
    from .image_utils import crop_letterbox, resize_long_side

    det = WoodBlockDetector(weights=weights, conf=conf, device=device, verbose=False)
    raw = cv2.imread(image_path)
    if raw is None:
        raise SystemExit(f"无法读取图片 {image_path}")
    img = resize_long_side(crop_letterbox(raw) if crop else raw, long_side)
    base = det.detect(img)
    if not base:
        return {"image": image_path, "error": "原图未检测到木块，无法进行缺角测试",
                "detected_before": 0, "detected_after": 0, "passed": False}

    target = base[0]
    rng = random.Random(0)
    out = make_chipped(img, (target.x1, target.y1, target.x2, target.y2), rng,
                       ratio_range=(cut_ratio, cut_ratio))
    if out is None:
        return {"image": image_path, "error": "木块框太小，无法做缺角测试", "passed": False}

    after = det.detect(out)
    hit = bool(after) and _iou([after[0].x1, after[0].y1, after[0].x2, after[0].y2],
                               [target.x1, target.y1, target.x2, target.y2]) >= 0.3
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        vis = out.copy()
        for d in after:
            x1, y1, x2, y2 = d.as_int_box()
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 3)
            cv2.putText(vis, f"wood_block {d.conf:.2f}", (x1, max(24, y1 - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        cv2.imwrite(save_to, vis)
    return {"image": image_path, "detected_before": len(base), "detected_after": len(after),
            "before_conf": round(target.conf, 4),
            "after_conf": round(after[0].conf, 4) if after else 0.0,
            "passed": hit, "saved": save_to}


# --------------------------------------------------------------------- 遮挡
def eval_occluded(image_path: str, weights: str | None = None,
                  conf: float = paths.CONF_THRESHOLD, save_to: str | None = None,
                  device: str | None = None, crop: bool = True,
                  long_side: int = 1280, rounds: int = 5) -> dict:
    """遮挡鲁棒性测试：随机挡住木块的一部分，看是否仍能框出木块。

    做 ``rounds`` 次随机遮挡（每次遮住 20%~45% 的面积），统计仍能检测到且框
    基本重合的比例 —— 对应"木块被人手/其他物体挡住一部分"的真实情况。
    """
    import cv2
    import random

    from .dataset import make_occluded
    from .detector import WoodBlockDetector
    from .image_utils import crop_letterbox, resize_long_side

    det = WoodBlockDetector(weights=weights, conf=conf, device=device, verbose=False)
    raw = cv2.imread(image_path)
    if raw is None:
        raise SystemExit(f"无法读取图片 {image_path}")
    img = resize_long_side(crop_letterbox(raw) if crop else raw, long_side)
    base = det.detect(img)
    if not base:
        return {"image": image_path, "error": "原图未检测到木块，无法进行遮挡测试", "passed": False}
    target = base[0]
    box = (target.x1, target.y1, target.x2, target.y2)

    ok = 0
    confs = []
    best_vis = None
    for i in range(rounds):
        rng = random.Random(100 + i)
        out = make_occluded(img, box, rng)
        if out is None:
            continue
        after = det.detect(out)
        hit = bool(after) and _iou([after[0].x1, after[0].y1, after[0].x2, after[0].y2],
                                   list(box)) >= 0.3
        ok += int(hit)
        confs.append(round(after[0].conf, 4) if after else 0.0)
        if best_vis is None:
            vis = out.copy()
            for d in after:
                x1, y1, x2, y2 = d.as_int_box()
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 3)
            best_vis = vis
    if save_to and best_vis is not None:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        cv2.imwrite(save_to, best_vis)
    return {"image": image_path, "rounds": rounds, "passed_rounds": ok,
            "pass_rate": ok / max(1, rounds), "confs": confs,
            "passed": ok >= max(1, rounds // 2), "saved": save_to}


# --------------------------------------------------------------------- 报告
def write_report(report: dict, calib: dict | None = None, extra: dict | None = None,
                 name: str = "evaluation") -> str:
    """把评估结果写成 JSON + Markdown。"""
    report = dict(report)
    if calib:
        report["calibration"] = calib
    if extra:
        report["extra"] = extra
    report["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(paths.REPORTS_DIR, exist_ok=True)
    jp = os.path.join(paths.REPORTS_DIR, f"{name}.json")
    with open(jp, "w", encoding="utf8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    lines = ["# 木块识别模型评估报告", "",
             f"- 生成时间: {report['generated_at']}",
             f"- 权重: `{report.get('weights')}`",
             f"- 置信度阈值: {report.get('conf')}", ""]
    if report.get("val"):
        v = report["val"]
        lines += ["## 1. 验证集指标（按场景切分）", "",
                  "| 指标 | 数值 |", "| --- | --- |",
                  f"| mAP@50 | {v['mAP50']:.4f} |",
                  f"| mAP@50-95 | {v['mAP50_95']:.4f} |",
                  f"| 精确率 P | {v['precision']:.4f} |",
                  f"| 召回率 R | {v['recall']:.4f} |", ""]
    n = report.get("negative")
    if n:
        lines += ["## 2. 负样本误检（BOP 官方测试集中不含木块的图像）", "",
                  "| 指标 | 数值 |", "| --- | --- |",
                  f"| 图像数 | {n['images']} |",
                  f"| 误检图像数 | {n['false_positive_images']} |",
                  f"| 误检率 | {n['fp_image_rate']*100:.2f}% |",
                  f"| 误检框总数 | {n['false_positive_boxes']} |",
                  f"| 其中框到**其他物体** | {n.get('boxes_on_other_objects', 'n/a')} |",
                  f"| 其中框到背景/桌面 | {n.get('boxes_on_background', 'n/a')} |",
                  f"| 最高置信度 | {n['max_conf']:.3f} |",
                  f"| p99 置信度 | {n['p99_conf']:.3f} |", ""]
        if n.get("per_object_hits"):
            lines += ["被误框的物体统计（IoU>=0.3 判定）：", "", "```",
                      json.dumps(n["per_object_hits"], ensure_ascii=False), "```", ""]
    sweep = report.get("threshold_sweep")
    if sweep:
        lines += ["## 直接可用的阈值取舍表（在 BOP 官方测试集上实测）", "",
                  "| 置信度阈值 | 误检图像率 | 误检框数 | 其中框到其他物体 | 木块召回率 | 平均 IoU |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for s in sweep:
            lines.append(f"| {s['threshold']:.2f} | {s['fp_image_rate']*100:.2f}% | {s['fp_boxes']} | "
                         f"{s['fp_boxes_on_other_objects']} | {s['recall']*100:.1f}% | {s['mean_iou']:.3f} |")
        lines.append("")
    p = report.get("positive")
    if p:
        lines += ["## 3. 木块召回（BOP 官方测试集，与真值框 IoU>=0.5 判定）", "",
                  "| 指标 | 数值 |", "| --- | --- |",
                  f"| 含木块图像数 | {p['images']} |",
                  f"| 真值框数 | {p['gt_boxes']} |",
                  f"| 召回率 | {p['recall']*100:.1f}% |",
                  f"| 平均 IoU | {p['mean_iou']:.3f} |",
                  f"| 多余检测框 | {p['extra_detections']} |", ""]
    if calib:
        lines += ["## 4. 阈值标定", "",
                  f"- 建议阈值: **{calib['suggested_threshold']:.2f}**（目标误检率 {calib['target_fp_rate']*100:.1f}%）",
                  f"- 负样本 p99 置信度: {calib['p99']:.3f}，最大值: {calib['max']:.3f}", ""]
    if extra:
        lines += ["## 5. 附加验证", "", "```json",
                  json.dumps(extra, ensure_ascii=False, indent=1), "```", ""]
    mp = os.path.join(paths.REPORTS_DIR, f"{name}.md")
    with open(mp, "w", encoding="utf8") as f:
        f.write("\n".join(lines))
    print(f"[评估] 报告已写入 {mp}")
    return mp
