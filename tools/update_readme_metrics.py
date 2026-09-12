"""把 reports/evaluation.json 的最新指标写进 README.md 的标记区块，保持文档与结果一致。"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")
REPORT = os.path.join(ROOT, "reports", "evaluation.json")
VIDEO = os.path.join(ROOT, "reports", "video_verification.json")

START, END = "<!-- METRICS:START -->", "<!-- METRICS:END -->"


def main() -> int:
    if not os.path.exists(REPORT):
        raise SystemExit("找不到 reports/evaluation.json，请先运行 python main.py eval")
    r = json.load(open(REPORT, encoding="utf8"))
    lines: list[str] = []

    v = r.get("val")
    if v:
        lines += ["**验证集（4 个留出场景，612 张，与训练场景不重叠）**", "",
                  f"- mAP@50 = **{v['mAP50']:.4f}**，mAP@50-95 = **{v['mAP50_95']:.4f}**",
                  f"- 精确率 P = {v['precision']:.4f}，召回率 R = {v['recall']:.4f}", ""]

    n, p = r.get("negative"), r.get("positive")
    if n and p:
        lines += [f"**BOP 官方测试集（{p['images']} 张含木块 + {n['images']} 张不含木块，来自 0048-0059 场景）**",
                  "",
                  f"- 不含木块图像误检率：**{n['fp_image_rate']*100:.2f}%**"
                  f"（{n['false_positive_images']}/{n['images']} 张，共 {n['false_positive_boxes']} 个框，"
                  f"其中框到其他物体 {n['boxes_on_other_objects']} 个）",
                  f"- 木块召回率：**{p['recall']*100:.1f}%**"
                  f"（{p['true_positives']}/{p['gt_boxes']}，IoU≥0.5），平均 IoU {p['mean_iou']:.3f}，"
                  f"多余检测框 {p['extra_detections']} 个",
                  f"- 负样本最高置信度 {n['max_conf']:.3f}，p99 = {n['p99_conf']:.3f}", ""]

    sweep = r.get("threshold_sweep")
    if sweep:
        lines += ["**阈值取舍表（BOP 官方测试集实测）**", "",
                  "| 置信度阈值 | 误检图像率 | 误检框 | 其中框到其他物体 | 木块召回率 |",
                  "| --- | --- | --- | --- | --- |"]
        for s in sweep:
            lines.append(f"| {s['threshold']:.2f} | {s['fp_image_rate']*100:.2f}% | {s['fp_boxes']} | "
                         f"{s['fp_boxes_on_other_objects']} | {s['recall']*100:.1f}% |")
        lines.append("")

    calib = r.get("calibration")
    if calib:
        lines += [f"**阈值标定**：在 {calib['images']} 张真实负样本上标定，"
                  f"建议阈值 **{calib['suggested_threshold']:.2f}**"
                  f"（目标误检率 {calib['target_fp_rate']*100:.1f}%）", ""]

    extra = r.get("extra") or {}
    bj = extra.get("block_jpg")
    if bj:
        dets = bj.get("detections", [])
        lines += ["**真实照片 `block.jpg`（手机截图，去黑边后 625×1280）**", ""]
        if dets:
            for d in dets:
                lines.append(f"- 检测到木块：置信度 {d['conf']:.3f}，框 {d['box']}")
        else:
            lines.append("- 未检测到木块")
        ch = extra.get("block_jpg_chipped")
        if ch:
            lines.append(f"- 缺角测试（敲掉一角）：{'通过' if ch.get('passed') else '未通过'}"
                         f"（置信度 {ch.get('before_conf')} → {ch.get('after_conf')}）")
        oc = extra.get("block_jpg_occluded")
        if oc and oc.get("rounds"):
            lines.append(f"- 遮挡测试（随机挡住 20%~45%）：{oc.get('passed_rounds')}/{oc.get('rounds')} 次通过"
                         f"（通过率 {oc.get('pass_rate', 0)*100:.0f}%）")
        lines.append("")

    if os.path.exists(VIDEO):
        vd = json.load(open(VIDEO, encoding="utf8"))
        t = vd.get("with_temporal") or {}
        if t:
            lines += ["**摄像头管线视频验证**（`tools/verify_video.py`，合成视频 90 帧）", "",
                      f"- 无木块段误检：{t.get('no_block_detected')}/{t.get('no_block_frames')} 帧"
                      f"（{t.get('no_block_fp_rate', 0)*100:.1f}%）",
                      f"- 有木块段命中：{t.get('block_detected')}/{t.get('block_frames')} 帧"
                      f"（{t.get('block_recall', 0)*100:.1f}%）",
                      f"- 处理速度：{t.get('fps')} FPS（GPU，含时序确认）", ""]

    lines.append(f"（生成时间：{r.get('generated_at', 'n/a')}；权重：`{r.get('weights', 'n/a')}`；"
                 f"置信度阈值：{r.get('conf')}）")

    text = open(README, encoding="utf8").read()
    block = START + "\n" + "\n".join(lines) + "\n" + END
    if START not in text or END not in text:
        raise SystemExit("README.md 中找不到 METRICS 标记区块")
    text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _m: block, text, flags=re.S)
    open(README, "w", encoding="utf8").write(text)
    print("README.md 指标区块已更新")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
