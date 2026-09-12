"""在合成测试视频上验证检测管线（误检抑制 + 时序确认 + 速度）。

视频由 tools/make_test_video.py 生成：前 34% 只有桌面（无木块），
之后木块出现。理想结果：
  * 前段（无木块）检测帧数 ≈ 0；
  * 后段（有木块）检测帧比例高；
  * 开启时序确认后，前段的偶发误检被完全抑制。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from woodblock.detector import TemporalFilter, WoodBlockDetector  # noqa: E402


def run(video: str, weights: str, conf: float, device: str, temporal: bool,
        imgsz: int = 640, save_video: str | None = None) -> dict:
    det = WoodBlockDetector(weights=weights, conf=conf, imgsz=imgsz, device=device, verbose=False)
    smoother = TemporalFilter(min_hits=3) if temporal else None
    cap = cv2.VideoCapture(video)
    writer = None
    total = 0
    neg_frames = neg_hits = 0
    pos_frames = pos_hits = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1
        is_neg = (total - 1) / max(1, n - 1) < 0.34
        dets = det.detect(frame)
        shown = smoother.update(dets) if smoother else dets
        if is_neg:
            neg_frames += 1
            neg_hits += 1 if shown else 0
        else:
            pos_frames += 1
            pos_hits += 1 if shown else 0
        vis = frame.copy()
        for d in shown:
            x1, y1, x2, y2 = d.as_int_box()
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        if save_video:
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            writer.write(vis)
    cap.release()
    if writer is not None:
        writer.release()
    dt = time.time() - t0
    res = {
        "video": video, "temporal": temporal, "conf": conf, "device": device,
        "frames": total, "fps": round(total / max(dt, 1e-6), 1),
        "no_block_frames": neg_frames, "no_block_detected": neg_hits,
        "no_block_fp_rate": round(neg_hits / max(1, neg_frames), 4),
        "block_frames": pos_frames, "block_detected": pos_hits,
        "block_recall": round(pos_hits / max(1, pos_frames), 4),
        "saved": save_video,
    }
    print(f"  temporal={temporal} conf={conf} -> fps={res['fps']}  "
          f"无木块段误检 {neg_hits}/{neg_frames} ({res['no_block_fp_rate']*100:.1f}%)  "
          f"有木块段命中 {pos_hits}/{pos_frames} ({res['block_recall']*100:.1f}%)")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join("data", "test_block_video.mp4"))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--save-video", default=os.path.join("reports", "test_video_detected.mp4"))
    ap.add_argument("--json", default=os.path.join("reports", "video_verification.json"))
    args = ap.parse_args()
    if not os.path.exists(args.video):
        raise SystemExit(f"找不到测试视频 {args.video}，请先运行 tools/make_test_video.py")

    print(f"[视频验证] {args.video}")
    no_temporal = run(args.video, args.weights, args.conf, args.device, temporal=False,
                      imgsz=args.imgsz, save_video=None)
    with_temporal = run(args.video, args.weights, args.conf, args.device, temporal=True,
                        imgsz=args.imgsz, save_video=args.save_video)
    if args.json:
        import json
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump({"no_temporal": no_temporal, "with_temporal": with_temporal},
                  open(args.json, "w", encoding="utf8"), ensure_ascii=False, indent=1)
        print("  结果已写入", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
