"""用 block.jpg 合成一段测试视频，用于验证摄像头检测管线。

视频结构：前 1/3 只有桌面（无木块，用于检验误检），中间 1/3 木块出现，
最后 1/3 木块缓慢平移（用于检验时序确认与框平滑）。

注意：这是**测试素材**，不参与任何训练。
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from woodblock.image_utils import crop_letterbox  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="block.jpg")
    ap.add_argument("--out", default=os.path.join("data", "test_block_video.mp4"))
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--size", type=int, default=640)
    args = ap.parse_args()

    img = crop_letterbox(cv2.imread(args.image))
    if img is None:
        raise SystemExit(f"无法读取 {args.image}")
    h, w = img.shape[:2]
    # 木块大致位置（block.jpg 中木块在画面左上区域）：取该区域作为"有木块"窗口
    block = img[int(h * 0.30):int(h * 0.52), int(w * 0.05):int(w * 0.55)]
    # 纯桌面区域（无木块）
    table = img[int(h * 0.06):int(h * 0.26), int(w * 0.35):int(w * 0.95)]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.out, fourcc, args.fps, (args.size, args.size))
    n = args.frames
    for i in range(n):
        phase = i / max(1, n - 1)
        if phase < 0.34:                      # 只有桌面
            patch = table
            dx = 0
        else:                                 # 木块出现并平移
            patch = block
            dx = int((phase - 0.34) / 0.66 * args.size * 0.25)
        ph, pw = patch.shape[:2]
        scale = args.size / max(ph, pw, 1)
        tile = cv2.resize(patch, (max(1, int(pw * scale)), max(1, int(ph * scale))))
        canvas = np.zeros((args.size, args.size, 3), np.uint8)
        th, tw = tile.shape[:2]
        y0 = max(0, (args.size - th) // 2)
        x0 = min(max(0, (args.size - tw) // 2 + dx), max(0, args.size - tw))
        canvas[y0:y0 + min(th, args.size - y0), x0:x0 + min(tw, args.size - x0)] = \
            tile[:min(th, args.size - y0), :min(tw, args.size - x0)]
        cv2.putText(canvas, f"frame {i}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        vw.write(canvas)
    vw.release()
    print(f"已生成测试视频: {args.out}  帧数={n}  尺寸={args.size}x{args.size}")
    print("  0-34%: 仅桌面（检验误检） | 34-100%: 含木块（检验检测与时序确认）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
