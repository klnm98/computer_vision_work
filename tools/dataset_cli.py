"""命令行入口：数据集扫描 / 拉取 / 构建。"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from woodblock import dataset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="YCB-Video 木块数据集工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scan", help="扫描场景，找出含木块的场景")
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--workers", type=int, default=10)

    p = sub.add_parser("fetch", help="拉取指定场景的标注与图片")
    p.add_argument("--scene", required=True)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--max-frames", type=int, default=None)

    p = sub.add_parser("build", help="构建 YOLO 数据集")
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--neg-scenes", type=int, default=18)
    p.add_argument("--neg-frames", type=int, default=20)

    args = ap.parse_args()

    if args.cmd == "scan":
        rep = dataset.scan_scenes(scene_ids=args.scenes, workers=args.workers, force=args.force)
        wood = dataset.wood_scenes(rep)
        print(f"\n含木块场景 {len(wood)} 个: {wood}")
        print(f"不含木块场景 {len(dataset.negative_scenes(rep))} 个")
        return 0

    if args.cmd == "fetch":
        meta = dataset.fetch_scene(args.scene, stride=args.stride, max_frames=args.max_frames)
        print(json.dumps({k: v for k, v in meta.items() if k != "frames"}, ensure_ascii=False))
        return 0

    if args.cmd == "build":
        rep = dataset.scan_scenes()
        stats = dataset.build_dataset(rep, stride=args.stride, max_scenes=args.max_scenes,
                                      neg_scenes=args.neg_scenes, neg_frames=args.neg_frames)
        print(json.dumps(stats, ensure_ascii=False, indent=1))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
