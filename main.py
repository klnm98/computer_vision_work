"""主进程：木块识别系统的统一入口。

子命令
------
    python main.py setup      # 安装/检查依赖（PyTorch、ultralytics 等）
    python main.py prepare    # 下载公开数据集(YCB-Video/BOP)并构建训练集
    python main.py train      # 训练木块检测模型
    python main.py eval       # 评估模型（mAP、误检率、召回率、缺角测试）
    python main.py camera     # 打开摄像头实时识别（默认动作）
    python main.py image <路径>  # 对单张图片识别（如 block.jpg）
    python main.py all        # 一键：准备数据 -> 训练 -> 评估

不带参数运行时：有模型就开摄像头，没有模型则先自动完成 prepare + train。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from woodblock import paths  # noqa: E402

BANNER = r"""
=========================================================
        木块识别系统  Wooden Block Detector
  数据集: YCB-Video / BOP YCB-V (真实、公开)
  模型:   YOLO11 单类别检测器 (wood_block)
=========================================================
"""

REQUIRED = [("torch", "PyTorch"), ("ultralytics", "Ultralytics YOLO")]


def check_deps(verbose: bool = True) -> bool:
    """检查关键依赖是否可用。"""
    missing = []
    for mod, name in REQUIRED:
        try:
            __import__(mod)
        except ImportError:
            missing.append((mod, name))
    if missing and verbose:
        print("缺少依赖:")
        for mod, name in missing:
            print(f"  - {name} ({mod})")
        print("\n请运行:  python main.py setup")
    return not missing


def cmd_setup(args) -> int:
    """安装依赖。"""
    from woodblock import env_setup

    return env_setup.install_requirements()


def cmd_prepare(args) -> int:
    """下载公开数据集并构建 YOLO 训练集。"""
    from woodblock import dataset as ds

    print("[1/2] 下载 BOP YCB-V 官方测试集（用于误检率/召回率评估）...")
    ds.fetch_bop_test()
    print("[2/2] 扫描 YCB-Video 场景并构建训练集...")
    report = ds.scan_scenes(workers=args.workers)
    woods = ds.wood_scenes(report)
    print(f"  含木块场景 {len(woods)} 个，不含木块场景 {len(ds.negative_scenes(report))} 个")
    stats = ds.build_dataset(report, stride=args.stride, max_scenes=args.max_scenes,
                             neg_scenes=args.neg_scenes, neg_frames=args.neg_frames)
    print(f"  训练集: {stats['train_images']} 张（负样本 {stats['neg_images']} 张）")
    print(f"  验证集: {stats['val_images']} 张（验证场景 {stats['val_scenes']}，与训练场景不重叠）")
    return 0


def cmd_train(args) -> int:
    from woodblock import train as tr

    tr.train(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
             device=args.device, base=args.base, name=args.name)
    return 0


def cmd_eval(args) -> int:
    from woodblock import evaluate as ev

    report = ev.evaluate(conf=args.conf, device=args.device,
                         max_negative=args.max_negative, max_positive=args.max_positive,
                         skip_val=args.skip_val)
    calib = ev.calibrate_threshold(device=args.device, max_negative=args.max_negative) \
        if not args.no_calibrate else None
    extra = {}
    sample = os.path.join(ROOT, "block.jpg")
    if os.path.exists(sample) and not args.skip_val:
        print("[评估] 附加：真实照片 block.jpg 检测 + 缺角鲁棒性测试")
        extra["block_jpg"] = ev.eval_single_image(
            sample, conf=args.conf, device=args.device,
            save_to=os.path.join(paths.REPORTS_DIR, "block_detected.jpg"))
        extra["block_jpg_chipped"] = ev.eval_chipped(
            sample, conf=args.conf, device=args.device,
            save_to=os.path.join(paths.REPORTS_DIR, "block_chipped_detected.jpg"))
        extra["block_jpg_occluded"] = ev.eval_occluded(
            sample, conf=args.conf, device=args.device,
            save_to=os.path.join(paths.REPORTS_DIR, "block_occluded_detected.jpg"))
    ev.write_report(report, calib=calib, extra=extra)
    return 0


def cmd_camera(args) -> int:
    import detect_camera

    if getattr(args, "probe", False):
        print("摄像头自检中（会依次尝试各设备 × 后端 × 分辨率，可能需要十几秒）...")
        detect_camera.probe_cameras(max_index=getattr(args, "max_index", 6))
        return 0
    detect_camera.run(source=args.source, weights=args.weights, conf=args.conf,
                      imgsz=args.imgsz, device=args.device,
                      temporal=not args.no_temporal, show=not args.no_show,
                      max_frames=args.max_frames, save_video=args.save_video,
                      backend=getattr(args, "backend", "auto"),
                      warmup=getattr(args, "warmup", 20),
                      rescan=getattr(args, "rescan", False),
                      auto_recover=not getattr(args, "no_recover", False))
    return 0


def cmd_image(args) -> int:
    import detect_camera

    detect_camera.run(source=args.path, weights=args.weights, conf=args.conf,
                      imgsz=args.imgsz, device=args.device, show=not args.no_show)
    return 0


def cmd_all(args) -> int:
    for step in (cmd_prepare, cmd_train, cmd_eval):
        t0 = time.time()
        rc = step(args)
        if rc != 0:
            print(f"步骤 {step.__name__} 失败（返回 {rc}）")
            return rc
        print(f"步骤 {step.__name__} 完成，用时 {time.time()-t0:.0f}s\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="木块识别系统主进程", formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("setup", help="安装/检查依赖")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("prepare", help="下载公开数据集并构建训练集")
    p.add_argument("--stride", type=int, default=paths.FRAME_STRIDE, help="抽帧步长")
    p.add_argument("--max-scenes", type=int, default=None, help="最多使用多少个含木块场景（默认全部）")
    p.add_argument("--neg-scenes", type=int, default=40, help="负样本场景数")
    p.add_argument("--neg-frames", type=int, default=20, help="每个负样本场景取多少帧")
    p.add_argument("--workers", type=int, default=12, help="下载并发数")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="训练模型")
    p.add_argument("--epochs", type=int, default=paths.EPOCHS)
    p.add_argument("--batch", type=int, default=paths.BATCH)
    p.add_argument("--imgsz", type=int, default=paths.IMG_SIZE)
    p.add_argument("--device", default=None)
    p.add_argument("--base", default=paths.BASE_WEIGHTS, help="预训练权重")
    p.add_argument("--name", default="wood_block", help="训练输出目录名（runs/<name>）")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", help="评估模型")
    p.add_argument("--conf", type=float, default=paths.CONF_THRESHOLD)
    p.add_argument("--device", default=None)
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--max-negative", type=int, default=400, help="负样本图像上限（加速调试）")
    p.add_argument("--max-positive", type=int, default=0, help="含木块图像上限，0=全部")
    p.add_argument("--skip-val", action="store_true", help="跳过验证集 mAP 与单图测试（快速调试）")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("camera", help="摄像头实时识别")
    p.add_argument("--source", default="0")
    p.add_argument("--weights", default=None)
    p.add_argument("--conf", type=float, default=paths.CONF_THRESHOLD)
    p.add_argument("--imgsz", type=int, default=paths.IMG_SIZE)
    p.add_argument("--device", default=None)
    p.add_argument("--no-temporal", action="store_true")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--save-video", default=None)
    p.add_argument("--backend", default="auto", help="摄像头后端：auto（自动排查）/dshow/msmf/any")
    p.add_argument("--warmup", type=int, default=20, help="丢弃多少帧预热画面")
    p.add_argument("--rescan", action="store_true", help="忽略上次记住的摄像头组合，重新自检")
    p.add_argument("--no-recover", action="store_true", help="关闭摄像头掉线/黑屏自动恢复")
    p.add_argument("--max-index", type=int, default=6, help="自检时最多枚举到第几号设备")
    p.add_argument("--probe", action="store_true", help="只做摄像头自检并给出建议，不启动识别")
    p.set_defaults(func=cmd_camera)

    p = sub.add_parser("image", help="单张图片识别")
    p.add_argument("path", nargs="?", default=os.path.join(ROOT, "block.jpg"))
    p.add_argument("--weights", default=None)
    p.add_argument("--conf", type=float, default=paths.CONF_THRESHOLD)
    p.add_argument("--imgsz", type=int, default=paths.IMG_SIZE)
    p.add_argument("--device", default=None)
    p.add_argument("--no-show", action="store_true")
    p.set_defaults(func=cmd_image)

    p = sub.add_parser("all", help="一键：prepare + train + eval")
    p.add_argument("--stride", type=int, default=paths.FRAME_STRIDE)
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--neg-scenes", type=int, default=40)
    p.add_argument("--neg-frames", type=int, default=20)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--epochs", type=int, default=paths.EPOCHS)
    p.add_argument("--batch", type=int, default=paths.BATCH)
    p.add_argument("--imgsz", type=int, default=paths.IMG_SIZE)
    p.add_argument("--device", default=None)
    p.add_argument("--base", default=paths.BASE_WEIGHTS)
    p.add_argument("--conf", type=float, default=paths.CONF_THRESHOLD)
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--max-negative", type=int, default=400)
    p.add_argument("--max-positive", type=int, default=0)
    p.add_argument("--skip-val", action="store_true")
    p.set_defaults(func=cmd_all)

    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser()
    args = ap.parse_args(argv)
    print(BANNER)

    if not check_deps():
        return 2

    if args.cmd is None:
        # 默认动作：有模型 -> 开摄像头；没有模型 -> 先准备数据并训练
        if os.path.exists(paths.DEFAULT_MODEL):
            print("已找到训练好的模型，进入摄像头识别模式（Ctrl+C 或按 q 退出）。\n")
            args = argparse.Namespace(
                source="0", weights=None, conf=paths.CONF_THRESHOLD,
                imgsz=paths.IMG_SIZE, device=None, no_temporal=False,
                no_show=False, max_frames=None, save_video=None)
            args.func = cmd_camera
        else:
            print("尚未训练模型，将自动执行：准备数据 -> 训练 -> 评估。")
            args.func = cmd_all

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n已被用户中断。")
        return 130
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"\n运行失败: {type(e).__name__}: {e}")
        if args.cmd in (None, "camera"):
            print("提示：摄像头模式需要可用摄像头；也可用 "
                  "`python main.py image block.jpg` 直接对图片测试。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
