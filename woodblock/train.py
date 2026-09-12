"""模型训练：在公开真实数据集（YCB-Video 木块）上微调 YOLO 检测器。

设计要点
--------
* 单类别（wood_block），从根本上避免把其他物体框成木块；
* 迁移学习：从 COCO 预训练的 YOLO11 权重开始微调，小数据集也能收敛；
* 强数据增强（HSV / 透视 / 旋转 / 缩放 / Mosaic / MixUp）弥补
  YCB 数据集（暗色木块、640x480 桌面场景）与手机摄像头实拍的域差异；
* 缺角增强：训练时对部分样本"擦掉木块的一角"，使模型不依赖完整方形轮廓。
"""
from __future__ import annotations

import os
import shutil

from . import paths


def _require_ultralytics():
    try:
        import ultralytics  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "未安装 ultralytics，请先执行：\n"
            "    python main.py setup\n"
            f"（原始错误：{e}）"
        ) from e
    return __import__("ultralytics")


def resolve_base_weights(name: str = paths.BASE_WEIGHTS) -> str:
    """返回可用的预训练权重路径（优先本地 models/pretrained）。"""
    local = os.path.join(paths.MODELS_DIR, "pretrained", os.path.basename(name))
    return local if os.path.exists(local) else name


def train(data_yaml: str | None = None, epochs: int = paths.EPOCHS,
          imgsz: int = paths.IMG_SIZE, batch: int = paths.BATCH,
          device: str | None = None, base: str | None = None,
          name: str = "wood_block", resume: bool = False, workers: int = 0) -> str:
    """训练模型并返回最佳权重路径。"""
    _require_ultralytics()
    from ultralytics import YOLO

    from . import compat

    compat.patch_ultralytics()
    workers = compat.safe_workers(workers)

    data_yaml = data_yaml or os.path.join(paths.DATASET_DIR, "data.yaml")
    if not os.path.exists(data_yaml):
        raise SystemExit(f"找不到数据集配置 {data_yaml}，请先运行：python main.py prepare")

    weights = resolve_base_weights(base or paths.BASE_WEIGHTS)
    print(f"[训练] 基础权重: {weights}")
    print(f"[训练] 数据集:   {data_yaml}")
    print(f"[训练] device={device or 'auto'} epochs={epochs} imgsz={imgsz} batch={batch} workers={workers}")

    model = YOLO(weights)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=paths.RUNS_DIR,
        name=name,
        exist_ok=True,
        resume=resume,
        # ---- 优化器 / 学习率
        optimizer="auto",
        lr0=0.008,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3.0,
        weight_decay=0.0005,
        patience=0,           # 关闭早停：配合 cos_lr 与最后关掉 mosaic，末段通常最好
        # ---- 颜色增强：跨越"数据集木块(暗光/固定相机) -> 实拍(手机/自然光)"的域差异
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        # ---- 几何增强：旋转/缩放/透视，模拟任意摆放角度
        degrees=20.0,
        translate=0.15,
        scale=0.6,
        shear=4.0,
        perspective=0.0008,
        fliplr=0.5,
        flipud=0.0,
        # ---- 组合增强
        mosaic=1.0,
        close_mosaic=15,
        mixup=0.10,
        copy_paste=0.0,
        # ---- 遮挡/缺角鲁棒性
        erasing=0.35,
        crop_fraction=1.0,
        # ---- 其他
        val=True,
        plots=True,
        verbose=True,
        seed=0,
        deterministic=True,
    )

    best = os.path.join(paths.RUNS_DIR, name, "weights", "best.pt")
    if not os.path.exists(best):
        raise SystemExit(f"训练结束但未找到权重 {best}")
    os.makedirs(paths.MODELS_DIR, exist_ok=True)
    shutil.copyfile(best, paths.DEFAULT_MODEL)
    print(f"[训练] 最佳权重已保存: {paths.DEFAULT_MODEL}")
    return paths.DEFAULT_MODEL
