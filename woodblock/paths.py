"""项目路径与默认参数集中配置。"""
from __future__ import annotations

import os

# ---------------------------------------------------------------- 目录结构
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")            # 下载的原始数据（缓存）
RAW_SCENES_DIR = os.path.join(RAW_DIR, "ycbv_scenes")
CD_CACHE_DIR = os.path.join(RAW_DIR, "cd_cache")   # 远程 zip 目录缓存
DATASET_DIR = os.path.join(DATA_DIR, "woodblock")  # 训练用 YOLO 格式数据集
MODELS_DIR = os.path.join(ROOT, "models")
RUNS_DIR = os.path.join(ROOT, "runs")
REPORTS_DIR = os.path.join(ROOT, "reports")

for _d in (DATA_DIR, RAW_DIR, RAW_SCENES_DIR, CD_CACHE_DIR, DATASET_DIR, MODELS_DIR, RUNS_DIR, REPORTS_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------- 数据参数
WOOD_BLOCK_OBJ_ID = 16          # YCB-Video classes.txt 第 16 行 = 036_wood_block
CLASS_NAMES = ["wood_block"]
FRAME_STRIDE = 20               # 抽帧步长（相邻帧高度重复，取大一点换取更多"姿态多样性"）
MIN_BOX_PIXELS = 22             # 最小框边长（640x480 下），过滤过小目标
MAX_NEG_FRAMES_PER_SCENE = 20   # 每个负样本（不含木块）场景最多取多少帧
VAL_SCENE_COUNT = 4             # 从含木块场景中划出的验证场景数（按场景切分，避免泄漏）

# ---------------------------------------------------------------- 训练参数
BASE_WEIGHTS = "yolo11s.pt"     # COCO 预训练权重（迁移学习起点，本地 models/pretrained/）
IMG_SIZE = 640
EPOCHS = 70
BATCH = 16
CHIPPED_RATIO = 0.25            # 缺角增强：对多少比例的训练样本"敲掉一个角"
OCCLUDED_RATIO = 0.20           # 遮挡增强：对多少比例的训练样本挡住木块的一部分
DEFAULT_MODEL = os.path.join(MODELS_DIR, "wood_block_yolo11s.pt")

# ---------------------------------------------------------------- 推理参数
CONF_THRESHOLD = 0.35           # 默认置信度阈值（由 evaluate 在负样本上标定）
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 20
# 多尺度重采样推理（默认关闭，仅用 1.0 单一尺度）。
# 打开后（例如 (1.0, 0.7, 0.45)）能额外命中"木块占满画面"的近景，但在 BOP 官方
# 负样本上误检率会从 0.85% 升到 8.7%（见 reports/evaluation.md 的对比表），
# 因此默认保持单尺度，按需用 --scales 开启。
MULTI_SCALES = (1.0,)
# 需要近景覆盖时可用的推荐配置：
MULTI_SCALES_RECOMMENDED = (1.0, 0.7, 0.45)
