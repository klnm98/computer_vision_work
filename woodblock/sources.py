"""公开数据集来源与出处说明。

本项目【只使用真实、公开的数据集】训练模型，不使用自采数据、不使用合成数据。

主数据集
--------
YCB-Video Dataset (YCB Benchmark, Yale GRAB Lab / NVIDIA)
  * 真实拍摄：Intel RealSense 采集的 92 个桌面场景视频（133,936 帧），
    21 类日常物体，含每帧 2D 边界框与 6D 位姿标注。
  * 本项目仅使用其中的 **036_wood_block（真实木块，obj_id = 16）**。
  * 官方主页: https://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/
  * 论文: Xiang et al., "PoseCNN: A Convolutional Neural Network for 6D Object
    Pose Estimation in Cluttered Scenes", RSS 2018.
  * 镜像: https://huggingface.co/datasets/Linpeng502502/YCB_Video_Dataset

BOP YCB-V (BOP Challenge 官方整理版，用于独立测试集与负样本)
  * 真实拍摄，含 RGB、实例掩码、scene_gt / scene_gt_info（2D 框 + 可见比例）。
  * 论文: Hodaň et al., "BOP: Benchmark for 6D Object Pose Estimation", ECCV 2018.
  * 镜像: https://huggingface.co/datasets/bop-benchmark/ycbv  (MIT License)

许可与引用
----------
使用 YCB 数据集请引用 YCB 及 YCB-Video 论文；BOP 部分遵循 MIT License。
本仓库代码仅用于学习/研究目的。
"""
from __future__ import annotations

HF_MIRROR = "https://hf-mirror.com/datasets"

# YCB-Video 场景压缩包（每个场景一个 zip，支持 HTTP Range 按需取单个文件）
YCBB_VIDEO_SCENE_URL = HF_MIRROR + "/Linpeng502502/YCB_Video_Dataset/resolve/main/data/{scene}.zip"
YCBB_VIDEO_POSES_URL = HF_MIRROR + "/Linpeng502502/YCB_Video_Dataset/resolve/main/poses.zip"
YCBB_VIDEO_IMAGE_SETS_URL = HF_MIRROR + "/Linpeng502502/YCB_Video_Dataset/resolve/main/image_sets.zip"

# BOP 官方整理的 YCB-V 测试集（含实例掩码与 2D 框标注）
BOP_YCBV_TEST_URL = HF_MIRROR + "/bop-benchmark/ycbv/resolve/main/ycbv_test_bop19.zip"
BOP_YCBV_MODELS_URL = HF_MIRROR + "/bop-benchmark/ycbv/resolve/main/ycbv_base.zip"

# YCB-Video 训练场景编号（不含 0048-0059 的官方验证/BOP 测试场景）
TRAIN_SCENE_IDS = [f"{i:04d}" for i in range(92) if not (48 <= i <= 59)]
TEST_SCENE_IDS = [f"{i:04d}" for i in range(48, 60)]

# YCB-Video 21 类物体（classes.txt 顺序，行号即 obj_id）
YCB_CLASSES = [
    "002_master_chef_can", "003_cracker_box", "004_sugar_box", "005_tomato_soup_can",
    "006_mustard_bottle", "007_tuna_fish_can", "008_pudding_box", "009_gelatin_box",
    "010_potted_meat_can", "011_banana", "019_pitcher_base", "021_bleach_cleanser",
    "024_bowl", "025_mug", "035_power_drill", "036_wood_block", "037_scissors",
    "040_large_marker", "051_large_clamp", "052_extra_large_clamp", "061_foam_brick",
]


def obj_name(obj_id: int) -> str:
    """obj_id（1 起）-> 物体名称。"""
    if 1 <= obj_id <= len(YCB_CLASSES):
        return YCB_CLASSES[obj_id - 1]
    return f"obj_{obj_id}"
