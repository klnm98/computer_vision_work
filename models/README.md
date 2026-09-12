# models/ 目录说明

| 文件 | 是否入库 | 说明 |
| --- | --- | --- |
| `wood_block_yolo11s.pt` | ✅ 是 | **最终上线的木块检测模型**（YOLO11s，单类别 `wood_block`）。验证集 mAP@50 = 0.828，BOP 官方测试集误检率 0.85%、木块召回 73.3%。clone 后直接可用，无需重新训练。 |
| `wood_block_v3_zoom.pt` | ❌ 否 | 实验版本（额外加了"近景放大"增强）：验证集更高，但 BOP 测试集召回更低，仅用于拍特写时临时切换。可由 `python main.py prepare` + `python main.py train` 复现。 |
| `pretrained/*.pt` | ❌ 否 | COCO 预训练权重（训练起点），首次训练时自动下载。 |
| `wheels/*.whl` | ❌ 否 | 离线下载的 PyTorch CUDA wheel（约 2.8 GB），仅用于无网/受限环境安装。 |

## 重新生成被忽略的权重

```bash
python main.py prepare     # 下载公开数据集并构建训练集
python main.py train       # 训练（RTX 4050 约 1.8 小时，70 epochs）
```

训练产物会在 `runs/<name>/weights/`，最佳权重会自动复制为 `models/wood_block_yolo11s.pt`。
