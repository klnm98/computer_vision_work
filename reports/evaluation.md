# 木块识别模型评估报告

- 生成时间: 2026-09-13 00:02:06
- 权重: `D:\computer_vision_work\models\wood_block_yolo11s.pt`
- 置信度阈值: 0.35

## 1. 验证集指标（按场景切分）

| 指标 | 数值 |
| --- | --- |
| mAP@50 | 0.8284 |
| mAP@50-95 | 0.4416 |
| 精确率 P | 0.8951 |
| 召回率 R | 0.7108 |

## 2. 负样本误检（BOP 官方测试集中不含木块的图像）

| 指标 | 数值 |
| --- | --- |
| 图像数 | 825 |
| 误检图像数 | 7 |
| 误检率 | 0.85% |
| 误检框总数 | 7 |
| 其中框到**其他物体** | 5 |
| 其中框到背景/桌面 | 2 |
| 最高置信度 | 0.587 |
| p99 置信度 | 0.339 |

被误框的物体统计（IoU>=0.3 判定）：

```
{"011_banana": 4, "035_power_drill": 1}
```

## 直接可用的阈值取舍表（在 BOP 官方测试集上实测）

| 置信度阈值 | 误检图像率 | 误检框数 | 其中框到其他物体 | 木块召回率 | 平均 IoU |
| --- | --- | --- | --- | --- | --- |
| 0.10 | 5.58% | 50 | 28 | 84.0% | 0.740 |
| 0.15 | 3.52% | 30 | 17 | 80.0% | 0.741 |
| 0.20 | 2.67% | 23 | 12 | 78.7% | 0.741 |
| 0.25 | 1.94% | 17 | 8 | 78.7% | 0.741 |
| 0.30 | 1.45% | 12 | 7 | 77.3% | 0.741 |
| 0.35 | 0.85% | 7 | 5 | 73.3% | 0.741 |
| 0.40 | 0.73% | 6 | 5 | 66.7% | 0.744 |
| 0.45 | 0.48% | 4 | 3 | 60.0% | 0.741 |
| 0.50 | 0.48% | 4 | 3 | 56.0% | 0.742 |
| 0.60 | 0.00% | 0 | 0 | 41.3% | 0.743 |
| 0.70 | 0.00% | 0 | 0 | 17.3% | 0.741 |
| 0.80 | 0.00% | 0 | 0 | 0.0% | 0.000 |

## 3. 木块召回（BOP 官方测试集，与真值框 IoU>=0.5 判定）

| 指标 | 数值 |
| --- | --- |
| 含木块图像数 | 75 |
| 真值框数 | 75 |
| 召回率 | 73.3% |
| 平均 IoU | 0.741 |
| 多余检测框 | 0 |

## 4. 阈值标定

- 建议阈值: **0.43**（目标误检率 0.5%）
- 负样本 p99 置信度: 0.339，最大值: 0.587

## 5. 附加验证

```json
{
 "block_jpg": {
  "image": "D:\\computer_vision_work\\block.jpg",
  "input_shape": [
   2376,
   1080
  ],
  "used_shape": [
   1280,
   625
  ],
  "cropped": true,
  "detections": [
   {
    "box": [
     0.0,
     9.2,
     625.0,
     847.7
    ],
    "conf": 0.833
   }
  ],
  "saved": "D:\\computer_vision_work\\reports\\block_detected.jpg"
 },
 "block_jpg_chipped": {
  "image": "D:\\computer_vision_work\\block.jpg",
  "detected_before": 1,
  "detected_after": 1,
  "before_conf": 0.833,
  "after_conf": 0.8304,
  "passed": true,
  "saved": "D:\\computer_vision_work\\reports\\block_chipped_detected.jpg"
 },
 "block_jpg_occluded": {
  "image": "D:\\computer_vision_work\\block.jpg",
  "rounds": 5,
  "passed_rounds": 5,
  "pass_rate": 1.0,
  "confs": [
   0.8222,
   0.8047,
   0.8099,
   0.7671,
   0.404
  ],
  "passed": true,
  "saved": "D:\\computer_vision_work\\reports\\block_occluded_detected.jpg"
 }
}
```
