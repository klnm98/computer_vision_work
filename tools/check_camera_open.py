"""验证 open_source 拿到的是真实画面（非黑屏），并打印选择结果。"""
from __future__ import annotations

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import detect_camera as dc  # noqa: E402

cap, is_camera, info = dc.open_source("0")
print("选择结果:", info, "是摄像头:", is_camera)
for i in range(5):
    ok, frame = cap.read()
    if not ok:
        print(f"  第 {i} 帧读取失败")
        break
    mean, std = dc.frame_stats(frame)
    print(f"  第 {i} 帧: {frame.shape[1]}x{frame.shape[0]} 亮度={mean:6.1f} 标准差={std:6.1f} "
          f"-> {'黑屏' if dc.is_blank(frame) else '正常'}")
    if i == 0:
        os.makedirs("reports", exist_ok=True)
        cv2.imwrite("reports/camera_frame_check.jpg", frame)
cap.release()
print("已保存首帧到 reports/camera_frame_check.jpg")
