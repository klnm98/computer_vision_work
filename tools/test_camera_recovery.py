"""用桩对象验证"运行中掉线/持续黑屏 -> 自动重新选择摄像头"的逻辑不会卡死。

不依赖真实硬件：把 open_source 换成会先给若干帧全黑、再给正常画面的假实现。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import detect_camera as dc  # noqa: E402

CALLS = {"n": 0}


class FakeCap:
    """先给 N 帧全黑，之后给正常画面；模拟"选到了坏组合 -> 自动恢复"。"""

    def __init__(self, black_frames: int, recover_after: bool):
        self.black_frames = black_frames
        self.i = 0
        self.recover_after = recover_after
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        self.i += 1
        if self.black_frames > 0:
            self.black_frames -= 1
            return True, np.zeros((480, 640, 3), np.uint8)
        if not self.recover_after:
            return True, np.zeros((480, 640, 3), np.uint8)
        rng = np.random.default_rng(self.i)
        return True, rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)

    def release(self):
        self.released = True

    def get(self, prop):
        return 640.0 if prop == dc.cv2.CAP_PROP_FRAME_WIDTH else 480.0

    def set(self, prop, value):
        return True


def fake_open_source(source, backend="auto", warmup=20, **kw):
    CALLS["n"] += 1
    print(f"    -> 第 {CALLS['n']} 次打开摄像头（模拟：首个组合给 95 帧黑屏后恢复）")
    if CALLS["n"] == 1:
        return FakeCap(black_frames=95, recover_after=True), True, "假设备 0 / 后端 dshow / 640x480"
    return FakeCap(black_frames=0, recover_after=True), True, "假设备 0 / 后端 msmf / 640x480"


dc.open_source = fake_open_source
det = None


class NoOpDetector:
    def __init__(self, **kw):
        self.weights = "fake"

    def detect(self, frame):
        return []


dc.WoodBlockDetector = NoOpDetector
dc.is_blank = dc.is_blank  # 保持真实判定（黑帧会被识别出来）

print("开始模拟运行（最多 200 帧）...")
n = dc.run(source="0", show=False, max_frames=200)
print(f"处理帧数={n}，打开摄像头次数={CALLS['n']}")
assert CALLS["n"] >= 2, "没有触发自动恢复逻辑"
print("通过：持续黑屏后自动重新选择摄像头，且未卡死。")
