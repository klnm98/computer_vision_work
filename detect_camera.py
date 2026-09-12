"""木块识别（摄像头实时 / 图片 / 视频）。

识别方案
--------
使用在**公开真实数据集 YCB-Video** 上微调得到的 YOLO 木块检测模型
（见 ``woodblock/``），而不是颜色/轮廓规则：

* 单类别 "wood_block"，网络只会输出木块，不会把桌面、手、其他积木框进来；
* 置信度阈值在真实负样本上标定，进一步压制误检；
* 摄像头模式下加入 N-of-M 时序确认，偶发的单帧抖动不会画出框；
* 木块缺角时，模型依据木质纹理与其余轮廓仍能给出完整框
  （训练时已加入遮挡/缺角增强）。

用法
----
    python detect_camera.py                     # 默认摄像头
    python detect_camera.py --source 1           # 指定摄像头序号
    python detect_camera.py --source block.jpg   # 单张图片
    python detect_camera.py --source video.mp4   # 视频文件
    python detect_camera.py --conf 0.45          # 调整置信度阈值
按 'q' 退出，按 's' 保存当前帧截图。
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from woodblock import paths  # noqa: E402
from woodblock.detector import TemporalFilter, WoodBlockDetector  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="camera_detection.log",
    filemode="a",
)
log = logging.getLogger("woodblock.camera")

WINDOW = "Block Recognition"


def draw_results(frame, detections, fps: float | None = None, note: str = ""):
    """绘制木块框与提示信息。"""
    for i, d in enumerate(detections, 1):
        x1, y1, x2, y2 = d.as_int_box()
        color = (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label = f"wood_block {d.conf:.2f}" if len(detections) == 1 else f"wood_block#{i} {d.conf:.2f}"
        ty = y1 - 12 if y1 - 12 > 20 else y2 + 26
        cv2.putText(frame, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    head = f"blocks: {len(detections)}"
    if fps is not None:
        head += f"   fps: {fps:.1f}"
    if note:
        head += f"   {note}"
    cv2.rectangle(frame, (0, 0), (int(11 * len(head)), 30), (0, 0, 0), -1)
    cv2.putText(frame, head, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame


BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "any": cv2.CAP_ANY,
}
if os.name != "nt":  # Linux/macOS 只有默认后端
    BACKENDS = {"any": cv2.CAP_ANY}


def frame_stats(frame) -> tuple[float, float]:
    """返回 (平均亮度, 标准差)。标准差≈0 说明是纯色帧。"""
    if frame is None:
        return 0.0, 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(gray.mean()), float(gray.std())


def is_blank(frame, min_std: float = 2.0, min_mean: float = 6.0) -> bool:
    """判断是否"黑屏/纯色"帧（摄像头预热、后端不兼容时常见）。"""
    mean, std = frame_stats(frame)
    return std < min_std and mean < min_mean


# 打开摄像头时依次尝试的分辨率（从高到低，最后一个是"不设置"）
SIZE_LADDER: list[tuple[int, int] | None] = [(1920, 1080), (1280, 720), (640, 480), None]
BACKEND_PREF = {"dshow": 0.30, "msmf": 0.25, "any": 0.10}  # 经验上 DShow 对网络摄像头最稳
CACHE_PATH = os.path.join(paths.ROOT, ".camera_config.json")


def _probe_combo(idx: int, backend_id: int, size, frames: int = 10):
    """试一个 (设备号, 后端, 分辨率) 组合，返回统计信息或 None。

    判定标准：能打开、能连续读出帧、且其中**大部分帧不是黑屏/纯色**。
    """
    cap = cv2.VideoCapture(idx, backend_id)
    if not cap.isOpened():
        cap.release()
        return None
    try:
        if size:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        means, stds, blanks, reads = [], [], 0, 0
        for _ in range(frames):
            ok, f = cap.read()
            if not ok:
                break
            reads += 1
            m, s = frame_stats(f)
            means.append(m)
            stds.append(s)
            if is_blank(f):
                blanks += 1
    finally:
        cap.release()
    if reads == 0:
        return None
    valid = reads - blanks
    # 亮度波动小 = 画面稳定（排除"偶尔闪一帧"的坏组合）
    jitter = float(np.std(means)) if len(means) > 1 else 0.0
    return {"index": idx, "backend_id": backend_id,
            "backend": next((n for n, v in BACKENDS.items() if v == backend_id), str(backend_id)),
            "resolution": res, "reads": reads, "valid_frames": valid, "blank_frames": blanks,
            "mean": round(float(np.mean(means)), 1), "jitter": round(jitter, 1),
            "asked_size": size}


def score_candidate(c: dict, requested_index: int) -> float:
    """给候选组合打分：分辨率越高越好、有效帧越多越好、画面越稳越好。"""
    px = max(1, c["resolution"][0] * c["resolution"][1])
    valid_ratio = c["valid_frames"] / max(1, c["reads"])
    res_score = math.log2(px / (640 * 480)) * 1.0           # 640x480 记 0 分，720p≈+0.83，1080p≈+1.75
    return (res_score
            + 1.2 * valid_ratio                              # 必须有画面
            + BACKEND_PREF.get(c["backend"], 0.0)
            + (0.15 if c["index"] == requested_index else 0.0)  # 轻微偏向用户指定的设备
            - 0.02 * min(c["jitter"], 50.0))                 # 抖动扣分


def scan_cameras(max_index: int = 5, probe_frames: int = 10, verbose: bool = True) -> list[dict]:
    """完整自检：枚举 设备号 × 后端 × 分辨率，返回全部可用候选（已按分数排序）。"""
    cands: list[dict] = []
    rejected: list[str] = []
    for idx in range(max_index):
        for name, be in BACKENDS.items():
            opened = False
            for size in SIZE_LADDER:
                c = _probe_combo(idx, be, size, frames=probe_frames)
                if c is None:
                    continue
                opened = True
                if c["valid_frames"] == 0:
                    rejected.append(f"设备 {idx} / {name} / {c['resolution'][0]}x{c['resolution'][1]}"
                                    f"：能打开但全黑（亮度 {c['mean']}）")
                    continue
                if size is not None and c["resolution"] != tuple(size):
                    # 驱动没有采用请求的分辨率：只保留"不设置分辨率"的那一份，避免重复
                    continue
                cands.append(c)
            if opened is False and verbose:
                pass  # 打不开的设备/后端不刷屏
    # 同一 (设备, 后端) 只保留最优的一条
    best_by_combo: dict[tuple[int, str], dict] = {}
    for c in cands:
        key = (c["index"], c["backend"])
        if key not in best_by_combo or c["resolution"] > best_by_combo[key]["resolution"]:
            best_by_combo[key] = c
    result = sorted(best_by_combo.values(),
                    key=lambda c: -score_candidate(c, requested_index=-1))
    if verbose:
        if result:
            print("[摄像头自检] 找到以下可用组合（按推荐度排序）：")
            for c in result:
                print(f"   设备 {c['index']} / {c['backend']:5s} / "
                      f"{c['resolution'][0]}x{c['resolution'][1]:<5d} "
                      f"有效帧 {c['valid_frames']}/{c['reads']}  亮度 {c['mean']:5.1f}  "
                      f"抖动 {c['jitter']:4.1f}  评分 {score_candidate(c, -1):.2f}")
        if rejected:
            print("[摄像头自检] 以下组合能打开但画面全黑/纯色（通常是虚拟摄像头）：")
            for r in rejected[:6]:
                print(f"   {r}")
    return result


def probe_cameras(max_index: int = 5, warmup: int = 10, verbose: bool = True) -> list[dict]:
    """评测/排查入口：打印自检结果并给出建议命令。"""
    cands = scan_cameras(max_index=max_index, probe_frames=max(6, warmup), verbose=verbose)
    if verbose:
        if cands:
            best = cands[0]
            print(f"\n建议使用: --source {best['index']} --backend {best['backend']}"
                  f"（{best['resolution'][0]}x{best['resolution'][1]}）")
            print("直接运行 `python main.py` 时也会自动选到这一组。")
        else:
            print("\n没有找到可用摄像头：请确认摄像头未被其他程序占用，"
                  "且已在『设置 > 隐私和安全性 > 相机』中允许桌面应用访问。")
    return cands


def _load_cached_camera() -> dict | None:
    try:
        with open(CACHE_PATH, encoding="utf8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "index" in data and "backend" in data:
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _save_cached_camera(index: int, backend: str, resolution) -> None:
    try:
        with open(CACHE_PATH, "w", encoding="utf8") as f:
            json.dump({"index": index, "backend": backend,
                       "resolution": list(resolution), "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                      f, ensure_ascii=False, indent=1)
    except OSError:
        pass


def open_camera(requested: int = 0, backend: str = "auto", warmup: int = 20,
                rescan: bool = False, auto_scan: bool = True, verbose: bool = True):
    """自动排查并选择摄像头，返回 (cap, 说明信息)。

    流程：先用上次成功的组合快速试一次（若仍有效就直接用）→ 否则完整自检、
    按"分辨率/有效帧/稳定性"打分选最优 → 记住结果供下次复用。
    """
    if backend == "auto":
        order = list(BACKENDS.items())
    else:
        if backend not in BACKENDS:
            raise SystemExit(f"--backend 只能是 {list(BACKENDS)} 或 auto")
        order = [(backend, BACKENDS[backend])]

    def try_combo(idx: int, name: str, warmup_frames: int, size=None):
        """按给定组合打开并预热，返回 (cap, info) 或 None。"""
        c = _probe_combo(idx, BACKENDS[name], size, frames=max(warmup_frames, 10))
        if c is None or c["valid_frames"] == 0:
            return None
        cap = cv2.VideoCapture(idx, BACKENDS[name])
        if not cap.isOpened():
            cap.release()
            return None
        if size:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        for _ in range(warmup_frames):  # 真正使用前把预热帧丢干净
            ok, _f = cap.read()
            if not ok:
                break
        res = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        return cap, f"设备 {idx} / 后端 {name} / {res}"

    # ① 快速路径：复用上次成功且仍然有效的组合
    cached = None if rescan else _load_cached_camera()
    if cached:
        got = try_combo(cached["index"], cached["backend"], warmup, cached.get("resolution"))
        if got is not None:
            if verbose:
                print(f"[摄像头] 复用上次可用的组合：{got[1]}")
            return got
        if verbose:
            print(f"[摄像头] 上次的组合（设备 {cached['index']} / {cached['backend']}）已失效，重新自检 ...")

    # ② 用户显式指定了后端：先只试该后端（含请求的设备号与其它设备号）
    indices = [requested] + ([i for i in range(6) if i != requested] if auto_scan else [])
    if backend != "auto":
        for idx in indices:
            for size in SIZE_LADDER:
                got = try_combo(idx, backend, warmup, size)
                if got is not None:
                    if idx != requested and verbose:
                        print(f"[摄像头] 指定的设备 {requested} 无有效画面，自动改用设备 {idx}。")
                    _save_cached_camera(idx, backend, size or (0, 0))
                    return got

    # ③ 自动自检 + 打分选优
    if verbose:
        print("[摄像头] 正在自动排查可用的摄像头（设备号 × 后端 × 分辨率）...")
    cands = scan_cameras(max_index=max(requested + 1, 6), verbose=verbose)
    if cands:
        ranked = sorted(cands, key=lambda c: -score_candidate(c, requested))
        best = ranked[0]
        got = try_combo(best["index"], best["backend"], warmup, best["asked_size"])
        if got is not None:
            if verbose:
                print(f"[摄像头] 自动选择：{got[1]}（评分 {score_candidate(best, requested):.2f}）")
                if best["index"] != requested:
                    print(f"[摄像头] 注意：你请求的是设备 {requested}，但它拿不到有效画面（"
                          f"很可能是虚拟摄像头），已自动改用设备 {best['index']}。")
            _save_cached_camera(best["index"], best["backend"], best["resolution"])
            return got

    # ④ 全部失败：给出可执行的诊断信息
    raise RuntimeError(
        "没有找到可用的摄像头（已排查 设备号 0-5 × dshow/msmf/any × 多种分辨率）。\n"
        "请依次检查：\n"
        "  1) 摄像头是否被相机应用 / 会议软件独占；\n"
        "  2) 『设置 > 隐私和安全性 > 相机』是否允许桌面应用访问；\n"
        "  3) 运行 `python detect_camera.py --probe` 查看每个设备的详细结果。\n"
        "也可以先用 `python detect_camera.py --source 视频或图片路径` 验证识别功能。")


def open_source(source: str, backend: str = "auto", warmup: int = 20,
                prefer_size: tuple[int, int] | None = None,
                auto_scan: bool = True, rescan: bool = False, verbose: bool = True):
    """打开摄像头或视频/图片文件，返回 (capture, 是否为摄像头, 说明信息)。"""
    if not source.isdigit():
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源 {source}")
        return cap, False, f"文件 {source}"
    cap, info = open_camera(int(source), backend=backend, warmup=warmup,
                            rescan=rescan, auto_scan=auto_scan, verbose=verbose)
    return cap, True, info


def parse_scales(text: str) -> tuple[float, ...]:
    """解析 ``--scales 1.0,0.7,0.45`` 形式的参数。"""
    vals = tuple(float(v) for v in text.replace(" ", "").split(",") if v)
    if not vals or any(not (0.05 < v <= 1.0) for v in vals):
        raise SystemExit("--scales 需为 0~1 之间的比例，例如 1.0,0.7,0.45")
    return vals


def run(source: str = "0", weights: str | None = None, conf: float = paths.CONF_THRESHOLD,
        iou: float = paths.IOU_THRESHOLD, imgsz: int = paths.IMG_SIZE, device: str | None = None,
        temporal: bool = True, min_hits: int = 3, show: bool = True,
        max_frames: int | None = None, save_video: str | None = None,
        snapshot_dir: str = "snapshots", tta: bool = False,
        scales: tuple[float, ...] | None = None, backend: str = "auto",
        warmup: int = 20, rescan: bool = False, auto_recover: bool = True) -> int:
    """执行识别主循环。返回处理帧数。"""
    detector = WoodBlockDetector(weights=weights, conf=conf, iou=iou, imgsz=imgsz,
                                 device=device, tta=tta, scales=scales)
    smoother = TemporalFilter(min_hits=min_hits) if temporal else None

    is_image = source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    if is_image:
        return _run_image(source, detector, show, snapshot_dir)

    cap, is_camera, info = open_source(source, backend=backend, warmup=warmup,
                                       rescan=rescan)
    print(f"摄像头/视频源已就绪（{'摄像头' if is_camera else '视频文件'}：{info}）。"
          f"把木块放在镜头前。按 'q' 退出，按 's' 保存截图。")
    log.info("开始识别: source=%s info=%s weights=%s conf=%.2f", source, info, detector.weights, conf)

    writer = None
    frames = 0
    fps = 0.0
    t_prev = time.time()
    det_frames = 0
    blank_streak = 0
    warned_blank = False
    recoveries = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # 摄像头掉线/被抢占：自动重新排查并接管，而不是直接退出
                if is_camera and auto_recover and recoveries < 3:
                    recoveries += 1
                    print(f"\n[摄像头] 读不到画面（第 {recoveries} 次），正在自动重新选择设备 ...")
                    log.warning("摄像头读取失败，尝试自动恢复（第 %d 次）", recoveries)
                    try:
                        cap.release()
                    except Exception:  # noqa: BLE001
                        pass
                    new_cap, _is_cam, new_info = open_source(source, backend=backend,
                                                             warmup=warmup, rescan=True)
                    cap = new_cap
                    print(f"[摄像头] 已恢复：{new_info}")
                    blank_streak = 0
                    continue
                if is_camera:
                    raise RuntimeError("无法读取画面")
                break  # 视频读完
            frames += 1

            # 画面全黑时给出明确提示（而不是让用户对着黑窗口猜）
            if is_camera and is_blank(frame):
                blank_streak += 1
                if blank_streak == 5:
                    # 存一张原始黑帧：用于区分"摄像头没出图"和"窗口没刷新"
                    os.makedirs(snapshot_dir, exist_ok=True)
                    p = os.path.join(snapshot_dir, f"black_frame_{int(time.time())}.jpg")
                    cv2.imwrite(p, frame)
                    print(f"检测到全黑画面，已保存原始帧用于排查: {p}")
                # 持续全黑（约 3 秒）也自动重新排查一次
                if blank_streak >= 90 and auto_recover and recoveries < 3:
                    recoveries += 1
                    print(f"\n[摄像头] 连续 {blank_streak} 帧全黑，正在自动重新选择设备 ...")
                    log.warning("持续黑屏，尝试自动恢复（第 %d 次）", recoveries)
                    try:
                        cap.release()
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        cap, _is_cam, new_info = open_source(source, backend=backend,
                                                             warmup=warmup, rescan=True)
                        print(f"[摄像头] 已恢复：{new_info}")
                    except RuntimeError as e:  # 恢复失败就保留提示，继续跑
                        print(f"[摄像头] 自动恢复失败：{e}")
                    blank_streak = 0
                    continue
                if blank_streak >= 30 and not warned_blank:
                    warned_blank = True
                    msg = ("画面全黑：可能打开了虚拟摄像头、后端不兼容，或设置分辨率后摄像头不出图。"
                           "程序会自动重新排查；也可运行 `python detect_camera.py --probe` 查看详情。")
                    print("\n" + msg)
                    log.warning(msg)
            else:
                blank_streak = 0

            dets = detector.detect(frame)
            shown = smoother.update(dets) if smoother else dets
            if shown:
                det_frames += 1

            t_now = time.time()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            note = "" if shown else "searching..."
            if warned_blank and blank_streak:
                note = "BLACK FRAME: try --probe / --backend msmf"
            vis = draw_results(frame, shown, fps=fps, note=note)
            if frames % 30 == 0:
                log.info("处理 %d 帧，检测到木块 %d 次，最近置信度 %.3f",
                         frames, len(shown), shown[0].conf if shown else 0.0)

            if save_video:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
                writer.write(vis)
            if show:
                cv2.imshow(WINDOW, vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    os.makedirs(snapshot_dir, exist_ok=True)
                    p = os.path.join(snapshot_dir, f"snapshot_{int(time.time())}.jpg")
                    cv2.imwrite(p, vis)
                    print(f"已保存截图: {p}")
            if max_frames and frames >= max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    print(f"程序已退出。共处理 {frames} 帧，其中 {det_frames} 帧检测到木块。")
    log.info("结束: 处理 %d 帧, 检测到 %d 帧", frames, det_frames)
    return frames


def _run_image(path: str, detector: WoodBlockDetector, show: bool, out_dir: str) -> int:
    """单张图片识别。"""
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"无法读取图片 {path}")
    t0 = time.time()
    dets = detector.detect(img)
    print(f"图片 {path}: 检测到 {len(dets)} 个木块（耗时 {(time.time()-t0)*1000:.0f} ms）")
    for d in dets:
        print(f"   置信度 {d.conf:.3f}  框 {d.as_int_box()}")
    vis = draw_results(img, dets)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + "_detected.jpg")
    cv2.imwrite(out, vis)
    print(f"结果已保存: {out}")
    log.info("图片 %s 检测到 %d 个木块, 结果=%s", path, len(dets), out)
    if show:
        cv2.imshow(WINDOW, vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="木块识别（训练好的 YOLO 模型）")
    ap.add_argument("--source", default="0", help="摄像头序号(0/1...)、视频文件或图片路径")
    ap.add_argument("--weights", default=None, help="模型权重路径")
    ap.add_argument("--conf", type=float, default=paths.CONF_THRESHOLD, help="置信度阈值")
    ap.add_argument("--iou", type=float, default=paths.IOU_THRESHOLD, help="NMS IoU 阈值")
    ap.add_argument("--imgsz", type=int, default=paths.IMG_SIZE, help="推理分辨率")
    ap.add_argument("--device", default=None, help="cpu 或 GPU 序号(0)")
    ap.add_argument("--no-temporal", action="store_true", help="关闭 N-of-M 时序确认")
    ap.add_argument("--min-hits", type=int, default=3, help="时序确认所需连续命中帧数")
    ap.add_argument("--tta", action="store_true", help="开启测试时增强（更稳但更慢）")
    ap.add_argument("--scales", default=None,
                    help="多尺度重采样推理比例，如 1.0,0.7,0.45。默认单尺度；"
                         "打开后可命中'木块占满画面'的近景，但误检率会升高")
    ap.add_argument("--no-show", action="store_true", help="不弹窗（无显示器环境）")
    ap.add_argument("--max-frames", type=int, default=None, help="最多处理多少帧（测试用）")
    ap.add_argument("--save-video", default=None, help="把带框结果写入视频文件")
    ap.add_argument("--snapshot-dir", default="snapshots", help="截图保存目录")
    ap.add_argument("--backend", default="auto", choices=["auto", *BACKENDS],
                    help="摄像头后端；auto 表示自动排查选择（默认）")
    ap.add_argument("--warmup", type=int, default=20,
                    help="打开摄像头后丢弃多少帧预热画面（默认 20，避开启动黑帧）")
    ap.add_argument("--rescan", action="store_true",
                    help="忽略上次记住的摄像头组合，重新完整自检")
    ap.add_argument("--no-recover", action="store_true",
                    help="关闭运行中的摄像头掉线/黑屏自动恢复")
    ap.add_argument("--max-index", type=int, default=6, help="自检时最多枚举到第几号设备")
    ap.add_argument("--probe", action="store_true",
                    help="只做摄像头自检：列出所有可用组合并给出建议，不启动识别")
    args = ap.parse_args(argv)

    if args.probe:
        print("摄像头自检中（会依次尝试各设备 × 后端 × 分辨率，可能需要十几秒）...")
        probe_cameras(max_index=args.max_index)
        return 0

    try:
        run(source=args.source, weights=args.weights, conf=args.conf, iou=args.iou,
            imgsz=args.imgsz, device=args.device, temporal=not args.no_temporal,
            min_hits=args.min_hits, show=not args.no_show, max_frames=args.max_frames,
            save_video=args.save_video, snapshot_dir=args.snapshot_dir, tta=args.tta,
            scales=parse_scales(args.scales) if args.scales else None,
            backend=args.backend, warmup=args.warmup, rescan=args.rescan,
            auto_recover=not args.no_recover)
        return 0
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        log.error("错误信息: %s", e, exc_info=True)
        print(f"运行失败: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
