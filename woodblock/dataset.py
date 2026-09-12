"""数据集构建：从公开真实数据集 YCB-Video / BOP YCB-V 生成 YOLO 训练集。

流程
----
1. ``scan_scenes``   —— 远程扫描 92 个场景，找出含有 036_wood_block 的场景；
2. ``fetch_scene``   —— 只按需拉取标注与抽帧后的彩色图（HTTP Range，不下载整包）；
3. ``build_dataset`` —— 转为 YOLO 目录结构，并按“场景”切分训练/验证集。

注意
----
* 标注文件 ``*-box.txt`` 每行格式为 ``<物体名称> <x1> <y1> <x2> <y2>``；
* 木块对应 ``036_wood_block``，在所有场景中位置一致（obj_id 16）；
* 负样本（不含木块的场景）会写入空标签文件，用于压制误检。
"""
from __future__ import annotations

import concurrent.futures as futures
import json
import os
import random
import shutil
import struct
import zlib

from . import paths, remote_zip, sources

WOOD_NAME = sources.obj_name(paths.WOOD_BLOCK_OBJ_ID)  # 036_wood_block


# --------------------------------------------------------------------- 工具
def _read_tiny(url: str, entry: dict, span: int = 700) -> bytes | None:
    """一次 Range 请求读取小文件（本地头 + 数据通常都在 span 字节内）。"""
    raw = remote_zip.fetch_range(url, entry["lho"], entry["lho"] + span - 1)
    if len(raw) < 30 or raw[:4] != b"PK\x03\x04":
        return None
    nlen, elen = struct.unpack("<HH", raw[26:30])
    start = 30 + nlen + elen
    if entry["method"] == 0:
        data = raw[start:start + entry["comp"]]
    else:
        try:
            data = zlib.decompressobj(-15).decompress(raw[start:])
        except zlib.error:
            return None
    return data if len(data) >= entry["usize"] else None


def _read_entry_fast(url: str, entry: dict) -> bytes:
    """优先单请求读取，失败则回退到精确读取。"""
    data = _read_tiny(url, entry)
    if data is not None:
        return data
    return remote_zip.read_entry(url, entry)


def parse_box_file(text: str) -> dict[str, tuple[float, float, float, float]]:
    """解析 ``*-box.txt`` -> {物体名: (x1, y1, x2, y2)}。"""
    out: dict[str, tuple[float, float, float, float]] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        name = parts[0]
        try:
            x1, y1, x2, y2 = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        out[name] = (x1, y1, x2, y2)
    return out


def _scene_url(scene: str) -> str:
    return sources.YCBB_VIDEO_SCENE_URL.format(scene=scene)


# --------------------------------------------------------------------- 扫描
def scan_scenes(scene_ids: list[str] | None = None, workers: int = 10,
                force: bool = False, verbose: bool = True) -> dict:
    """扫描场景，判断是否含有木块。结果缓存到 ``data/raw/scene_scan.json``。"""
    cache = os.path.join(paths.RAW_DIR, "scene_scan.json")
    scene_ids = scene_ids or sources.TRAIN_SCENE_IDS
    report = {}
    if os.path.exists(cache):
        try:
            report = json.load(open(cache, encoding="utf8"))
        except (json.JSONDecodeError, OSError):
            report = {}
    if not force:
        missing = [s for s in scene_ids if s not in report]
        if not missing:
            return report
        scene_ids = missing

    def probe(scene: str):
        try:
            url = _scene_url(scene)
            entries = remote_zip.central_directory(url, cache_key=scene)
            boxes = remote_zip.find_entries(entries, suffix="-box.txt")
            if not boxes:
                return scene, None
            boxes.sort(key=lambda e: e["name"])
            picks = [boxes[0], boxes[len(boxes) // 2]]
            objs: set[str] = set()
            for e in picks:
                objs.update(parse_box_file(_read_entry_fast(url, e).decode("utf8", "ignore")))
            return scene, {"frames": len(boxes), "objects": sorted(objs),
                           "has_wood": WOOD_NAME in objs}
        except Exception as exc:  # noqa: BLE001  场景缺失/网络异常都不应中断整体扫描
            return scene, {"error": f"{type(exc).__name__}: {exc}", "frames": 0,
                           "objects": [], "has_wood": False}

    done = 0
    with futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for scene, info in ex.map(probe, scene_ids):
            done += 1
            if info is None:
                report[scene] = {"error": "无标注", "frames": 0, "objects": [], "has_wood": False}
                if verbose:
                    print(f"  场景 {scene}: 无标注，跳过", flush=True)
                continue
            report[scene] = info
            if verbose:
                if "error" in info:
                    print(f"  场景 {scene}: 跳过（{info['error'][:70]}）", flush=True)
                else:
                    mark = "  <== 含木块" if info["has_wood"] else ""
                    print(f"  场景 {scene}: {info['frames']:5d} 帧, 物体数={len(info['objects']):2d}{mark}",
                          flush=True)
            if done % 10 == 0:  # 增量落盘，避免中途失败丢失进度
                with open(cache, "w", encoding="utf8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=1)

    with open(cache, "w", encoding="utf8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    return report


def wood_scenes(report: dict) -> list[str]:
    """含木块的场景编号（按帧数从多到少）。"""
    items = [(s, r) for s, r in report.items() if r.get("has_wood")]
    items.sort(key=lambda kv: -kv[1]["frames"])
    return [s for s, _ in items]


def negative_scenes(report: dict) -> list[str]:
    return sorted(s for s, r in report.items() if not r.get("has_wood"))


# --------------------------------------------------------------------- 拉取
def fetch_scene(scene: str, stride: int = paths.FRAME_STRIDE, workers: int = 12,
                with_wood: bool = True, max_frames: int | None = None,
                verbose: bool = True) -> dict:
    """拉取单个场景的标注 + 抽帧彩色图，保存到 ``data/raw/ycbv_scenes/<scene>``。

    返回 ``{"scene", "frames": [(frame, box or None)], "images": n}``。
    """
    url = _scene_url(scene)
    out_dir = os.path.join(paths.RAW_SCENES_DIR, scene)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    entries = remote_zip.central_directory(url, cache_key=scene)
    box_entries = {os.path.basename(e["name"])[:-8]: e
                   for e in remote_zip.find_entries(entries, suffix="-box.txt")}
    color_entries = {os.path.basename(e["name"])[:-10]: e
                     for e in remote_zip.find_entries(entries, suffix="-color.png")}
    frames = sorted(set(box_entries) & set(color_entries))
    frames = frames[::stride]
    if max_frames:
        frames = frames[:max_frames]

    annotations: dict[str, dict] = {}
    lock_errors: list[str] = []

    def read_box(frame: str):
        try:
            text = _read_entry_fast(url, box_entries[frame]).decode("utf8", "ignore")
            return frame, parse_box_file(text)
        except Exception as e:  # noqa: BLE001
            lock_errors.append(f"{frame}: {e}")
            return frame, {}

    with futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for frame, boxes in ex.map(read_box, frames):
            annotations[frame] = boxes

    todo = []
    for frame, boxes in annotations.items():
        if with_wood and WOOD_NAME not in boxes:
            continue
        dst = os.path.join(img_dir, f"{frame}.png")
        if not os.path.exists(dst):
            todo.append((frame, boxes, dst))

    def grab(item):
        frame, boxes, dst = item
        try:
            data = _read_entry_fast(url, color_entries[frame])
            tmp = dst + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dst)
            return frame, boxes
        except Exception:  # noqa: BLE001
            return frame, None

    kept = []
    if todo:
        with futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for frame, boxes in ex.map(grab, todo):
                if boxes is not None:
                    kept.append((frame, boxes))

    # 已存在的图片也要计入
    for frame, boxes in annotations.items():
        if with_wood and WOOD_NAME not in boxes:
            continue
        if os.path.exists(os.path.join(img_dir, f"{frame}.png")) and not any(k[0] == frame for k in kept):
            kept.append((frame, boxes))

    kept.sort()
    meta = {"scene": scene, "stride": stride, "wood_name": WOOD_NAME,
            "annotated_frames": len(annotations),
            "frames": {f: b for f, b in kept}}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf8") as f:
        json.dump(meta, f, ensure_ascii=False)

    if verbose:
        print(f"  场景 {scene}: 抽帧 {len(frames)}, 含木块且有图 {len(kept)}"
              + (f", 读取失败 {len(lock_errors)}" if lock_errors else ""))
    return meta


# --------------------------------------------------------------------- 构建
def _clamp_box(box: tuple[float, float, float, float], w: int = 640, h: int = 480):
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0.0, min(x1, w)), max(0.0, min(x2, w))))
    y1, y2 = sorted((max(0.0, min(y1, h)), max(0.0, min(y2, h))))
    return x1, y1, x2, y2


def _yolo_line(box: tuple[float, float, float, float], w: int, h: int) -> str | None:
    x1, y1, x2, y2 = _clamp_box(box, w, h)
    bw, bh = x2 - x1, y2 - y1
    if bw < paths.MIN_BOX_PIXELS or bh < paths.MIN_BOX_PIXELS:
        return None
    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
    return f"0 {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}"


# 每个角对应的三角形切口顶点（相对切口矩形）
_CHIP_TRIANGLES = {
    "tl": [(0, 0), (1, 0), (0, 1)],
    "tr": [(1, 0), (1, 1), (0, 0)],
    "bl": [(0, 1), (1, 1), (0, 0)],
    "br": [(1, 1), (0, 1), (1, 0)],
}


def make_chipped(img, box: tuple[float, float, float, float], rng: random.Random,
                 ratio_range: tuple[float, float] = (0.18, 0.35)):
    """缺角增强：把木块框的一个角"敲掉"，用旁边的背景纹理填补。

    用来模拟真实木块缺角（或手/物体挡住一角）的情况，避免模型只依赖完整的
    方形轮廓。标签保持不变 —— 框仍然覆盖木块的整体范围。
    """
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in _clamp_box(box, w, h))
    bw, bh = x2 - x1, y2 - y1
    if bw < 28 or bh < 28:
        return None

    ratio = rng.uniform(*ratio_range)
    cw, ch = max(6, int(bw * ratio)), max(6, int(bh * ratio))
    corner = rng.choice(list(_CHIP_TRIANGLES))
    cx1, cy1 = {"tl": (x1, y1), "tr": (x2 - cw, y1),
                "bl": (x1, y2 - ch), "br": (x2 - cw, y2 - ch)}[corner]
    cx1, cy1 = max(0, cx1), max(0, cy1)
    cw, ch = min(cw, w - cx1), min(ch, h - cy1)
    if cw < 6 or ch < 6:
        return None

    # 从框外取背景纹理（优先下方 / 左侧 / 右侧，通常是桌面）作为填补内容
    pad = max(8, int(min(bw, bh) * 0.15))
    regions = [
        (max(0, y2 + pad), min(h, y2 + pad + ch), max(0, x1), min(w, x1 + cw)),
        (max(0, y1 - pad - ch), max(1, y1 - pad), max(0, x1), min(w, x1 + cw)),
        (max(0, y1), min(h, y1 + ch), max(0, x2 + pad), min(w, x2 + pad + cw)),
        (max(0, y1), min(h, y1 + ch), max(0, x1 - pad - cw), max(1, x1 - pad)),
    ]
    rng.shuffle(regions)
    fill = None
    for ry1, ry2, rx1, rx2 in regions:
        if ry2 - ry1 >= 4 and rx2 - rx1 >= 4:
            fill = cv2.resize(img[ry1:ry2, rx1:rx2], (cw, ch))
            break
    if fill is None:
        fill = np.tile(img.reshape(-1, 3).mean(axis=0).astype(img.dtype), (ch, cw, 1))

    pts = np.array([[cx1 + int(px * (cw - 1)), cy1 + int(py * (ch - 1))]
                    for px, py in _CHIP_TRIANGLES[corner]], np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    sub = mask[cy1:cy1 + ch, cx1:cx1 + cw]
    out = img.copy()
    out[cy1:cy1 + ch, cx1:cx1 + cw] = np.where(sub[..., None] > 0, fill, out[cy1:cy1 + ch, cx1:cx1 + cw])
    return out


def make_occluded(img, box: tuple[float, float, float, float], rng: random.Random,
                  ratio_range: tuple[float, float] = (0.2, 0.45)):
    """遮挡增强：用图像中其他位置的纹理块挡住木块的一部分。

    真实场景里木块常被瓶子、手、其他积木挡住（YCB-Video 里就常有这种情况），
    该增强让模型学会"只看到一部分也要认出来"。标签保持为木块整体范围。
    """
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in _clamp_box(box, w, h))
    bw, bh = x2 - x1, y2 - y1
    if bw < 28 or bh < 28:
        return None

    ratio = rng.uniform(*ratio_range)
    cw = max(8, int(bw * rng.uniform(0.35, 0.7)))
    ch = max(8, int(bh * ratio))
    side = rng.choice(["left", "right", "top", "bottom"])
    if side == "left":
        cx, cy = x1, rng.randint(y1, max(y1, y2 - ch))
    elif side == "right":
        cx, cy = x2 - cw, rng.randint(y1, max(y1, y2 - ch))
    elif side == "top":
        cx, cy = rng.randint(x1, max(x1, x2 - cw)), y1
    else:
        cx, cy = rng.randint(x1, max(x1, x2 - cw)), y2 - ch
    cx, cy = max(0, min(cx, w - cw)), max(0, min(cy, h - ch))
    cw, ch = min(cw, w - cx), min(ch, h - cy)
    if cw < 8 or ch < 8:
        return None

    # 遮挡物纹理从图像别处随机取（大概率是桌面/其他物体），并做一点模糊更像真实遮挡
    for _ in range(12):
        sx = rng.randint(0, max(0, w - cw - 1))
        sy = rng.randint(0, max(0, h - ch - 1))
        if sx + cw < x1 or sx > x2 or sy + ch < y1 or sy > y2:
            patch = img[sy:sy + ch, sx:sx + cw].copy()
            break
    else:
        patch = img[max(0, y2 + 8):max(0, y2 + 8) + ch, x1:x1 + cw]
        if patch.size == 0:
            return None
        patch = cv2.resize(patch, (cw, ch))

    out = img.copy()
    patch = cv2.GaussianBlur(patch, (5, 5), 0)
    if rng.random() < 0.5:  # 一半概率用不规则边缘，更像真实遮挡轮廓
        mask = np.zeros((ch, cw), np.uint8)
        pts = np.array([[0, rng.randint(0, ch // 2)], [cw, rng.randint(0, ch // 2)],
                        [cw, ch], [0, ch]], np.int32)
        cv2.fillPoly(mask, [pts], 255)
        region = out[cy:cy + ch, cx:cx + cw]
        out[cy:cy + ch, cx:cx + cw] = np.where(mask[..., None] > 0, patch, region)
    else:
        out[cy:cy + ch, cx:cx + cw] = patch
    return out


def add_chipped_variants(ratio: float = paths.CHIPPED_RATIO, seed: int = 0,
                         verbose: bool = True) -> int:
    """为训练集生成缺角增强样本（在已构建好的数据集上离线执行，无需重新下载）。"""
    return _add_variants(make_chipped, "_chip", ratio, seed, "缺角增强", verbose)


def add_occluded_variants(ratio: float = paths.OCCLUDED_RATIO, seed: int = 1,
                          verbose: bool = True) -> int:
    """为训练集生成遮挡增强样本。"""
    return _add_variants(make_occluded, "_occ", ratio, seed, "遮挡增强", verbose)


def make_zoomed(img, box: tuple[float, float, float, float], rng: random.Random,
                zoom_range: tuple[float, float] = (1.4, 2.6)):
    """放大增强：围绕木块裁一个更小的窗口再放大回原尺寸。

    YCB-Video 里木块通常只占画面高度的一小部分；而摄像头贴近拍摄时木块可能占满
    画面。该增强补上"大目标"样本，避免模型只在中小尺度上有效。
    返回 (新图, 新标签行所需的信息)。
    """
    import cv2

    h, w = img.shape[:2]
    x1, y1, x2, y2 = _clamp_box(box, w, h)
    bw, bh = x2 - x1, y2 - y1
    if bw < 16 or bh < 16:
        return None

    k = rng.uniform(*zoom_range)
    cw, ch = int(w / k), int(h / k)
    # 保证裁剪窗口能完整包含木块（留 12% 余量）
    cw = min(w, max(cw, int(bw * 1.12)))
    ch = min(h, max(ch, int(bh * 1.12)))
    bcx, bcy = (x1 + x2) / 2, (y1 + y2) / 2
    jx, jy = rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15)
    x0 = int(round(bcx - cw / 2 + jx * cw))
    y0 = int(round(bcy - ch / 2 + jy * ch))
    x0 = max(0, min(x0, w - cw))
    y0 = max(0, min(y0, h - ch))
    crop = img[y0:y0 + ch, x0:x0 + cw]
    if crop.size == 0:
        return None
    out = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    sx, sy = w / cw, h / ch
    nx1, ny1 = (x1 - x0) * sx, (y1 - y0) * sy
    nx2, ny2 = (x2 - x0) * sx, (y2 - y0) * sy
    nc_x, nc_y = (nx1 + nx2) / 2 / w, (ny1 + ny2) / 2 / h
    nw, nh = (nx2 - nx1) / w, (ny2 - ny1) / h
    if nw <= 0 or nh <= 0:
        return None
    return out, f"0 {nc_x:.6f} {nc_y:.6f} {nw:.6f} {nh:.6f}"


def add_zoomed_variants(ratio: float = 0.35, seed: int = 2, verbose: bool = True) -> int:
    """为训练集生成"大目标/近景"增强样本。"""
    import cv2

    img_dir = os.path.join(paths.DATASET_DIR, "images", "train")
    lbl_dir = os.path.join(paths.DATASET_DIR, "labels", "train")
    names = [f for f in sorted(os.listdir(img_dir))
             if f.lower().endswith((".png", ".jpg")) and not f.startswith("neg_")
             and not any(s in f for s in ("_chip", "_occ", "_zoom"))]
    rng = random.Random(seed)
    rng.shuffle(names)
    n = int(len(names) * ratio)
    made = 0
    for name in names[:n]:
        stem = os.path.splitext(name)[0]
        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        lines = [ln.split() for ln in open(lbl_path, encoding="utf8").read().splitlines() if ln.split()]
        if not lines:
            continue
        img = cv2.imread(os.path.join(img_dir, name))
        if img is None:
            continue
        h, w = img.shape[:2]
        _, cx, cy, bw, bh = (float(v) for v in lines[0][:5])
        box = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
        res = make_zoomed(img, box, rng)
        if res is None:
            continue
        out, new_line = res
        cv2.imwrite(os.path.join(img_dir, stem + "_zoom.png"), out)
        with open(os.path.join(lbl_dir, stem + "_zoom.txt"), "w", encoding="utf8") as f:
            f.write(new_line + "\n")
        made += 1
    if verbose:
        print(f"[数据集] 放大增强: 基于 {n} 张训练图生成 {made} 张近景样本")
    return made


def _add_variants(fn, suffix: str, ratio: float, seed: int, label: str, verbose: bool) -> int:
    import cv2

    img_dir = os.path.join(paths.DATASET_DIR, "images", "train")
    lbl_dir = os.path.join(paths.DATASET_DIR, "labels", "train")
    names = [f for f in sorted(os.listdir(img_dir))
             if f.lower().endswith((".png", ".jpg")) and not f.startswith("neg_")
             and "_chip" not in f and "_occ" not in f]
    rng = random.Random(seed)
    rng.shuffle(names)
    n = int(len(names) * ratio)
    made = 0
    for name in names[:n]:
        stem = os.path.splitext(name)[0]
        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        lines = [ln.split() for ln in open(lbl_path, encoding="utf8").read().splitlines() if ln.split()]
        if not lines:
            continue
        img = cv2.imread(os.path.join(img_dir, name))
        if img is None:
            continue
        h, w = img.shape[:2]
        _, cx, cy, bw, bh = (float(v) for v in lines[0][:5])
        box = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
        out = fn(img, box, rng)
        if out is None:
            continue
        cv2.imwrite(os.path.join(img_dir, stem + suffix + ".png"), out)
        with open(os.path.join(lbl_dir, stem + suffix + ".txt"), "w", encoding="utf8") as f:
            f.write("\n".join(" ".join(ln) for ln in lines) + "\n")
        made += 1
    if verbose:
        print(f"[数据集] {label}: 基于 {n} 张训练图生成 {made} 张样本")
    return made


# --------------------------------------------------------------------- 测试集
def fetch_bop_test(verbose: bool = True) -> str:
    """下载并解压 BOP YCB-V 官方测试集（用于误检率/召回率评估）。

    该测试集来自 YCB-Video 的官方验证场景（0048-0059），与训练/验证场景不重叠。
    """
    import urllib.request
    import zipfile

    dest_dir = os.path.join(paths.RAW_DIR, "ycbv_test_bop19")
    if os.path.isdir(os.path.join(dest_dir, "test")):
        if verbose:
            n = len(os.listdir(os.path.join(dest_dir, "test")))
            print(f"[测试集] 已存在: {dest_dir}（{n} 个场景）")
        return dest_dir

    zip_path = os.path.join(paths.RAW_DIR, "ycbv_test_bop19.zip")
    if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 600_000_000:
        if verbose:
            print("[测试集] 下载 BOP YCB-V 测试集（约 660 MB，支持断点续传）...")
        _download(sources.BOP_YCBV_TEST_URL, zip_path, verbose=verbose)

    if verbose:
        print("[测试集] 解压 ...")
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    if verbose:
        n = len(os.listdir(os.path.join(dest_dir, "test")))
        print(f"[测试集] 完成: {dest_dir}（{n} 个场景）")
    return dest_dir


def _download(url: str, dest: str, verbose: bool = True, retries: int = 20) -> None:
    """带断点续传与重试的下载。"""
    import time
    import urllib.error
    import urllib.request

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    total = None
    try:
        with urllib.request.urlopen(
                urllib.request.Request(url, headers=remote_zip.UA, method="HEAD"), timeout=60) as r:
            total = int(r.headers.get("Content-Length", -1))
    except Exception:  # noqa: BLE001
        pass

    for attempt in range(1, retries + 1):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if total and total > 0 and have >= total:
            break
        req = urllib.request.Request(url, headers=dict(remote_zip.UA))
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if have and r.status != 206:
                    have = 0
                with open(tmp, "ab" if have else "wb") as f:
                    last = time.time()
                    while True:
                        buf = r.read(4 << 20)
                        if not buf:
                            break
                        f.write(buf)
                        have += len(buf)
                        if verbose and time.time() - last > 15:
                            last = time.time()
                            pct = f"{have/1e6:.0f}/{total/1e6:.0f} MB" if total else f"{have/1e6:.0f} MB"
                            print(f"    {pct}", flush=True)
        except Exception as e:  # noqa: BLE001
            if attempt >= retries:
                raise
            if verbose:
                print(f"    下载中断（{type(e).__name__}），3 秒后续传 ...", flush=True)
            time.sleep(3)
            continue
        if not total or total < 0 or os.path.getsize(tmp) >= total:
            break
    os.replace(tmp, dest)


def build_dataset(report: dict, stride: int = paths.FRAME_STRIDE, workers: int = 12,
                  val_scenes: int = paths.VAL_SCENE_COUNT, max_scenes: int | None = None,
                  neg_scenes: int = 18, neg_frames: int = 20, verbose: bool = True) -> dict:
    """构建 YOLO 格式数据集，返回统计信息。"""
    ds = paths.DATASET_DIR
    if os.path.isdir(ds):
        shutil.rmtree(ds)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(ds, sub), exist_ok=True)

    woods = wood_scenes(report)
    if max_scenes:
        woods = woods[:max_scenes]
    if verbose:
        print(f"[数据集] 含木块场景 {len(woods)} 个: {woods}")

    val_ids = set(woods[:val_scenes]) if val_scenes else set()

    stats = {"train_images": 0, "train_boxes": 0, "val_images": 0, "val_boxes": 0,
             "neg_images": 0, "scenes": woods, "val_scenes": sorted(val_ids), "skipped": 0}

    for scene in woods:
        meta = fetch_scene(scene, stride=stride, workers=workers, verbose=verbose)
        split = "val" if scene in val_ids else "train"
        img_src = os.path.join(paths.RAW_SCENES_DIR, scene, "images")
        boxes_by_frame = meta["frames"]
        for frame, boxes in boxes_by_frame.items():
            box = boxes.get(WOOD_NAME)
            if box is None:
                continue
            line = _yolo_line(box, 640, 480)
            if line is None:
                stats["skipped"] += 1
                continue
            name = f"{scene}_{frame}"
            shutil.copyfile(os.path.join(img_src, f"{frame}.png"),
                            os.path.join(ds, "images", split, name + ".png"))
            with open(os.path.join(ds, "labels", split, name + ".txt"), "w", encoding="utf8") as f:
                f.write(line + "\n")
            stats[f"{split}_images"] += 1
            stats[f"{split}_boxes"] += 1

    # 负样本（不含木块的真实场景，空标签）
    negs = negative_scenes(report)[:neg_scenes]
    random.Random(0).shuffle(negs)
    for scene in negs:
        try:
            meta = fetch_scene(scene, stride=max(stride * 4, 24), workers=workers,
                               with_wood=False, max_frames=neg_frames, verbose=False)
        except Exception:  # noqa: BLE001
            continue
        img_src = os.path.join(paths.RAW_SCENES_DIR, scene, "images")
        for frame in list(meta["frames"])[:neg_frames]:
            src = os.path.join(img_src, f"{frame}.png")
            if not os.path.exists(src):
                continue
            name = f"neg_{scene}_{frame}"
            shutil.copyfile(src, os.path.join(ds, "images/train", name + ".png"))
            open(os.path.join(ds, "labels/train", name + ".txt"), "w").close()
            stats["neg_images"] += 1
            stats["train_images"] += 1

    yaml_path = os.path.join(ds, "data.yaml")
    with open(yaml_path, "w", encoding="utf8") as f:
        f.write(f"path: {ds.replace(os.sep, '/')}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n  0: wood_block\n")
    stats["yaml"] = yaml_path

    # 缺角 / 遮挡 / 近景放大增强（基于真实图片离线生成，标签不变或同步更新）
    stats["chipped_images"] = add_chipped_variants(verbose=verbose)
    stats["occluded_images"] = add_occluded_variants(verbose=verbose)
    stats["zoomed_images"] = add_zoomed_variants(verbose=verbose)
    stats["train_images"] += (stats["chipped_images"] + stats["occluded_images"]
                              + stats["zoomed_images"])

    if verbose:
        print(f"[数据集] 完成: 训练 {stats['train_images']} 张（木块 {stats['train_boxes']} + "
              f"缺角 {stats['chipped_images']} + 遮挡 {stats['occluded_images']} + "
              f"近景 {stats['zoomed_images']} + 负样本 {stats['neg_images']}）, "
              f"验证 {stats['val_images']} 张, 过滤小框 {stats['skipped']}")
    return stats
