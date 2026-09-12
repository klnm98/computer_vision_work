"""运行环境兼容补丁。

部分受限环境（例如只允许写工作区、禁止进程间命名管道通信的沙箱）下，
Ultralytics 的两处默认行为会直接报权限错误：

1. ``YOLODataset.cache_labels`` 使用 ``multiprocessing.pool.ThreadPool`` 扫描标注，
   而它的内部队列依赖命名管道 —— 会被拒绝（WinError 5）。这里替换成等价的
   **顺序实现**，产出的 ``labels.cache`` 格式与官方完全一致。
2. ``DataLoader`` 的多进程 worker 同样依赖管道，因此训练/验证时把 ``workers``
   强制设为 0（在主进程内加载数据）。

这两项在普通环境下也是安全的：只是牺牲一点加载速度，不影响训练结果。
"""
from __future__ import annotations

import os
from pathlib import Path

_PATCHED = False


def _sequential_cache_labels(self, path: Path = Path("./labels.cache")) -> dict:
    """``YOLODataset.cache_labels`` 的无多进程版本。"""
    from ultralytics.data.dataset import DATASET_CACHE_VERSION
    from ultralytics.data.utils import HELP_URL, save_dataset_cache_file
    from ultralytics.utils import LOGGER, TQDM

    x: dict = {"labels": []}
    nm = nf = ne = nc = 0  # missing, found, empty, corrupt
    msgs: list[str] = []
    desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
    total = len(self.im_files)

    func, iterable = self.verify_args()
    pbar = TQDM(iterable, desc=desc, total=total)
    for args in pbar:
        # 与官方 ``pool.imap(func, iterable)`` 语义一致：每次把一个参数元组交给 func
        result = func(args)
        label, nm_f, nf_f, ne_f, nc_f, msg = self.result_to_label(result)
        nm += nm_f
        nf += nf_f
        ne += ne_f
        nc += nc_f
        if label is not None:
            x["labels"].append(label)
        if msg:
            msgs.append(msg)
        pbar.desc = f"{desc} {self.scan_summary(nf, nm, ne, nc)}"
    pbar.close()

    if msgs:
        LOGGER.info("\n".join(msgs))
    if nf == 0:
        if self.augment:
            raise ValueError(f"{self.prefix}No labels found in {path}. {HELP_URL}")
        LOGGER.warning(f"{self.prefix}No labels found in {path}. {HELP_URL}")

    x["hash"] = self.get_cache_hash()
    x["results"] = nf, nm, ne, nc, total
    x["msgs"] = msgs
    if x["labels"]:
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
    return x


def patch_ultralytics() -> bool:
    """打上兼容补丁（可重复调用）。返回是否为本次调用首次生效。"""
    global _PATCHED
    if _PATCHED:
        return False
    try:
        from ultralytics.data.dataset import YOLODataset
    except ImportError:  # 还没装 ultralytics 时直接跳过
        return False
    YOLODataset.cache_labels = _sequential_cache_labels
    _PATCHED = True
    return True


def safe_workers(requested: int = 0) -> int:
    """返回当前环境可用的 DataLoader worker 数。"""
    if os.environ.get("WOODBLOCK_ALLOW_WORKERS") == "1":
        return requested
    return 0
