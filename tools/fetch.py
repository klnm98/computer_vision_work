"""可断点续传、自动重试的大文件下载器。"""
from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
RETRIES = 40


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", -1))


def download(url: str, dest: str, chunk: int = 4 << 20) -> None:
    """下载 url 到 dest，支持断点续传与自动重试。"""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    total = None
    try:
        total = remote_size(url)
    except Exception:  # noqa: BLE001
        pass
    if total and os.path.exists(dest) and os.path.getsize(dest) == total:
        print(f"已存在完整文件 {dest} ({total/1e6:.1f} MB)")
        return

    t_start = time.time()
    for attempt in range(1, RETRIES + 1):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if total and have >= total:
            break
        req = urllib.request.Request(url, headers=UA)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                if have and r.status != 206:  # 服务端不支持续传，重头来
                    have = 0
                with open(tmp, "ab" if have else "wb") as f:
                    last = time.time()
                    while True:
                        buf = r.read(chunk)
                        if not buf:
                            break
                        f.write(buf)
                        have += len(buf)
                        now = time.time()
                        if now - last > 10:
                            last = now
                            speed = have / max(now - t_start, 1e-6) / 1e6
                            tail = f"/{total/1e6:.1f}" if total else ""
                            print(f"  {have/1e6:8.1f}{tail} MB  {speed:5.2f} MB/s", flush=True)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            print(f"  第 {attempt} 次中断（{type(e).__name__}: {str(e)[:60]}），3 秒后续传 ...", flush=True)
            time.sleep(3)
            continue
        if not total or os.path.getsize(tmp) >= total:
            break
    else:
        raise RuntimeError(f"下载失败: {url}")

    size = os.path.getsize(tmp)
    if total and size != total:
        raise RuntimeError(f"下载不完整: {size}/{total} 字节")
    os.replace(tmp, dest)
    print(f"DONE {dest} {size/1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    download(sys.argv[1], sys.argv[2])
