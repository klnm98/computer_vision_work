"""验证局域网 APK 下载服务是否可用（本机 + 局域网地址各测一次）。"""
from __future__ import annotations

import urllib.request

NAME = "wood_block_detector_debug.apk"
for host in ("127.0.0.1", "10.138.37.62"):
    url = f"http://{host}:8765/{NAME}"
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=8) as r:
            size = int(r.headers.get("Content-Length", 0))
        print(f"OK   {url}   HTTP {r.status}   {size/1e6:.1f} MB")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {url}   {type(e).__name__}: {str(e)[:70]}")
