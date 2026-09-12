"""通过 HTTP Range 远程读取 ZIP 内的单个文件。

YCB-Video 每个场景压缩包 1~3 GB，但标注文件（*-box.txt）只有约 160 字节。
本模块只请求所需字节：先取 ZIP 中央目录，再按需读取指定条目，
从而避免下载整包。
"""
from __future__ import annotations

import json
import os
import re
import struct
import time
import urllib.error
import urllib.request
import zlib

from . import paths

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
RETRIES = 4


def _request(url: str, headers: dict | None = None, method: str = "GET"):
    h = dict(UA)
    if headers:
        h.update(headers)
    for attempt in range(RETRIES):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=h, method=method), timeout=180)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == RETRIES - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def total_size(url: str) -> int:
    """远程文件总长度（字节）。"""
    with _request(url, method="HEAD") as r:
        n = r.headers.get("Content-Length")
    if n:
        return int(n)
    with _request(url, {"Range": "bytes=0-0"}) as r:
        m = re.search(r"/(\d+)$", r.headers.get("Content-Range", ""))
    if not m:
        raise RuntimeError(f"无法获取文件大小: {url}")
    return int(m.group(1))


def fetch_range(url: str, start: int, end: int) -> bytes:
    """读取 [start, end] 闭区间的字节。"""
    with _request(url, {"Range": f"bytes={start}-{end}"}) as r:
        return r.read()


def central_directory(url: str, cache_key: str | None = None, tail: int = 1_500_000) -> list[dict]:
    """返回 ZIP 中央目录条目列表（带本地缓存）。"""
    cache_path = None
    if cache_key:
        cache_path = os.path.join(paths.CD_CACHE_DIR, f"{cache_key}.json")
        if os.path.exists(cache_path):
            try:
                return json.load(open(cache_path, encoding="utf8"))
            except (json.JSONDecodeError, OSError):
                pass

    n = total_size(url)
    win = min(tail, n)
    buf = fetch_range(url, n - win, n - 1)
    idx = buf.rfind(b"PK\x05\x06")
    if idx < 0:
        raise RuntimeError(f"未找到 ZIP 结尾记录: {url}")
    cd_size, cd_off = struct.unpack("<II", buf[idx + 12: idx + 20])
    if cd_off + cd_size > n:
        raise RuntimeError(f"中央目录越界: {url}")
    cd = fetch_range(url, cd_off, cd_off + cd_size - 1)

    entries, p = [], 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", cd[p + 10: p + 12])
        comp, usize = struct.unpack("<II", cd[p + 20: p + 28])
        nlen, elen, clen = struct.unpack("<HHH", cd[p + 28: p + 34])
        lho, = struct.unpack("<I", cd[p + 42: p + 46])
        name = cd[p + 46: p + 46 + nlen].decode("utf8", "replace")
        entries.append({"name": name, "method": method, "comp": comp, "usize": usize, "lho": lho})
        p += 46 + nlen + elen + clen

    if cache_path:
        with open(cache_path, "w", encoding="utf8") as f:
            json.dump(entries, f)
    return entries


def read_entry(url: str, entry: dict, limit: int | None = None) -> bytes:
    """读取 ZIP 中某个条目的内容（自动解压）。"""
    head = fetch_range(url, entry["lho"], entry["lho"] + 29)
    nlen, elen = struct.unpack("<HH", head[26:30])
    start = entry["lho"] + 30 + nlen + elen
    size = entry["comp"] if limit is None else min(entry["comp"], limit)
    raw = fetch_range(url, start, start + size - 1)
    if entry["method"] == 0:
        return raw
    if entry["method"] != 8:
        raise RuntimeError(f"不支持的压缩方式 {entry['method']}")
    return zlib.decompress(raw, -15)


def find_entries(entries: list[dict], suffix: str | None = None, pattern: str | None = None) -> list[dict]:
    """按后缀或正则筛选条目。"""
    out = []
    rx = re.compile(pattern) if pattern else None
    for e in entries:
        if e["name"].endswith("/"):
            continue
        if suffix and not e["name"].endswith(suffix):
            continue
        if rx and not rx.search(e["name"]):
            continue
        out.append(e)
    return out
