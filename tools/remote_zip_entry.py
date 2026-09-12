"""Range-read a single small entry from a remote ZIP (no full download).

Usage:
    python tools/remote_zip_entry.py <url> <entry-name-substring> [max-bytes]

Finds the entry whose name ends with the given substring, range-fetches its local
header + compressed payload, decompresses it and prints the text.
"""
import re
import struct
import sys
import urllib.request
import zlib

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def total_size(url):
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        n = r.headers.get("Content-Length")
    if n:
        return int(n)
    h = dict(UA)
    h["Range"] = "bytes=0-0"
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(re.search(r"/(\d+)$", r.headers.get("Content-Range", "")).group(1))


def fetch(url, start, end):
    h = dict(UA)
    h["Range"] = f"bytes={start}-{end}"
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def central_directory(url, tail=4_000_000):
    n = total_size(url)
    buf = fetch(url, max(0, n - tail), n - 1)
    i = buf.rfind(b"PK\x05\x06")
    if i < 0:
        raise RuntimeError("no EOCD")
    cd_size, cd_off = struct.unpack("<II", buf[i + 12: i + 20])
    cd = fetch(url, cd_off, cd_off + cd_size - 1)
    entries, p = [], 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        (method,) = struct.unpack("<H", cd[p + 10: p + 12])
        comp, usize = struct.unpack("<II", cd[p + 20: p + 28])
        nlen, elen, clen = struct.unpack("<HHH", cd[p + 28: p + 34])
        (lho,) = struct.unpack("<I", cd[p + 42: p + 46])
        name = cd[p + 46: p + 46 + nlen].decode("utf8", "replace")
        entries.append(dict(name=name, method=method, comp=comp, usize=usize, lho=lho))
        p += 46 + nlen + elen + clen
    return entries


def read_entry(url, ent, limit=1 << 20):
    head = fetch(url, ent["lho"], ent["lho"] + 29)
    nlen, elen = struct.unpack("<HH", head[26:30])
    start = ent["lho"] + 30 + nlen + elen
    raw = fetch(url, start, start + min(ent["comp"], limit) - 1)
    if ent["method"] == 0:
        return raw
    return zlib.decompress(raw, -15)


if __name__ == "__main__":
    url, pattern = sys.argv[1], sys.argv[2]
    entries = central_directory(url)
    hits = [e for e in entries if e["name"].endswith(pattern)]
    print(f"entries={len(entries)} matches={len(hits)}")
    for e in hits[:3]:
        print("----", e["name"], e["comp"], "bytes")
        print(read_entry(url, e).decode("utf8", "ignore")[:2000])
