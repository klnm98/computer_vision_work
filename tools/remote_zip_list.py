"""List a remote ZIP's central directory via HTTP range requests (works for huge archives)."""
import re
import struct
import sys
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def total_size(url):
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        n = r.headers.get("Content-Length")
    if n:
        return int(n)
    # some mirrors omit Content-Length on HEAD: try a range GET
    h = dict(UA)
    h["Range"] = "bytes=0-0"
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range", "")
    m = re.search(r"/(\d+)$", cr)
    return int(m.group(1)) if m else -1


def fetch(url, start, end):
    h = dict(UA)
    h["Range"] = f"bytes={start}-{end}"
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def eocd(url, n, tail=2_000_000):
    buf = fetch(url, max(0, n - tail), n - 1)
    i = buf.rfind(b"PK\x05\x06")
    if i < 0:
        raise RuntimeError("no EOCD found")
    cd_size, cd_off = struct.unpack("<II", buf[i + 12: i + 20])
    return cd_off, cd_size


def parse_cd(cd):
    out, p = [], 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        comp, usize = struct.unpack("<II", cd[p + 20: p + 28])
        nlen, elen, clen = struct.unpack("<HHH", cd[p + 28: p + 34])
        name = cd[p + 46: p + 46 + nlen].decode("utf8", "replace")
        out.append((name, comp, usize))
        p += 46 + nlen + elen + clen
    return out


def list_zip(url):
    n = total_size(url)
    cd_off, cd_size = eocd(url, n)
    cd = fetch(url, cd_off, cd_off + cd_size - 1)
    return n, parse_cd(cd)


if __name__ == "__main__":
    for url in sys.argv[1:]:
        try:
            n, entries = list_zip(url)
            files = [e for e in entries if not e[0].endswith("/")]
            total = sum(e[1] for e in files)
            print("=" * 72)
            print(url)
            print(f"  archive={n/1e6:.1f} MB  entries={len(entries)}  files={len(files)}  "
                  f"uncompressed={total/1e6:.1f} MB")
            exts = {}
            for name, comp, usize in files:
                ext = name.rsplit(".", 1)[-1].lower() if "." in name else "<none>"
                exts.setdefault(ext, [0, 0])
                exts[ext][0] += 1
                exts[ext][1] += comp
            for ext, (cnt, size) in sorted(exts.items(), key=lambda kv: -kv[1][1])[:10]:
                print(f"    .{ext:6s} count={cnt:7d}  compressed={size/1e6:9.1f} MB")
            print("  first entries:")
            for name, comp, usize in entries[:12]:
                print(f"    {name}  ({comp/1e6:.2f} MB)")
        except Exception as e:  # noqa: BLE001
            print("ERR", url, type(e).__name__, e)
