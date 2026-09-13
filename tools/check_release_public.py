"""从"未登录的普通访客"视角核验 Release 附件是否真的可下载。

检查三件事：
  1) API 里附件的状态（state/size/content_type）；
  2) 不带头部凭据直接 GET 下载直链，看是否 200 且字节数对得上；
  3) 拉取 Release 页面 HTML，确认页面里出现了附件与下载链接。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

REPO = "klnm98/computer_vision_work"
TAG = "v1.0.0"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

print("=== 1) API 里的 Release 与附件 ===")
with urllib.request.urlopen(urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}", headers=UA), timeout=60) as r:
    rel = json.load(r)
print("draft:", rel["draft"], "| prerelease:", rel["prerelease"], "| tag:", rel["tag_name"])
print("assets 数量:", len(rel["assets"]))
for a in rel["assets"]:
    print(f"  名称={a['name']}")
    print(f"  state={a.get('state')}  size={a['size']}  type={a.get('content_type')}")
    print(f"  创建={a['created_at']}  下载次数={a['download_count']}")
    print(f"  直链={a['browser_download_url']}")

print("\n=== 2) 未登录直接下载附件（跟随重定向） ===")
url = rel["assets"][0]["browser_download_url"] if rel["assets"] else ""
if url:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        print(f"  HTTP {r.status}  收到 {len(data)/1e6:.1f} MB  最终地址={r.geturl()[:90]}")
        print(f"  是 APK(ZIP) 头: {data[:2] == b'PK'}  SHA256 前 8 位: ", end="")
        import hashlib
        print(hashlib.sha256(data).hexdigest()[:8])
    except urllib.error.HTTPError as e:
        print(f"  下载失败 HTTP {e.code} {e.reason}")
else:
    print("  没有附件")

print("\n=== 3) Release 页面 HTML 里是否出现附件 ===")
page = f"https://github.com/{REPO}/releases/tag/{TAG}"
try:
    with urllib.request.urlopen(urllib.request.Request(page, headers=UA), timeout=60) as r:
        html = r.read().decode("utf8", "ignore")
    print(f"  页面 HTTP {r.status}，长度 {len(html)}")
    name = rel["assets"][0]["name"] if rel["assets"] else "wood_block_detector_debug.apk"
    for key in (name, "Assets", "releases/download"):
        print(f"  HTML 含 {key!r}: {key in html}")
except urllib.error.HTTPError as e:
    print(f"  页面读取失败 HTTP {e.code} {e.reason}")

print("\n=== 4) releases 列表（latest） ===")
with urllib.request.urlopen(urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/releases", headers=UA), timeout=60) as r:
    lst = json.load(r)
for x in lst:
    print(f"  {x['tag_name']:8s} draft={x['draft']} 附件={len(x['assets'])} 标题={x['name']}")
