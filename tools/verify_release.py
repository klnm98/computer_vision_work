"""核验线上 Release 与附件（只读）。"""
from __future__ import annotations

import json
import urllib.request

TOK = open(".git-tmp/token.txt", encoding="utf8").read().strip()


def api(path: str):
    r = urllib.request.Request("https://api.github.com" + path)
    r.add_header("Authorization", "Bearer " + TOK)
    r.add_header("User-Agent", "verify-release")
    r.add_header("Accept", "application/vnd.github+json")
    with urllib.request.urlopen(r, timeout=60) as resp:
        return json.load(resp)


rel = api("/repos/klnm98/computer_vision_work/releases/tags/v1.0.0")
print("标题      :", rel["name"])
print("tag       :", rel["tag_name"], "| 草稿:", rel["draft"], "| 预发布:", rel["prerelease"])
print("发布时间  :", rel["published_at"])
print("说明长度  :", len(rel["body"]), "字符")
print("页面      :", rel["html_url"])
for a in rel["assets"]:
    print(f"附件      : {a['name']}  {a['size']/1e6:.1f} MB  下载次数={a['download_count']}")
    print("            ", a["browser_download_url"])
tags = api("/repos/klnm98/computer_vision_work/tags")
print("仓库 tags :", [t["name"] for t in tags])
