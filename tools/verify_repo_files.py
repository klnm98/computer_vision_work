"""核验远程仓库根目录文件列表（只读）。"""
from __future__ import annotations

import json
import urllib.request

TOK = open(".git-tmp/token.txt", encoding="utf8").read().strip()


def api(path: str):
    r = urllib.request.Request("https://api.github.com" + path)
    r.add_header("Authorization", "Bearer " + TOK)
    r.add_header("User-Agent", "verify-repo")
    r.add_header("Accept", "application/vnd.github+json")
    with urllib.request.urlopen(r, timeout=60) as resp:
        return json.load(resp)


files = api("/repos/klnm98/computer_vision_work/contents/")
print("远程仓库根目录:")
for f in sorted(files, key=lambda x: (x["type"] != "dir", x["name"])):
    print(f"  [{'目录' if f['type'] == 'dir' else '文件'}] {f['name']}")

names = [f["name"] for f in files]
print("\n积木识别初步.py 是否还在远程:", "在（未删除！）" if "积木识别初步.py" in names else "已删除 ✓")

targets = ["README.md", "RELEASE_NOTES.md", "main.py", "detect_camera.py", "block.jpg"]
print("关键文件是否都在:", {t: (t in names) for t in targets})

commits = api("/repos/klnm98/computer_vision_work/commits?per_page=3")
print("\n最近提交:")
for c in commits:
    print(f"  {c['sha'][:7]}  {c['commit']['message'].splitlines()[0]}")
