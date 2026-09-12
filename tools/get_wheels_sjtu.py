"""从上海交大镜像下载 PyTorch CUDA wheels（速度快）。"""
from __future__ import annotations

import os
import re
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch import download  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
BASE = "https://mirror.sjtu.edu.cn/pytorch-wheels/cu128"
OUT = r"D:\computer_vision_work\models\wheels"


def list_pkg(pkg: str) -> list[str]:
    url = f"{BASE}/{pkg}/"
    html = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode("utf8", "ignore")
    return [urllib.parse.urljoin(url, h) for h in re.findall(r'href="([^"]+\.whl)"', html)]


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "cp314-cp314-win_amd64"
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if "#" in f:
            os.remove(os.path.join(OUT, f))

    for pkg in ("torch", "torchvision"):
        try:
            urls = list_pkg(pkg)
        except Exception as e:  # noqa: BLE001
            print(f"{pkg}: 列表获取失败 {e}")
            continue
        hits = [u for u in urls if tag in u and "+" in urllib.parse.unquote(u)]
        if not hits:
            print(f"{pkg}: 没有匹配 {tag} 的 wheel（共 {len(urls)} 个）")
            continue
        url = sorted(hits)[-1]
        name = os.path.basename(urllib.parse.unquote(url))
        dest = os.path.join(OUT, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
            print(f"{pkg}: 已存在 {name}")
            continue
        download(url, dest)
    print("\nwheel 目录:")
    for f in sorted(os.listdir(OUT)):
        print(f"   {f}  ({os.path.getsize(os.path.join(OUT, f))/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
