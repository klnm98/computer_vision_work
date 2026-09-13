"""列出 Android 仓库清单里 platform-34 / build-tools 34 的真实下载项。"""
from __future__ import annotations

import re
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
xml = urllib.request.urlopen(
    urllib.request.Request("https://dl.google.com/android/repository/repository2-3.xml", headers=UA),
    timeout=60).read().decode("utf8", "ignore")

print("全部 platform zip:")
for h in sorted(set(re.findall(r"platform-3[0-9][^\"<>]*\.zip", xml))):
    print("   ", h)

for want in ("platforms;android-34", "platforms;android-35", "build-tools;34.0.0"):
    m = re.search(r'<remotePackage path="' + re.escape(want) + r'".*?</remotePackage>', xml, re.S)
    print(f"\n--- {want} ---")
    if not m:
        print("   未找到该条目")
        continue
    seg = m.group(0)
    print("   archives:", re.findall(r"<url>([^<]+\.zip)</url>", seg)[:5])
    print("   windows :", [u for u in re.findall(r"<url>([^<]+\.zip)</url>", seg) if "windows" in u][:3])
