"""探测 Android 构建所需的主机可达性，以及 onnx 相关 wheel 对 Python 3.14 的支持情况。"""
from __future__ import annotations

import json
import socket
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def https_ok(url: str, timeout: int = 20) -> str:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return f"OK {r.status} len={len(r.read(2000))}"
    except Exception as e:  # noqa: BLE001
        return f"FAIL {type(e).__name__}: {str(e)[:70]}"


print("=== TCP 443 ===")
for host in ["dl.google.com", "maven.google.com", "repo1.maven.org", "services.gradle.org",
             "api.adoptium.net", "github.com", "objects.githubusercontent.com"]:
    try:
        s = socket.create_connection((host, 443), timeout=10)
        s.close()
        print(f"  {host:32s} TCP OK")
    except Exception as e:  # noqa: BLE001
        print(f"  {host:32s} FAIL {type(e).__name__}")

print("\n=== HTTPS 取样 ===")
for url in [
    "https://dl.google.com/android/repository/repository2-3.xml",
    "https://maven.google.com/com/android/tools/build/gradle/maven-metadata.xml",
    "https://repo1.maven.org/maven2/org/pytorch/pytorch_android/maven-metadata.xml",
    "https://services.gradle.org/distributions/",
    "https://api.adoptium.net/v3/info/available_releases",
]:
    print(f"  {url[:62]:64s} {https_ok(url)}")

print("\n=== PyPI: onnx / onnxruntime / tensorflow 的 cp314 wheel ===")
for pkg, ver in [("onnx", None), ("onnxruntime", None), ("tensorflow", None), ("onnxslim", None),
                 ("onnx2tf", None)]:
    try:
        url = f"https://pypi.org/pypi/{pkg}/json" if not ver else f"https://pypi.org/pypi/{pkg}/{ver}/json"
        d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25))
        latest = d["info"]["version"]
        files = d["releases"].get(latest, [])
        win = [f["filename"] for f in files if "cp314" in f["filename"] and "win_amd64" in f["filename"]]
        anyw = [f["filename"] for f in files if f["filename"].endswith(".whl")]
        print(f"  {pkg:14s} latest={latest:12s} cp314-win={win[:2] if win else '无'}  通用wheel数={len(anyw)}")
    except Exception as e:  # noqa: BLE001
        print(f"  {pkg:14s} ERR {type(e).__name__} {str(e)[:60]}")
