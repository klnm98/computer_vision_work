"""下载并安装 Android 构建所需工具链（全部放在工作区内，不污染系统）。

包含：JDK 17、Android command-line tools + SDK 平台/构建工具、Gradle。
在受限网络下会按顺序尝试多个镜像源。

用法:
    python tools/setup_android_toolchain.py            # 全部安装
    python tools/setup_android_toolchain.py --only jdk  # 只装某一步
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, ".android-build")
JDK_DIR = os.path.join(BASE, "jdk")
SDK_DIR = os.path.join(BASE, "sdk")
GRADLE_DIR = os.path.join(BASE, "gradle")
DL = os.path.join(BASE, "dl")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

JDK_SOURCES = [
    "https://mirrors.tuna.tsinghua.edu.cn/Adoptium/17/jdk/x64/windows/OpenJDK17U-jdk_x64_windows_hotspot_17.0.13_11.zip",
    "https://mirrors.huaweicloud.com/openjdk/17.0.2/openjdk-17.0.2_windows-x64_bin.zip",
    "https://aka.ms/download-jdk/microsoft-jdk-17-windows-x64.zip",
    "https://api.adoptium.net/v3/binary/latest/17/ga/windows/x64/jdk/hotspot/normal/eclipse",
]
CMDLINE_TOOLS = [
    "https://dl.google.com/android/repository/commandlinetools-win-11076708_latest.zip",
    "https://mirrors.cloud.tencent.com/AndroidSDK/commandlinetools-win-11076708_latest.zip",
]
GRADLE_SOURCES = [
    "https://services.gradle.org/distributions/gradle-8.7-bin.zip",
    "https://mirrors.cloud.tencent.com/gradle/gradle-8.7-bin.zip",
    "https://mirrors.aliyun.com/macports/distfiles/gradle/gradle-8.7-bin.zip",
]


def _download(url: str, dest: str, min_size: int = 1_000_000, retries: int = 3) -> bool:
    if os.path.exists(dest) and os.path.getsize(dest) > min_size:
        print(f"  已存在 {os.path.basename(dest)} ({os.path.getsize(dest)/1e6:.1f} MB)")
        return True
    for attempt in range(1, retries + 1):
        try:
            print(f"  下载 {url}")
            req = urllib.request.Request(url, headers=UA)
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=120) as r, open(dest + ".part", "wb") as f:
                total = int(r.headers.get("Content-Length", 0))
                done = 0
                last = time.time()
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
                    done += len(buf)
                    if time.time() - last > 10:
                        last = time.time()
                        pct = f"{done/1e6:.0f}/{total/1e6:.0f} MB" if total else f"{done/1e6:.0f} MB"
                        print(f"    {pct}", flush=True)
            if os.path.getsize(dest + ".part") < min_size:
                raise RuntimeError("文件过小")
            os.replace(dest + ".part", dest)
            print(f"  完成 {os.path.basename(dest)} ({os.path.getsize(dest)/1e6:.1f} MB, "
                  f"{time.time()-t0:.0f}s)")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  失败({attempt}/{retries}) {type(e).__name__}: {str(e)[:80]}")
            if os.path.exists(dest + ".part"):
                os.remove(dest + ".part")
            time.sleep(2)
    return False


def _unzip(src: str, dest: str, strip_top: bool = True) -> str:
    """解压 zip；strip_top=True 时去掉压缩包里的顶层目录。"""
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        top = names[0].split("/")[0] if names else ""
        for n in names:
            if n.endswith("/"):
                continue
            rel = n[len(top) + 1:] if strip_top and top and n.startswith(top + "/") else n
            if not rel:
                continue
            out = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with z.open(n) as s, open(out, "wb") as f:
                shutil.copyfileobj(s, f)
    return dest


def install_jdk() -> str | None:
    print("[1/3] JDK 17")
    java = os.path.join(JDK_DIR, "bin", "java.exe")
    if os.path.exists(java):
        print(f"  已安装: {java}")
        return JDK_DIR
    for url in JDK_SOURCES:
        name = os.path.basename(url.split("?")[0]) or "jdk.zip"
        if not name.endswith(".zip"):
            name = "jdk17.zip"
        dest = os.path.join(DL, name)
        if _download(url, dest, min_size=50_000_000):
            print("  解压 ...")
            _unzip(dest, JDK_DIR, strip_top=True)
            if os.path.exists(java):
                print(f"  JDK 就绪: {JDK_DIR}")
                return JDK_DIR
            print("  解压后未找到 bin/java.exe，换下一个源")
    print("  JDK 安装失败")
    return None


def install_sdk(jdk_dir: str) -> str | None:
    print("[2/3] Android SDK")
    sdkmanager = os.path.join(SDK_DIR, "cmdline-tools", "latest", "bin", "sdkmanager.bat")
    if not os.path.exists(sdkmanager):
        ok = False
        for url in CMDLINE_TOOLS:
            dest = os.path.join(DL, os.path.basename(url))
            if _download(url, dest, min_size=50_000_000):
                print("  解压 ...")
                tmp = os.path.join(SDK_DIR, "cmdline-tools", "latest")
                _unzip(dest, tmp, strip_top=True)
                if os.path.exists(sdkmanager):
                    ok = True
                    break
        if not ok:
            print("  cmdline-tools 安装失败")
            return None

    env = dict(os.environ, JAVA_HOME=jdk_dir, ANDROID_HOME=SDK_DIR,
               PATH=os.path.join(jdk_dir, "bin") + os.pathsep + os.environ.get("PATH", ""))
    pkgs = ["platform-tools", "platforms;android-34", "build-tools;34.0.0"]
    print(f"  安装 SDK 组件: {', '.join(pkgs)}（首次较慢）")
    yes = subprocess.run(f'echo y| "{sdkmanager}" --sdk_root="{SDK_DIR}" --licenses',
                         shell=True, env=env, capture_output=True, text=True)
    print(f"  接受许可: rc={yes.returncode}")
    proc = subprocess.run([sdkmanager, f"--sdk_root={SDK_DIR}", *pkgs],
                          env=env, capture_output=True, text=True)
    tail = (proc.stdout or "")[-800:]
    print(f"  sdkmanager rc={proc.returncode}")
    print("  " + tail.replace("\n", "\n  ")[-700:])
    if proc.returncode != 0:
        print("  stderr: " + (proc.stderr or "")[-500:])
        return None
    print(f"  SDK 就绪: {SDK_DIR}")
    return SDK_DIR


def install_gradle() -> str | None:
    print("[3/3] Gradle 8.7")
    gradle_bin = os.path.join(GRADLE_DIR, "bin", "gradle.bat")
    if os.path.exists(gradle_bin):
        print(f"  已安装: {gradle_bin}")
        return GRADLE_DIR
    for url in GRADLE_SOURCES:
        dest = os.path.join(DL, os.path.basename(url))
        if _download(url, dest, min_size=50_000_000):
            print("  解压 ...")
            _unzip(dest, GRADLE_DIR, strip_top=True)
            if os.path.exists(gradle_bin):
                print(f"  Gradle 就绪: {GRADLE_DIR}")
                return GRADLE_DIR
    print("  Gradle 安装失败")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["jdk", "sdk", "gradle"], default=None)
    args = ap.parse_args()
    os.makedirs(DL, exist_ok=True)

    jdk = install_jdk() if args.only in (None, "jdk") else JDK_DIR
    sdk = install_sdk(jdk) if args.only in (None, "sdk") and jdk else (SDK_DIR if args.only == "gradle" else None)
    gradle = install_gradle() if args.only in (None, "gradle") else None
    print("\n=== 结果 ===")
    print(f"JDK:    {jdk}")
    print(f"SDK:    {sdk}")
    print(f"Gradle: {gradle}")
    return 0 if all([jdk, sdk is not None or args.only != "sdk"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
