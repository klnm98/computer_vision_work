"""一键构建 Android APK（自动设置 JDK / SDK / Gradle 环境变量）。

工具链装在 .android-build/ 下（见 tools/setup_android_toolchain.py 与
tools/install_android_sdk_direct.py），不污染系统环境。

用法:
    python tools/build_apk.py              # 构建 debug APK（可直接安装）
    python tools/build_apk.py --release    # 构建 release APK（用自动生成的签名）
    python tools/build_apk.py --test       # 先跑单元测试再打包
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, ".android-build")
JDK = os.path.join(BASE, "jdk")
SDK = os.path.join(BASE, "sdk")
GRADLE = os.path.join(BASE, "gradle", "bin", "gradle.bat")
ANDROID_HOME = os.path.join(BASE, "android-home")
GRADLE_HOME = os.path.join(BASE, "gradle-home")
PROJECT = os.path.join(ROOT, "android")
DIST = os.path.join(ROOT, "dist")


def build_env() -> dict:
    env = dict(os.environ)
    env["JAVA_HOME"] = JDK
    env["ANDROID_HOME"] = SDK
    env["ANDROID_SDK_ROOT"] = SDK
    env["ANDROID_USER_HOME"] = ANDROID_HOME      # 否则 AGP 会去写 C:\Users\<user>\.android
    env["GRADLE_USER_HOME"] = GRADLE_HOME        # 缓存也放工作区内
    env["PATH"] = os.path.join(JDK, "bin") + os.pathsep + env.get("PATH", "")
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", action="store_true", help="构建 release 版")
    ap.add_argument("--test", action="store_true", help="构建前先跑单元测试")
    ap.add_argument("--clean", action="store_true", help="先 clean")
    args = ap.parse_args()

    for path, name in ((JDK, "JDK"), (SDK, "Android SDK"), (GRADLE, "Gradle")):
        if not os.path.exists(path):
            print(f"缺少 {name}: {path}")
            print("请先运行: python tools/setup_android_toolchain.py 与 "
                  "python tools/install_android_sdk_direct.py")
            return 2
    os.makedirs(ANDROID_HOME, exist_ok=True)
    os.makedirs(GRADLE_HOME, exist_ok=True)

    tasks = []
    if args.clean:
        tasks.append("clean")
    if args.test:
        tasks.append(":app:testDebugUnitTest")
    tasks.append(":app:assembleRelease" if args.release else ":app:assembleDebug")

    cmd = [GRADLE, "-p", PROJECT, *tasks, "--no-daemon", "--console=plain"]
    print("[构建]", " ".join(tasks))
    rc = subprocess.run(cmd, env=build_env(), cwd=ROOT).returncode
    if rc != 0:
        print(f"[构建] 失败（退出码 {rc}）")
        return rc

    out_dir = os.path.join(PROJECT, "app", "build", "outputs", "apk")
    apks = []
    for root, _dirs, files in os.walk(out_dir):
        apks += [os.path.join(root, f) for f in files if f.endswith(".apk")]
    if not apks:
        print("[构建] 没有找到 APK 产物")
        return 1
    os.makedirs(DIST, exist_ok=True)
    for apk in apks:
        name = "wood_block_detector" + ("_release" if args.release else "_debug") + ".apk"
        dst = os.path.join(DIST, name)
        shutil.copyfile(apk, dst)
        print(f"[构建] APK: {dst}  ({os.path.getsize(dst)/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
