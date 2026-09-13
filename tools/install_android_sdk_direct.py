"""直接下载 Android SDK 组件（不依赖 sdkmanager / Java 网络栈），并处理 JDK 信任库。

受限网络下 sdkmanager 会因为 Java 自带 cacerts 不信任拦截证书而失败，
因此这里：
  1. 把系统信任的根证书导入我们自己的 JDK（.android-build/jdk，不动系统 JDK）；
  2. 用 Python（TLS 正常）直接下载 platform-tools / build-tools / platforms 的官方 zip；
  3. 写入 licenses 目录，避免 Gradle 因未接受许可而报错。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from fetch import download  # noqa: E402  (可断点续传的下载器)

BASE = os.path.join(ROOT, ".android-build")
JDK = os.path.join(BASE, "jdk")
SDK = os.path.join(BASE, "sdk")
DL = os.path.join(BASE, "dl")

REPO = "https://dl.google.com/android/repository"
COMPONENTS = [
    ("platform-tools", f"{REPO}/platform-tools-latest-windows.zip", "platform-tools"),
    ("build-tools;34.0.0", f"{REPO}/build-tools_r34-windows.zip", os.path.join("build-tools", "34.0.0")),
    # 官方仓库里 API 34 的平台包名就是 ext7 这一版（platform-34_r0x.zip 已不再提供）
    ("platforms;android-34", f"{REPO}/platform-34-ext7_r03.zip", os.path.join("platforms", "android-34")),
]

LICENSE_HASHES = {
    "android-sdk-license": ["24333f8a63b6825ea9c5514f83c2829b004d1fee",
                            "d56f5187479451eabf01fb78af6dfcb131a6481e",
                            "8933bad161af4178b1185d1a37fbf41ea5269c55"],
    "android-sdk-preview-license": ["84831b9409646a918e30573bab4c9c91346d8abd"],
    "android-sdk-arm-dbt-license": ["859f317696f67ef3d7f30a50a5560e7834b43903"],
}


def import_ca_into_jdk() -> bool:
    """把系统根证书导入本工具链 JDK 的 cacerts（让 Gradle/sdkmanager 能正常 HTTPS）。"""
    cacerts = os.path.join(JDK, "lib", "security", "cacerts")
    keytool = os.path.join(JDK, "bin", "keytool.exe")
    if not (os.path.exists(cacerts) and os.path.exists(keytool)):
        print(f"[CA] 找不到 JDK 信任库或 keytool: {cacerts}")
        return False

    import base64
    import ssl

    pem = os.path.join(BASE, "system-ca.pem")
    certs = ssl.create_default_context().get_ca_certs(binary_form=True)
    with open(pem, "wb") as f:
        for der in certs:
            b64 = base64.encodebytes(der).decode("ascii")
            f.write(b"-----BEGIN CERTIFICATE-----\n")
            for i in range(0, len(b64), 64):
                f.write(b64[i:i + 64].encode("ascii"))
            f.write(b"-----END CERTIFICATE-----\n")
    print(f"[CA] 导出 {len(certs)} 张系统根证书 -> {pem}")

    # 逐张导入，已存在则忽略
    imported = 0
    for idx, der in enumerate(certs):
        alias = f"sysroot{idx}"
        one = os.path.join(BASE, f"ca-{idx}.pem")
        b64 = base64.encodebytes(der).decode("ascii")
        with open(one, "w", encoding="ascii") as f:
            f.write("-----BEGIN CERTIFICATE-----\n")
            for i in range(0, len(b64), 64):
                f.write(b64[i:i + 64] + "\n")
            f.write("-----END CERTIFICATE-----\n")
        rc = subprocess.run([keytool, "-importcert", "-noprompt", "-trustcacerts",
                             "-alias", alias, "-file", one, "-keystore", cacerts,
                             "-storepass", "changeit"],
                            capture_output=True, text=True)
        if rc.returncode == 0:
            imported += 1
        os.remove(one)
    print(f"[CA] 导入完成：新增 {imported} 张（其余为已存在）")
    return True


def install_components() -> bool:
    os.makedirs(SDK, exist_ok=True)
    ok = True
    for name, url, rel in COMPONENTS:
        dest_dir = os.path.join(SDK, rel)
        marker = os.path.join(dest_dir, ".installed")
        if os.path.exists(marker):
            print(f"[SDK] {name} 已安装")
            continue
        zip_path = os.path.join(DL, os.path.basename(url))
        print(f"[SDK] 下载 {name} ...")
        try:
            download(url, zip_path)
        except Exception as e:  # noqa: BLE001
            print(f"[SDK] 下载失败 {name}: {e}")
            ok = False
            continue
        print(f"[SDK] 解压 {name} -> {dest_dir}")
        os.makedirs(dest_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            top = names[0].split("/")[0] if names else ""
            for n in names:
                if n.endswith("/"):
                    continue
                rel_name = n[len(top) + 1:] if top and n.startswith(top + "/") else n
                if not rel_name:
                    continue
                out = os.path.join(dest_dir, rel_name)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with z.open(n) as s, open(out, "wb") as f:
                    shutil.copyfileobj(s, f)
        open(marker, "w").close()
    # licenses
    lic_dir = os.path.join(SDK, "licenses")
    os.makedirs(lic_dir, exist_ok=True)
    for fname, hashes in LICENSE_HASHES.items():
        with open(os.path.join(lic_dir, fname), "w", encoding="utf8") as f:
            f.write("\n".join(hashes) + "\n")
        print(f"[SDK] 写入许可 {fname}")
    # 版本标记文件（Gradle 需要 build-tools 的 source.properties，zip 里通常自带）
    for rel in ("build-tools/34.0.0", "platforms/android-34"):
        sp = os.path.join(SDK, rel, "source.properties")
        print(f"[SDK] {rel}: source.properties {'存在' if os.path.exists(sp) else '缺失!'}")
    return ok


def main() -> int:
    import_ca_into_jdk()
    ok = install_components()
    print("\n=== SDK 目录 ===")
    for rel in ("platform-tools", "build-tools/34.0.0", "platforms/android-34"):
        p = os.path.join(SDK, rel)
        print(f"  {rel:24s} {'OK' if os.path.isdir(p) else '缺失'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
