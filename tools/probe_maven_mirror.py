"""测试 Google Maven / Maven Central 的可用镜像（Android 构建依赖来源）。"""
from __future__ import annotations

import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# 需要能取到：AGP、AndroidX、CameraX（Google Maven）；ONNX Runtime、Kotlin（Maven Central）
TESTS = [
    ("google", "https://maven.aliyun.com/repository/google/com/android/tools/build/gradle/maven-metadata.xml"),
    ("google", "https://maven.aliyun.com/repository/google/androidx/camera/camera-core/maven-metadata.xml"),
    ("google", "https://mirrors.cloud.tencent.com/nexus/repository/maven-public/com/android/tools/build/gradle/maven-metadata.xml"),
    ("google", "https://maven.google.com/com/android/tools/build/gradle/maven-metadata.xml"),
    ("central", "https://maven.aliyun.com/repository/public/com/microsoft/onnxruntime/onnxruntime-android/maven-metadata.xml"),
    ("central", "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/maven-metadata.xml"),
    ("central", "https://maven.aliyun.com/repository/central/org/jetbrains/kotlin/kotlin-stdlib/maven-metadata.xml"),
    ("gradle-plugin", "https://maven.aliyun.com/repository/gradle-plugin/com/android/tools/build/gradle/maven-metadata.xml"),
]

for kind, url in TESTS:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25) as r:
            body = r.read()
        vers = [v for v in body.decode("utf8", "ignore").split("<version>")[1:]][:1]
        latest = vers[0].split("</version>")[0] if vers else "?"
        print(f"OK   [{kind:13s}] {url[:70]:72s} 样例版本={latest}")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL [{kind:13s}] {url[:70]:72s} {type(e).__name__}: {str(e)[:50]}")
