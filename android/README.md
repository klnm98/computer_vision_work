# 木块识别 · Android 版

把桌面端的木块检测模型搬到手机上：**CameraX 实时取帧 → ONNX Runtime 推理 → 画框**，
与桌面端使用**同一个模型、同一套后处理与阈值**。

## 直接安装（推荐）

1. 把 `dist/wood_block_detector_debug.apk`（约 70 MB）传到手机
   （微信/QQ 传文件、数据线、网盘都可以）。
2. 手机上点开这个 APK，按提示允许「安装未知应用」（设置 → 应用 → 特殊权限）。
3. 安装后打开「木块识别」，首次进入会请求相机权限，允许即可。

> 系统要求：Android 7.0（API 24）及以上，arm64 或 armeabi-v7a 机型。

## 使用

| 功能 | 说明 |
| --- | --- |
| 实时识别 | 打开即在摄像头画面上实时框出木块，左上角显示检测数量、右上角显示帧率 |
| 切换镜头 | 前后摄像头切换 |
| 相册选图 | 对相册里的照片做一次检测（适合验证模型在你拍的照片上是否有效） |
| 示例图片 | 用内置的 `block.jpg` 快速验证识别功能是否正常 |
| 置信度阈值 | 默认 0.35（在 BOP 官方负样本上标定，误检率 0.85%）。**误框了别的东西就调高**（如 0.5）；木块漏检就调低（如 0.25） |

与桌面端一致，实时模式带 **N-of-M 时序确认**（连续 3 帧命中同一位置才画框），
偶发的单帧误检不会显示出来。

## 性能预期

手机端是 **CPU 推理**（YOLO11s，640×640）：

- 预览画面始终流畅（相机预览与推理分离，只处理最新一帧）；
- 检测框刷新约 **每秒 2~5 次**（取决于手机 CPU），对"把木块放到镜头前看框"这类用法足够；
- 想更快可以改用更小的模型（见下文重新训练/导出）。

## 与桌面端的一致性验证

Android 端的后处理逻辑抽成了纯 Kotlin 的 `YoloPostProcess`，并用 JVM 单元测试
（`app/src/test/java/.../YoloPostProcessTest.kt`）与桌面端做了交叉验证：

| 测试 | 内容 |
| --- | --- |
| letterbox 参数 | scale=0.269360、padX=174、padY=0 与 Python 一致 |
| 输出布局 | 模型输出为 (1, 5, 8400)，符合单类别 YOLO11 检测头 |
| **端到端** | 桌面端基准：conf=**0.8302**、框 (0,209,1080,1586)；Android 端得到同一结果，**IoU > 0.9** |
| 阈值行为 | 阈值提到 0.95 时不再输出框 |
| 形态过滤 | 500×10 的狭长框被正确丢弃 |

跑测试：

```bash
python tools/build_apk.py --test
```

## 重新构建 APK

工具链（JDK 17 / Android SDK / Gradle）会装在项目的 `.android-build/` 下，不动系统环境：

```bash
python tools/setup_android_toolchain.py        # JDK + Gradle + cmdline-tools
python tools/install_android_sdk_direct.py     # platform-tools / build-tools 34 / android-34
python tools/build_apk.py --test               # 跑单测并打包，产物在 dist/
python tools/build_apk.py --release            # 需要签名时用（见下）
```

模型来自 `android/app/src/main/assets/wood_block_yolo11s.onnx`，由桌面端权重导出：

```bash
python tools/export_onnx.py            # models/wood_block_yolo11s.pt -> ONNX，并校验一致性
```

## 换成更小/更快的模型

若想进一步提升手机端帧率，可以用 `yolo11n` 重新训练（桌面端 `main.py train --base
yolo11n.pt`），再导出 ONNX 覆盖 assets 里的文件即可；APK 会自动带上新模型。

## 已知限制

- 目前是 **debug 签名**的 APK：可正常安装使用，但更新时需卸载重装（若换签名）。
  如需正式签名，生成 keystore 后在 `app/build.gradle.kts` 里配置 `signingConfigs` 再
  `python tools/build_apk.py --release`。
- 只打包了 `arm64-v8a` 与 `armeabi-v7a`（覆盖几乎所有手机）；如需 x86 模拟器，
  在 `app/build.gradle.kts` 的 `abiFilters` 里加上即可。
- 近景（木块占满画面）情况与桌面端一样偏弱：桌面端可用多尺度推理缓解，
  手机端为控制耗时未开启。
