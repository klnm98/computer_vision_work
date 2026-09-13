# v1.0.0 · 木块识别（桌面端 + Android APK）

## ⬇️ 直接下载 APK

**➡️ [点击下载 wood_block_detector_debug.apk（70.4 MB）](https://github.com/klnm98/computer_vision_work/releases/download/v1.0.0/wood_block_detector_debug.apk)**

> 链接点不动的话，往下滚到本页**最底部的 `Assets` 折叠框**，
> 那里有同一个 `wood_block_detector_debug.apk`（GitHub 的发布附件总是显示在说明正文下方）。

安装：下载后点开 APK → 按提示允许「安装未知应用」→ 打开「木块识别」→ 允许相机权限。

用**真实、公开数据集**训练的单类别木块检测器：桌面端摄像头实时识别 + 手机端 APK。

## 下载与安装（手机）

1. 下载 **`wood_block_detector_debug.apk`**（70.4 MB）
2. 传到手机后点击安装，按提示允许「安装未知应用」
3. 打开「木块识别」，允许相机权限即可

* 系统要求：Android 7.0（API 24）及以上；已包含 `arm64-v8a` 与 `armeabi-v7a`
* APK SHA256：`e19e51e43894e47132a762ac04e339207ea5d0a56d5b3e1c6e33095f13c8721f`

App 内功能：实时摄像头识别（带 N-of-M 时序确认）、相册选图检测、内置示例图、
置信度阈值调节（误框别的东西就调高，漏检就调低）。

## 实测指标

| 项目 | 结果 |
| --- | --- |
| 验证集（4 个留出场景，612 张） | mAP@50 **0.828**，mAP@50-95 0.442，P 0.895 |
| BOP 官方测试集 · 不含木块的 825 张 | 误检图像率 **0.85%**（其中框到其他物体仅 5 个框） |
| BOP 官方测试集 · 含木块的 75 张 | 召回 **73.3%**，平均 IoU 0.741，多余框 0 个 |
| `block.jpg`（真实手机截图） | 置信度 **0.83**，只输出 1 个框 |
| 缺角鲁棒性 | 敲掉一角后置信度 0.833 → **0.830**，仍检出 |
| 遮挡鲁棒性 | 随机挡住 20%~45%，**5/5 次**仍检出 |
| 桌面端速度 | GPU 实时 **30~52 FPS**；手机端 CPU 检测框约 2~5 次/秒（预览始终流畅） |

阈值取舍表（误检率 ↔ 召回率）见仓库 `reports/evaluation.md`。

## 数据来源（真实、公开）

* **YCB-Video Dataset**（YCB Benchmark / Yale GRAB Lab，真实拍摄 92 个场景）：
  扫描 92 个场景，其中 **15 个含 036_wood_block**，全部用于训练/验证
* **BOP YCB-V**（BOP Challenge 官方整理版，MIT）：官方测试场景 0048-0059 作独立测试，
  与训练场景**完全不重叠**
* 不使用自采数据、不使用合成数据

## 仓库内容

| 目录/文件 | 说明 |
| --- | --- |
| `main.py` | 主进程：`setup / prepare / train / eval / camera / image / all` |
| `detect_camera.py` | 摄像头识别；启动时**自动排查并选择可用摄像头**（设备号 × 后端 × 分辨率打分选优），运行中掉线/黑屏自动恢复 |
| `woodblock/` | 数据集构建（含缺角/遮挡/近景增强）、训练、评估、推理 |
| `android/` | Android 工程（CameraX + ONNX Runtime），含与桌面端一致性的 JVM 单测 |
| `models/wood_block_yolo11s.pt` | 上线模型（18.3 MB），clone 后可直接使用 |
| `reports/` | 评估报告、阈值表、缺角/遮挡/近景增强核对图、视频验证结果 |
| `tools/` | 数据准备、调试、Android 工具链与一键打包脚本 |

## 从源码构建 APK

```bash
python tools/setup_android_toolchain.py     # JDK 17 + Gradle + cmdline-tools（装在 .android-build/）
python tools/install_android_sdk_direct.py  # platform-tools / build-tools 34 / android-34
python tools/build_apk.py --test            # 跑单测并打包，产物在本地 dist/（构建产物不入库）
```

## 已知限制

* APK 为 **debug 签名**：可直接安装使用；若日后要正式签名发布，配置 keystore 后
  `python tools/build_apk.py --release`
* **近景（木块占满画面）**偏弱：桌面端可用 `--scales 1.0,0.7,0.45` 缓解（代价是误检升高），
  手机端为控制耗时未开启
* `block.jpg` 这类「木块与木桌同色」的手机截图上，框会略松（含周围桌面），
  但不会框到其他物体
* 极端遮挡/掠射角（如 BOP 场景 000055 中被瓶子挡住）召回偏低，整体 73.3% 的漏检主要来自这里
