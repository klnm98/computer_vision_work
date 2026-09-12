# tools/ 开发与调试脚本

这些脚本不是运行系统所必需的，用于数据准备、环境搭建与结果核查。

## 数据准备与检查
| 脚本 | 用途 |
| --- | --- |
| `dataset_cli.py` | 数据集命令行：`scan`（扫描哪些场景含木块）/ `fetch`（拉取单个场景）/ `build`（构建 YOLO 数据集） |
| `check_labels.py` | 把 YOLO 标签画回图片并拼图输出，人工核对标注是否正确 |
| `check_chipped.py` | 可视化"缺角增强"效果（左：原图，右：敲掉一角） |
| `check_block.py` | 在 `block.jpg` 上可视化检测结果（可指定权重与阈值） |
| `make_test_video.py` | 用 `block.jpg` 合成一段测试视频（前半只有桌面、后半含木块），用于验证摄像头管线 |
| `verify_video.py` | 在测试视频上量化验证：无木块段误检率、有木块段命中率、FPS（对比开/关时序确认） |
| `inspect_image.py` | 纯数值方式分析图片（尺寸、色彩聚类、亮度图），不依赖任何模型 |

## 摄像头排查与验证
| 脚本 | 用途 |
| --- | --- |
| `python detect_camera.py --probe` | 摄像头自检：枚举 设备号 × 后端 × 分辨率，打印有效帧/亮度/抖动/评分，并给出建议命令（不需要本脚本） |
| `check_camera_open.py` | 验证 `open_source` 选到的组合能连续出图（打印亮度/标准差，非黑屏） |
| `test_camera_recovery.py` | 用桩对象模拟"选到黑屏组合"，验证运行中自动重新选择摄像头的逻辑不会卡死 |

## 远程数据访问
| 脚本 | 用途 |
| --- | --- |
| `remote_zip_entry.py` | 用 HTTP Range 读取远程 ZIP 里的**单个文件**（YCB-Video 场景包 1~3 GB，标注只有 160 字节） |
| `remote_zip_ls.py` / `remote_zip_list.py` | 列出远程 ZIP 的目录结构、条目数量与压缩后大小 |
| `head_sizes.py` | 探测远程文件大小（部分镜像不支持 HEAD，可用 Range GET 代替） |
| `fetch.py` | 可断点续传 + 自动重试的大文件下载器 |

## 环境搭建（受限环境）
| 脚本 | 用途 |
| --- | --- |
| `pip_workaround.py` | 在"只允许写工作区、且禁止往 0700 临时目录写文件"的沙箱里正常调用 pip |
| `get_wheels_sjtu.py` | 从镜像直接下载 PyTorch CUDA wheel（比 pip 更可控、可断点续传） |

> 说明：正式流程里这些能力已经收敛进 `woodblock/` 包
> （`remote_zip.py`、`dataset.py`、`env_setup.py`、`compat.py`），
> `tools/` 下的脚本便于单独调试与排查。
