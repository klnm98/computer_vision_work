"""分析训练结果曲线，找出真正最优的 epoch 与最终指标。

用法: python tools/analyze_training.py [runs/<name>]
"""
import csv
import os
import sys

run_name = sys.argv[1] if len(sys.argv) > 1 else "runs/wood_block"
path = os.path.join(run_name, "results.csv")
rows = list(csv.DictReader(open(path, encoding="utf8")))
print("epochs:", len(rows))
best = None
for r in rows:
    e = int(r["epoch"])
    p = float(r["metrics/precision(B)"])
    rc = float(r["metrics/recall(B)"])
    m50 = float(r["metrics/mAP50(B)"])
    m = float(r["metrics/mAP50-95(B)"])
    fit = 0.1 * m50 + 0.9 * m
    if best is None or fit > best[0]:
        best = (fit, e, m50, m, p, rc)
print("best fitness: epoch=%d mAP50=%.4f mAP50-95=%.4f P=%.3f R=%.3f" % best[1:])
print("\nprogress (every 5th epoch):")
for r in rows[::5]:
    e = int(r["epoch"])
    m50 = float(r["metrics/mAP50(B)"])
    m = float(r["metrics/mAP50-95(B)"])
    print(f"  epoch {e:3d}  mAP50={m50:.4f}  mAP50-95={m:.4f}  fit={0.1*m50+0.9*m:.4f}")
last = rows[-1]
print(f"\nlast epoch {last['epoch']}: mAP50={float(last['metrics/mAP50(B)']):.4f} "
      f"mAP50-95={float(last['metrics/mAP50-95(B)']):.4f} "
      f"P={float(last['metrics/precision(B)']):.4f} R={float(last['metrics/recall(B)']):.4f}")
print("total train time: %.2f h" % (float(last["time"]) / 3600))
