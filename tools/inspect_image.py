"""Numerically inspect block.jpg without any vision model.

Prints objective pixel statistics: dimensions, global stats, dominant colour
clusters (k-means), and a coarse tile map so a human can reason about layout.
"""
import sys
import numpy as np
from PIL import Image

path = sys.argv[1] if len(sys.argv) > 1 else "block.jpg"
im = Image.open(path).convert("RGB")
a = np.asarray(im)
h, w, _ = a.shape
print(f"path={path}")
print(f"size={w}x{h}  aspect(w/h)={w/h:.4f}")
print(f"mean RGB={a.reshape(-1,3).mean(0).round(2).tolist()}")
print(f"std  RGB={a.reshape(-1,3).std(0).round(2).tolist()}")

# Coarse tile luminance map (8 cols x 16 rows)
rows, cols = 16, 8
th, tw = h // rows, w // cols
lum = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])
print("\nluminance tile map (rows x cols, 0-255):")
for r in range(rows):
    vals = []
    for c in range(cols):
        tile = lum[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
        vals.append(f"{tile.mean():5.0f}")
    print(" ".join(vals))

# Saturation map: raw wood is low saturation, painted blocks are high
mx = a.max(2).astype(np.float32)
mn = a.min(2).astype(np.float32)
sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1) * 255, 0)
print(f"\nmean saturation={sat.mean():.1f}  p90={np.percentile(sat,90):.1f}")

# k-means on subsampled pixels -> dominant colours
import cv2
small = a[::4, ::4].reshape(-1, 3).astype(np.float32)
K = 6
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
_, labels, centers = cv2.kmeans(small, K, None, crit, 3, cv2.KMEANS_PP_CENTERS)
counts = np.bincount(labels.flatten(), minlength=K)
order = np.argsort(-counts)
print("\ndominant colours (k-means K=6), sorted by share:")
for i in order:
    b, g, r = centers[i]
    print(f"  RGB=({r:3.0f},{g:3.0f},{b:3.0f})  share={counts[i]/counts.sum()*100:5.1f}%  "
          f"sat~{max(centers[i])-min(centers[i]):3.0f}")
