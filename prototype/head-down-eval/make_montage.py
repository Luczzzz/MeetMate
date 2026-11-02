#!/usr/bin/env python3
"""把人按 bbox 裁成小图拼成表格，方便肉眼标注 head_down / head_up / 不可判定。

用法: python make_montage.py --img ../../../真实场景2.jpg --csv out/persons.csv --out out/montage.jpg
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--img", required=True)
ap.add_argument("--csv", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cols", type=int, default=6)
ap.add_argument("--cell-w", type=int, default=260)
ap.add_argument("--cell-h", type=int, default=330)
ap.add_argument("--head-crop", type=float, default=1.0, help="只裁 bbox 上部比例，用于细看头部")
a = ap.parse_args()

img = cv2.imread(a.img)
df = pd.read_csv(a.csv)
n = len(df)
rows = int(np.ceil(n / a.cols))
canvas = np.full((rows * a.cell_h, a.cols * a.cell_w, 3), 30, np.uint8)

for i, r in df.reset_index(drop=True).iterrows():
    pad = int((r.y2 - r.y1) * 0.35) + 12
    y2c = r.y1 + (r.y2 - r.y1) * a.head_crop if a.head_crop < 1 else r.y2
    pad = pad if a.head_crop >= 1 else int((r.y2 - r.y1) * 0.12) + 10
    x1, y1 = max(0, int(r.x1) - pad), max(0, int(r.y1) - pad)
    x2, y2 = min(img.shape[1], int(r.x2) + pad), min(img.shape[0], int(y2c) + pad)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        continue
    s = min(a.cell_w / crop.shape[1], (a.cell_h - 42) / crop.shape[0])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))
    bx = [int(v) for v in ((r.x1 - x1) * s, (r.y1 - y1) * s, (r.x2 - x1) * s, (r.y2 - y1) * s)]
    cv2.rectangle(crop, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)
    cx, cy = (i % a.cols) * a.cell_w, (i // a.cols) * a.cell_h
    canvas[cy + 42:cy + 42 + crop.shape[0], cx:cx + crop.shape[1]] = crop
    txt = f"#{int(r.person)} v={r.vert_ratio:.2f} a={r.lean_deg:.0f} n={r.neck_ratio:.2f}"
    cv2.putText(canvas, txt, (cx + 4, cy + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
    cv2.rectangle(canvas, (cx, cy), (cx + a.cell_w - 1, cy + a.cell_h - 1), (80, 80, 80), 1)

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(a.out, canvas)
print(a.out, canvas.shape)
