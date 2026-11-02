#!/usr/bin/env python3
"""按已确认的 Distraction 判定规则，把结果画在图上（给人看效果用）。

规则（#6 结论）：
  nose_ear_dy > NOSE_THR                      -> 低头
  否则 vert_ratio < VERT_THR                  -> 低头
  否则                                        -> 抬头
  任一指标缺值 -> 不可判

用法:
  python render_verdict.py --img ../../真实场景2.jpg --csv out/s2/persons.csv \
      --labels labels-s2-provisional.csv --out out/s2/verdict.jpg
"""
import argparse

import cv2
import numpy as np
import pandas as pd

DOWN, UP, UNK = 0, 1, 2
COLORS = {DOWN: (0, 0, 235), UP: (0, 175, 0), UNK: (140, 140, 140)}
ZH = {DOWN: "低头", UP: "抬头", UNK: "不可判"}

ap = argparse.ArgumentParser()
ap.add_argument("--img", required=True)
ap.add_argument("--csv", required=True)
ap.add_argument("--labels", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--nose-thr", type=float, default=0.124)
ap.add_argument("--vert-thr", type=float, default=0.462)
ap.add_argument("--width", type=int, default=2000, help="输出图宽度")
ap.add_argument("--thickness", type=int, default=4)
ap.add_argument("--font-scale", type=float, default=1.0)
ap.add_argument("--head-crop", type=float, default=1.0)
ap.add_argument("--montage", default=None, help="同时输出人头特写拼表")
ap.add_argument("--verbose", action="store_true", help="逐人显示指标数值（默认只标低头）")
a = ap.parse_args()


def verdict(r):
    nd, vr = r.get("nose_ear_dy"), r.get("vert_ratio")
    if pd.notna(nd) and nd > a.nose_thr:
        return DOWN
    if pd.notna(vr) and vr < a.vert_thr:
        return DOWN
    if pd.isna(nd) and pd.isna(vr):
        return UNK
    return UP


img = cv2.imread(a.img)
H, W = img.shape[:2]
df = pd.read_csv(a.csv)
df["verdict"] = df.apply(verdict, axis=1)
if a.labels:
    lab = pd.read_csv(a.labels)
    df = df.merge(lab, on=["frame", "person"], how="left")
    gt = {"down": DOWN, "up": UP}
    df["gt_v"] = df.label.map(gt)
    df["hit"] = np.where(df.gt_v.isna(), "", np.where(df.gt_v == df.verdict, "OK", "MISS"))

s = a.width / W
base = cv2.resize(img, (a.width, int(H * s)))
th = max(2, int(a.thickness * s * 2))

for _, r in df.iterrows():
    x1, y1, x2, y2 = [int(v * s) for v in (r.x1, r.y1, r.x2, r.y2)]
    v = int(r.verdict)
    c = COLORS[v]
    if not a.verbose and v != DOWN:
        # 默认视图：只把"低头"标出来，抬头只留细框，避免文字糊成一片
        cv2.rectangle(base, (x1, y1), (x2, y2), c, max(1, th // 2))
        continue
    cv2.rectangle(base, (x1, y1), (x2, y2), c, th)
    fs = a.font_scale * max(0.55, a.width / 2000.0)
    txt = ZH[v]
    if a.verbose:
        if pd.notna(r.nose_ear_dy):
            txt += f" n={r.nose_ear_dy:.2f}"
        if pd.notna(r.vert_ratio):
            txt += f" v={r.vert_ratio:.2f}"
        if r.get("hit"):
            txt += f" [{r.get('hit')}]"
    (tw, tl), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, th // 2))
    ty = max(y1 - 6, tl + 6)
    cv2.rectangle(base, (x1 - 2, ty - tl - 6), (x1 + tw + 8, ty + 6), c, -1)
    cv2.putText(base, txt, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, th // 2))

n = {v: int((df.verdict == v).sum()) for v in (DOWN, UP, UNK)}
legend = (f"rule: nose_ear_dy>{a.nose_thr} -> DOWN | vert_ratio<{a.vert_thr} -> DOWN   "
          f"|  低头 {n[DOWN]}  抬头 {n[UP]}  不可判 {n[UNK]}  (共 {len(df)} 人)")
bar = np.full((46, base.shape[1], 3), 24, np.uint8)
cv2.putText(bar, legend, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (235, 235, 235), 2)
out = np.vstack([base, bar])
cv2.imwrite(a.out, out, [cv2.IMWRITE_JPEG_QUALITY, 88])
print(a.out, out.shape, f"down={n[DOWN]} up={n[UP]} unk={n[UNK]}")
if a.labels:
    print(df[df.hit != ""].hit.value_counts().to_string(), "\n",
          df[df.hit == "MISS"][["person", "label", "verdict", "nose_ear_dy", "vert_ratio"]].to_string(index=False))

# 人头特写拼表（同样带裁决）
if a.montage:
    cols, cw, ch = 6, 300, 330
    rows = int(np.ceil(len(df) / cols))
    canvas = np.full((rows * ch, cols * cw, 3), 30, np.uint8)
    for i, (_, r) in enumerate(df.iterrows()):
        pad = int((r.y2 - r.y1) * 0.12) + 10
        y2c = r.y1 + (r.y2 - r.y1) * a.head_crop if a.head_crop < 1 else r.y2
        x1c, y1c = max(0, int(r.x1) - pad), max(0, int(r.y1) - pad)
        x2c, y2c = min(W, int(r.x2) + pad), min(H, int(y2c) + pad)
        crop = img[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            continue
        k = min(cw / crop.shape[1], (ch - 46) / crop.shape[0])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * k)), max(1, int(crop.shape[0] * k))))
        bx = [int(v) for v in ((r.x1 - x1c) * k, (r.y1 - y1c) * k, (r.x2 - x1c) * k, (r.y2 - y1c) * k)]
        cv2.rectangle(crop, (bx[0], bx[1]), (bx[2], bx[3]), COLORS[int(r.verdict)], 2)
        cx, cy = (i % cols) * cw, (i // cols) * ch
        canvas[cy + 46:cy + 46 + crop.shape[0], cx:cx + crop.shape[1]] = crop
        txt = f"#{int(r.person)} {ZH[int(r.verdict)]}"
        if r.get("hit"):
            txt += f" [{r.hit}]"
        cv2.putText(canvas, txt, (cx + 4, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    COLORS[int(r.verdict)], 2)
        cv2.rectangle(canvas, (cx, cy), (cx + cw - 1, cy + ch - 1), (80, 80, 80), 1)
    cv2.imwrite(a.montage, canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(a.montage, canvas.shape)
