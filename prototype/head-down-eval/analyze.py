#!/usr/bin/env python3
"""用人工标注（up/down）评估各候选指标的区分力 + 阈值扫描。

用法: python analyze.py --csv /tmp/hd/S2b/persons.csv --labels labels-s2-provisional.csv
"""
import argparse

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--metrics", default="vert_ratio,neck_ratio,lean_deg,nose_ear_dy,eye_ear_dy")
a = ap.parse_args()

df = pd.read_csv(a.csv).merge(pd.read_csv(a.labels), on=["frame", "person"], how="inner")
df = df[df.label.isin(["up", "down"])].copy()
df["y"] = (df.label == "down").astype(int)
print(f"labeled persons: {len(df)}  down={int(df.y.sum())}  up={int((1 - df.y).sum())}\n")


def auc(scores, y):
    """rank-based AUC，score 越大越像 down。"""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    y = np.asarray(y)
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


rows = []
for m in a.metrics.split(","):
    if m not in df:
        continue
    s = df[m].astype(float)
    s = s.fillna(s.median())
    for direction, sc in (("↑down", s), ("↓down", -s)):
        A = auc(sc.values, df.y.values)
        rows.append({"metric": m, "direction": direction, "auc": round(A, 3)})
print(pd.DataFrame(rows).sort_values("auc", ascending=False).to_string(index=False))
print()

# 阈值扫描（只看我们预期方向：vert_ratio / neck_ratio 越小越可能低头；nose_ear_dy 越大越可能低头）
sweeps = [("vert_ratio", "<"), ("neck_ratio", "<"), ("nose_ear_dy", ">")]
for m, dirn in sweeps:
    if m not in df:
        continue
    print(f"--- {m} ({'低于' if dirn == '<' else '高于'}阈值判为 down) ---")
    vals = np.round(sorted(df[m].dropna().unique()), 3)
    out = []
    for t in vals:
        pred = (df[m] < t) if dirn == "<" else (df[m] > t)
        tp = int((pred & (df.y == 1)).sum())
        fp = int((pred & (df.y == 0)).sum())
        fn = int((~pred & (df.y == 1)).sum())
        rec = tp / max(tp + fn, 1)
        prec = tp / max(tp + fp, 1)
        out.append({"thr": t, "TP": tp, "FP": fp, "FN": fn,
                    "recall": round(rec, 2), "precision": round(prec, 2)})
    print(pd.DataFrame(out).to_string(index=False))
    print()

print("--- 各指标分组统计（down vs up）---")
print(df.groupby("label")[["vert_ratio", "neck_ratio", "lean_deg", "nose_ear_dy", "eye_ear_dy"]]
      .agg(["min", "median", "max"]).round(3).to_string())
