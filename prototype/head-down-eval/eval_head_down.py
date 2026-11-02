#!/usr/bin/env python3
"""低头（Distraction）判定原型评估脚本 —— 粗糙版，只为拿数字和踩坑。

跑 YOLOv8-pose，对每个 person 计算若干候选"低头"指标，输出 CSV/JSON + 标注图，
用于讨论：在真实会议室视角/分辨率下，肩颈-头部关键点方案到底work不work。

用法:
  python eval_head_down.py --source ../../../真实场景2.jpg --imgsz 1280 --out out/
  python eval_head_down.py --source some_video.mp4 --imgsz 960 --stride 5 --out out/vid/
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

# COCO-17 关键点索引
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHO, R_SHO = 0, 1, 2, 3, 4, 5, 6
HEAD_PTS = [NOSE, L_EYE, R_EYE, L_EAR, R_EAR]
KP_CONF = 0.3  # 单点可用阈值


def _pt(kp, i, thr):
    if kp[i][2] < thr:
        return None
    return kp[i][:2].astype(np.float64)


def person_metrics(kp, box, kp_thr=KP_CONF):
    """返回该 person 的候选低头指标；数据不足返回 None。"""
    sho_l, sho_r = _pt(kp, L_SHO, kp_thr), _pt(kp, R_SHO, kp_thr)
    head = [(i, _pt(kp, i, kp_thr)) for i in HEAD_PTS]
    head = [(i, p) for i, p in head if p is not None]
    if sho_l is None or sho_r is None or not head:
        return None

    sho_mid = (sho_l + sho_r) / 2.0
    w = float(np.linalg.norm(sho_r - sho_l))            # 肩宽，作为尺度归一化基准
    if w < 8:
        return None

    # 头部中心：可用头部关键点按置信度加权
    ws = np.array([kp[i][2] for i, _ in head])
    pts = np.stack([p for _, p in head])
    head_c = (pts * ws[:, None]).sum(0) / ws.sum()

    v = head_c - sho_mid                                  # 肩中点 -> 头部中心
    # 1) 倾斜角：与图像"上"方向的夹角（侧视时≈pitch，正视时可能不明显）
    lean_deg = float(np.degrees(np.arctan2(abs(v[0]), -v[1])))
    # 2) 颈长比：肩中点到头中心距离 / 肩宽（低头时塌缩）
    neck_ratio = float(np.linalg.norm(v) / w)
    # 3) 竖直颈长比：只用 y 分量（更稳健，不受侧倾影响）
    vert_ratio = float((sho_mid[1] - head_c[1]) / w)
    # 4) 有符号倾斜（>0 头偏画面右侧）
    lean_signed = float(np.degrees(np.arctan2(v[0], -v[1])))

    nose = _pt(kp, NOSE, kp_thr)
    # 正视机位下更靠得住的信号：低头时鼻子相对耳线下降、眼-耳竖直间距塌缩
    ears = [p for i, p in ((L_EAR, _pt(kp, L_EAR, kp_thr)), (R_EAR, _pt(kp, R_EAR, kp_thr))) if p is not None]
    eyes = [p for i, p in ((L_EYE, _pt(kp, L_EYE, kp_thr)), (R_EYE, _pt(kp, R_EYE, kp_thr))) if p is not None]
    ear_mid = np.mean(ears, axis=0) if ears else None
    eye_mid = np.mean(eyes, axis=0) if eyes else None
    nose_ear_dy = round(float((nose[1] - ear_mid[1]) / w), 3) if (nose is not None and ear_mid is not None) else None
    eye_ear_dy = round(float((eye_mid[1] - ear_mid[1]) / w), 3) if (eye_mid is not None and ear_mid is not None) else None
    return {
        "nose_ear_dy": nose_ear_dy,
        "eye_ear_dy": eye_ear_dy,
        "n_eye_pts": len(eyes),
        "n_ear_pts": len(ears),
        "lean_deg": round(lean_deg, 1),
        "lean_signed": round(lean_signed, 1),
        "neck_ratio": round(neck_ratio, 3),
        "vert_ratio": round(vert_ratio, 3),
        "n_head_pts": len(head),
        "has_nose": int(nose is not None),
        "nose_conf": round(float(kp[NOSE][2]), 2),
        "shoulder_w_px": round(w, 1),
        "box_h_px": round(float(box[3] - box[1]), 1),
        "box_w_px": round(float(box[2] - box[0]), 1),
        "head_c": head_c,
        "sho_mid": sho_mid,
        "sho": (sho_l, sho_r),
    }


def draw(img, idx, m, box, kp):
    x1, y1, x2, y2 = [int(v) for v in box]
    col = (0, 0, 255) if m["vert_ratio"] < 0.55 else (0, 200, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
    label = f"#{idx} v={m['vert_ratio']:.2f} a={m['lean_deg']:.0f} h={m['box_h_px']:.0f}"
    cv2.putText(img, label, (x1, max(y1 - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.circle(img, tuple(m["head_c"].astype(int)), 6, (255, 0, 255), -1)
    for p in m["sho"]:
        cv2.circle(img, tuple(p.astype(int)), 6, (255, 128, 0), -1)
    cv2.line(img, tuple(m["sho_mid"].astype(int)), tuple(m["head_c"].astype(int)), (255, 0, 255), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="图片/视频/目录")
    ap.add_argument("--model", default="/tmp/mm-models/yolov8n-pose.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--kp-thr", type=float, default=KP_CONF)
    ap.add_argument("--stride", type=int, default=1, help="视频抽帧步长")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from ultralytics import YOLO

    out = Path(a.out)
    (out / "vis").mkdir(parents=True, exist_ok=True)
    model = YOLO(a.model)

    src = Path(a.source)
    if src.is_dir():
        files = sorted([p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
        frames = [(p.name, cv2.imread(str(p))) for p in files]
    elif src.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        frames = [(src.name, cv2.imread(str(src)))]
    else:
        cap = cv2.VideoCapture(str(src))
        frames, i = [], 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            if i % a.stride == 0:
                frames.append((f"f{i:06d}", im))
            i += 1
            if a.max_frames and len(frames) >= a.max_frames:
                break
        cap.release()

    rows, lat = [], []
    for name, img in frames:
        t0 = time.perf_counter()
        r = model.predict(img, imgsz=a.imgsz, conf=a.conf, verbose=False, device="cpu")[0]
        lat.append((time.perf_counter() - t0) * 1000.0)

        n_det = len(r.boxes)
        n_usable = 0
        for idx in range(n_det):
            box = r.boxes.xyxy[idx].cpu().numpy()
            kp = r.keypoints.data[idx].cpu().numpy()
            m = person_metrics(kp, box, a.kp_thr)
            if m is None:
                continue
            n_usable += 1
            row = {"frame": name, "person": idx, "conf": round(float(r.boxes.conf[idx]), 2),
                   "x1": round(float(box[0]), 1), "y1": round(float(box[1]), 1),
                   "x2": round(float(box[2]), 1), "y2": round(float(box[3]), 1)}
            row.update({k: v for k, v in m.items() if not isinstance(v, (np.ndarray, tuple))})
            rows.append(row)
            draw(img, idx, m, box, kp)

        if n_det:
            vis = img.copy()
            cv2.imwrite(str(out / "vis" / f"{Path(name).stem}.jpg"), vis)
        print(f"{name}: det={n_det} usable={n_usable} lat={lat[-1]:.0f}ms")

    fields = ["frame", "person", "conf", "x1", "y1", "x2", "y2", "box_h_px", "box_w_px", "shoulder_w_px",
              "lean_deg", "lean_signed", "neck_ratio", "vert_ratio",
              "n_head_pts", "has_nose", "nose_conf", "nose_ear_dy", "eye_ear_dy", "n_eye_pts", "n_ear_pts"]
    with open(out / "persons.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    summary = {
        "source": str(a.source), "imgsz": a.imgsz, "conf": a.conf, "kp_thr": a.kp_thr,
        "frames": len(frames), "persons_usable": len(rows),
        "latency_ms_mean": round(float(np.mean(lat)), 1) if lat else None,
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 1) if lat else None,
        "det_per_frame_mean": round(len(rows) / max(len(frames), 1), 1),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
