# 低头（Distraction）判定效果验证 —— 原型结果

> 对应 ticket：#6「分心检测算法效果验证」（wayfinder:prototype, HITL）
> 目的：把调研（#4）选出的方案在真实画面上跑一遍，看误判/漏检/延迟，判断能不能用，要不要改指标或阈值。
> 状态：**指标口径已确认**（wayfinder 会话 2026-11-02）：
> Distraction = `nose_ear_dy > 0.124` 或 `vert_ratio < 0.462` → 低头；架构保留"一次 pose 推理供 Occupancy + Distraction"；
> spec 需把「按排/座位 ROI 裁剪推理」写进前提。标注仍为代理标注，待人工复核（见"局限"）。

## 结论摘要（TL;DR）

1. **调研推荐的"肩颈—头顶向量与竖直方向夹角"在真实机位下基本失效**：24 人里低头组角度 1.0°–20.2°（中位 5.9°），抬头组 0.9°–20.7°（中位 7.6°），AUC ≈ 0.51（等于抛硬币）。原因是本场景为**高位正面俯视**，低头主要体现在"鼻/眼相对耳线的竖直位移"和"头-肩竖直间距塌缩"，而不在向量倾角上。
2. **换用竖直方向指标后有明显区分力**（同一批 24 人）：
   - `nose_ear_dy`（鼻尖相对耳线中点、按肩宽归一化的竖直偏移）**AUC 0.938**；阈值 >0.117 时 recall 0.88 / precision 0.88，>0.124 时 recall 0.88 / precision 1.00（8 个低头漏 1 个）。
   - `vert_ratio`（肩中点到头中心的竖直距离 / 肩宽，低头时塌缩）AUC 0.859；阈值 <0.46 时 recall 0.88 / precision 0.78。
3. **全帧降采样推理不可行**：同一张 4K 画面，`imgsz=640` 只检出 4 人，`1280` 13 人，`1920` 25 人，`2560` 29 人（画面里可见约 40+ 人），**后两排几乎全丢**。把后排条带裁出来单独跑 `imgsz=1920`，同一条带检出 14 人，而全帧 1920 在该条带只有 6 人 → **必须按行/座位 ROI 裁剪推理**，不能靠"全帧 + 大 imgsz"。
4. **缺视频，所以"持续超过 Distraction Threshold"这一层没验证**：抖动、短暂低头负样本、遮挡掉帧都测不了。误报率的最终数字必须用自录视频（#5 的 P0 素材）来定。
5. **延迟只有 CPU 参考值**（本机 16 核，yolov8n-pose）：`imgsz` 640/960/1280/1920 对应推理 50/91/395/830 ms，前后处理 <10 ms。Jetson Orin Nano 实测仍缺（属 map 未决项），这组数字只能当"分辨率代价随 imgsz 近似平方增长"的相对参考。

## 方法

- 模型：`yolov8n-pose.pt`（Ultralytics 8.4.198，torch 2.15 CPU），`conf=0.25`，关键点可用阈值 `conf>0.3`。
- 素材：仓库根目录 `真实场景1/2/3.jpg`（4080×2296，目标会议厅实拍，静态图）。
- 每人指标（全部按肩宽 `W` 归一化，尺度无关）：
  - `lean_deg`：肩中点 → 头部中心向量与"图像向上"方向的夹角（= 调研推荐的向量角度）
  - `vert_ratio`：该向量的**竖直分量** / W（低头 → 塌缩）
  - `neck_ratio`：该向量的模 / W
  - `nose_ear_dy`：`(y_nose − y_耳中点) / W`（低头 → 鼻尖相对耳线下降，值变大）
  - `eye_ear_dy`：`(y_眼中点 − y_耳中点) / W`（低头 → 眼-耳竖直间距塌缩/变号）
- 人工标注：`真实场景2.jpg` @1920 的 25 个检出中，1 人不可判（#10），其余 24 人标为 up(16) / down(8)，见 `labels-s2-provisional.csv`。

## 结果 1：分辨率 → 检出人数（conf 0.25，关键点可用率 100%）

| imgsz | 真实场景1 | 真实场景2 | 真实场景3 |
|---|---|---|---|
| 960 | 7 | 10 | 2 |
| 1280 | 13 | 13 | 5 |
| 1920 | 16 | 25 | 5 |
| 2560 | — | 29 | — |

后排条带（`y ∈ [26%, 50%]`，4080×552）单独裁切推理：

| 方式 | 该条带检出 |
|---|---|
| 全帧 @1920 | 6 |
| 条带裁剪 @1280 | 10 |
| 条带裁剪 @1920 | 14 |

→ 单次全帧推理在 4K 会议厅里召回不够；**按排/座位 ROI 裁剪 + 局部更大尺度**是同分辨率下唯一有效的办法。

## 结果 2：各候选指标的区分力（24 人：down 8 / up 16）

| 指标 | 方向 | AUC |
|---|---|---|
| `nose_ear_dy` | 越大越可能低头 | **0.938** |
| `eye_ear_dy` | 越大越可能低头 | 0.914 |
| `vert_ratio` | 越小越可能低头 | 0.859 |
| `neck_ratio` | 越小越可能低头 | 0.844 |
| `lean_deg`（调研推荐） | — | **0.508 / 0.492（无区分力）** |

分组统计（min / median / max）：

| 组 | lean_deg | vert_ratio | nose_ear_dy |
|---|---|---|---|
| down (8) | 1.0 / 5.9 / 20.2 | 0.260 / 0.378 / 0.704 | 0.053 / 0.160 / 0.297 |
| up (16) | 0.9 / 7.6 / 20.7 | 0.428 / 0.599 / 0.696 | 0.019 / 0.056 / 0.124 |

阈值扫描（节选）：

| 规则 | TP | FP | FN | recall | precision |
|---|---|---|---|---|---|
| `nose_ear_dy > 0.117` | 7 | 1 | 1 | 0.88 | 0.88 |
| `nose_ear_dy > 0.124` | 7 | 0 | 1 | 0.88 | **1.00** |
| `vert_ratio < 0.462` | 7 | 1 | 1 | 0.88 | 0.88 |
| `vert_ratio < 0.459` | 6 | 1 | 2 | 0.75 | 0.86 |
| 线性组合（`nose_ear_dy` + `vert_ratio` 网格搜索） | — | — | — | AUC 0.938，**未超过单指标** |

**已知漏检案例**：#21（低头看手机，手机举在胸前）`nose_ear_dy=0.053`、`vert_ratio=0.70`，两个指标都判为抬头 —— 头只微低、脸仍完整可见，是这个口径下最难的一类。若要覆盖，需要单独标定"手持设备"子类或引入耳-眼-鼻的关键点排序关系，属于下一轮（视频）的事。

## 结果 3：延迟（本机 CPU，16 核，输入为 numpy 数组，已 warmup）

| imgsz | preprocess | inference | postprocess | 墙钟 | 检出人数 |
|---|---|---|---|---|---|
| 640 | 1.5 ms | 50 ms | 1.0 ms | 55 ms | 4 |
| 960 | 2.7 ms | 91 ms | 0.8 ms | 69 ms | 10 |
| 1280 | 4.0 ms | 395 ms | 0.7 ms | 400 ms | 13 |
| 1920 | 8.5 ms | 830 ms | 1.7 ms | 840 ms | 25 |

- 推理耗时随 imgsz 近似平方增长，前后处理可忽略。
- CPU 上多次测量抖动明显（同一配置 840 ms–1400 ms），不要拿这组数字直接当产能预算；**Orin Nano + TensorRT 的实测仍缺**。
- 若按"每座位低频采样"而不是全帧全速，1–5 FPS 目标下 1920 全帧在 CPU 上都难，Jetson GPU 上预期宽松，但必须实测。

## 效果展示（已确认规则的可视化）

规则：`nose_ear_dy > 0.124` → 低头；否则 `vert_ratio < 0.462` → 低头；否则抬头。渲染脚本 `render_verdict.py`。

> **所有 jpg/png 效果图只在本地生成，不进仓库**（真实场景照片含出镜人员，且体积大；`.gitignore` 已排除 `prototype/**/artifacts/*.jpg|png` 与仓库根目录的 `真实场景*.jpg`）。
> 需要看图时在本地跑 `render_verdict.py` 生成，见本节下方"复现"。

| 图 | 内容 | 结果 |
|---|---|---|
| `artifacts/scene1-verdict.jpg` | 真实场景1（imgsz 1920） | 检出 16 人：低头 11、抬头 5 |
| `artifacts/scene2-verdict.jpg` | 真实场景2（imgsz 1920） | 检出 25 人：低头 8、抬头 17 |
| `artifacts/scene3-verdict.jpg` | 真实场景3（imgsz 1920） | 检出 5 人：低头 1、抬头 4 |
| `artifacts/scene1-heads.jpg` / `scene2-heads.jpg` | 逐人头部特写 + 裁决 + `[OK]/[MISS]`（与代理标注比对） | scene2：**22/24 判对**，1 假阳（#14 抬头误判低头）、1 漏检（#21 低头看手机） |

看效果时请注意：后排/边侧没框的人不是"抬头"，是**根本没被检出**（见结果 1）。

## 局限（读结论前请先看这里）

- **24 人、1 帧、代理标注**：标签是本会话按肉眼判断打的，`#10` 标为不可判。样本极小，AUC 与阈值只能当方向性证据，不能当验收数字。
- **没有视频**：无法评估抖动、遮挡掉帧、"持续低头 > Distraction Threshold"的时序判定，也无法算真实的每小时误报次数（这是产品最关心的指标）。
- **模型只测了 yolov8n-pose**，未与 yolov8s-pose / YOLO11-pose / RTMPose 对比。
- 静态图无遮挡/光照变化，也不含"低头→抬头"的过渡帧。
- 延迟为 x86 CPU 数字，不代表 Jetson。

## 建议（待 human 决策）

1. **改指标**：spec 的 Distraction 口径从"肩颈—头顶向量角度"改为"`nose_ear_dy`（主）+ `vert_ratio`（辅/兜底）"，保留"一次 pose 推理同时供 Occupancy + Distraction"的架构不变。
2. **改推理架构**：spec 需明确「按排/座位 ROI 裁剪推理」作为前提（可每座位低频采样），否则 4K 全帧降采样看不到后排。
3. **补素材**：录 #5 里列的 10 类最小场景（尤其"持续低头 / 短暂低头负样本 / 遮挡"），才能定 Distraction Threshold 与误报率。
4. 阈值/持续时长的最终标定另开 ticket 做（本 ticket 只交付"指标选型 + 首轮数字"）。

## 复现

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python ultralytics pandas
# 模型权重：https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt

python eval_head_down.py --source ../../真实场景2.jpg --imgsz 1920 --out out/s2
python make_montage.py --img ../../真实场景2.jpg --csv out/s2/persons.csv --out out/s2/montage.jpg
python make_montage.py --img ../../真实场景2.jpg --csv out/s2/persons.csv --out out/s2/heads.jpg --head-crop 0.45
python analyze.py --csv out/s2/persons.csv --labels labels-s2-provisional.csv

# 效果图（只写在本地，不要提交）
python render_verdict.py --img ../../真实场景2.jpg --csv out/s2/persons.csv \
    --labels labels-s2-provisional.csv --out out/s2/verdict.jpg --montage out/s2/heads.jpg --head-crop 0.45
```

## 文件

| 文件 | 说明 |
|---|---|
| `eval_head_down.py` | 跑 pose，算每人候选指标，出 CSV/JSON + 标注图 |
| `make_montage.py` | 按 bbox 裁人拼表，供肉眼标注（`--head-crop` 只看头部） |
| `analyze.py` | AUC + 阈值扫描（TP/FP/FN/recall/precision） |
| `render_verdict.py` | 按已确认规则出效果图（默认只标"低头"；`--verbose` 带指标数值）+ 逐人拼表 |
| `labels-s2-provisional.csv` | `真实场景2` 24 人的 up/down 代理标注（**待人确认**） |
| `artifacts/s2-annotated-full.jpg`（仅本地） | 全图标注（框 + vert_ratio/角度/框高） |
| `artifacts/s2-heads-montage.jpg`（仅本地） | 25 人头部特写拼表，可直接肉眼复核标签 |
| `artifacts/persons-s2-imgsz1920.csv` | 25 人全部指标原始数据 |
