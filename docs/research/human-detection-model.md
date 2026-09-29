# 人体检测模型选型与 Jetson 推理可行性调研

调研目的：为 MeetMate（单固定摄像头、预标定 Seat 区域、判定 Check-in/Early Leave 的 Occupancy 检测）选择开源人体检测模型，并评估 Jetson Orin Nano 上的实时性可行性。

## 1. 推荐模型

推荐 **YOLO11n**（Ultralytics 开源，AGPL/商业双许可）。理由：

- Ultralytics 官方对比页明确指出 YOLO11 相比 YOLOv8 参数更少、精度相当或更好，延迟更低（[docs.ultralytics.com/compare/yolov8-vs-yolo11](https://docs.ultralytics.com/compare/yolov8-vs-yolo11)）。在算力受限的 Jetson 上，同精度下更少参数意味着更大帧率余量。
- 只需二分类"有人/无人"，nano 尺寸模型足够；不需要 s/m/l 更大模型。
- COCO 预训练的 person 类本身包含大量遮挡、半身、坐姿样本，作为起点可直接试用；**是否需要微调**取决于实测：建议先用预训练权重在真实会议室机位录像上跑一轮，若因俯视角/桌面遮挡导致漏检率明显偏高，再用现场标注的少量坐姿/半身样本做小规模微调（这是基于场景推断的工程建议，非官方文档结论，无法给出具体来源）。
- 备选：YOLOv8n（生态更成熟、Ultralytics 官方 Jetson 指南历史版本benchmark数据更完整），可作为 YOLO11n 的对照基线。

## 2. supervision 库 PolygonZone 能力评估

Roboflow `supervision` 库的 `sv.PolygonZone` 已是成熟可用组件（[supervision.roboflow.com/latest/detection/tools/polygon_zone](https://supervision.roboflow.com/latest/detection/tools/polygon_zone/)）：

- 用法：`PolygonZone(polygon=顶点数组, triggering_anchors=(sv.Position.BOTTOM_CENTER,))`，`zone.trigger(detections)` 返回逐检测框的布尔命中数组，`zone.current_count` 给出当前帧命中数——可直接映射为 Seat 的 occupancy 二值状态。
- **关键局限（官方文档明确说明）**：PolygonZone 的判断是**基于锚点（anchor-based），而非真正的几何框/多边形相交测试**。即只检测框上指定的锚点（如底边中心）是否落在多边形内，而不是判断检测框与 Seat 区域是否有重叠/包含关系。文档原话建议如需要"完整重叠语义"应使用 mask 或 IoU 方法。
- 对 MeetMate 场景的影响：固定俯视/侧视角度下，坐姿/半身人体检测框的底边中心可能落在座位区域之外（遮挡、透视形变），需要针对实际机位调试 `triggering_anchors`（如改用 `CENTER` 或多锚点 `require_all_anchors=False`），或自行用 IoU/重叠比例实现更贴合"重叠/包含判断"需求的过滤逻辑，而不能直接依赖默认锚点判断。这是需要在集成时验证的实际风险点，而非可以忽略的细节。
- 遮挡、俯视角、半身入镜对检测精度的影响属于检测模型（YOLO）本身的问题，supervision 只做后处理过滤，不能弥补漏检；因此模型选型与阈值调优仍是关键。

## 3. Jetson 上的预期帧率

一手来源汇总（单位均为 640×640 输入）：

- **Ultralytics 官方 GitHub Issue #22479**（[github.com/ultralytics/ultralytics/issues/22479](https://github.com/ultralytics/ultralytics/issues/22479)）引用 Ultralytics 官方文档基准：YOLO11n 在 Jetson Orin Nano Super 上 TensorRT FP16 推理约 **4.53 ms/帧（约 220 FPS）**，PyTorch 约 13.7 ms/帧（约 73 FPS）。该 issue 中用户自测（25W 功耗模式）实测约 16 ms（约 62 FPS，TensorRT FP16）、33 ms（约 30 FPS，PyTorch），仍远高于场景所需。
- **Ultralytics 官方 Jetson 部署指南历史版本**（YOLOv8n，Jetson Orin NX 16GB，FP32，[GitHub 镜像](https://github.com/ledinhtuyen/yolov8_custom/blob/main/ultralytics/docs/en/guides/nvidia-jetson.md)）：TensorRT 推理 8.7 ms/帧（约 115 FPS），PyTorch 14.3 ms/帧（约 70 FPS）。
- **第三方学术基准 AgriJetsonBench**（arXiv 2608.00927，[arxiv.org/abs/2608.00927](https://arxiv.org/abs/2608.00927)）：Jetson Orin Nano Super 在 15W 功耗下运行 YOLO11s（比 nano 更重）TensorRT INT8 达到 **145.82 FPS**，能耗 0.0767 J/次推理。

**结论**：无论取哪个数据点，YOLO11n/YOLOv8n 在 Jetson Orin Nano（含 Super 版）上经 TensorRT 加速后帧率量级在**几十到两百多 FPS**，远超 MeetMate 场景实际所需的 **1-5 FPS**（因为座位状态变化是分钟级事件，不需要视频帧率级刷新）。性能余量非常充裕，甚至 FP32/PyTorch 未加速路径（约 30-75 FPS）也可能满足需求，量化加速更多是为了降低功耗和给多路/多任务留余量，而不是"能不能跑起来"的问题。

## 4. 量化与加速方案建议

- **导出为 TensorRT engine**（`model.export(format="engine")`，Ultralytics 官方支持）是 Jetson 上推荐的加速路径，收益如上节数据所示（PyTorch → TensorRT 通常有 2-3 倍加速）。
- **FP16 vs INT8**：本场景帧率需求极低（1-5 FPS），FP16 已有巨大余量，精度损失小、无需额外校准数据集；**不建议优先上 INT8**——INT8 需要校准集且精度风险更高，收益（更高 FPS）在本场景无实际意义。仅在未来需要单机多路摄像头/多任务共享算力时才考虑 INT8。
- **DeepStream vs 纯 Python + supervision**：DeepStream（NVIDIA 官方视频分析框架，[docs.ultralytics.com/guides/deepstream-nvidia-jetson](https://docs.ultralytics.com/guides/deepstream-nvidia-jetson)）面向多路视频流、高吞吐生产级流水线，学习成本和部署复杂度高。MeetMate 是单摄像头、低帧率、逻辑简单（框-区域重叠判断），**纯 Python（Ultralytics TensorRT 推理）+ supervision 完全够用**，工程复杂度更低、更易调试和迭代，推荐作为首选方案；只有当未来扩展到多摄像头/多会议室集中部署时才重新评估 DeepStream。

## 5. 结论置信度与依据来源

- 模型选型（YOLO11n 优于 YOLOv8n）：**高置信度**，直接依据官方对比文档。
- supervision PolygonZone 成熟度与"锚点判断非真实几何重叠"局限：**高置信度**，直接引用官方文档原文。
- Jetson 帧率数量级：**高置信度**（多个独立来源交叉验证，且场景所需帧率远低于任何数据点下限）；**具体数字置信度中等**（官方文档表格页面本次调研中多次抓取超时未能拿到完整原始表格，采用了 GitHub Issue 中引用官方数字 + 第三方学术基准 + 历史版本官方指南三方交叉验证）。
- 坐姿/半身场景是否需要微调：**低置信度/工程推断**，无直接一手文档支持，建议以实测验证。

参考来源列表：
1. Ultralytics YOLOv8 vs YOLO11 对比 — https://docs.ultralytics.com/compare/yolov8-vs-yolo11
2. Ultralytics NVIDIA Jetson 部署指南 — https://docs.ultralytics.com/guides/nvidia-jetson/
3. Ultralytics DeepStream on Jetson 指南 — https://docs.ultralytics.com/guides/deepstream-nvidia-jetson
4. Ultralytics GitHub Issue #22479（YOLO11n Orin Nano 实测与官方数字引用） — https://github.com/ultralytics/ultralytics/issues/22479
5. Ultralytics Jetson 指南历史版本（YOLOv8n on Orin NX，官方FP32表格） — https://github.com/ledinhtuyen/yolov8_custom/blob/main/ultralytics/docs/en/guides/nvidia-jetson.md
6. AgriJetsonBench（YOLO11s Orin Nano Super INT8 基准） — https://arxiv.org/abs/2608.00927
7. supervision 官方文档 PolygonZone — https://supervision.roboflow.com/latest/detection/tools/polygon_zone/
8. NVIDIA Jetson Orin Nano Super 开发者套件官方页面（TOPS 规格） — https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/

## 6. 若最终选型非 Jetson Orin Nano

本调研基准假设是 Jetson Orin Nano（8GB，约 20 TOPS INT8 dense / 40 TOPS sparse，Super 模式 67 TOPS）。若最终选型变化：

- **Orin NX / AGX Orin**（算力更高，157/275 TOPS 级别）：帧率数字会更高，结论方向不变（余量更大），**无需重新核对**，只需更新文档中的具体数字即可，不影响可行性判断。
- **老款 Jetson Nano（非 Orin 系列，Maxwell GPU，约 0.5 TOPS 级）/ Xavier NX（约 21 TOPS）**：算力显著低于 Orin Nano，**必须重新核对帧率数据**——老款 Nano 很可能无法用 TensorRT 达到本文引用的百级 FPS，需重新查证 Ultralytics/NVIDIA 针对该具体型号的官方基准；但由于场景仅需 1-5 FPS，即使实际帧率降低一个数量级，大概率仍能满足需求，方向性结论（可行）大概率不变，需要核实的是"余量有多大"而非"是否可行"。
