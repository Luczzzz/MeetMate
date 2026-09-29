# 低头姿态检测（Distraction 判定）技术方案调研

> **TL;DR**：推荐 **YOLOv8-pose（或同系 YOLO11/YOLO26-pose）+ Roboflow `supervision`**，用人体姿态关键点中的 **肩颈—头顶（耳/鼻）向量角度** 作为低头判定依据，并让它与座位占用检测**复用同一次姿态检测推理**（同一模型同时输出 person bbox + keypoints，供 Occupancy 和 Distraction 两个判定逻辑分别消费），而不是叠加人脸关键点/头部姿态回归模型。置信度：**中**（架构选择置信度高，具体阈值与 Orin Nano 实测数字置信度低，需实测验证）。

## 1. 背景约束简述

依据 `CONTEXT.md` 与 ADR：Distraction 唯一判定口径是"低头姿态持续超过阈值"（不含视线/交谈）；不做人脸/身份识别（ADR 0001）；Jetson 本地推理、暂以 Orin Nano 为基准（ADR 0002）；只留结构化数据 + Key Frame（ADR 0003）。因此候选方案需要在**不依赖人脸识别**、**单机与座位占用模型共享算力**的前提下比较。

## 2. 候选方案鲁棒性对比

| 方案 | 遮挡（桌子挡下半身/手挡脸） | 侧脸 | 低分辨率/远距离 | 是否违反 ADR0001（人脸相关） |
|---|---|---|---|---|
| ①头部姿态估计（人脸关键点回归 6DoF，如 PnP/6DRepNet/WHENet） | 脸部被手/文件遮挡时关键点或直接回归会失真甚至失效 | 侧脸时人脸检测器召回骤降，PnP 求解误差增大 | 远距离下人脸区域像素极少，landmark 精度急剧下降 | 依赖人脸检测/关键点，技术上接近人脸相关处理，需谨慎避免变成事实上的人脸识别（虽非身份识别，但风险边界模糊） |
| ②人脸关键点（MediaPipe Face Mesh 468点 / dlib 68点） | 同上，遮挡下 468 点里大量点不可见，头部姿态解算不稳定；IJSR 研究报告在非受控/遮挡场景下 landmark 检测准确率约 90%（整体，非低头专项）([IJSR][1]) | 侧脸只能看到半张脸，Face Mesh 官方声明面向正面/近正面场景，非全角度设计([MediaPipe][2]) | 远距离时脸部分辨率不足，Face Mesh 需要相对完整清晰的人脸框才能稳定输出 | 与①相同顾虑 |
| ③人体姿态关键点：肩颈-头部向量（YOLOv8-pose/RTMPose 等 COCO 17点：nose/eyes/ears + shoulders） | **上半身（肩、颈、头）通常不被会议桌遮挡**（桌子挡的是下半身/手），核心可用点仍可见；即使部分脸部关键点（眼/鼻）被手遮挡，仍可用耳朵+肩膀近似角度 | 侧脸时耳朵、肩膀关键点仍稳定可检测（人体姿态模型对全身朝向鲁棒性远高于人脸关键点模型），是三者中最抗侧脸方案 | 人体关键点检测在低分辨率下比人脸关键点更稳健，因为依赖的是躯干大尺度结构而非五官细节；YOLO-pose 系列本身即为应对多尺度设计 | 完全不涉及人脸特征提取，最贴合 ADR0001 |
| ④启发式（bbox 长宽比变化 / 头部区域下移量） | 对遮挡不敏感（只用 person bbox 整体形状），但精度低、易与"弯腰捡东西""趴桌子睡觉"等混淆 | 不受侧脸影响（不依赖脸部） | 极低分辨率下仍可用（只需 bbox），但噪声大，误报率高 | 不涉及人脸，最省算力，但**判定精度和可解释性最弱**，一般作为③的辅助/兜底信号而非主方案 |

**结论（鲁棒性维度）**：③肩颈-头顶向量方案在三种典型场景（遮挡、侧脸、低分辨率）下综合表现最优，且与 ADR 0001 的"不做人脸识别"原则天然契合；①②在遮挡/侧脸/低分辨率任一场景下都明显脆弱，且存在触及人脸处理边界的隐私顾虑；④可作为③的低成本兜底/交叉验证信号。

## 3. Jetson Orin Nano 算力开销估算

**硬件基准（NVIDIA 官方）**：Jetson Orin Nano（原始版）AI 性能最高 40 INT8 TOPS，5–15W 功耗；经 "Super" 固件升级后最高 67 INT8 TOPS，7–25W；1024 CUDA cores + 32 Tensor cores，8GB（或4GB）统一内存，与 CPU/GPU 共享。([NVIDIA][3]) 同系列 Orin NX 最高 100–157 TOPS，AGX Orin 最高 275 TOPS，可作为"同架构不同算力档位"的插值参考。([NVIDIA][3])

**Ultralytics YOLOv8-pose 模型规模**（官方文档 `docs.ultralytics.com/tasks/pose`，640px 输入；pose 头与检测头共享同一 backbone/neck，只是多一个 keypoint 分支）：

| 模型 | 参数量 | FLOPs |
|---|---|---|
| YOLOv8n-pose | 3.3M | 9.2B |
| YOLOv8s-pose | 11.6M | 30.2B |
| YOLOv8m-pose | 26.4M | 81.0B |

（数字来自公开文档记忆整理，未在本次调研中重新逐字核对官方表格，建议实测前用 `YOLO("yolov8n-pose.pt").info()` 核对一次。）Ultralytics 官方 Jetson 部署指南本身聚焦检测模型（当前版本已转向 YOLO26），未直接给出 Orin Nano 上 pose 模型的 FPS/延迟表格，本次调研未能抓取到该页表格数据（页面抓取超时），是本报告的主要数据缺口。([Ultralytics Jetson Guide][4])

**可类比的官方/半官方 Jetson 姿态估计实测数据**（用于量级推算，非 Orin Nano 直测）：
- NVIDIA-AI-IOT `trt_pose`（TensorRT 加速的 ResNet18/DenseNet121 姿态估计，官方仓库）：ResNet18-baseline 224×224 在 **Jetson Nano** 约 22 FPS（纯推理），在 **Jetson Xavier** 约 251 FPS；DenseNet121 版本在 Nano 约 12 FPS，Xavier 约 101 FPS。社区实测普遍比 README 数字低 30–40%（如 Nano 实测 12–14 FPS）。([NVIDIA-AI-IOT trt_pose][5], [NVIDIA 开发者论坛][6])
- 按 TOPS 量级换算：Jetson Nano ≈ 0.5 TOPS(INT8, 未启用DLA时约472GFLOPS FP16)，Orin Nano 40–67 TOPS，量级提升约 **50–80 倍**；即便考虑内存带宽/软件栈成熟度等非线性因素打折，Orin Nano 跑同等规模的 ResNet18 级姿态模型（远小于 YOLOv8n-pose 的 9.2GFLOPs量级，实际上 trt_pose 的 backbone 更大）大概率可轻松达到实时（>30FPS）甚至有较大余量，但**这是外推估算，非直测数据**。
- RTMPose（OpenMMLab 官方 benchmark，非 Jetson 环境）：RTMPose-m 在 Intel i7-11700 CPU 上 90+FPS，在 NVIDIA GTX1660Ti 上 430+FPS，仅 75.8% AP（COCO）；轻量版 RTMPose-s 在 Snapdragon 865（ncnn）上 70+FPS。([RTMPose 官方博客][7]) 桌面级 GPU 性能远高于 Orin Nano 且不代表 TensorRT/Jetson 实际表现，仅可作为模型本身"轻量、专为实时设计"的旁证。

**能否与座位占用模型共用一次推理**：这是本方案节省总算力的关键。YOLOv8-pose（以及 YOLO11/YOLO26-pose、RTMPose 搭配的检测器）本身输出即包含 **person bounding box**，可直接同时满足"座位是否有人"（Occupancy，用 bbox 与座位标定区域的交并判断）与"是否低头"（用该 bbox 内的 keypoints 判断角度）两个需求，**理论上只需部署一个姿态估计模型，而不必再单独跑一个人体检测模型**。这比"检测模型 + 独立头部姿态/人脸关键点模型"两个模型的组合，节省了一整套 backbone 推理开销（量级上等于减少 30–50% 的总算力占用，具体取决于第二模型大小）。显存/算力占用量级：YOLOv8n-pose（9.2GFLOPs）+ TensorRT INT8/FP16 量化后，预计在 Orin Nano 上单帧延迟应在个位数至十几毫秒量级（对比检测模型 YOLOv8n 在 Jetson 系列的公开量级数据），实时性（每座位每秒数帧采样，非每帧全速处理）应有较大余量，但**这一具体延迟/显存数字未经 Orin Nano 实测确认，需要在真实硬件上跑 `yolo benchmark` 验证**。

## 4. 最终推荐方案

**推荐**：**YOLOv8-pose（n 或 s 档，视精度/速度取舍）+ Roboflow `supervision` 库**，规则层用 COCO 关键点中的 `nose / left_ear / right_ear / left_shoulder / right_shoulder` 计算"头部中心—肩颈中线"向量与垂直方向的夹角（pitch 近似角），角度超过阈值并持续超过 `Distraction Threshold` 触发 Distraction 事件。

理由：
1. **一模型两用**：同一次 pose 推理的 bbox 输出可直接喂给 Occupancy 判定，keypoints 输出喂给 Distraction 判定，避免部署第二个人体检测模型，直接命中"总算力预算有限、需与座位占用模型共享算力"这一核心约束。
2. **鲁棒性最优**：如第2节表格所示，肩颈-头部方案在遮挡（会议桌挡下半身不影响肩颈头部关键点）、侧脸（人体姿态模型对朝向的鲁棒性远高于人脸关键点模型）、低分辨率（依赖大尺度躯干结构而非五官细节）三个场景下均优于人脸关键点/6DoF头部姿态方案。
3. **合规契合**：完全不提取人脸特征，与 ADR 0001"不做人脸识别"的精神最贴合，不存在"技术上不是身份识别但仍在处理人脸数据"的灰色地带。
4. **生态成熟、团队倾向一致**：Ultralytics YOLOv8-pose 有官方 TensorRT/Jetson 部署路径，`supervision` 对 Ultralytics 输出有原生 `KeyPoints` 支持，工程落地成本低，符合团队"开源 + supervision + YOLOv8-pose/RTMPose/MediaPipe"的倾向。
5. RTMPose 作为备选：精度/速度（尤其 CPU）表现优秀，但官方生态（MMDeploy + TensorRT）在 Jetson 上的部署案例和公开 benchmark 少于 Ultralytics 路线，工程成熟度略低，可作为二期"若 YOLOv8-pose 精度不够"时的升级候选，而非首选。

不推荐①②的核心原因：即使不考虑算力，遮挡/侧脸/低分辨率三个约束场景下鲁棒性明显弱于③，且需要额外部署一个人脸检测/关键点模型，与"总算力预算有限、需与人体检测模型协同"的约束直接冲突。④启发式可作为③的低成本辅助信号（例如用于快速排除"离座"或过滤抖动），不建议作为主判定方案。

## 5. 置信度评级与依据来源

**置信度：中**
- 高置信度部分："肩颈-头部关键点角度"方案在遮挡/侧脸/低分辨率下优于人脸关键点方案——这是姿态估计与人脸关键点算法的通用特性结论，有相对稳固的第一手证据支持。"一次姿态推理同时服务 Occupancy + Distraction"的架构合理性——由 YOLOv8-pose 官方输出格式（bbox+keypoints 同源）直接支持，逻辑上确定。
- 低置信度 / 需实测验证部分：
  - **Jetson Orin Nano 上 YOLOv8-pose 的具体延迟/FPS/显存数字**——本次调研未能获取官方直测表格（Ultralytics Jetson 指南页面抓取超时、当前版本已转向 YOLO26），只能用 TOPS 量级换算和 trt_pose 在 Nano/Xavier 上的社区实测数据做外推，误差可能较大，**建议上线前在实际 Orin Nano（或候选型号）上跑 `yolo benchmark model=yolov8n-pose.pt` 实测**。
  - **低头判定的具体角度阈值**（多少度算"低头"）——未在业界找到会议室场景的公开标定数据，需要结合实际摄像头架设角度、座椅朝向做现场标定。
  - YOLOv8-pose 参数量/FLOPs 表格为记忆整理，未逐字核对官方最新页面（官方页面已迁移至 YOLO26 为主角），建议使用前用 `.info()` 或官方最新文档核对。

**来源列表**：
- Ultralytics Pose 任务文档：https://docs.ultralytics.com/tasks/pose/
- Ultralytics NVIDIA Jetson 部署指南：https://docs.ultralytics.com/guides/nvidia-jetson/
- NVIDIA Jetson Orin Nano 产品页/开发者博客：https://developer.nvidia.com/embedded/learn/jetson-orin-nano-devkit-user-guide/index.html ，https://developer.nvidia.com/blog/solving-entry-level-edge-ai-challenges-with-nvidia-jetson-orin-nano/
- NVIDIA-AI-IOT `trt_pose` 官方仓库（Jetson Nano/Xavier 实测 FPS）：https://github.com/NVIDIA-AI-IOT/trt_pose ；社区实测讨论：https://forums.developer.nvidia.com/t/performance-of-trt-pose-on-jetson-nano/112913
- OpenMMLab RTMPose 官方博客（精度/速度 benchmark）：https://openmmlab.medium.com/rtmpose-the-all-in-one-real-time-pose-estimation-solution-for-application-and-research-6404f17cd52f ，论文：https://arxiv.org/html/2303.07399v1
- MMPose 官方 Inference Speed Summary（历史模型对比，背景参考）：https://mmpose.readthedocs.io/en/0.x/inference_speed_summary.html
- MediaPipe Face Mesh 官方说明：https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md
- 人脸关键点遮挡场景研究（约90%检测准确率，非低头专项）：IJSR, "Environmental Risk Handling on Face Landmark Detection Using Computer Vision" https://www.ijsr.net/archive/v14i12/MR251215121031.pdf
- Roboflow `supervision` 官方仓库与文档（KeyPoints 类、keypoint NMS）：https://github.com/roboflow/supervision ，https://supervision.roboflow.com/
- 课堂行为识别相关工程参考（YOLO-pose/OpenPose 用于姿态类行为识别，非低头专项但方法论可类比）：https://pmc.ncbi.nlm.nih.gov/articles/PMC12099366/ ，https://www.mdpi.com/1424-8220/25/17/5386

[1]: https://www.ijsr.net/archive/v14i12/MR251215121031.pdf
[2]: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md
[3]: https://developer.nvidia.com/blog/solving-entry-level-edge-ai-challenges-with-nvidia-jetson-orin-nano/
[4]: https://docs.ultralytics.com/guides/nvidia-jetson/
[5]: https://github.com/NVIDIA-AI-IOT/trt_pose
[6]: https://forums.developer.nvidia.com/t/performance-of-trt-pose-on-jetson-nano/112913
[7]: https://openmmlab.medium.com/rtmpose-the-all-in-one-real-time-pose-estimation-solution-for-application-and-research-6404f17cd52f
