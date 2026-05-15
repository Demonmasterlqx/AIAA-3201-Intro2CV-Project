# Project 2 Part 1 Plan B 总结报告

## 1. 项目目标与复现范围

本次工作的目标是基于当前仓库复现 Project 2 Part 1 的 Plan B，即基于 Vision-Language Frontier Map (VLFM) 的 frontier-based zero-shot object navigation 方法。在实现上，我们没有重写 VLFM 的核心算法，而是保留现有 `HabitatITMPolicyV2`、`ObstacleMap`、`ValueMap`、`ObjectPointCloudMap` 与 `PointNav` 主链路，重点补齐以下工程闭环：

- HM3D 与 MP3D 数据接入
- 结构化批量评测
- per-episode 日志与三路视频导出
- 结果汇总与课程报告材料生成
- 失败案例保留与分析

冻结时刻说明：

- 本报告基于已停止的后台实验结果撰写，不再随着后续后台任务变化而更新。
- 冻结时刻 HM3D `val` 已完整跑完，MP3D `val` 跑到部分结果后停止。

## 2. 实验环境与代码版本

- 代码版本：`24e3d0c`
- Python / Conda 环境：`3.9.23` / `vlfm`
- 核心运行入口：`project/run_plan_b.py`
- VLM 服务：GroundingDINO、BLIP2ITM、MobileSAM、YOLOv7
- 主要评测输出根目录：
  - `project/plan_b_runs/hm3d/val/full_val_20260411/`
  - `project/plan_b_runs/mp3d/val/full_val_20260411/`

使用的数据：

- HM3D：`data/datasets/objectnav/hm3d/v1/val/val.json.gz`
- MP3D：`data/datasets/objectnav/mp3d/v1/val/val.json.gz`

场景根目录：

- HM3D：`data/scene_datasets`
- MP3D：`/data/home/sim6g/code/aiaa3201_cv_project/data/MatterPort3D`

## 3. 方法落地说明

当前仓库中的 Plan B 落地方式如下：

- 使用深度图构建 `ObstacleMap`，维护 explored area、obstacle map 与 frontier 候选点。
- 使用 BLIP2ITM 对当前 RGB 观测与文本目标做 image-text matching，将语义分数投影并融合到 `ValueMap`。
- 只在 frontier 上取局部 value，选择语义潜力最高的 frontier 作为当前探索目标。
- 一旦 `ObjectPointCloudMap` 确认目标对象位置，策略从 `explore` 切换到 `navigate`，由 pointnav 策略直接导航至目标。

与标准 VLFM 思路的一致点：

- 保持 frontier-based exploration，而不是纯 dense map retrieval。
- 保持 vision-language semantic promise 评估。
- 保持 zero-shot object-goal navigation 设定。

本次工程性适配主要包括：

- scene-sharded 批量评测，支持断点续跑
- 结构化 per-episode JSON
- 每个 episode 导出 `composite / egocentric / topdown` 三路视频
- preflight 检查与运行时数据工作目录
- 失败原因归类与总报告生成

## 4. 数据集与评测设置

### HM3D

- Split：`val`
- 目标 episode 数：`2000`
- 冻结时刻完成：`2000 / 2000`

### MP3D

- Split：`val`
- 目标 episode 数：`2195`
- 冻结时刻完成：`1310 / 2195`
- 说明：MP3D 为**部分结果**，报告中所有 MP3D 数值均应按 partial evaluation 理解

### 统一指标

- Success Rate (SR)
- Success weighted by Path Length (SPL)
- Soft SPL
- Avg Path Length
- Avg Step Count
- Avg Episode Wall Time
- Avg Inference Time

## 5. HM3D 定量结果

冻结时刻 HM3D 汇总如下：

| Metric | Value |
| --- | ---: |
| Episodes | 2000 |
| Success Rate | 52.30% |
| SPL | 30.34% |
| Soft SPL | 36.26% |
| Avg Path Length (m) | 18.06 |
| Avg Step Count | 156.11 |
| Avg Episode Time (s) | 79.25 |
| Avg Inference Time (s) | 0.4611 |

失败原因分布：

- `success`: 1046
- `false_positive`: 443
- `frontier_collapse`: 267
- `target_not_found_timeout`: 242
- `stuck_low_progress`: 2

对 HM3D 的总体判断：

- Plan B 的核心行为已明显复现出来。
- agent 能稳定地执行“语义驱动 frontier 搜索 -> 目标确认 -> 目标导航”的链路。
- 指标层面已足够支撑课程报告中的主结果与定量分析。

## 6. MP3D 当前定量结果（Partial Evaluation）

冻结时刻 MP3D 汇总如下：

| Metric | Value |
| --- | ---: |
| Episodes | 1310 / 2195 |
| Success Rate | 26.18% |
| SPL | 12.58% |
| Soft SPL | 19.54% |
| Avg Path Length (m) | 30.11 |
| Avg Step Count | 246.40 |
| Avg Episode Time (s) | 127.31 |
| Avg Inference Time (s) | 0.4719 |

失败原因分布：

- `false_positive`: 480
- `target_not_found_timeout`: 420
- `success`: 343
- `frontier_collapse`: 62
- `stuck_low_progress`: 5

对 MP3D 的当前判断：

- MP3D 上方法已经能够运行并产出成功案例，但整体效果明显弱于 HM3D。
- 目前更适合被描述为“部分复现”或“跨数据集泛化不足”，不适合写成“完整成功复现”。
- 报告中应明确标注该部分是 partial evaluation，而非完整全量结果。

## 7. 定性案例分析

### HM3D 成功案例

- `zt1RVoi7PcG / ep 106 / bed`
  - `SPL = 0.974`
  - 视频：`episodes/zt1RVoi7PcG__ep_106/composite.mp4`
- `Nfvxx8J5NCo / ep 35 / couch`
  - `SPL = 0.969`
  - 视频：`episodes/Nfvxx8J5NCo__ep_35/composite.mp4`
- `6s7QHgap2fW / ep 91 / chair`
  - `SPL = 0.968`
  - 视频：`episodes/6s7QHgap2fW__ep_91/composite.mp4`

这些样例表明：

- 当目标类别具有较强语义先验或较稳定室内共现结构时，Plan B 能迅速聚焦到高语义潜力 frontier。
- 一旦点云确认目标，pointnav 接管后的路径效率较高。

### HM3D 失败案例

- `XB4GS9ShBRE / ep 50 / couch`
  - `failure_reason = false_positive`
  - `soft_spl = 0.880`
  - 视频：`episodes/XB4GS9ShBRE__ep_50/composite.mp4`
- `zt1RVoi7PcG / ep 34 / tv`
  - `failure_reason = false_positive`
  - `soft_spl = 0.845`
  - 视频：`episodes/zt1RVoi7PcG__ep_34/composite.mp4`

这些失败说明：

- 语义 promise 已经足够强，会把 agent 拉到“看起来很像目标的区域”。
- 但一旦共现物体或房间上下文与目标高度相似，就可能产生显著假阳性。

### MP3D 成功案例

- `TbHJrupSAjP / ep 106 / pillow`
  - `SPL = 0.960`
  - 视频：`episodes/TbHJrupSAjP__ep_106/composite.mp4`
- `2azQ1b91cZZ / ep 95 / chair`
  - `SPL = 0.931`
  - 视频：`episodes/2azQ1b91cZZ__ep_95/composite.mp4`
- `TbHJrupSAjP / ep 20 / potted plant`
  - `SPL = 0.923`
  - 视频：`episodes/TbHJrupSAjP__ep_20/composite.mp4`

### MP3D 失败案例

- `EU6Fwq7SyZv / ep 73 / pillow`
  - `failure_reason = false_positive`
  - `soft_spl = 0.913`
  - 视频：`episodes/EU6Fwq7SyZv__ep_73/composite.mp4`
- `Z6MFQCViBuw / ep 19 / chair`
  - `failure_reason = false_positive`
  - `soft_spl = 0.876`
  - 视频：`episodes/Z6MFQCViBuw__ep_19/composite.mp4`
- `EU6Fwq7SyZv / ep 66 / pillow`
  - `failure_reason = false_positive`
  - `soft_spl = 0.875`
  - 视频：`episodes/EU6Fwq7SyZv__ep_66/composite.mp4`

MP3D 的成功样例证明该方法不是完全失效；但失败分布说明它在更复杂类别和更长路径场景中更容易被误导。

## 8. 失败模式总结

### 1. False Positive

这是当前最主要的问题。

- HM3D：443
- MP3D：480

原因推断：

- BLIP2ITM 的局部图像-文本匹配容易被房间布局、共现家具和局部上下文触发高分。
- frontier 语义评分会把 agent 引向“语义上有希望”的区域，但该区域不一定真有目标实例。

### 2. Target Not Found Timeout

这类失败表示 agent 在预算内没有建立足够可靠的语义证据或没有真正看到目标。

- HM3D：242
- MP3D：420

在 MP3D 中更明显，说明跨数据集泛化不足和目标外观多样性会显著削弱在线匹配稳定性。

### 3. Frontier Collapse

这类失败表示局部地图或 frontier 集合在某些阶段退化，导致探索策略失去有效候选。

- HM3D：267
- MP3D：62

这通常与复杂几何结构、局部视角受限、地图边界以及探索区域被过早裁剪有关。

### 4. Stuck Low Progress

数量很少，但反映 agent 在局部区域震荡或长时间低效移动。

- HM3D：2
- MP3D：5

## 9. 复现问题与修复记录

本次复现中，主要修复了以下工程问题：

1. **日志、视频、结果汇总缺失**
- 补齐了 scene-sharded 批跑、per-episode JSON、三路视频和 summary/report 输出。

2. **HM3D 语义注释不完整导致兼容问题**
- 使用 `frontier_exploration_compat.py` 做兼容处理，使缺失语义 bbox 的环境仍可评测。

3. **MP3D 数据接入问题**
- 增加 `--scenes-dir` 支持，使评测脚本能接入本机 MP3D 场景根目录。

4. **Value map 投影越界**
- 修复 `place_img_in_img()` 和相关半径取值逻辑，避免地图边界触发断言崩溃。

5. **Top-down 点云可视化越界**
- 修复 `color_point_cloud_on_map()` 的边界裁剪，避免 `IndexError` 使批跑中断。

6. **HM3D 0.2 `ascent` 数据接入**
- 为 Plan B 结构化评测链路新增 `--hm3d-source ascent` 与 runtime data workspace 机制，并完成 smoke 验证。

仍未完全解决的问题：

- MP3D 指标明显弱于 HM3D，说明现有语义匹配与 frontier 评分在跨数据集泛化上仍然不足。
- false positive 数量过高，说明当前 Plan B 仍缺少更强的置信度抑制与时间一致性机制。

## 10. 当前结论与后续建议

当前结论：

- 在 HM3D 上，本次 Plan B 复现已经**基本成功**。方法不仅能运行，而且已经复现出核心行为模式，并取得可接受的 SR/SPL。
- 在 MP3D 上，本次结果应表述为**部分复现**。方法已能跑通并产生成功样例，但整体效果明显偏弱，不能与 HM3D 质量等同。

建议在最终课程 PDF 中采用如下表述：

> VLFM Plan B 在 HM3D 上已表现出稳定的语义驱动 frontier exploration 能力，并成功复现出中等水平的 Success Rate 和 SPL；而在 MP3D 上，方法虽已跑通并出现成功案例，但整体性能显著下降，主要受 false positive 和目标未发现超时影响，因此该部分更适合表述为 partial reproduction rather than full reproduction.

后续建议：

- 引入更严格的 semantic confidence filtering，抑制假阳性。
- 增加跨帧一致性机制，如 EMA 或 target-centric memory。
- 将 frontier 评分与几何可达性约束更紧地耦合。
- 对 MP3D 单独调参或增加更鲁棒的目标类别文本模板。

## 11. 结果文件索引

主结果目录：

- HM3D：`project/plan_b_runs/hm3d/val/full_val_20260411/`
- MP3D：`project/plan_b_runs/mp3d/val/full_val_20260411/`

关键汇总文件：

- HM3D 指标：`project/plan_b_runs/hm3d/val/full_val_20260411/summary/metrics.json`
- MP3D 指标：`project/plan_b_runs/mp3d/val/full_val_20260411/summary/metrics.json`
- HM3D 报告：`project/plan_b_runs/hm3d/val/full_val_20260411/report.md`
- MP3D 报告：`project/plan_b_runs/mp3d/val/full_val_20260411/report.md`

