import csv
import json
import os
import re
from collections import Counter
from statistics import mean
from typing import Dict, Iterable, List

import numpy as np


def slugify(value: str) -> str:
    value = value.strip().replace("/", "_")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("_")


def classify_failure(metrics: Dict[str, object], semantic_threshold: float) -> str:
    if float(metrics.get("success", 0.0)) >= 1.0:
        return "success"
    if bool(metrics.get("stop_called", False)):
        return "false_stop"
    if int(metrics.get("stuck_events", 0)) > 0:
        return "stuck"
    if float(metrics.get("max_similarity_seen", -1.0)) < semantic_threshold:
        return "target_not_found"
    if int(metrics.get("failed_goal_count", 0)) > 0:
        return "false_semantic_peak"
    if bool(metrics.get("timed_out", False)):
        return "timeout"
    return "timeout"


def write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(data, file, indent=2, default=_to_jsonable)


def write_trace(path: str, trace: Iterable[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        for row in trace:
            file.write(json.dumps(row, default=_to_jsonable) + "\n")


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def aggregate_metrics(records: List[Dict[str, object]]) -> Dict[str, object]:
    if len(records) == 0:
        return {
            "episode_count": 0,
            "success_rate": 0.0,
            "spl": 0.0,
            "softspl": 0.0,
            "avg_steps": 0.0,
            "avg_path_length": 0.0,
            "avg_runtime_sec": 0.0,
            "failure_reasons": {},
        }

    failure_counts = Counter(record["failure_reason"] for record in records)
    return {
        "episode_count": len(records),
        "success_rate": mean(float(record["success"]) for record in records),
        "spl": mean(float(record["spl"]) for record in records),
        "softspl": mean(float(record["softspl"]) for record in records),
        "avg_steps": mean(float(record["steps"]) for record in records),
        "avg_path_length": mean(float(record["path_length"]) for record in records),
        "avg_runtime_sec": mean(float(record["runtime_sec"]) for record in records),
        "failure_reasons": dict(sorted(failure_counts.items())),
    }


def write_summary_csv(path: str, records: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "scene_id",
        "episode_id",
        "goal_category",
        "success",
        "spl",
        "softspl",
        "distance_to_goal",
        "steps",
        "path_length",
        "runtime_sec",
        "failure_reason",
        "video_path",
        "semantic_image_path",
    ]
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})


def build_repo_audit_markdown() -> str:
    return """# Repo Audit

## Core Findings
- 入口链路是 `run.py -> zson.config.get_config -> ZSONTrainer`。
- 当前官方实现是 `RGB/global feature + text goal + RNN PPO`，不是 dense semantic map。
- `ObjectGoalPromptSensor` 已提供 ObjectNav 的文本目标 token。
- `SimpleRLEnv`、Habitat dataset、`AGENT_POSITION`/`AGENT_ROTATION`、`TOP_DOWN_MAP` 已可复用。
- 原仓库自带视频仅是 Habitat frame 拼接，没有 per-episode artifact / report pipeline。

## Plan A Mapping
- 新增独立 `strict Plan A` 入口，不改原 PPO 评测语义。
- 主结果使用 frozen CLIP、depth、pose、semantic similarity map 和 shortest-path follower。
- 官方 `zson_conf_B.pth` 仅保留为参考，不混入主结果。
"""


def build_report_markdown(
    config,
    env_info: Dict[str, object],
    aggregate: Dict[str, object],
    records: List[Dict[str, object]],
) -> str:
    success_examples = [r for r in records if float(r["success"]) >= 1.0][:3]
    failure_examples = [r for r in records if float(r["success"]) < 1.0][:3]
    lines = [
        f"# Plan A Report ({config.PLAN_A.DATASET} / {config.PLAN_A.SPLIT})",
        "",
        "## Experiment Setup",
        f"- Commit: {env_info.get('git_commit', 'unknown')}",
        f"- Python: {env_info.get('python_version', 'unknown')}",
        f"- Habitat: {env_info.get('habitat_version', 'unknown')}",
        f"- Habitat-Sim: {env_info.get('habitat_sim_version', 'unknown')}",
        f"- CLIP model: {config.PLAN_A.CLIP_MODEL}",
        f"- Dataset path: {config.TASK_CONFIG.DATASET.DATA_PATH}",
        f"- Scenes dir: {config.TASK_CONFIG.DATASET.SCENES_DIR}",
        f"- Episode count: {aggregate['episode_count']}",
        "",
        "## Quantitative Results",
        f"- SR: {aggregate['success_rate']:.4f}",
        f"- SPL: {aggregate['spl']:.4f}",
        f"- SoftSPL: {aggregate['softspl']:.4f}",
        f"- Avg steps: {aggregate['avg_steps']:.2f}",
        f"- Avg path length: {aggregate['avg_path_length']:.2f}",
        f"- Avg runtime (sec): {aggregate['avg_runtime_sec']:.3f}",
        f"- Failure reasons: {aggregate['failure_reasons']}",
        "",
        "## Qualitative Notes",
        "- Semantic heatmap is produced from CLIP patch similarities projected into the top-down map.",
        "- Local planning uses Habitat shortest-path follower toward semantic peaks or frontier cells.",
        "- Stop uses center-patch similarity with depth gating, not GT distance.",
        "",
        "## Success Examples",
    ]
    if success_examples:
        for record in success_examples:
            lines.append(
                f"- {record['scene_id']} / {record['episode_id']} / {record['goal_category']} -> {record['video_path']}"
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Failure Examples"])
    if failure_examples:
        for record in failure_examples:
            lines.append(
                f"- {record['scene_id']} / {record['episode_id']} / {record['goal_category']} / {record['failure_reason']} -> {record['video_path']}"
            )
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Bug / Fix / Residual Risks",
            "- This run uses a newly added strict Plan A pipeline and does not depend on the legacy PPO checkpoint.",
            "- MP3D and HM3D both rely on depth + pose projection; semantic asset files are not required.",
            "- Full-val runtime is substantial because CLIP patch extraction runs at every step in a single environment.",
        ]
    )
    return "\n".join(lines) + "\n"
