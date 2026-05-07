#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from plan_b_common import load_episode_records


def average(records: List[Dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return sum(float(record.get(key, 0.0)) for record in records) / len(records)


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    failure_counts = Counter(str(record.get("failure_reason", "unknown")) for record in records)
    return {
        "episode_count": len(records),
        "success_rate": average(records, "success"),
        "spl": average(records, "spl"),
        "soft_spl": average(records, "soft_spl"),
        "avg_path_length_m": average(records, "path_length_m"),
        "avg_step_count": average(records, "step_count"),
        "avg_episode_wall_time_sec": average(records, "episode_wall_time_sec"),
        "avg_inference_time_sec": average(records, "avg_inference_time_sec"),
        "failure_reason_counts": dict(failure_counts),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_episode_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        row = dict(record)
        video_paths = row.pop("video_paths", {}) or {}
        row["composite_video"] = video_paths.get("composite", "")
        row["egocentric_video"] = video_paths.get("egocentric", "")
        row["topdown_video"] = video_paths.get("topdown", "")
        rows.append(row)
    return rows


def metrics_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for key in (
        "episode_count",
        "success_rate",
        "spl",
        "soft_spl",
        "avg_path_length_m",
        "avg_step_count",
        "avg_episode_wall_time_sec",
        "avg_inference_time_sec",
    ):
        rows.append({"metric": key, "value": summary[key]})
    for reason, count in summary["failure_reason_counts"].items():
        rows.append({"metric": f"failure_reason::{reason}", "value": count})
    return rows


def choose_case(records: List[Dict[str, Any]], success: bool) -> Dict[str, Any]:
    filtered = [record for record in records if bool(record.get("success", 0.0)) is success]
    if not filtered:
        return {}
    if success:
        return sorted(filtered, key=lambda record: float(record.get("spl", 0.0)), reverse=True)[0]
    return sorted(filtered, key=lambda record: float(record.get("path_length_m", 0.0)), reverse=True)[0]


def build_metrics_md(summary: Dict[str, Any]) -> str:
    return "\n".join(
        [
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Episodes | {summary['episode_count']} |",
            f"| Success Rate | {summary['success_rate'] * 100:.2f}% |",
            f"| SPL | {summary['spl'] * 100:.2f}% |",
            f"| Soft SPL | {summary['soft_spl'] * 100:.2f}% |",
            f"| Avg Path Length (m) | {summary['avg_path_length_m']:.2f} |",
            f"| Avg Step Count | {summary['avg_step_count']:.2f} |",
            f"| Avg Episode Time (s) | {summary['avg_episode_wall_time_sec']:.2f} |",
            f"| Avg Inference Time (s) | {summary['avg_inference_time_sec']:.4f} |",
        ]
    )


def build_report(
    run_root: Path,
    records: List[Dict[str, Any]],
    summary: Dict[str, Any],
    preflight: Dict[str, Any],
    run_spec: Dict[str, Any],
) -> str:
    commit_display = str(
        run_spec.get("git_commit_short", preflight.get("git_commit_short", preflight.get("git_commit", "unknown")))
    ).split()[0]
    success_case = choose_case(records, success=True)
    failure_case = choose_case(records, success=False)
    failure_lines = [
        f"- `{reason}`: {count}"
        for reason, count in sorted(summary["failure_reason_counts"].items(), key=lambda item: item[1], reverse=True)
    ]
    if not failure_lines:
        failure_lines = ["- 无"]

    return "\n".join(
        [
            f"# {run_spec.get('dataset', '').upper()} Plan B 测试报告",
            "",
            "## 1. 实验设置",
            "",
            f"- 代码版本：`{commit_display}`",
            f"- Python / 环境：`{preflight.get('python', {}).get('version', 'unknown')}` / `{preflight.get('python', {}).get('conda_env', '')}`",
            f"- 数据集：`{run_spec.get('dataset', '')}`",
            f"- split：`{run_spec.get('split', '')}`",
            f"- dataset json：`{run_spec.get('dataset_json', '')}`",
            f"- scenes dir：`{run_spec.get('scenes_dir', '')}`",
            f"- 策略：`{run_spec.get('policy_name', 'HabitatITMPolicyV2')}`",
            f"- 评测 episode 数：`{summary['episode_count']}`",
            f"- 随机种子：`{run_spec.get('seed', '')}`",
            f"- 视频视角：`{', '.join(run_spec.get('video_views', []))}`",
            "",
            "## 2. 方法说明",
            "",
            "- 直接复用仓库内的 `HabitatITMPolicyV2` 作为 Plan B 主策略。",
            "- 地图部分沿用 `ObstacleMap + ValueMap`：深度构图、前沿点提取、BLIP2ITM 语义分数投影与融合。",
            "- 探索部分沿用 frontier-restricted 语义排序：只在 frontier 上取局部 value，选择语义潜力最高的 frontier。",
            "- 工程适配新增了 dataset preflight、scene-sharded 批跑、per-episode 结构化日志与三路视频导出。",
            "",
            "## 3. 定量结果",
            "",
            build_metrics_md(summary),
            "",
            "## 4. 定性分析",
            "",
            f"- 成功案例：`{success_case.get('scene_short', '')} / ep {success_case.get('episode_id', '')}`，视频 `{success_case.get('video_paths', {}).get('composite', 'N/A')}`"
            if success_case
            else "- 成功案例：本次运行中无成功 episode。",
            f"- 失败案例：`{failure_case.get('scene_short', '')} / ep {failure_case.get('episode_id', '')}`，原因 `{failure_case.get('failure_reason', '')}`，视频 `{failure_case.get('video_paths', {}).get('composite', 'N/A')}`"
            if failure_case
            else "- 失败案例：本次运行中无失败 episode。",
            "- 失败原因统计：",
            *failure_lines,
            "- `target_not_found_timeout` 与 `timeout_after_detection` 主要反映语义证据不足或检测后导航未完成。",
            "- `frontier_collapse` 与 `mapping_issue` 用于识别地图/前沿提取链路异常，便于和普通 timeout 分开分析。",
            "",
            "## 5. 问题排查记录",
            "",
            *([f"- 预检警告：{warning}" for warning in preflight.get("warnings", [])] or ["- 无额外预检警告。"]),
            "",
            "## 6. 结果文件",
            "",
            f"- episodes.csv: `{(run_root / 'summary' / 'episodes.csv').relative_to(run_root)}`",
            f"- metrics.json: `{(run_root / 'summary' / 'metrics.json').relative_to(run_root)}`",
            f"- metrics.md: `{(run_root / 'summary' / 'metrics.md').relative_to(run_root)}`",
            "",
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a structured VLFM Plan B run.")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    episodes_root = run_root / "episodes"
    summary_root = run_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    records = load_episode_records(episodes_root)
    summary = summarize(records)

    episodes_json = summary_root / "episodes.json"
    episodes_csv = summary_root / "episodes.csv"
    metrics_json = summary_root / "metrics.json"
    metrics_csv = summary_root / "metrics.csv"
    metrics_md = summary_root / "metrics.md"
    report_md = run_root / "report.md"

    episodes_json.write_text(json.dumps(records, indent=2) + "\n")
    write_csv(episodes_csv, flatten_episode_rows(records))
    metrics_json.write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(metrics_csv, metrics_rows(summary))
    metrics_md.write_text(build_metrics_md(summary) + "\n")

    preflight_path = run_root / "preflight.json"
    run_spec_path = run_root / "run_spec.json"
    preflight = json.loads(preflight_path.read_text()) if preflight_path.exists() else {}
    run_spec = json.loads(run_spec_path.read_text()) if run_spec_path.exists() else {}
    report_md.write_text(build_report(run_root, records, summary, preflight, run_spec))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
