#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from zson.plan_a.reporting import aggregate_metrics, write_json, write_summary_csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="directory to write the merged summary",
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="one or more plan_a run directories containing summary.json",
    )
    return parser.parse_args()


def load_records(run_dirs: List[str]) -> List[Dict[str, object]]:
    records = []
    for run_dir in run_dirs:
        summary_path = os.path.join(run_dir, "summary.json")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(summary_path)
        with open(summary_path, "r") as file:
            payload = json.load(file)
        records.extend(payload.get("episodes", []))
    return records


def load_json_if_exists(path: str) -> Optional[Dict[str, object]]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as file:
        return json.load(file)


def build_report(
    aggregate: Dict[str, object],
    records: List[Dict[str, object]],
    env_info: Optional[Dict[str, object]],
    dataset_check: Optional[Dict[str, object]],
    source_run_dirs: List[str],
) -> str:
    lines = ["# Aggregated Plan A Report", ""]
    if dataset_check is not None:
        lines.extend(
            [
                "## Experiment Setup",
                f"- Dataset: {dataset_check.get('dataset', 'unknown')}",
                f"- Split: {dataset_check.get('split', 'unknown')}",
                f"- Data path: {dataset_check.get('data_path', 'unknown')}",
                f"- Scenes dir: {dataset_check.get('scenes_dir', 'unknown')}",
                f"- Episode count: {aggregate['episode_count']}",
                "",
            ]
        )
    if env_info is not None:
        lines.extend(
            [
                "## Environment",
                f"- Commit: {env_info.get('git_commit', 'unknown')}",
                f"- Python: {env_info.get('python_version', 'unknown')}",
                f"- Torch: {env_info.get('torch_version', 'unknown')}",
                f"- Habitat: {env_info.get('habitat_version', 'unknown')}",
                f"- Habitat-Sim: {env_info.get('habitat_sim_version', 'unknown')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Quantitative Results",
            f"- SR: {aggregate['success_rate']:.4f}",
            f"- SPL: {aggregate['spl']:.4f}",
            f"- SoftSPL: {aggregate['softspl']:.4f}",
            f"- Avg steps: {aggregate['avg_steps']:.2f}",
            f"- Avg path length: {aggregate['avg_path_length']:.2f}",
            f"- Avg runtime: {aggregate['avg_runtime_sec']:.3f}",
            f"- Failure reasons: {aggregate['failure_reasons']}",
            "",
            "## Sample Episodes",
        ]
    )
    for record in records[:5]:
        lines.append(
            f"- {record['scene_id']} / {record['episode_id']} / {record['goal_category']} / {record['failure_reason']} -> {record.get('video_path', '')}"
        )
    lines.extend(
        [
            "",
            "## Source Runs",
        ]
    )
    for run_dir in source_run_dirs:
        lines.append(f"- {run_dir}")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    records = load_records(args.run_dirs)
    aggregate = aggregate_metrics(records)
    env_info = load_json_if_exists(os.path.join(args.run_dirs[0], "env_info.json"))
    dataset_check = load_json_if_exists(
        os.path.join(args.run_dirs[0], "dataset_check.json")
    )
    payload = {
        "aggregate": aggregate,
        "episodes": records,
        "source_run_dirs": args.run_dirs,
    }
    write_json(os.path.join(args.output_dir, "summary.json"), payload)
    write_summary_csv(os.path.join(args.output_dir, "summary.csv"), records)
    if env_info is not None:
        write_json(os.path.join(args.output_dir, "env_info.json"), env_info)
    if dataset_check is not None:
        write_json(os.path.join(args.output_dir, "dataset_check.json"), dataset_check)
    with open(os.path.join(args.output_dir, "report.md"), "w") as file:
        file.write(
            build_report(
                aggregate=aggregate,
                records=records,
                env_info=env_info,
                dataset_check=dataset_check,
                source_run_dirs=args.run_dirs,
            )
        )


if __name__ == "__main__":
    main()
