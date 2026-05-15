#!/usr/bin/env python3
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import argparse
import json
import os
from typing import Dict, Iterable, List, Tuple


MetricSummary = Dict[str, float]
EpisodeStats = Dict[str, object]


def load_episode_stats(directory: str) -> Dict[Tuple[str, str], EpisodeStats]:
    stats: Dict[Tuple[str, str], EpisodeStats] = {}
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(directory, filename)
        if os.path.getsize(path) == 0:
            continue
        with open(path, "r") as handle:
            episode = json.load(handle)
        key = (str(episode["scene_id"]), str(episode["episode_id"]))
        stats[key] = episode
    return stats


def summarize(stats: Iterable[EpisodeStats]) -> MetricSummary:
    stat_list = list(stats)
    if not stat_list:
        raise ValueError("No episode JSON files were found.")

    summary: MetricSummary = {"episodes": float(len(stat_list))}
    for metric in ("success", "spl", "soft_spl"):
        values = [float(episode.get(metric, 0.0)) for episode in stat_list]
        summary[metric] = sum(values) / len(values)
    return summary


def format_metric(metric: str, value: float) -> str:
    if metric == "episodes":
        return str(int(value))
    return f"{value * 100:.2f}%"


def print_comparison(
    label_a: str,
    label_b: str,
    summary_a: MetricSummary,
    summary_b: MetricSummary,
) -> None:
    rows = [
        ("episodes", "Episodes"),
        ("success", "Success Rate"),
        ("spl", "SPL"),
        ("soft_spl", "Soft SPL"),
    ]
    header = f"{'Metric':<14} {label_a:<18} {label_b:<18} {'Delta (B-A)':<14}"
    print(header)
    print("-" * len(header))
    for metric_key, metric_name in rows:
        delta = summary_b[metric_key] - summary_a[metric_key]
        delta_str = format_metric(metric_key, delta) if metric_key != "episodes" else str(int(delta))
        print(
            f"{metric_name:<14} "
            f"{format_metric(metric_key, summary_a[metric_key]):<18} "
            f"{format_metric(metric_key, summary_b[metric_key]):<18} "
            f"{delta_str:<14}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two VLFM evaluation directories.")
    parser.add_argument("plan_a_dir", type=str, help="Episode JSON directory for Plan A")
    parser.add_argument("plan_b_dir", type=str, help="Episode JSON directory for Plan B")
    parser.add_argument("--plan-a-label", default="Plan A", help="Label for the first run")
    parser.add_argument("--plan-b-label", default="Plan B", help="Label for the second run")
    args = parser.parse_args()

    plan_a_stats = load_episode_stats(args.plan_a_dir)
    plan_b_stats = load_episode_stats(args.plan_b_dir)
    common_keys = sorted(set(plan_a_stats) & set(plan_b_stats))

    if common_keys:
        subset_a = [plan_a_stats[key] for key in common_keys]
        subset_b = [plan_b_stats[key] for key in common_keys]
        print(f"Comparing {len(common_keys)} shared episodes.")
    else:
        subset_a = list(plan_a_stats.values())
        subset_b = list(plan_b_stats.values())
        print("No shared episode IDs found, comparing aggregate run averages.")

    print_comparison(
        args.plan_a_label,
        args.plan_b_label,
        summarize(subset_a),
        summarize(subset_b),
    )


if __name__ == "__main__":
    main()

