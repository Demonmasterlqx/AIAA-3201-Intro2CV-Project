#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from parse_100ep_failures import load_manifest_episodes, parse_log


def choose_manifest(baseline_dir: Path, variant_dir: Path) -> List[Dict[str, Any]]:
    for result_dir in (variant_dir, baseline_dir):
        episodes = load_manifest_episodes(result_dir / "episodes_fixed.json")
        if episodes:
            return episodes
    return []


def choose_episode_records(baseline: Dict[str, Any], variant: Dict[str, Any]) -> List[Dict[str, Any]]:
    for parsed in (variant, baseline):
        records = parsed.get("episode_records", [])
        if isinstance(records, list) and records:
            return records
    return []


def failure_by_index(parsed: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    failures: Dict[int, Dict[str, Any]] = {}
    for record in parsed.get("failure_records", []):
        index = record.get("episode_index")
        if isinstance(index, int) and index >= 0:
            failures[index] = record
    return failures


def episode_status(index: int, completed: Optional[int], failures: Dict[int, Dict[str, Any]]) -> str:
    if completed is not None and index >= completed:
        return "not_completed"
    if index in failures:
        return str(failures[index].get("failure_cause", "failed"))
    return "success"


def short_status(status: str) -> str:
    return "success" if status == "success" else "failed"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare fixed-subset per-episode outcomes between two ASCENT runs.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    baseline = parse_log(args.baseline / "run.log", args.baseline / "episodes_fixed.json")
    variant = parse_log(args.variant / "run.log", args.variant / "episodes_fixed.json")
    episode_records = choose_episode_records(baseline, variant)
    manifest = choose_manifest(args.baseline, args.variant)

    baseline_failures = failure_by_index(baseline)
    variant_failures = failure_by_index(variant)
    baseline_completed = baseline.get("metrics", {}).get("episodes")
    variant_completed = variant.get("metrics", {}).get("episodes")
    max_len = max(
        len(manifest),
        int(baseline_completed or 0),
        int(variant_completed or 0),
        max(baseline_failures.keys(), default=-1) + 1,
        max(variant_failures.keys(), default=-1) + 1,
    )

    rows: List[Dict[str, Any]] = []
    transitions = Counter()
    for index in range(max_len):
        manifest_record = {}
        if index < len(episode_records):
            manifest_record = episode_records[index]
        elif index < len(manifest):
            manifest_record = manifest[index]
        baseline_status = episode_status(index, baseline_completed, baseline_failures)
        variant_status = episode_status(index, variant_completed, variant_failures)
        transition = f"{short_status(baseline_status)}->{short_status(variant_status)}"
        transitions[transition] += 1
        rows.append(
            {
                "episode_index": index,
                "scene": manifest_record.get("scene", ""),
                "scene_episode_index": manifest_record.get("scene_episode_index", ""),
                "episode_id": manifest_record.get("episode_id", ""),
                "goal": manifest_record.get("object_category", ""),
                "baseline_status": baseline_status,
                "variant_status": variant_status,
                "transition": transition,
            }
        )

    summary = {
        "baseline_dir": str(args.baseline),
        "variant_dir": str(args.variant),
        "baseline_metrics": baseline.get("metrics", {}),
        "variant_metrics": variant.get("metrics", {}),
        "transitions": dict(sorted(transitions.items())),
        "gained_success": [row for row in rows if row["baseline_status"] != "success" and row["variant_status"] == "success"],
        "lost_success": [row for row in rows if row["baseline_status"] == "success" and row["variant_status"] != "success"],
        "changed_failures": [
            row
            for row in rows
            if row["baseline_status"] != "success"
            and row["variant_status"] != "success"
            and row["baseline_status"] != row["variant_status"]
        ],
        "episodes": rows,
    }

    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
