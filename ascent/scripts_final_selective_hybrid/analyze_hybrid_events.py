#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


SCENE_RE = re.compile(r"This is Scene ID:\s*([^,]+), Episode ID:\s*([^\.]+)\. The goal is\s+(.+?)\s+for this episode")
PROGRESS_RE = re.compile(r"Till Now Average Success rate:.*\(([-+0-9.]+) out of ([0-9]+)\)")
FAIL_RE = re.compile(r"Episode\s+([^ ]+)\s+in scene\s+([^ ]+)\s+failed due to '([^']+)'")
FINAL_RE = re.compile(r"Final check mode=([^ ]+) confidence=([-+0-9.eE]+), decision=([^,]+), tristate=([^,]+), patches=([0-9.]+)")


def load_compare(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"lost_success": [], "gained_success": []}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_log(log_path: Path) -> Dict[int, Dict[str, Any]]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    records: Dict[int, Dict[str, Any]] = {}
    current_index = 0
    last_progress_count = 0

    for line in lines:
        progress = PROGRESS_RE.search(line)
        if progress:
            last_progress_count = int(progress.group(2))
        scene = SCENE_RE.search(line)
        if scene:
            scene_id, episode_id, goal = scene.groups()
            current_index = last_progress_count
            records[current_index] = {
                "idx": current_index,
                "scene": scene_id,
                "episode_id": episode_id,
                "goal": goal,
                "finals": [],
                "fallback": 0,
                "qwen_accepts": 0,
                "holds": 0,
                "rejects": 0,
            }
        rec = records.get(current_index)
        if rec is not None:
            if "Selective hybrid accepted high-confidence BLIP2 after Qwen-VL rejection" in line:
                rec["fallback"] += 1
            if "Qwen-VL patch strong confirmation accepted" in line or "Qwen-VL patch repeated confirmation accepted" in line:
                rec["qwen_accepts"] += 1
            if "Qwen-VL patch positive held" in line:
                rec["holds"] += 1
            final = FINAL_RE.search(line)
            if final:
                mode, confidence, decision, tristate, patches = final.groups()
                rec["finals"].append(
                    {
                        "mode": mode,
                        "confidence": float(confidence),
                        "decision": decision,
                        "tristate": tristate,
                        "patches": float(patches),
                    }
                )
                if tristate == "reject":
                    rec["rejects"] += 1
        failure = FAIL_RE.search(line)
        if failure:
            episode_id, scene_id, cause = failure.groups()
            index = last_progress_count - 1 if last_progress_count else current_index
            fail_rec = records.setdefault(
                index,
                {
                    "idx": index,
                    "scene": scene_id,
                    "episode_id": episode_id,
                    "goal": "",
                    "finals": [],
                    "fallback": 0,
                    "qwen_accepts": 0,
                    "holds": 0,
                    "rejects": 0,
                },
            )
            fail_rec["failure_cause"] = cause
    return records


def summarize_compare(compare: Dict[str, Any]) -> Dict[str, Any]:
    gained = compare.get("gained_success", [])
    lost = compare.get("lost_success", [])
    return {
        "gained_by_goal": dict(Counter(row.get("goal", "") for row in gained)),
        "lost_by_goal": dict(Counter(row.get("goal", "") for row in lost)),
        "net_by_goal": {
            goal: Counter(row.get("goal", "") for row in gained)[goal] - Counter(row.get("goal", "") for row in lost)[goal]
            for goal in sorted(set([row.get("goal", "") for row in gained] + [row.get("goal", "") for row in lost]))
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze final-check event attribution for a hybrid ASCENT run.")
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--compare", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    records = parse_log(args.run_log)
    compare = load_compare(args.compare)
    interesting_indices = sorted(
        {int(row["episode_index"]) for row in compare.get("lost_success", [])}
        | {int(row["episode_index"]) for row in compare.get("gained_success", [])}
    )
    compare_rows = {
        int(row["episode_index"]): row
        for row in compare.get("lost_success", []) + compare.get("gained_success", [])
    }

    false_positive_records = []
    for idx, record in sorted(records.items()):
        if record.get("failure_cause") == "false_positive":
            false_positive_records.append(
                {
                    "idx": idx,
                    "goal": record.get("goal", ""),
                    "scene": record.get("scene", ""),
                    "fallback": record.get("fallback", 0),
                    "qwen_accepts": record.get("qwen_accepts", 0),
                    "holds": record.get("holds", 0),
                    "rejects": record.get("rejects", 0),
                    "last_final": record.get("finals", [])[-1] if record.get("finals") else None,
                }
            )

    interesting = []
    for idx in interesting_indices:
        row = compare_rows[idx]
        record = records.get(idx, {})
        interesting.append(
            {
                "idx": idx,
                "transition": row.get("transition", ""),
                "baseline_status": row.get("baseline_status", ""),
                "variant_status": row.get("variant_status", ""),
                "goal": row.get("goal", ""),
                "scene": row.get("scene", ""),
                "fallback": record.get("fallback", 0),
                "qwen_accepts": record.get("qwen_accepts", 0),
                "holds": record.get("holds", 0),
                "rejects": record.get("rejects", 0),
                "last_final": record.get("finals", [])[-1] if record.get("finals") else None,
            }
        )

    by_goal_fp = defaultdict(Counter)
    for item in false_positive_records:
        goal = item["goal"] or "unknown"
        by_goal_fp[goal]["count"] += 1
        by_goal_fp[goal]["fallback"] += int(item["fallback"] > 0)
        by_goal_fp[goal]["qwen_accept"] += int(item["qwen_accepts"] > 0)

    report = {
        "event_counts": {
            "fallback_total": sum(record.get("fallback", 0) for record in records.values()),
            "qwen_accept_total": sum(record.get("qwen_accepts", 0) for record in records.values()),
            "held_total": sum(record.get("holds", 0) for record in records.values()),
        },
        "compare_summary": summarize_compare(compare),
        "false_positive_by_goal": {goal: dict(counter) for goal, counter in sorted(by_goal_fp.items())},
        "interesting": interesting,
        "false_positive_records": false_positive_records,
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
