#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SCENE_RE = re.compile(r"This is Scene ID:\s*([^,]+), Episode ID:\s*([^\.]+)\. The goal is\s+(.+?)\s+for this episode")
PROGRESS_RE = re.compile(r"Till Now Average Success rate:.*\(([-+0-9.]+) out of ([0-9]+)\)")
FAIL_RE = re.compile(r"Episode\s+([^ ]+)\s+in scene\s+([^ ]+)\s+failed due to '([^']+)'")
FINAL_RE = re.compile(
    r"Final check mode=([^ ]+) confidence=([-+0-9.eE]+), decision=([^,]+), tristate=([^,]+), patches=([0-9.]+)"
)
BLIP_RE = re.compile(r"Blip2 match score:\s*([-+0-9.eE]+)")
QWEN_RAW_RE = re.compile(r"Qwen-VL raw response:\s*(.*)")


def parse_run(log_path: Path) -> list[dict[str, Any]]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    episodes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    last_completed = 0

    for line in lines:
        progress = PROGRESS_RE.search(line)
        if progress:
            last_completed = int(progress.group(2))

        scene = SCENE_RE.search(line)
        if scene:
            current = {
                "episode_index": last_completed,
                "scene": scene.group(1),
                "episode_id": scene.group(2),
                "goal": scene.group(3),
                "final_events": [],
                "blip_scores": [],
                "qwen_raw_tail": [],
                "accepted_high_blip_after_reject": 0,
                "gate_only_accepts": 0,
                "qwen_accepts": 0,
                "qwen_rejects": 0,
                "might_false_positive": 0,
            }
            episodes.append(current)

        if current is None:
            continue

        blip = BLIP_RE.search(line)
        if blip:
            current["blip_scores"].append(float(blip.group(1)))

        final = FINAL_RE.search(line)
        if final:
            event = {
                "mode": final.group(1),
                "confidence": float(final.group(2)),
                "decision": final.group(3).strip(),
                "tristate": final.group(4).strip(),
                "patches": float(final.group(5)),
            }
            current["final_events"].append(event)
            if event["mode"] == "selective_hybrid_gate_only" and event["decision"] == "True":
                current["gate_only_accepts"] += 1
            if event["mode"] == "qwenvl_objectpatch":
                if event["decision"] == "True":
                    current["qwen_accepts"] += 1
                else:
                    current["qwen_rejects"] += 1

        raw = QWEN_RAW_RE.search(line)
        if raw:
            current["qwen_raw_tail"].append(raw.group(1)[:300])
            current["qwen_raw_tail"] = current["qwen_raw_tail"][-5:]

        if "accepted high-confidence BLIP2 after Qwen-VL rejection" in line:
            current["accepted_high_blip_after_reject"] += 1
        if "Might false positive, change to look for the true goal." in line:
            current["might_false_positive"] += 1

        failure = FAIL_RE.search(line)
        if failure:
            # Failure is printed after the progress line, so it belongs to the most
            # recently completed episode.
            index = last_completed - 1 if last_completed else current["episode_index"]
            for episode in reversed(episodes):
                if episode["episode_index"] == index:
                    episode["failure_cause"] = failure.group(3)
                    break
            else:
                current["failure_cause"] = failure.group(3)

    for episode in episodes:
        finals = episode["final_events"]
        episode["last_final"] = finals[-1] if finals else None
        episode["status"] = episode.get("failure_cause", "success")
    return episodes


def main() -> int:
    parser = argparse.ArgumentParser(description="Attribute final-check events to individual ASCENT episodes.")
    parser.add_argument("run_log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    episodes = parse_run(args.run_log)
    text = json.dumps(episodes, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
