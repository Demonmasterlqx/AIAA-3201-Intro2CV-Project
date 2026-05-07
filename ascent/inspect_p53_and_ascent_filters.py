#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

ROOT = Path("/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent")
DATASET = ROOT / "data/datasets/objectnav/hm3d/v1/val/content/p53SfW6mjZe.json.gz"

print(f"dataset={DATASET}")
with gzip.open(DATASET, "rt", encoding="utf-8") as f:
    data = json.load(f)

episodes = data.get("episodes", [])
print(f"episode_count={len(episodes)}")
for i, ep in enumerate(episodes[:30]):
    goals = ep.get("goals") or []
    goal_info = []
    for g in goals[:3]:
        goal_info.append({
            "object_category": g.get("object_category"),
            "position": g.get("position"),
        })
    print(json.dumps({
        "idx": i,
        "episode_id": ep.get("episode_id"),
        "scene_id": ep.get("scene_id"),
        "object_category": ep.get("object_category"),
        "start_position": ep.get("start_position"),
        "start_rotation": ep.get("start_rotation"),
        "goals_head": goal_info,
    }, ensure_ascii=False))

needles = [
    "episode_id",
    "episodes_allowed",
    "episode_indices",
    "content_scenes",
    "num_episode_sample",
    "test_episode_count",
]
print("\ncode_matches:")
for base in [ROOT / "ascent", ROOT / "habitat_extensions", ROOT / "experiments"]:
    if not base.exists():
        continue
    for path in base.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".yaml", ".yml"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for needle in needles:
            if needle in text:
                rel = path.relative_to(ROOT)
                lines = [j + 1 for j, line in enumerate(text.splitlines()) if needle in line][:5]
                print(f"{rel}: {needle}: lines={lines}")
                break
