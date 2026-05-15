#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


AVERAGE_RE = re.compile(r"Average episode ([A-Za-z0-9_]+):\s*([-+0-9.eE]+)")
FAIL_RE = re.compile(r"Episode\s+([^ ]+)\s+in scene\s+([^ ]+)\s+failed due to '([^']+)'")
SCENE_RE = re.compile(r"This is Scene ID:\s*([^,]+), Episode ID:\s*([^\.]+)\. The goal is\s+(.+?)\s+for this episode")
SCENE_PROGRESS_RE = re.compile(r"Success rate of Scene .*?/([A-Za-z0-9]+)\.basis\.glb:")
BLIP_RE = re.compile(r"Blip2 match score:\s*([-+0-9.eE]+)")
FINAL_MODE_RE = re.compile(r"Final check mode=([^ ]+) confidence=([-+0-9.eE]+), decision=([^,]+), tristate=([^,]+), patches=([0-9.]+)")
PROGRESS_RE = re.compile(r"Till Now Average Success rate:\s*[-+0-9.]+%\s*\(([-+0-9.]+)\s+out of\s+([0-9]+)\)")


def load_manifest_episodes(manifest_path: Path) -> List[Dict[str, Any]]:
    if not manifest_path.exists():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    episodes = payload.get("episodes", []) if isinstance(payload, dict) else []
    return episodes if isinstance(episodes, list) else []


def _load_manifest_payload(manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _scene_from_content_path(path: Path) -> str:
    return path.name.removesuffix(".json.gz")


def load_filtered_content_episodes(manifest_path: Optional[Path]) -> Dict[str, List[Dict[str, Any]]]:
    if manifest_path is None:
        return {}
    payload = _load_manifest_payload(manifest_path)
    output_data_path = str(payload.get("output_data_path", ""))
    if not output_data_path:
        return {}
    val_path = Path(output_data_path.replace("{split}", "val"))
    if not val_path.is_absolute():
        val_path = Path.cwd() / val_path
    content_root = val_path.parent / "content"
    if not content_root.exists():
        return {}

    by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for content_path in sorted(content_root.glob("*.json.gz")):
        scene = _scene_from_content_path(content_path)
        try:
            with gzip.open(content_path, "rt", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue
        records = []
        for filtered_index, episode in enumerate(data.get("episodes", [])):
            records.append(
                {
                    "scene": scene,
                    "scene_episode_index": filtered_index,
                    "episode_id": str(episode.get("episode_id", "")),
                    "object_category": str(episode.get("object_category", "") or ""),
                }
            )
        if records:
            by_scene[scene] = records
    return by_scene


def manifest_goal(manifest_episodes: List[Dict[str, Any]], episode_index: Optional[int]) -> str:
    if episode_index is None or episode_index < 0 or episode_index >= len(manifest_episodes):
        return ""
    episode = manifest_episodes[episode_index]
    return str(episode.get("object_category", "") or "")


def episode_record_goal(episode_records: List[Dict[str, Any]], episode_index: Optional[int]) -> str:
    if episode_index is None or episode_index < 0 or episode_index >= len(episode_records):
        return ""
    return str(episode_records[episode_index].get("object_category", "") or "")


def parse_log(log_path: Path, manifest_path: Optional[Path] = None) -> Dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    manifest_episodes = load_manifest_episodes(manifest_path) if manifest_path is not None else []
    content_episodes = load_filtered_content_episodes(manifest_path)
    averages: Dict[str, float] = {}
    failures: List[Dict[str, Any]] = []
    episode_records: List[Dict[str, Any]] = []
    goals_by_key: Dict[tuple[str, str], str] = {}
    final_modes = Counter()
    tristates = Counter()
    blip_scores: List[float] = []
    completed_episodes = None
    last_progress_episode_count = None
    last_completed_scene = ""
    scene_counts: Dict[str, int] = defaultdict(int)

    for line in text.splitlines():
        avg = AVERAGE_RE.search(line)
        if avg:
            averages[avg.group(1)] = float(avg.group(2))
        scene_match = SCENE_RE.search(line)
        if scene_match:
            scene, episode_id, goal = scene_match.groups()
            goals_by_key[(scene, episode_id)] = goal
        scene_progress = SCENE_PROGRESS_RE.search(line)
        if scene_progress:
            last_completed_scene = scene_progress.group(1)
        failure = FAIL_RE.search(line)
        if failure:
            episode_id, scene, cause = failure.groups()
            episode_index = (last_progress_episode_count - 1) if last_progress_episode_count is not None else None
            goal = (
                goals_by_key.get((scene, episode_id), "")
                or episode_record_goal(episode_records, episode_index)
                or manifest_goal(manifest_episodes, episode_index)
            )
            failures.append(
                {
                    "episode_id": episode_id,
                    "episode_index": episode_index,
                    "scene": scene,
                    "failure_cause": cause,
                    "goal": goal,
                }
            )
        blip = BLIP_RE.search(line)
        if blip:
            blip_scores.append(float(blip.group(1)))
        progress = PROGRESS_RE.search(line)
        if progress:
            completed_episodes = int(progress.group(2))
            last_progress_episode_count = completed_episodes
            episode_index = completed_episodes - 1
            if episode_index >= len(episode_records):
                scene = last_completed_scene
                scene_count = scene_counts[scene]
                scene_records = content_episodes.get(scene, [])
                if 0 <= scene_count < len(scene_records):
                    record = dict(scene_records[scene_count])
                else:
                    record = manifest_episodes[episode_index] if episode_index < len(manifest_episodes) else {}
                    record = {
                        "scene": str(record.get("scene", scene) or scene),
                        "scene_episode_index": record.get("scene_episode_index", scene_count),
                        "episode_id": str(record.get("episode_id", "")),
                        "object_category": str(record.get("object_category", "") or ""),
                    }
                record["episode_index"] = episode_index
                episode_records.append(record)
                if scene:
                    scene_counts[scene] += 1
        final_mode = FINAL_MODE_RE.search(line)
        if final_mode:
            final_modes[final_mode.group(1)] += 1
            tristates[final_mode.group(4)] += 1

    failure_buckets = Counter(item["failure_cause"] for item in failures)
    goal_breakdown: Dict[str, Counter] = defaultdict(Counter)
    scene_breakdown: Dict[str, Counter] = defaultdict(Counter)
    for item in failures:
        goal_breakdown[item.get("goal", "unknown") or "unknown"][item["failure_cause"]] += 1
        scene_breakdown[item["scene"]][item["failure_cause"]] += 1

    metrics = {
        "episodes": completed_episodes,
        "sr": averages.get("success"),
        "sr_percent": averages.get("success", 0.0) * 100.0 if "success" in averages else None,
        "spl": averages.get("spl"),
        "spl_percent": averages.get("spl", 0.0) * 100.0 if "spl" in averages else None,
        "dtg": averages.get("distance_to_goal"),
        "steps": averages.get("num_steps"),
        "target_detected": averages.get("target_detected"),
        "stop_called": averages.get("stop_called"),
        "traveled_stairs": averages.get("traveled_stairs"),
        "final_check_call_count": averages.get("final_check_call_count"),
        "final_check_success_count": averages.get("final_check_success_count"),
        "final_check_parse_error_count": averages.get("final_check_parse_error_count"),
        "num_failures_logged": len(failures),
        "blip_score_count": len(blip_scores),
        "blip_score_avg": sum(blip_scores) / len(blip_scores) if blip_scores else None,
        "final_modes": dict(final_modes),
        "final_tristates": dict(tristates),
    }

    return {
        "metrics": metrics,
        "averages_raw": averages,
        "failure_buckets": dict(failure_buckets),
        "goal_failure_breakdown": {goal: dict(counter) for goal, counter in sorted(goal_breakdown.items())},
        "scene_failure_breakdown": {scene: dict(counter) for scene, counter in sorted(scene_breakdown.items())},
        "failure_records": failures,
        "episode_records": episode_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse ASCENT 100ep run.log into metrics and failure buckets.")
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()

    result_dir = args.result_dir
    parsed = parse_log(result_dir / "run.log", result_dir / "episodes_fixed.json")
    (result_dir / "metrics_summary.json").write_text(
        json.dumps(parsed["metrics"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (result_dir / "failure_buckets.json").write_text(
        json.dumps(
            {
                "failure_buckets": parsed["failure_buckets"],
                "goal_failure_breakdown": parsed["goal_failure_breakdown"],
                "scene_failure_breakdown": parsed["scene_failure_breakdown"],
                "failure_records": parsed["failure_records"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(parsed["metrics"], indent=2, sort_keys=True))
    print(json.dumps(parsed["failure_buckets"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
