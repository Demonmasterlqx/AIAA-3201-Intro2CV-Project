#!/usr/bin/env python3

import argparse
import csv
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2


EpisodeStats = Dict[str, Any]


def load_episode_stats(directory: Path) -> Dict[Tuple[str, str], EpisodeStats]:
    stats: Dict[Tuple[str, str], EpisodeStats] = {}
    if not directory.exists():
        return stats

    for path in sorted(directory.glob("*.json")):
        if path.stat().st_size == 0:
            continue
        with path.open("r") as handle:
            episode = json.load(handle)
        key = (str(episode["scene_id"]), str(episode["episode_id"]))
        stats[key] = episode
    return stats


def summarize(stats: Dict[Tuple[str, str], EpisodeStats]) -> Dict[str, Any]:
    if not stats:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "spl": 0.0,
            "soft_spl": 0.0,
            "failure_causes": {},
        }

    episodes = list(stats.values())
    failure_causes = Counter(str(ep.get("failure_cause", "unknown")) for ep in episodes)

    def avg(key: str) -> float:
        values = [float(ep.get(key, 0.0)) for ep in episodes]
        return sum(values) / len(values)

    return {
        "episodes": len(episodes),
        "success_rate": avg("success"),
        "spl": avg("spl"),
        "soft_spl": avg("soft_spl"),
        "failure_causes": dict(failure_causes),
    }


def write_episode_comparison(
    csv_path: Path,
    dense_stats: Dict[Tuple[str, str], EpisodeStats],
    frontier_stats: Dict[Tuple[str, str], EpisodeStats],
) -> None:
    common_keys = sorted(set(dense_stats) & set(frontier_stats))
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scene_id",
                "episode_id",
                "target_object",
                "dense_success",
                "frontier_success",
                "dense_spl",
                "frontier_spl",
                "dense_soft_spl",
                "frontier_soft_spl",
                "dense_failure_cause",
                "frontier_failure_cause",
            ]
        )
        for key in common_keys:
            dense_ep = dense_stats[key]
            frontier_ep = frontier_stats[key]
            writer.writerow(
                [
                    key[0],
                    key[1],
                    frontier_ep.get("target_object", dense_ep.get("target_object", "")),
                    dense_ep.get("success", 0),
                    frontier_ep.get("success", 0),
                    dense_ep.get("spl", 0.0),
                    frontier_ep.get("spl", 0.0),
                    dense_ep.get("soft_spl", 0.0),
                    frontier_ep.get("soft_spl", 0.0),
                    dense_ep.get("failure_cause", ""),
                    frontier_ep.get("failure_cause", ""),
                ]
            )


def parse_video_metrics(path: Path) -> Tuple[float, int]:
    name = path.name
    spl_match = re.search(r"spl=([0-9.]+)", name)
    succ_match = re.search(r"succ=([01])", name)
    spl = float(spl_match.group(1)) if spl_match else -1.0
    succ = int(succ_match.group(1)) if succ_match else -1
    return spl, succ


def choose_representative_video(video_dir: Path) -> Optional[Path]:
    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        return None
    ranked = sorted(
        videos,
        key=lambda path: parse_video_metrics(path),
        reverse=True,
    )
    return ranked[0]


def extract_last_frame(video_path: Path, output_path: Path) -> Optional[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    last_frame = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        last_frame = frame
    cap.release()

    if last_frame is None:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), last_frame)
    return output_path


def extract_map_crop(frame_path: Path, output_path: Path) -> Optional[Path]:
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return None
    width = frame.shape[1]
    crop = frame[:, width // 2 :]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), crop)
    return output_path


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def sorted_failure_lines(summary: Dict[str, Any]) -> List[str]:
    items = sorted(summary["failure_causes"].items(), key=lambda item: item[1], reverse=True)
    if not items:
        return ["- none"]
    return [f"- {cause}: {count}" for cause, count in items]


def build_report(
    report_path: Path,
    dense_summary: Dict[str, Any],
    frontier_summary: Dict[str, Any],
    dense_video: Optional[Path],
    frontier_video: Optional[Path],
    dense_map: Optional[Path],
    frontier_map: Optional[Path],
    comparison_csv: Path,
) -> None:
    lines = [
        "# VLFM Project Report",
        "",
        "## Exploration Efficiency Under Different Conditions",
        "",
        "The project compares two VLFM exploration conditions on the same split:",
        "",
        "- Plan A: dense value matching (`HabitatDenseValueMatchingPolicy`)",
        "- Plan B: frontier-restricted semantic exploration (`HabitatITMPolicyV2`)",
        "",
        "| Condition | Episodes | Success Rate | SPL | Soft SPL |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Plan A Dense | {dense_summary['episodes']} | {pct(dense_summary['success_rate'])} | {pct(dense_summary['spl'])} | {pct(dense_summary['soft_spl'])} |",
        f"| Plan B Frontier | {frontier_summary['episodes']} | {pct(frontier_summary['success_rate'])} | {pct(frontier_summary['spl'])} | {pct(frontier_summary['soft_spl'])} |",
        "",
        "A higher SPL indicates more efficient exploration because success is discounted by path inefficiency.",
        "",
        "## Failure and Occlusion Analysis",
        "",
        "The occlusion limitation of VLFM comes from the policy design itself:",
        "",
        "- The semantic value map is projected into a 2D top-down map, so vertical structure is discarded.",
        "- A high VLM score can be triggered by room context, partial target visibility, or nearby co-occurring objects instead of a fully reachable target.",
        "- Under severe occlusion, the projected score can still make a hidden or partially visible region look promising, even when the target is not yet directly observable from that frontier.",
        "- Frontier restriction improves discipline, but it only constrains where the agent explores next. It does not fix incorrect semantic evidence.",
        "",
        "Observed failure-cause counts in this run:",
        "",
        "Plan A Dense:",
        *sorted_failure_lines(dense_summary),
        "",
        "Plan B Frontier:",
        *sorted_failure_lines(frontier_summary),
        "",
        "If false negatives or repeated non-detection failures appear, they are consistent with the expected limitation of relying on 2D projections and simple VLM similarity when the goal is heavily occluded.",
        "",
        "## Visual Artifacts",
        "",
        f"- Per-episode comparison table: `{comparison_csv.relative_to(report_path.parent)}`",
        f"- Representative dense trajectory video: `{dense_video.relative_to(report_path.parent)}`" if dense_video else "- Representative dense trajectory video: not available",
        f"- Representative frontier trajectory video: `{frontier_video.relative_to(report_path.parent)}`" if frontier_video else "- Representative frontier trajectory video: not available",
        f"- Dense top-down semantic map figure: `{dense_map.relative_to(report_path.parent)}`" if dense_map else "- Dense top-down semantic map figure: not available",
        f"- Frontier top-down semantic map figure: `{frontier_map.relative_to(report_path.parent)}`" if frontier_map else "- Frontier top-down semantic map figure: not available",
        "",
        "The extracted map figures come from the right half of the saved trajectory frames, where VLFM renders the Habitat top-down map together with the obstacle map and semantic value map.",
    ]
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize paired VLFM evaluation runs.")
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--dense-dir", required=True)
    parser.add_argument("--frontier-dir", required=True)
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    dense_dir = Path(args.dense_dir)
    frontier_dir = Path(args.frontier_dir)
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else project_dir / "artifacts"
    maps_dir = artifacts_dir / "maps"
    videos_dir = artifacts_dir / "videos"
    report_path = Path(args.report_path) if args.report_path else project_dir / "REPORT.md"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    dense_stats = load_episode_stats(dense_dir / "episode_stats")
    frontier_stats = load_episode_stats(frontier_dir / "episode_stats")
    dense_summary = summarize(dense_stats)
    frontier_summary = summarize(frontier_stats)

    summary = {
        "plan_a_dense": dense_summary,
        "plan_b_frontier": frontier_summary,
    }
    (artifacts_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    comparison_csv = artifacts_dir / "episode_comparison.csv"
    write_episode_comparison(comparison_csv, dense_stats, frontier_stats)

    dense_video_src = choose_representative_video(dense_dir / "videos")
    frontier_video_src = choose_representative_video(frontier_dir / "videos")

    dense_video_dst = None
    frontier_video_dst = None
    if dense_video_src is not None:
        dense_video_dst = videos_dir / "plan_a_dense_representative.mp4"
        shutil.copy2(dense_video_src, dense_video_dst)
    if frontier_video_src is not None:
        frontier_video_dst = videos_dir / "plan_b_frontier_representative.mp4"
        shutil.copy2(frontier_video_src, frontier_video_dst)

    dense_frame = extract_last_frame(
        dense_video_src, maps_dir / "plan_a_dense_last_frame.png"
    ) if dense_video_src is not None else None
    frontier_frame = extract_last_frame(
        frontier_video_src, maps_dir / "plan_b_frontier_last_frame.png"
    ) if frontier_video_src is not None else None

    dense_map = extract_map_crop(dense_frame, maps_dir / "plan_a_dense_maps.png") if dense_frame else None
    frontier_map = extract_map_crop(frontier_frame, maps_dir / "plan_b_frontier_maps.png") if frontier_frame else None

    build_report(
        report_path,
        dense_summary,
        frontier_summary,
        dense_video_dst,
        frontier_video_dst,
        dense_map,
        frontier_map,
        comparison_csv,
    )


if __name__ == "__main__":
    main()
