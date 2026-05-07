#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from vlfm.utils.eval_artifacts import parse_video_views

from plan_b_common import (
    PLAN_B_RUNS_ROOT,
    REPO_ROOT,
    build_runtime_env,
    count_scene_episode_dirs,
    create_runtime_data_workspace,
    ensure_dir,
    get_dataset_json_path,
    get_default_scenes_dir,
    get_source_data_root,
    git_commit,
    git_commit_short,
    list_content_scene_counts,
    require_dataset,
    runtime_dataset_json_path,
    runtime_scenes_dir,
    runtime_workspace_root,
)
from plan_b_preflight import run_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structured VLFM Plan B evaluation.")
    parser.add_argument("--dataset", choices=sorted({"hm3d", "mp3d"}), required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--hm3d-source", choices=["repo", "ascent"], default="repo")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--episodes", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--scenes-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--video-views", default="composite,egocentric,topdown")
    parser.add_argument("--max-episode-steps", type=int, default=None)
    return parser.parse_args()


def build_run_spec(
    dataset: str,
    hm3d_source: str,
    split: str,
    run_name: str,
    episodes: int,
    seed: int,
    scenes_dir: Path,
    dataset_json: Path,
    video_views: List[str],
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "hm3d_source": hm3d_source if dataset == "hm3d" else "repo",
        "split": split,
        "run_name": run_name,
        "episodes": episodes,
        "seed": seed,
        "scenes_dir": str(scenes_dir),
        "dataset_json": str(dataset_json),
        "policy_name": "HabitatITMPolicyV2",
        "video_views": video_views,
        "git_commit": git_commit(),
        "git_commit_short": git_commit_short(),
        "python_executable": sys.executable,
    }


def build_eval_command(
    dataset_json: Path,
    split: str,
    scenes_dir: Path,
    scene_short: str,
    target_count: int,
    scene_run_dir: Path,
    seed: int,
    max_episode_steps: Optional[int],
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "vlfm.run",
        "--config-name",
        "experiments/vlfm_frontier_objectnav_hm3d",
        "habitat_baselines.evaluate=True",
        "habitat_baselines.load_resume_state_config=False",
        "habitat_baselines.num_environments=1",
        f"habitat.seed={seed}",
        f"habitat.dataset.data_path={dataset_json}",
        f"habitat.dataset.scenes_dir={scenes_dir}",
        f"habitat.dataset.content_scenes=['{scene_short}']",
        f"habitat_baselines.eval.split={split}",
        f"habitat_baselines.test_episode_count={target_count}",
        f"habitat_baselines.video_dir={scene_run_dir / 'legacy_videos'}",
        'habitat_baselines.eval.video_option=["disk"]',
        "habitat.task.measurements.frontier_exploration_map.draw_goal_aabbs=False",
        f"hydra.run.dir={scene_run_dir / 'hydra'}",
    ]
    if max_episode_steps is not None:
        command.extend(
            [
                f"habitat.environment.max_episode_steps={max_episode_steps}",
                f"habitat.task.measurements.frontier_exploration_map.max_episode_steps={max_episode_steps}",
            ]
        )
    return command


def main() -> None:
    args = parse_args()
    dataset_defaults = require_dataset(args.dataset)
    split = args.split or dataset_defaults["split"]
    scenes_dir = Path(args.scenes_dir) if args.scenes_dir else get_default_scenes_dir(args.dataset, args.hm3d_source)
    dataset_root = get_source_data_root(args.dataset, args.hm3d_source)
    dataset_json = get_dataset_json_path(args.dataset, split, args.hm3d_source)
    run_root = ensure_dir(PLAN_B_RUNS_ROOT / args.dataset / split / args.run_name)
    episodes_root = ensure_dir(run_root / "episodes")
    scene_runs_root = ensure_dir(run_root / "scene_runs")
    video_views = parse_video_views(args.video_views)
    runtime_workspace = create_runtime_data_workspace(run_root, dataset_root, scenes_dir)
    runtime_dataset_json = runtime_dataset_json_path(run_root, args.dataset, split)
    runtime_scene_root = runtime_scenes_dir(run_root)

    preflight_path = run_root / "preflight.json"
    preflight = run_preflight(args.dataset, split, scenes_dir, preflight_path, hm3d_source=args.hm3d_source)
    if preflight["status"] != "ok":
        raise SystemExit("Preflight failed. See preflight.json for details.")

    content_scene_counts = list_content_scene_counts(dataset_json)
    requested_total = sum(count for _, count in content_scene_counts)
    if args.episodes >= 0:
        requested_total = min(requested_total, args.episodes)

    run_spec = build_run_spec(
        args.dataset,
        args.hm3d_source,
        split,
        args.run_name,
        args.episodes,
        args.seed,
        scenes_dir,
        dataset_json,
        video_views,
    )
    run_spec["requested_total_episodes"] = requested_total
    (run_root / "run_spec.json").write_text(json.dumps(run_spec, indent=2) + "\n")

    remaining = None if args.episodes < 0 else args.episodes
    for scene_short, scene_episode_count in content_scene_counts:
        if remaining is not None and remaining <= 0:
            break

        target_count = scene_episode_count if remaining is None else min(scene_episode_count, remaining)
        completed_count = count_scene_episode_dirs(episodes_root, scene_short)
        scene_run_dir = ensure_dir(scene_runs_root / scene_short)
        scene_status_path = scene_run_dir / "status.json"
        if args.resume and completed_count >= target_count:
            scene_status_path.write_text(
                json.dumps(
                    {
                        "scene": scene_short,
                        "status": "skipped_complete",
                        "expected_episode_count": target_count,
                        "completed_episode_count": completed_count,
                    },
                    indent=2,
                )
                + "\n"
            )
            if remaining is not None:
                remaining -= target_count
            continue

        command = build_eval_command(
            dataset_json=runtime_dataset_json,
            split=split,
            scenes_dir=runtime_scene_root,
            scene_short=scene_short,
            target_count=target_count,
            scene_run_dir=scene_run_dir,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
        )
        env = build_runtime_env(os.environ.copy())
        env.update(
            {
                "VLFM_EPISODE_ROOT_DIR": str(episodes_root),
                "VLFM_VIDEO_VIEWS": ",".join(video_views),
                "VLFM_DATASET_NAME": args.dataset,
                "VLFM_RUN_SEED": str(args.seed),
            }
        )
        log_path = scene_run_dir / "run.log"
        with log_path.open("w") as log_handle:
            process = subprocess.run(
                command,
                cwd=runtime_workspace,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )

        completed_after = count_scene_episode_dirs(episodes_root, scene_short)
        scene_status = {
            "scene": scene_short,
            "status": "ok" if process.returncode == 0 and completed_after >= target_count else "failed",
            "expected_episode_count": target_count,
            "completed_episode_count_before": completed_count,
            "completed_episode_count_after": completed_after,
            "returncode": process.returncode,
            "log_path": str(log_path.relative_to(run_root)),
            "command": command,
        }
        scene_status_path.write_text(json.dumps(scene_status, indent=2) + "\n")
        if process.returncode != 0:
            raise SystemExit(f"Scene run failed for {scene_short}. See {log_path}.")
        if completed_after < target_count:
            raise SystemExit(
                f"Scene {scene_short} finished without enough episodes. Expected {target_count}, got {completed_after}."
            )

        if remaining is not None:
            remaining -= target_count

    summarize_command = [sys.executable, str(REPO_ROOT / "project" / "summarize_plan_b.py"), "--run-root", str(run_root)]
    subprocess.run(summarize_command, cwd=REPO_ROOT, check=True)
    print(f"Plan B run finished. Outputs are under {run_root}")


if __name__ == "__main__":
    main()
