#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "results" / "as_ascent" / "eval_1pct"
DEFAULT_CONFIG = "eval_as_ascent_hm3d_smoke"
DEFAULT_EPISODE_COUNT = 20
DEFAULT_MAX_SCENE_REPEAT_STEPS = 1
DEFAULT_PORTS = {
    "BLIP2ITM_PORT": "15182",
    "SAM_PORT": "15183",
    "GROUNDING_DINO_PORT": "15184",
    "RAM_PORT": "15185",
    "DFINE_PORT": "15186",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 1% HM3D eval for adaptive and naive_shared.")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--episode-count", type=int, default=DEFAULT_EPISODE_COUNT)
    parser.add_argument("--max-scene-repeat-steps", type=int, default=DEFAULT_MAX_SCENE_REPEAT_STEPS)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--parallelism", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--episodes-per-shard", type=int, default=None)
    parser.add_argument("--gpu-pool", default="")
    parser.add_argument("--gpu-map", default="")
    parser.add_argument(
        "--methods",
        default="adaptive,naive_shared",
        help="Comma-separated subset of methods to run: adaptive,naive_shared",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip runs and only build comparison artifacts from existing method outputs.",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Run selected methods but do not generate comparison outputs yet.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    commit_id = run_git(["git", "rev-parse", "HEAD"]).strip()
    run_root = RESULT_ROOT / args.run_id
    comparison_dir = run_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    all_methods = {
        "adaptive": "adaptive",
        "naive_shared": "naive_shared",
    }
    method_names = [name.strip() for name in args.methods.split(",") if name.strip()]
    unknown = [name for name in method_names if name not in all_methods]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")

    env = os.environ.copy()
    env.setdefault("HYDRA_FULL_ERROR", "1")
    for key, value in DEFAULT_PORTS.items():
        env.setdefault(key, value)
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    scene_ids = discover_hm3d_val_scene_ids(args.episode_count)
    episodes_per_shard = args.episodes_per_shard or max(
        1, (args.episode_count + args.parallelism - 1) // args.parallelism
    )
    shard_scenes = build_scene_shards(scene_ids, episodes_per_shard)
    gpu_pool = parse_gpu_pool(args.parallelism, args.gpu_pool)
    gpu_map = parse_gpu_map(args.gpu_map)

    if not args.aggregate_only:
        tasks: List[Dict[str, object]] = []
        for method_name in method_names:
            coordination_mode = all_methods[method_name]
            method_dir = run_root / method_name
            prepare_method_dir(
                method_dir=method_dir,
                method_name=method_name,
                config_name=args.config_name,
                episode_count=args.episode_count,
                max_scene_repeat_steps=args.max_scene_repeat_steps,
                max_episode_steps=args.max_episode_steps,
                commit_id=commit_id,
                parallelism=args.parallelism,
                shard_scenes=shard_scenes,
                gpu_pool=gpu_pool,
                gpu_map=gpu_map.get(method_name, []),
            )
            for shard_index, scenes in enumerate(shard_scenes):
                tasks.append(
                    {
                        "method_name": method_name,
                        "coordination_mode": coordination_mode,
                        "method_dir": method_dir,
                        "shard_index": shard_index,
                        "scene_ids": scenes,
                        "max_episode_steps": args.max_episode_steps,
                        "max_scene_repeat_steps": args.max_scene_repeat_steps,
                        "allowed_gpus": gpu_map.get(method_name, []) or gpu_pool,
                    }
                )

        run_sharded_tasks(
            tasks=tasks,
            config_name=args.config_name,
            commit_id=commit_id,
            base_env=env,
        )

        for method_name in method_names:
            method_dir = run_root / method_name
            consolidate_method_outputs(method_dir)
            with (method_dir / "run.log").open("a", encoding="utf-8") as handle:
                handle.write(f"finished_at={datetime.now().isoformat(timespec='seconds')}\n")
                handle.write("exit_code=0\n")

    if not args.skip_aggregate:
        aggregate_results(
            run_root=run_root,
            comparison_dir=comparison_dir,
            config_name=args.config_name,
            episode_count=args.episode_count,
            max_scene_repeat_steps=args.max_scene_repeat_steps,
            commit_id=commit_id,
            method_names=method_names,
        )
    return 0


def prepare_method_dir(
    *,
    method_dir: Path,
    method_name: str,
    config_name: str,
    episode_count: int,
    max_scene_repeat_steps: int,
    max_episode_steps: Optional[int],
    commit_id: str,
    parallelism: int,
    shard_scenes: List[List[str]],
    gpu_pool: List[str],
    gpu_map: List[str],
) -> None:
    if method_dir.exists():
        shutil.rmtree(method_dir)
    method_dir.mkdir(parents=True, exist_ok=True)
    (method_dir / "shards").mkdir(parents=True, exist_ok=True)
    log_path = method_dir / "run.log"
    header_lines = [
        f"method={method_name}",
        f"config_name={config_name}",
        f"test_episode_count={episode_count}",
        f"max_episode_steps={max_episode_steps if max_episode_steps is not None else 'config_default'}",
        "shuffle=False",
        f"max_scene_repeat_steps={max_scene_repeat_steps}",
        f"commit_id={commit_id}",
        f"parallelism={parallelism}",
        f"episodes_per_shard={max(1, len(shard_scenes[0])) if shard_scenes else 0}",
        f"gpu_pool={','.join(gpu_pool)}",
        f"gpu_map={','.join(gpu_map) if gpu_map else 'shared_pool'}",
        f"scene_ids={','.join(scene for scenes in shard_scenes for scene in scenes)}",
        f"started_at={datetime.now().isoformat(timespec='seconds')}",
    ]
    log_path.write_text("\n".join(header_lines) + "\n\n", encoding="utf-8")

def build_shard_command(
    *,
    method_dir: Path,
    coordination_mode: str,
    config_name: str,
    max_scene_repeat_steps: int,
    max_episode_steps: Optional[int],
    shard_index: int,
    scene_ids: List[str],
) -> List[str]:
    shard_dir = method_dir / "shards" / f"shard_{shard_index:02d}"
    videos_dir = shard_dir / "videos"
    tb_dir = shard_dir / "tb"
    episodes_dir = shard_dir / "episodes"
    shard_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        "-m",
        "ascent.run",
        f"--config-name={config_name}",
        "hydra.output_subdir=null",
        "hydra.run.dir=.",
        f"habitat_baselines.test_episode_count={len(scene_ids)}",
        "habitat.environment.iterator_options.shuffle=False",
        f"habitat.environment.iterator_options.max_scene_repeat_steps={max_scene_repeat_steps}",
        f"habitat_baselines.video_dir={videos_dir}",
        f"habitat_baselines.tensorboard_dir={tb_dir}",
        f"habitat_baselines.rl.policy.main_agent.as_ascent.coordination_mode={coordination_mode}",
        f"habitat.dataset.content_scenes=[{','.join(scene_ids)}]",
        (
            "habitat_baselines.rl.policy.main_agent.as_ascent.logging.log_dir_template="
            f"{episodes_dir}/scene-__SCENE_ID__--episode-__EPISODE_ID__--goal-__GOAL_NAME__"
        ),
    ]
    if max_episode_steps is not None:
        command.append(f"habitat.environment.max_episode_steps={max_episode_steps}")
    return command


def run_sharded_tasks(
    *,
    tasks: List[Dict[str, object]],
    config_name: str,
    commit_id: str,
    base_env: Dict[str, str],
) -> None:
    queue = list(tasks)
    running: List[Dict[str, object]] = []
    free_gpus = sorted({gpu for task in tasks for gpu in task["allowed_gpus"]})  # type: ignore[index]
    while queue or running:
        launched = False
        for task in list(queue):
            allowed = [gpu for gpu in task["allowed_gpus"] if gpu in free_gpus]  # type: ignore[index]
            if not allowed:
                continue
            gpu = allowed[0]
            free_gpus.remove(gpu)
            queue.remove(task)
            shard_index = int(task["shard_index"])  # type: ignore[arg-type]
            method_dir = task["method_dir"]  # type: ignore[assignment]
            method_name = str(task["method_name"])
            coordination_mode = str(task["coordination_mode"])
            scene_ids = list(task["scene_ids"])  # type: ignore[arg-type]
            max_scene_repeat_steps = int(task["max_scene_repeat_steps"])  # type: ignore[arg-type]
            max_episode_steps = task["max_episode_steps"]  # type: ignore[assignment]
            shard_dir = method_dir / "shards" / f"shard_{shard_index:02d}"
            shard_log = shard_dir / "run.log"
            command = build_shard_command(
                method_dir=method_dir,
                coordination_mode=coordination_mode,
                config_name=config_name,
                max_scene_repeat_steps=max_scene_repeat_steps,
                max_episode_steps=max_episode_steps,
                shard_index=shard_index,
                scene_ids=scene_ids,
            )
            env = dict(base_env)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            shard_header = [
                f"method={method_name}",
                f"config_name={config_name}",
                f"commit_id={commit_id}",
                f"shard_index={shard_index}",
                f"scene_ids={','.join(scene_ids)}",
                f"episode_count={len(scene_ids)}",
                f"coordination_mode={coordination_mode}",
                f"cuda_visible_devices={gpu}",
                f"command={' '.join(command)}",
                f"started_at={datetime.now().isoformat(timespec='seconds')}",
            ]
            shard_log.write_text("\n".join(shard_header) + "\n\n", encoding="utf-8")
            handle = shard_log.open("a", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            running.append(
                {
                    "gpu": gpu,
                    "task": task,
                    "process": process,
                    "handle": handle,
                    "shard_dir": shard_dir,
                    "method_dir": method_dir,
                }
            )
            launched = True
        if launched:
            continue
        time.sleep(5)
        for entry in list(running):
            process = entry["process"]  # type: ignore[assignment]
            return_code = process.poll()
            if return_code is None:
                continue
            handle = entry["handle"]  # type: ignore[assignment]
            handle.write(f"\nexit_code={return_code}\n")
            handle.write(f"finished_at={datetime.now().isoformat(timespec='seconds')}\n")
            handle.close()
            if return_code != 0:
                raise RuntimeError(
                    f"Shard failed for {entry['task']['method_name']} shard {entry['task']['shard_index']} with exit code {return_code}"
                )
            flatten_artifacts(entry["shard_dir"])  # type: ignore[arg-type]
            finalize_episode_artifacts(entry["shard_dir"])  # type: ignore[arg-type]
            free_gpus.append(str(entry["gpu"]))
            free_gpus = sorted(set(free_gpus))
            running.remove(entry)


def discover_hm3d_val_scene_ids(episode_count: int) -> List[str]:
    content_dir = REPO_ROOT / "data" / "datasets" / "objectnav" / "hm3d" / "v1" / "val" / "content"
    if not content_dir.exists():
        raise FileNotFoundError(content_dir)
    scene_ids = sorted(path.stem.split(".")[0] for path in content_dir.glob("*.json.gz"))
    if episode_count > len(scene_ids):
        raise ValueError(f"Requested {episode_count} episodes but only {len(scene_ids)} HM3D val content scenes are available")
    return scene_ids[:episode_count]


def build_scene_shards(scene_ids: List[str], episodes_per_shard: int) -> List[List[str]]:
    if episodes_per_shard <= 0:
        raise ValueError("episodes_per_shard must be positive")
    return [
        scene_ids[index : index + episodes_per_shard]
        for index in range(0, len(scene_ids), episodes_per_shard)
    ]


def parse_gpu_pool(parallelism: int, gpu_pool: str) -> List[str]:
    if gpu_pool:
        return [gpu.strip() for gpu in gpu_pool.split(",") if gpu.strip()]
    if parallelism == 4:
        return ["0", "4", "6", "7"]
    if parallelism == 2:
        return ["0", "6"]
    return ["0"]


def parse_gpu_map(gpu_map: str) -> Dict[str, List[str]]:
    parsed: Dict[str, List[str]] = {}
    if not gpu_map:
        return parsed
    for chunk in gpu_map.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid gpu map entry: {chunk}")
        method, gpus = chunk.split(":", 1)
        parsed[method.strip()] = [gpu.strip() for gpu in gpus.split(",") if gpu.strip()]
    return parsed


def flatten_artifacts(method_dir: Path) -> None:
    for subdir_name in ("videos", "tb"):
        subdir = method_dir / subdir_name
        if not subdir.exists():
            continue
        files = [p for p in subdir.rglob("*") if p.is_file()]
        for path in files:
            if path.parent == subdir:
                continue
            target = subdir / path.name
            if target.exists():
                stem = path.stem
                suffix = path.suffix
                counter = 1
                while target.exists():
                    target = subdir / f"{stem}_{counter}{suffix}"
                    counter += 1
            shutil.move(str(path), str(target))
        for child in sorted(subdir.iterdir(), reverse=True):
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)


def finalize_episode_artifacts(method_dir: Path) -> None:
    episodes_dir = method_dir / "episodes"
    current_dir = method_dir / "current"
    current_dir.mkdir(parents=True, exist_ok=True)
    summary_lines: List[str] = []
    event_lines: List[str] = []

    for episode_dir in sorted(p for p in episodes_dir.iterdir() if p.is_dir()):
        summary_path = episode_dir / "episode_summary.jsonl"
        events_path = episode_dir / "episode_events.jsonl"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8").strip())
        video_path = summary.get("video_path", "") or summary.get("rendered_video_path", "")
        resolved_video = ""
        if video_path:
            candidate = method_dir / "videos" / Path(str(video_path)).name
            if candidate.exists():
                episode_video = episode_dir / "video.mp4"
                shutil.copy2(candidate, episode_video)
                resolved_video = str(episode_video.as_posix())
        summary["video_path"] = resolved_video
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        summary_lines.append(json.dumps(summary, ensure_ascii=False))

        if events_path.exists():
            for raw in events_path.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                record = json.loads(raw)
                record.setdefault("scene_id", summary.get("scene_id", ""))
                record.setdefault("episode_id", summary.get("episode_id", ""))
                record.setdefault("goal_name", summary.get("goal_name", ""))
                record.setdefault("method", method_dir.name)
                event_lines.append(json.dumps(record, ensure_ascii=False))

    (current_dir / "episode_summary.jsonl").write_text(
        "\n".join(summary_lines) + ("\n" if summary_lines else ""),
        encoding="utf-8",
    )
    (current_dir / "episode_events.jsonl").write_text(
        "\n".join(event_lines) + ("\n" if event_lines else ""),
        encoding="utf-8",
    )


def consolidate_method_outputs(method_dir: Path) -> None:
    current_dir = method_dir / "current"
    videos_dir = method_dir / "videos"
    episodes_dir = method_dir / "episodes"
    shards_dir = method_dir / "shards"
    current_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)

    summary_lines: List[str] = []
    event_lines: List[str] = []
    for shard_dir in sorted(p for p in shards_dir.iterdir() if p.is_dir()):
        shard_current = shard_dir / "current" / "episode_summary.jsonl"
        shard_episodes = shard_dir / "episodes"
        shard_videos = shard_dir / "videos"
        if shard_episodes.exists():
            for episode_dir in sorted(p for p in shard_episodes.iterdir() if p.is_dir()):
                target_dir = episodes_dir / episode_dir.name
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.copytree(episode_dir, target_dir)
        if shard_videos.exists():
            for video_path in shard_videos.glob("*.mp4"):
                target = videos_dir / video_path.name
                if target.exists():
                    target.unlink()
                shutil.copy2(video_path, target)
        if shard_current.exists():
            for raw in shard_current.read_text(encoding="utf-8").splitlines():
                if raw.strip():
                    record = json.loads(raw)
                    record["video_path"] = _resolve_method_episode_video_path(method_dir, record)
                    summary_lines.append(json.dumps(record, ensure_ascii=False))
        shard_events = shard_dir / "current" / "episode_events.jsonl"
        if shard_events.exists():
            for raw in shard_events.read_text(encoding="utf-8").splitlines():
                if raw.strip():
                    event_lines.append(raw)

    summary_lines = sort_jsonl_records(summary_lines)
    event_lines = sort_jsonl_records(event_lines, event_log=True)
    (current_dir / "episode_summary.jsonl").write_text(
        "\n".join(summary_lines) + ("\n" if summary_lines else ""),
        encoding="utf-8",
    )
    (current_dir / "episode_events.jsonl").write_text(
        "\n".join(event_lines) + ("\n" if event_lines else ""),
        encoding="utf-8",
    )


def _resolve_method_episode_video_path(method_dir: Path, record: Dict[str, object]) -> str:
    scene_id = str(record.get("scene_id", ""))
    episode_id = str(record.get("episode_id", ""))
    goal_name = _slugify(str(record.get("goal_name", "")))
    episode_dir = method_dir / "episodes" / f"scene-{scene_id}--episode-{episode_id}--goal-{goal_name}"
    episode_video = episode_dir / "video.mp4"
    if episode_video.exists():
        return str(episode_video)
    rendered_video_path = str(record.get("rendered_video_path", "") or record.get("video_path", ""))
    if rendered_video_path:
        candidate = method_dir / "videos" / Path(rendered_video_path).name
        if candidate.exists():
            return str(candidate)
    return ""


def _slugify(value: str) -> str:
    return (
        str(value)
        .strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def aggregate_results(
    *,
    run_root: Path,
    comparison_dir: Path,
    config_name: str,
    episode_count: int,
    max_scene_repeat_steps: int,
    commit_id: str,
    method_names: List[str],
) -> None:
    method_records: Dict[str, List[Dict[str, object]]] = {}
    for method in method_names:
        consolidate_method_outputs(run_root / method)
        summary_path = run_root / method / "current" / "episode_summary.jsonl"
        records = [
            json.loads(line)
            for line in summary_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(records) != episode_count:
            raise RuntimeError(
                f"{method} expected {episode_count} summaries, found {len(records)}"
            )
        method_records[method] = records

    key_sets = {
        method: episode_keys(records)
        for method, records in method_records.items()
    }
    reference_keys = next(iter(key_sets.values()), set())
    for method, keys in key_sets.items():
        if keys != reference_keys:
            raise RuntimeError(f"{method} episode keys do not match the comparison reference set")

    rows: List[Dict[str, object]] = []
    summary: Dict[str, object] = {
        "config_name": config_name,
        "commit_id": commit_id,
        "test_episode_count": episode_count,
        "shuffle": False,
        "max_scene_repeat_steps": max_scene_repeat_steps,
        "episode_key_match": True,
        "scene_count": len({scene_id for scene_id, _ in reference_keys}),
        "episode_count": len(reference_keys),
        "methods": {},
    }

    for method, records in method_records.items():
        success_values = []
        spl_values = []
        spawn_total = 0.0
        reclaim_total = 0.0
        scenes = set()
        for record in records:
            metrics = record.get("metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}
            row = {
                "scene_id": record.get("scene_id", ""),
                "episode_id": record.get("episode_id", ""),
                "goal_name": record.get("goal_name", ""),
                "method": method,
                "success": metrics.get("success", 0.0),
                "spl": metrics.get("spl", 0.0),
                "num_steps": metrics.get("num_steps", 0.0),
                "spawn_count": record.get("spawn_count", metrics.get("spawn_count", 0.0)),
                "reclaim_count": record.get("reclaim_count", metrics.get("reclaim_count", 0.0)),
                "overlap_ratio": metrics.get(
                    "overlap_ratio",
                    record.get("policy_info", {}).get("overlap_ratio", 0.0),
                ),
                "video_path": record.get("video_path", ""),
                "failure_cause": record.get("failure_cause", ""),
            }
            rows.append(row)
            success_values.append(to_float(row["success"]))
            spl_values.append(to_float(row["spl"]))
            spawn_total += to_float(row["spawn_count"])
            reclaim_total += to_float(row["reclaim_count"])
            scenes.add(row["scene_id"])

        summary["methods"][method] = {
            "avg_success": mean(success_values),
            "avg_spl": mean(spl_values),
            "total_spawn_count": spawn_total,
            "total_reclaim_count": reclaim_total,
            "episode_count": len(records),
            "scene_count": len(scenes),
        }

    if "adaptive" in summary["methods"]:
        adaptive_spawn_total = summary["methods"]["adaptive"]["total_spawn_count"]  # type: ignore[index]
        summary["adaptive_zero_spawn"] = bool(adaptive_spawn_total == 0)
        if summary["adaptive_zero_spawn"]:
            summary["adaptive_note"] = "formal 1% slice had zero adaptive spawns"

    csv_path = comparison_dir / "per_episode_join.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "scene_id",
            "episode_id",
            "goal_name",
            "method",
            "success",
            "spl",
            "num_steps",
            "spawn_count",
            "reclaim_count",
            "overlap_ratio",
            "video_path",
            "failure_cause",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary_path = comparison_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sort_jsonl_records(lines: List[str], event_log: bool = False) -> List[str]:
    records = [json.loads(line) for line in lines if line.strip()]
    if event_log:
        records.sort(
            key=lambda record: (
                str(record.get("scene_id", "")),
                str(record.get("episode_id", "")),
                str(record.get("goal_name", "")),
                str(record.get("event_type", "")),
            )
        )
    else:
        records.sort(
            key=lambda record: (
                str(record.get("scene_id", "")),
                str(record.get("episode_id", "")),
                str(record.get("goal_name", "")),
            )
        )
    return [json.dumps(record, ensure_ascii=False) for record in records]


def episode_keys(records: Iterable[Dict[str, object]]) -> set[Tuple[str, str]]:
    return {
        (str(record.get("scene_id", "")), str(record.get("episode_id", "")))
        for record in records
    }


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    finite = [value for value in values if value == value and value not in (float("inf"), float("-inf"))]
    if not finite:
        return 0.0
    return sum(finite) / len(finite)


def to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return 0.0


def run_git(command: List[str]) -> str:
    return subprocess.check_output(command, cwd=REPO_ROOT, text=True)


if __name__ == "__main__":
    raise SystemExit(main())
