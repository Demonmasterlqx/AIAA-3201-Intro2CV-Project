#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from plan_b_common import (
    build_runtime_env,
    create_runtime_data_workspace,
    REPO_ROOT,
    ensure_dir,
    get_default_scene_dataset_config,
    get_dataset_json_path,
    get_default_scenes_dir,
    get_source_data_root,
    git_commit,
    git_commit_short,
    list_content_scene_counts,
    load_sample_episode,
    require_dataset,
    resolve_scene_path,
    resolve_runtime_scene_dataset_config,
    runtime_dataset_json_path,
    runtime_scenes_dir,
)


REQUIRED_MODULES = ("habitat", "habitat_baselines", "habitat_sim", "frontier_exploration", "hydra", "open3d", "torch")
REQUIRED_WEIGHTS = (
    "data/mobile_sam.pt",
    "data/groundingdino_swint_ogc.pth",
    "data/yolov7-e6e.pt",
    "data/pointnav_weights.pth",
    "data/dummy_policy.pth",
)
REQUIRED_PORTS = {
    "grounding_dino": 12181,
    "blip2_itm": 12182,
    "mobile_sam": 12183,
    "yolov7": 12184,
}
PROCESS_PATTERNS = {
    "grounding_dino": "vlfm.vlm.grounding_dino --port 12181",
    "blip2_itm": "vlfm.vlm.blip2itm --port 12182",
    "mobile_sam": "vlfm.vlm.sam --port 12183",
    "yolov7": "vlfm.vlm.yolov7 --port 12184",
}
REQUIRED_SENSOR_MARKERS = (
    "frontier_sensor:",
    "gps_sensor:",
    "compass_sensor:",
    "heading_sensor:",
    "frontier_exploration_map:",
    "traveled_stairs:",
)


def check_port(port: int) -> bool:
    try:
        sock = socket.socket()
        sock.settimeout(0.25)
        sock.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        try:
            sock.close()
        except Exception:
            pass


def check_process(pattern: str) -> bool:
    result = subprocess.run(
        ["pgrep", "-af", pattern],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def collect_port_info() -> Dict[str, Dict[str, Any]]:
    ports: Dict[str, Dict[str, Any]] = {}
    for name, port in REQUIRED_PORTS.items():
        open_via_socket = check_port(port)
        open_via_process = check_process(PROCESS_PATTERNS[name])
        ports[name] = {
            "port": port,
            "open": open_via_socket or open_via_process,
            "socket_open": open_via_socket,
            "process_open": open_via_process,
        }
    return ports


def collect_module_info() -> Dict[str, Dict[str, Any]]:
    modules: Dict[str, Dict[str, Any]] = {}
    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            modules[module_name] = {"ok": True, "version": getattr(module, "__version__", "unknown")}
        except Exception as exc:
            modules[module_name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return modules


def collect_weight_info() -> Dict[str, Dict[str, Any]]:
    weights: Dict[str, Dict[str, Any]] = {}
    for relative_path in REQUIRED_WEIGHTS:
        path = REPO_ROOT / relative_path
        weights[relative_path] = {
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }
    return weights


def run_config_dry_run(config_name: str, dataset_json: Path, scenes_dir: Path, split: str, workspace_root: Path) -> Dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "vlfm.run",
        "--config-name",
        config_name,
        "--cfg",
        "job",
        f"habitat.dataset.data_path={dataset_json}",
        f"habitat.dataset.scenes_dir={scenes_dir}",
        f"habitat_baselines.eval.split={split}",
    ]
    result = subprocess.run(
        command,
        cwd=workspace_root,
        env=build_runtime_env(),
        capture_output=True,
        text=True,
    )
    stdout = result.stdout
    markers = {marker.rstrip(":"): marker in stdout for marker in REQUIRED_SENSOR_MARKERS}
    return {
        "ok": result.returncode == 0 and all(markers.values()),
        "returncode": result.returncode,
        "markers": markers,
        "stdout_tail": stdout.splitlines()[-80:],
        "stderr_tail": result.stderr.splitlines()[-40:],
        "command": command,
    }


def run_preflight(
    dataset: str,
    split: str,
    scenes_dir: Path,
    output_path: Path,
    hm3d_source: str = "repo",
) -> Dict[str, Any]:
    dataset_config = require_dataset(dataset)
    dataset_root = get_source_data_root(dataset, hm3d_source)
    dataset_json = get_dataset_json_path(dataset, split, hm3d_source)
    run_root = output_path.parent
    workspace_root = create_runtime_data_workspace(run_root, dataset_root, scenes_dir)
    runtime_dataset_json = runtime_dataset_json_path(run_root, dataset, split)
    runtime_scene_root = runtime_scenes_dir(run_root)
    result: Dict[str, Any] = {
        "dataset": dataset,
        "hm3d_source": hm3d_source if dataset == "hm3d" else "repo",
        "split": split,
        "repo_root": str(REPO_ROOT),
        "git_commit": git_commit(),
        "git_commit_short": git_commit_short(),
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        },
        "dataset_json": str(dataset_json),
        "scenes_dir": str(scenes_dir),
        "runtime_workspace": str(workspace_root),
        "runtime_dataset_json": str(runtime_dataset_json),
        "runtime_scenes_dir": str(runtime_scene_root),
        "modules": collect_module_info(),
        "weights": collect_weight_info(),
        "ports": collect_port_info(),
        "errors": [],
        "warnings": [],
    }

    if not dataset_json.exists():
        result["errors"].append(f"Missing dataset json: {dataset_json}")
    else:
        content_scene_counts = list_content_scene_counts(dataset_json)
        result["content_scene_counts"] = [{"scene": scene, "episodes": count} for scene, count in content_scene_counts]
        result["content_scene_count"] = len(content_scene_counts)
        result["episode_count"] = sum(count for _, count in content_scene_counts)
        if not content_scene_counts:
            result["errors"].append(f"No content shard files found under {dataset_json.parent / 'content'}")
        else:
            sample_episode = load_sample_episode(dataset_json, content_scene_counts[0][0])
            resolved_scene = resolve_scene_path(sample_episode["scene_id"], scenes_dir)
            runtime_resolved_scene = resolve_scene_path(sample_episode["scene_id"], runtime_scene_root)
            result["sample_episode"] = {
                "episode_id": str(sample_episode["episode_id"]),
                "scene_id": sample_episode["scene_id"],
                "object_category": sample_episode.get("object_category", ""),
                "scene_dataset_config": sample_episode.get("scene_dataset_config"),
                "resolved_scene_path": str(resolved_scene),
                "resolved_scene_exists": resolved_scene.exists(),
                "runtime_resolved_scene_path": str(runtime_resolved_scene),
                "runtime_resolved_scene_exists": runtime_resolved_scene.exists(),
            }
            if not resolved_scene.exists():
                result["errors"].append(f"Resolved scene path does not exist: {resolved_scene}")
            scene_dataset_config = sample_episode.get("scene_dataset_config")
            if scene_dataset_config:
                resolved_scene_dataset_config = resolve_runtime_scene_dataset_config(run_root, scene_dataset_config)
                if not resolved_scene_dataset_config.exists():
                    result["warnings"].append(
                        f"Referenced scene_dataset_config is missing locally: {resolved_scene_dataset_config}"
                    )
                result["sample_episode"]["resolved_scene_dataset_config"] = str(resolved_scene_dataset_config)
                result["sample_episode"]["resolved_scene_dataset_config_exists"] = resolved_scene_dataset_config.exists()
            else:
                default_scene_dataset_config = get_default_scene_dataset_config(dataset, hm3d_source)
                if default_scene_dataset_config is not None:
                    result["sample_episode"]["default_scene_dataset_config"] = str(default_scene_dataset_config)

    for module_name, info in result["modules"].items():
        if not info["ok"]:
            result["errors"].append(f"Missing Python module '{module_name}': {info['error']}")

    for relative_path, info in result["weights"].items():
        if not info["exists"]:
            result["errors"].append(f"Missing weight/checkpoint file: {relative_path}")

    for port_name, info in result["ports"].items():
        if not info["open"]:
            result["errors"].append(f"Required VLM server port is closed: {port_name} ({info['port']})")

    if dataset_json.exists():
        dry_run = run_config_dry_run(
            dataset_config["config_name"],
            runtime_dataset_json,
            runtime_scene_root,
            split,
            workspace_root,
        )
        result["config_dry_run"] = dry_run
        if not dry_run["ok"]:
            result["errors"].append("Hydra/Habitat config dry-run failed or required sensors/measurements are missing.")

    result["status"] = "ok" if len(result["errors"]) == 0 else "error"
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight checks for VLFM Plan B evaluation.")
    parser.add_argument("--dataset", choices=sorted({"hm3d", "mp3d"}), required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--hm3d-source", choices=["repo", "ascent"], default="repo")
    parser.add_argument("--scenes-dir", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    dataset_defaults = require_dataset(args.dataset)
    split = args.split or dataset_defaults["split"]
    scenes_dir = Path(args.scenes_dir) if args.scenes_dir else get_default_scenes_dir(args.dataset, args.hm3d_source)
    output_path = Path(args.output) if args.output else REPO_ROOT / "project" / "preflight.json"
    result = run_preflight(args.dataset, split, scenes_dir, output_path, hm3d_source=args.hm3d_source)
    print(json.dumps(result, indent=2))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
