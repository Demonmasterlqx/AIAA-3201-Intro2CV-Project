from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parent
PLAN_B_RUNS_ROOT = PROJECT_ROOT / "plan_b_runs"
REPO_DATA_ROOT = REPO_ROOT / "data"
ASCENT_HM3D_DATA_ROOT = Path("/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent/data")

DATASET_DEFAULTS = {
    "hm3d": {
        "split": "val",
        "dataset_path_template": "datasets/objectnav/hm3d/v1/{split}/{split}.json.gz",
        "scenes_dir": "scene_datasets",
        "config_name": "experiments/vlfm_frontier_objectnav_hm3d",
    },
    "mp3d": {
        "split": "val",
        "dataset_path_template": "datasets/objectnav/mp3d/v1/{split}/{split}.json.gz",
        "scenes_dir": "scene_datasets",
        "config_name": "experiments/vlfm_frontier_objectnav_hm3d",
    },
}


def require_dataset(dataset: str) -> Dict[str, str]:
    if dataset not in DATASET_DEFAULTS:
        raise ValueError(f"Unsupported dataset '{dataset}'. Expected one of {sorted(DATASET_DEFAULTS)}.")
    return DATASET_DEFAULTS[dataset]


def normalize_hm3d_source(dataset: str, hm3d_source: str) -> str:
    if dataset != "hm3d":
        return "repo"
    if hm3d_source not in {"repo", "ascent"}:
        raise ValueError(f"Unsupported hm3d source '{hm3d_source}'. Expected 'repo' or 'ascent'.")
    return hm3d_source


def get_source_data_root(dataset: str, hm3d_source: str = "repo") -> Path:
    if dataset == "hm3d" and normalize_hm3d_source(dataset, hm3d_source) == "ascent":
        return ASCENT_HM3D_DATA_ROOT
    return REPO_DATA_ROOT


def get_dataset_json_path(dataset: str, split: str, hm3d_source: str = "repo") -> Path:
    config = require_dataset(dataset)
    return get_source_data_root(dataset, hm3d_source) / config["dataset_path_template"].format(split=split)


def get_default_scenes_dir(dataset: str, hm3d_source: str = "repo") -> Path:
    config = require_dataset(dataset)
    return get_source_data_root(dataset, hm3d_source) / config["scenes_dir"]


def get_default_scene_dataset_config(dataset: str, hm3d_source: str = "repo") -> Optional[Path]:
    if dataset != "hm3d":
        return None
    return get_default_scenes_dir(dataset, hm3d_source) / "hm3d" / "hm3d_annotated_basis.scene_dataset_config.json"


def load_json_gz(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def list_content_scene_counts(dataset_json_path: Path) -> List[Tuple[str, int]]:
    content_dir = dataset_json_path.parent / "content"
    scene_counts: List[Tuple[str, int]] = []
    for path in sorted(content_dir.glob("*.json.gz")):
        payload = load_json_gz(path)
        scene_counts.append((path.stem.replace(".json", ""), len(payload.get("episodes", []))))
    return scene_counts


def load_sample_episode(dataset_json_path: Path, scene_name: Optional[str] = None) -> Dict[str, Any]:
    content_dir = dataset_json_path.parent / "content"
    if scene_name is None:
        content_files = sorted(content_dir.glob("*.json.gz"))
        if not content_files:
            raise FileNotFoundError(f"No content shard files found under {content_dir}")
        target_path = content_files[0]
    else:
        target_path = content_dir / f"{scene_name}.json.gz"
        if not target_path.exists():
            raise FileNotFoundError(f"Missing content shard {target_path}")

    payload = load_json_gz(target_path)
    episodes = payload.get("episodes", [])
    if not episodes:
        raise ValueError(f"No episodes found in {target_path}")
    return episodes[0]


def resolve_scene_path(scene_id: str, scenes_dir: Path) -> Path:
    return scenes_dir / scene_id


def runtime_workspace_root(run_root: Path) -> Path:
    return run_root / ".runtime_data"


def runtime_data_root(run_root: Path) -> Path:
    return runtime_workspace_root(run_root) / "data"


def runtime_dataset_json_path(run_root: Path, dataset: str, split: str) -> Path:
    config = require_dataset(dataset)
    return runtime_data_root(run_root) / config["dataset_path_template"].format(split=split)


def runtime_scenes_dir(run_root: Path) -> Path:
    return runtime_data_root(run_root) / "scene_datasets"


def create_runtime_data_workspace(
    run_root: Path,
    dataset_root: Path,
    scenes_dir: Path,
) -> Path:
    workspace_root = runtime_workspace_root(run_root)
    data_root = ensure_dir(runtime_data_root(run_root))
    _replace_with_symlink(data_root / "datasets", dataset_root / "datasets")
    _replace_with_symlink(data_root / "scene_datasets", scenes_dir)
    _replace_with_symlink(data_root / "dummy_policy.pth", REPO_DATA_ROOT / "dummy_policy.pth")
    _replace_with_symlink(data_root / "pointnav_weights.pth", REPO_DATA_ROOT / "pointnav_weights.pth")
    return workspace_root


def build_runtime_env(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = dict(base_env or os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) if existing_pythonpath == "" else str(REPO_ROOT) + os.pathsep + existing_pythonpath
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-vlfm")
    env.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache-vlfm")
    return env


def resolve_runtime_scene_dataset_config(run_root: Path, scene_dataset_config: str) -> Path:
    return (runtime_workspace_root(run_root) / scene_dataset_config).resolve()


def _replace_with_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.is_symlink():
        if link_path.resolve() == target_path.resolve():
            return
        link_path.unlink()
    elif link_path.exists():
        if link_path.is_dir():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    ensure_dir(link_path.parent)
    link_path.symlink_to(target_path)


def count_scene_episode_dirs(episodes_root: Path, scene_short: str) -> int:
    if not episodes_root.exists():
        return 0
    count = 0
    for candidate in episodes_root.glob(f"{scene_short}__ep_*"):
        if (candidate / "episode_stats.json").exists():
            count += 1
    return count


def load_episode_records(episodes_root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not episodes_root.exists():
        return records
    for path in sorted(episodes_root.glob("*__ep_*/episode_stats.json")):
        records.append(json.loads(path.read_text()))
    return records


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_commit_short() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def relpath(path: Path, start: Path) -> str:
    return str(path.resolve().relative_to(start.resolve()))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
