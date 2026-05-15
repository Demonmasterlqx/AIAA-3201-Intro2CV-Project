#!/usr/bin/env python3
"""Validate the local ASCENT data layout."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
PRETRAINED_DIR = REPO_ROOT / "pretrained_weights"


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: str


def _has_any(paths: Iterable[Path]) -> bool:
    return any(path.exists() for path in paths)


def _format_result(result: CheckResult) -> str:
    status = "OK" if result.ok else "MISSING"
    return f"[{status}] {result.name}: {result.details}"


def check_common_assets() -> List[CheckResult]:
    required_weights = [
        PRETRAINED_DIR / "mobile_sam.pt",
        PRETRAINED_DIR / "groundingdino_swint_ogc.pth",
        PRETRAINED_DIR / "dfine_x_obj2coco.pth",
        PRETRAINED_DIR / "rednet_semmap_mp3d_40.pth",
        PRETRAINED_DIR / "ram_plus_swin_large_14m.pth",
        PRETRAINED_DIR / "resnet50_places365.pth.tar",
    ]
    return [
        CheckResult(
            "DDPPO backbone",
            (DATA_DIR / "ddppo-models" / "gibson-2plus-resnet50.pth").exists(),
            "requires data/ddppo-models/gibson-2plus-resnet50.pth",
        ),
        CheckResult(
            "PointNav policy",
            (REPO_ROOT / "third_party" / "vlfm" / "data" / "pointnav_weights.pth").exists(),
            "requires third_party/vlfm/data/pointnav_weights.pth",
        ),
        CheckResult(
            "Core model weights",
            all(path.exists() for path in required_weights) and (PRETRAINED_DIR / "Qwen2.5-7b").exists(),
            "requires pretrained_weights plus Qwen2.5-7b",
        ),
    ]


def check_hm3d() -> List[CheckResult]:
    scene_root = DATA_DIR / "scene_datasets" / "hm3d"
    episode_root = DATA_DIR / "datasets" / "objectnav" / "hm3d" / "v1"
    return [
        CheckResult(
            "HM3D scene dataset config",
            (scene_root / "hm3d_annotated_basis.scene_dataset_config.json").exists(),
            f"looked for {(scene_root / 'hm3d_annotated_basis.scene_dataset_config.json').relative_to(REPO_ROOT)}",
        ),
        CheckResult(
            "HM3D scene assets",
            any(scene_root.glob("minival/*/*.basis.glb")) and any(scene_root.glob("minival/*/*.semantic.glb")),
            "requires *.basis.glb and *.semantic.glb under data/scene_datasets/hm3d",
        ),
        CheckResult(
            "HM3D ObjectNav episodes",
            any(episode_root.glob("*/*.json.gz")),
            "requires data/datasets/objectnav/hm3d/v1/{split}/{split}.json.gz",
        ),
    ]


def check_mp3d() -> List[CheckResult]:
    scene_root = DATA_DIR / "scene_datasets" / "mp3d"
    episode_root = DATA_DIR / "datasets" / "objectnav" / "mp3d" / "v1"
    scene_dirs = [path for path in scene_root.iterdir() if path.is_dir()] if scene_root.exists() else []
    missing_semantics = [
        path.name
        for path in scene_dirs
        if not ((path / f"{path.name}.scn").exists() or (path / "info_semantic.json").exists())
    ]

    semantic_detail = "all scene directories have semantic sidecars"
    if missing_semantics:
        preview = ", ".join(missing_semantics[:5])
        if len(missing_semantics) > 5:
            preview += ", ..."
        semantic_detail = f"missing semantic files for: {preview}"

    return [
        CheckResult(
            "MP3D scene geometry",
            any(scene_root.glob("*/*.glb")),
            "requires data/scene_datasets/mp3d/{scene_id}/{scene_id}.glb",
        ),
        CheckResult(
            "MP3D scene semantics",
            len(missing_semantics) == 0 and len(scene_dirs) > 0,
            semantic_detail,
        ),
        CheckResult(
            "MP3D ObjectNav episodes",
            any(episode_root.glob("*/*.json.gz")),
            "requires data/datasets/objectnav/mp3d/v1/{split}/{split}.json.gz",
        ),
    ]


def check_hssd() -> List[CheckResult]:
    scene_root = DATA_DIR / "scene_datasets" / "hssd-hab"
    episode_root = DATA_DIR / "datasets" / "objectnav" / "hssd-hab"
    return [
        CheckResult(
            "HSSD-Hab scene dataset config",
            (scene_root / "hssd-hab.scene_dataset_config.json").exists(),
            f"looked for {(scene_root / 'hssd-hab.scene_dataset_config.json').relative_to(REPO_ROOT)}",
        ),
        CheckResult(
            "HSSD-Hab scene instances",
            any((scene_root / "scenes").glob("*.scene_instance.json")),
            "requires data/scene_datasets/hssd-hab/scenes/*.scene_instance.json",
        ),
        CheckResult(
            "HSSD-Hab ObjectNav episodes",
            any(episode_root.glob("*/*.json.gz")),
            "requires data/datasets/objectnav/hssd-hab/{split}/{split}.json.gz",
        ),
    ]


def print_section(title: str, results: List[CheckResult]) -> bool:
    print(f"\n{title}")
    print("-" * len(title))
    for result in results:
        print(_format_result(result))
    return all(result.ok for result in results)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ASCENT dataset readiness.")
    parser.add_argument(
        "--dataset",
        choices=["all", "hm3d", "mp3d", "hssd-hab"],
        default="all",
        help="Limit checks to a single dataset.",
    )
    args = parser.parse_args()

    all_ok = print_section("Common Assets", check_common_assets())

    if args.dataset in ("all", "hm3d"):
        all_ok = print_section("HM3D", check_hm3d()) and all_ok
    if args.dataset in ("all", "mp3d"):
        all_ok = print_section("MP3D", check_mp3d()) and all_ok
    if args.dataset in ("all", "hssd-hab"):
        all_ok = print_section("HSSD-Hab", check_hssd()) and all_ok

    if all_ok:
        print("\nASCENT data check passed.")
        return 0

    print("\nASCENT data check found missing requirements.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
