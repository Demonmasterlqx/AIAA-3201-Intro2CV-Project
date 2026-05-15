#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_SCENES = [
    "4ok3usBNeis",
    "5cdEh9F2hJL",
    "6s7QHgap2fW",
    "DYehNKdT76V",
    "Dd4bFSTQ8gi",
    "Nfvxx8J5NCo",
    "QaLdnwvtxbs",
    "TEEsavR23oF",
    "XB4GS9ShBRE",
    "bxsVRursffK",
    "cvZr5TUy5C5",
    "mL8ThkuaVTM",
    "mv2HUxq3B53",
    "p53SfW6mjZe",
    "q3zU7Yy5E5s",
    "qyAac8rV8Zk",
    "svBbv1Pavdk",
    "wcojb4TFT35",
    "ziup5kvtCCR",
    "zt1RVoi7PcG",
]


def read_json_gz(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_gz(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))


def scene_from_episode(episode: Dict[str, Any], fallback: str) -> str:
    scene_id = str(episode.get("scene_id", ""))
    for chunk in scene_id.split("/"):
        if "-" in chunk:
            candidate = chunk.split("-")[-1]
            if candidate:
                return candidate
    if scene_id.endswith(".basis.glb"):
        return Path(scene_id).stem.replace(".basis", "")
    return fallback


def load_scene_episodes(source_root: Path, scenes: Iterable[str]) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    scene_payloads: Dict[str, Dict[str, Any]] = {}
    records: List[Dict[str, Any]] = []
    for scene in scenes:
        scene_path = source_root / "content" / f"{scene}.json.gz"
        if not scene_path.exists():
            raise FileNotFoundError(f"Missing HM3D content file: {scene_path}")
        payload = read_json_gz(scene_path)
        scene_payloads[scene] = payload
        for scene_episode_index, episode in enumerate(payload.get("episodes", [])):
            records.append(
                {
                    "scene": scene_from_episode(episode, scene),
                    "scene_episode_index": scene_episode_index,
                    "episode_id": str(episode.get("episode_id")),
                    "object_category": episode.get("object_category", ""),
                    "episode": episode,
                }
            )
    return scene_payloads, records


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a deterministic 100-episode HM3D ObjectNav split.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/data/home/sim6g/code/aiaa3201_cv_project/data/ascent_imported/datasets/objectnav/hm3d/v2/val"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/localqwen_100ep_fixed/v2/val"),
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/localqwen_100ep_fixed/episodes_100_localqwen_fixed.json"))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3201)
    parser.add_argument("--scenes", nargs="*", default=DEFAULT_SCENES)
    args = parser.parse_args()

    top_payload = read_json_gz(args.source_root / "val.json.gz")
    scene_payloads, all_records = load_scene_episodes(args.source_root, args.scenes)
    if len(all_records) < args.sample_size:
        raise ValueError(f"Requested {args.sample_size} episodes, but only found {len(all_records)}.")

    rng = random.Random(args.seed)
    shuffled = list(all_records)
    rng.shuffle(shuffled)
    selected = shuffled[: args.sample_size]

    selected_by_scene: Dict[str, set[int]] = defaultdict(set)
    for record in selected:
        selected_by_scene[record["scene"]].add(record["scene_episode_index"])

    output_root = args.output_root
    write_json_gz(
        output_root / "val.json.gz",
        {
            key: value
            for key, value in top_payload.items()
            if key != "episodes"
        }
        | {"episodes": []},
    )

    for scene, payload in scene_payloads.items():
        filtered_payload = copy.deepcopy(payload)
        keep_indices = selected_by_scene.get(scene, set())
        filtered_payload["episodes"] = [
            episode
            for scene_episode_index, episode in enumerate(payload.get("episodes", []))
            if scene_episode_index in keep_indices
        ]
        if filtered_payload["episodes"]:
            write_json_gz(output_root / "content" / f"{scene}.json.gz", filtered_payload)

    manifest = {
        "seed": args.seed,
        "sample_size": args.sample_size,
        "source_root": str(args.source_root),
        "output_data_path": str(output_root / "{split}.json.gz"),
        "content_scenes": sorted(selected_by_scene.keys()),
        "episodes": [
            {
                "scene": record["scene"],
                "scene_episode_index": record["scene_episode_index"],
                "episode_id": record["episode_id"],
                "object_category": record["object_category"],
            }
            for record in selected
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(selected)} fixed episodes to {args.manifest}")
    print(f"Filtered Habitat split root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
