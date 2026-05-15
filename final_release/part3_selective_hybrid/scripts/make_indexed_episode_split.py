#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def read_json_gz(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_gz(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))


def load_manifest(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("episodes"), list):
        raise ValueError(f"Manifest has no episodes list: {path}")
    return payload


def parse_indices(raw: str) -> List[int]:
    indices: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        indices.append(int(item))
    if not indices:
        raise ValueError("At least one episode index is required.")
    return indices


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an HM3D split from explicit fixed-manifest episode indices.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--indices", required=True, help="Comma-separated episode indices from source manifest.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--name", default="indexed_subset")
    args = parser.parse_args()

    source_manifest = load_manifest(args.source_manifest)
    all_records = source_manifest["episodes"]
    selected_indices = parse_indices(args.indices)
    selected_records = []
    for index in selected_indices:
        if index < 0 or index >= len(all_records):
            raise IndexError(f"Episode index {index} outside manifest length {len(all_records)}")
        record = dict(all_records[index])
        record["source_episode_index"] = index
        selected_records.append(record)

    top_payload = read_json_gz(args.source_root / "val.json.gz")
    selected_by_scene: Dict[str, set[int]] = defaultdict(set)
    for record in selected_records:
        selected_by_scene[str(record["scene"])].add(int(record["scene_episode_index"]))

    write_json_gz(
        args.output_root / "val.json.gz",
        {key: value for key, value in top_payload.items() if key != "episodes"} | {"episodes": []},
    )

    for scene, scene_episode_indices in selected_by_scene.items():
        scene_path = args.source_root / "content" / f"{scene}.json.gz"
        payload = read_json_gz(scene_path)
        filtered_payload = copy.deepcopy(payload)
        filtered_payload["episodes"] = [
            episode
            for scene_episode_index, episode in enumerate(payload.get("episodes", []))
            if scene_episode_index in scene_episode_indices
        ]
        if filtered_payload["episodes"]:
            write_json_gz(args.output_root / "content" / f"{scene}.json.gz", filtered_payload)

    output_manifest = {
        "name": args.name,
        "source_manifest": str(args.source_manifest),
        "source_root": str(args.source_root),
        "selected_indices": selected_indices,
        "sample_size": len(selected_records),
        "content_scenes": sorted(selected_by_scene.keys()),
        "episodes": selected_records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(output_manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(selected_records)} indexed episodes to {args.manifest}")
    print(f"Filtered Habitat split root: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
