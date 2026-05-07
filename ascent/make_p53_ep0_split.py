#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
from pathlib import Path

base = Path("/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent/data/datasets/objectnav/hm3d/v1")
src_split = base / "val"
dst_split = base / "val_p53_ep0"
dst_content = dst_split / "content"
dst_content.mkdir(parents=True, exist_ok=True)

with gzip.open(src_split / "val.json.gz", "rt", encoding="utf-8") as f:
    val_meta = json.load(f)
with gzip.open(src_split / "content/p53SfW6mjZe.json.gz", "rt", encoding="utf-8") as f:
    p53 = json.load(f)

episode = p53["episodes"][0]
assert episode["object_category"] == "sofa", episode["object_category"]
assert episode["scene_id"].endswith("p53SfW6mjZe.basis.glb"), episode["scene_id"]

single_content = dict(p53)
single_content["episodes"] = [episode]

single_meta = dict(val_meta)
single_meta["episodes"] = []

with gzip.open(dst_split / "val_p53_ep0.json.gz", "wt", encoding="utf-8") as f:
    json.dump(single_meta, f)
with gzip.open(dst_content / "p53SfW6mjZe.json.gz", "wt", encoding="utf-8") as f:
    json.dump(single_content, f)

print(f"wrote {dst_split / 'val_p53_ep0.json.gz'}")
print(f"wrote {dst_content / 'p53SfW6mjZe.json.gz'}")
print(json.dumps({
    "episode_id": episode["episode_id"],
    "scene_id": episode["scene_id"],
    "object_category": episode["object_category"],
    "start_position": episode["start_position"],
    "start_rotation": episode["start_rotation"],
}, indent=2))
