# ASCENT Data Guide

This repository uses `data/` for three different kinds of assets:

- Scene datasets under `data/scene_datasets/`
- ObjectNav episode datasets under `data/datasets/objectnav/`
- Policy backbone weights under `data/ddppo-models/`

## Dataset Mapping

- `HM3D`
  - Scenes: `data/scene_datasets/hm3d`
  - Episodes: `data/datasets/objectnav/hm3d/v1/{split}/{split}.json.gz`
  - Eval config: `experiments/eval_ascent_hm3d.yaml`
  - Current repo status: HM3D scene assets are already present through the `data/versioned_data/hm3d-0.2` link target, but ObjectNav episodes still need to be added.

- `MP3D`
  - Scenes: `data/scene_datasets/mp3d`
  - Episodes: `data/datasets/objectnav/mp3d/v1/{split}/{split}.json.gz`
  - Eval config: `experiments/eval_ascent_mp3d.yaml`
  - Current repo status: ObjectNav episodes may be present while scene semantics are still missing. For ObjectNav evaluation, each scene needs the Habitat semantic sidecar files, not just `{scene_id}.glb`.

- `HSSD-Hab`
  - Scenes: `data/scene_datasets/hssd-hab`
  - Episodes: `data/datasets/objectnav/hssd-hab/{split}/{split}.json.gz`
  - Eval config: `experiments/eval_ascent_hssd.yaml`
  - Current repo status: ASCENT now recognizes `hssd-hab` as a dataset type, but floor priors fall back to zero when no HSSD-specific prior file exists.

## Repo-Local Layout

- `data/scene_datasets/hm3d` can point to `data/versioned_data/hm3d-0.2/hm3d`
  - expected scene files include:
    - `*.basis.glb`
    - `*.basis.navmesh`
    - `*.semantic.glb`
    - `*.semantic.txt`

- `data/scene_datasets/mp3d` is expected to contain one directory per scene:
  - `{scene_id}/{scene_id}.glb`
  - plus Habitat semantic scene files such as `{scene_id}.scn` or `info_semantic.json`

- `data/scene_datasets/hssd-hab` is expected to expose:
  - `hssd-hab.scene_dataset_config.json`
  - `scenes/*.scene_instance.json`

## Quick Readiness Check

Run:

```bash
python scripts/check_ascent_data.py --dataset all
```

This prints which datasets are ready, partial, or blocked, and highlights the missing files required for ASCENT evaluation.
