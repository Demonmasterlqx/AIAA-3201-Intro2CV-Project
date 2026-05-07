# VLFM Project Usage

This directory contains the reusable project scripts and documentation for the VLFM
course-project evaluation.

The committed source files here are:

- `project/README.md`
- `project/run_eval_pair.sh`
- `project/summarize_results.py`

The evaluation outputs are generated locally and are intentionally ignored by git.

## What This Produces

The paired evaluation compares:

- `Plan A`: dense value matching
- `Plan B`: frontier-restricted semantic exploration

For each run, the scripts generate:

- per-episode JSON stats
- aggregate metrics such as `Success Rate`, `SPL`, and `Soft SPL`
- representative trajectory videos
- extracted top-down map figures
- a short auto-generated report

## Prerequisites

Before running:

1. Activate the `vlfm` conda environment.
2. Start the VLM service processes with `./scripts/launch_vlm_servers.sh`.
3. Make sure the local dataset links exist:
   - `data/datasets`
   - `data/scene_datasets`

The repo currently uses local symlinks for those dataset paths, and they are ignored by
git because they are machine-specific.

If your machine does not match the default 4-GPU layout, override the server bindings
before launch. For example:

```bash
GROUNDING_DINO_CUDA_DEVICE=0 \
BLIP2ITM_CUDA_DEVICE=0 \
SAM_CUDA_DEVICE=0 \
YOLOV7_CUDA_DEVICE=0 \
./scripts/launch_vlm_servers.sh
```

## Run Commands

From the repo root:

```bash
conda run -n vlfm bash -lc './project/run_eval_pair.sh'
```

Useful examples:

```bash
# Small paired run on val_mini
conda run -n vlfm bash -lc 'EPISODES=5 SPLIT=val_mini ./project/run_eval_pair.sh'

# Fixed-step-budget comparison for exploration-efficiency analysis
conda run -n vlfm bash -lc \
  'EPISODES=10 SPLIT=val_mini MAX_EPISODE_STEPS=80 \
   RESULTS_DIR=project/results_budget80_ep10 \
   ARTIFACTS_DIR=project/artifacts_budget80_ep10 \
   REPORT_PATH=project/REPORT_budget80_ep10.md \
   ./project/run_eval_pair.sh'
```

## Output Layout

By default, outputs are written to:

- `project/results/plan_a_dense/`
- `project/results/plan_b_frontier/`
- `project/artifacts/summary.json`
- `project/artifacts/episode_comparison.csv`
- `project/artifacts/maps/`
- `project/artifacts/videos/`
- `project/REPORT.md`

When `RESULTS_DIR`, `ARTIFACTS_DIR`, or `REPORT_PATH` are overridden, the outputs are
written to those locations instead.

## Git Guidance

The generated data should not be uploaded.

Ignored by git:

- `project/results*/`
- `project/artifacts*/`
- `project/REPORT*.md`
- local dataset links under `data/`
- runtime caches such as `tb/` and `lockfiles/`

Files that should be kept and submitted are the code and documentation only:

- `project/README.md`
- `project/run_eval_pair.sh`
- `project/summarize_results.py`
- `scripts/launch_vlm_servers.sh`
- `vlfm/run.py`
- `vlfm/measurements/frontier_exploration_compat.py`

Suggested staging command:

```bash
git add .gitignore \
  project/.gitignore \
  project/README.md \
  project/run_eval_pair.sh \
  project/summarize_results.py \
  scripts/launch_vlm_servers.sh \
  vlfm/run.py \
  vlfm/measurements/frontier_exploration_compat.py
```

## Notes On The Compatibility Patch

The local HM3D install on this machine does not include semantic annotation files. To
keep the evaluation runnable, the repo adds a lightweight compatibility patch in
`vlfm/measurements/frontier_exploration_compat.py` and disables `draw_goal_aabbs` at
runtime. This keeps the metrics and videos usable without requiring changes to the
external dataset installation.

## Structured Plan B Pipeline

For the course-project Part 1 delivery, use the structured Plan B pipeline instead of
the older paired comparison helper.

Preflight only:

```bash
conda run -n vlfm python project/plan_b_preflight.py \
  --dataset hm3d \
  --split val_mini \
  --output project/plan_b_runs/hm3d/val_mini/smoke/preflight.json
```

Preflight using the HM3D 0.2 data shipped under `ascent/ascent/data`:

```bash
conda run -n vlfm python project/plan_b_preflight.py \
  --dataset hm3d \
  --hm3d-source ascent \
  --split val_mini \
  --output project/plan_b_runs/hm3d/val_mini/smoke_ascent/preflight.json
```

Run HM3D smoke test:

```bash
conda run -n vlfm python project/run_plan_b.py \
  --dataset hm3d \
  --split val_mini \
  --run-name smoke \
  --episodes 1 \
  --resume
```

Run HM3D smoke test with the `ascent` HM3D 0.2 dataset:

```bash
conda run -n vlfm python project/run_plan_b.py \
  --dataset hm3d \
  --hm3d-source ascent \
  --split val_mini \
  --run-name smoke_ascent \
  --episodes 1 \
  --resume
```

Run MP3D smoke test:

```bash
conda run -n vlfm python project/run_plan_b.py \
  --dataset mp3d \
  --split val \
  --run-name smoke \
  --episodes 1 \
  --scenes-dir /data/home/sim6g/code/aiaa3201_cv_project/data/MatterPort3D \
  --resume
```

The structured outputs are written to:

```text
project/plan_b_runs/{dataset}/{split}/{run_name}/
├── preflight.json
├── run_spec.json
├── report.md
├── summary/
│   ├── episodes.csv
│   ├── episodes.json
│   ├── metrics.csv
│   ├── metrics.json
│   └── metrics.md
├── episodes/
│   └── {scene_short}__ep_{episode_id}/
│       ├── episode_stats.json
│       ├── composite.mp4
│       ├── egocentric.mp4
│       └── topdown.mp4
└── scene_runs/
```

Notes:

- Always launch the VLM servers before `run_plan_b.py`.
- HM3D continues to use `data/scene_datasets`.
- For HM3D, `--hm3d-source ascent` switches both the task dataset and scene dataset
  to `/data/home/sim6g/code/aiaa3201_cv_project/ascent/ascent/data` through a
  per-run runtime workspace; the repo default remains unchanged.
- MP3D scene roots are not hardcoded; pass them through `--scenes-dir`.
- `--resume` skips scene shards whose expected episode count is already present in
  `episodes/`.
