import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

from plan_b_common import ASCENT_HM3D_DATA_ROOT, create_runtime_data_workspace, get_dataset_json_path, get_default_scenes_dir
from vlfm.utils.img_utils import pixel_value_within_radius, place_img_in_img
from vlfm.utils.eval_artifacts import parse_video_views, scene_id_to_short
from vlfm.utils.habitat_visualizer import color_point_cloud_on_map


def test_parse_video_views_and_scene_short() -> None:
    assert parse_video_views("all") == ["composite", "egocentric", "topdown"]
    assert parse_video_views("composite,topdown") == ["composite", "topdown"]
    assert scene_id_to_short("hm3d/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb") == "4ok3usBNeis"


def test_img_utils_out_of_bounds_helpers_do_not_crash() -> None:
    base = np.zeros((6, 6), dtype=np.float32)
    patch = np.ones((4, 4), dtype=np.float32)
    placed = place_img_in_img(base.copy(), patch, row=8, col=3)
    assert placed.shape == base.shape
    assert np.count_nonzero(placed) == 0
    partial = place_img_in_img(base.copy(), patch, row=5, col=5)
    assert np.count_nonzero(partial) > 0
    assert pixel_value_within_radius(base, (20, 20), 1) == -1


def test_color_point_cloud_on_map_ignores_out_of_bounds_points() -> None:
    infos = [
        {
            "top_down_map": {
                "upper_bound": (10, 10),
                "lower_bound": (0, 0),
                "grid_resolution": (4, 4),
                "tf_episodic_to_global": np.eye(4),
                "map": np.zeros((4, 4), dtype=np.uint8),
            }
        }
    ]
    policy_info = [{"target_point_cloud": np.array([[100.0, 100.0, 0.0]])}]
    color_point_cloud_on_map(infos, policy_info)
    assert infos[0]["top_down_map"]["map"].shape == (4, 4)
    assert np.count_nonzero(infos[0]["top_down_map"]["map"]) == 0


def test_hm3d_ascent_paths_and_runtime_workspace(tmp_path: Path) -> None:
    repo_dataset_json = get_dataset_json_path("hm3d", "val", hm3d_source="repo")
    ascent_dataset_json = get_dataset_json_path("hm3d", "val", hm3d_source="ascent")
    assert str(repo_dataset_json).endswith("data/datasets/objectnav/hm3d/v1/val/val.json.gz")
    assert ascent_dataset_json == ASCENT_HM3D_DATA_ROOT / "datasets/objectnav/hm3d/v1/val/val.json.gz"
    assert get_default_scenes_dir("hm3d", hm3d_source="ascent") == ASCENT_HM3D_DATA_ROOT / "scene_datasets"

    source_root = tmp_path / "source_data"
    (source_root / "datasets" / "objectnav" / "hm3d" / "v1" / "val").mkdir(parents=True)
    (source_root / "scene_datasets" / "hm3d").mkdir(parents=True)
    run_root = tmp_path / "run"
    workspace = create_runtime_data_workspace(run_root, source_root, source_root / "scene_datasets")
    assert workspace == run_root / ".runtime_data"
    assert (workspace / "data" / "datasets").is_symlink()
    assert (workspace / "data" / "scene_datasets").is_symlink()
    assert (workspace / "data" / "dummy_policy.pth").is_symlink()
    assert (workspace / "data" / "pointnav_weights.pth").is_symlink()


def test_summarize_plan_b_script(tmp_path: Path) -> None:
    run_root = tmp_path / "plan_b_run"
    episode_dir = run_root / "episodes" / "sceneA__ep_1"
    episode_dir.mkdir(parents=True)
    (episode_dir / "episode_stats.json").write_text(
        json.dumps(
            {
                "episode_id": "1",
                "scene_id": "sceneA.glb",
                "scene_short": "sceneA",
                "success": 1.0,
                "spl": 0.5,
                "soft_spl": 0.6,
                "path_length_m": 4.0,
                "step_count": 12,
                "episode_wall_time_sec": 2.0,
                "avg_inference_time_sec": 0.05,
                "failure_reason": "success",
                "video_paths": {
                    "composite": "episodes/sceneA__ep_1/composite.mp4",
                    "egocentric": "episodes/sceneA__ep_1/egocentric.mp4",
                    "topdown": "episodes/sceneA__ep_1/topdown.mp4",
                },
            }
        )
        + "\n"
    )
    (run_root / "preflight.json").write_text(json.dumps({"python": {"version": "3.9", "conda_env": "vlfm"}}) + "\n")
    (run_root / "run_spec.json").write_text(
        json.dumps({"dataset": "hm3d", "split": "val_mini", "policy_name": "HabitatITMPolicyV2", "video_views": ["composite"]})
        + "\n"
    )

    subprocess.run(
        [sys.executable, "project/summarize_plan_b.py", "--run-root", str(run_root)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert (run_root / "summary" / "metrics.json").exists()
    assert (run_root / "summary" / "episodes.csv").exists()
    assert (run_root / "report.md").exists()
