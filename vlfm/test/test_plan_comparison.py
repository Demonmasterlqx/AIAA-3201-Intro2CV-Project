# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import json
import subprocess
import sys
from pathlib import Path


def test_compare_plan_runs_script(tmp_path: Path) -> None:
    plan_a = tmp_path / "plan_a"
    plan_b = tmp_path / "plan_b"
    plan_a.mkdir()
    plan_b.mkdir()

    for directory, success, spl, soft_spl in (
        (plan_a, 1.0, 0.6, 0.8),
        (plan_b, 0.0, 0.4, 0.5),
    ):
        with open(directory / "1_scene.json", "w") as handle:
            json.dump(
                {
                    "episode_id": 1,
                    "scene_id": "scene",
                    "success": success,
                    "spl": spl,
                    "soft_spl": soft_spl,
                },
                handle,
            )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compare_plan_runs.py",
            str(plan_a),
            str(plan_b),
            "--plan-a-label",
            "Dense",
            "--plan-b-label",
            "Frontier",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Comparing 1 shared episodes." in result.stdout
    assert "Success Rate" in result.stdout
    assert "Dense" in result.stdout
    assert "Frontier" in result.stdout

