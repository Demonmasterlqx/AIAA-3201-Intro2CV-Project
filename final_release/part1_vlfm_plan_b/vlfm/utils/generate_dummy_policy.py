# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from pathlib import Path

import torch

from vlfm.run import get_config


def save_dummy_policy(filename: str) -> None:
    # Save a dummy state_dict using torch.save
    repo_root = Path(__file__).resolve().parents[2]
    config = get_config(str(repo_root / "config" / "experiments" / "vlfm_objectnav_hm3d.yaml"))
    dummy_dict = {
        "config": config,
        "extra_state": {"step": 0},
        "state_dict": {},
    }

    torch.save(dummy_dict, filename)


if __name__ == "__main__":
    save_dummy_policy("data/dummy_policy.pth")
    print("Dummy policy weights saved to data/dummy_policy.pth")
