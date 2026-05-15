import os
import hydra  # noqa
from habitat import get_config  # noqa
from habitat.config import read_write
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.run import execute_exp
from habitat_baselines.common import habitat_env_factory as habitat_env_factory_module
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import DictConfig
from omegaconf import OmegaConf
import sys
sys.path.insert(0, "third_party/frontier_exploration")
sys.path.insert(0, "third_party/depth_camera_filtering")
sys.path.insert(0, "third_party/vlfm")
import vlfm.measurements.traveled_stairs  # noqa: F401
import vlfm.obs_transformers.resize  # noqa: F401
import vlfm.policy.action_replay_policy  # noqa: F401
import vlfm.policy.habitat_policies  # noqa: F401
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)

from model_api.llm_backends import load_env_file
from ascent import ascent_policy
from ascent import ascent_trainer 


def _set_default_service_ports() -> None:
    os.environ.setdefault("QWEN2_5_PORT", "13181")
    os.environ.setdefault("BLIP2ITM_PORT", "13182")
    os.environ.setdefault("SAM_PORT", "13183")
    os.environ.setdefault("GROUNDING_DINO_PORT", "13184")
    os.environ.setdefault("RAM_PORT", "13185")
    os.environ.setdefault("DFINE_PORT", "13186")


class HabitatConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="ascent", path="config/")


# 只注册一次，在这里
register_hydra_plugin(HabitatConfigPlugin)
load_env_file()
_set_default_service_ports()


_ORIGINAL_CONSTRUCT_ENVS = habitat_env_factory_module.HabitatVectorEnvFactory.construct_envs


def _patched_construct_envs(self, config, *args, **kwargs):
    with read_write(config):
        return _ORIGINAL_CONSTRUCT_ENVS(self, config, *args, **kwargs)


habitat_env_factory_module.HabitatVectorEnvFactory.construct_envs = _patched_construct_envs

@hydra.main(
    version_base=None,
    config_path="../experiments",  
    config_name="eval_ascent_hm3d.yaml",  # 修改：文件名不带 .yaml 后缀
)
def main(cfg: DictConfig) -> None:
    cfg = patch_config(cfg)
    execute_exp(cfg, "eval" if cfg.habitat_baselines.evaluate else "train")


if __name__ == "__main__":
    register_hydra_plugin(HabitatBaselinesConfigPlugin)  # 如果需要的话保留
    main()
