from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from vlfm.policy.habitat_policies import cs

from ascent.utils import AscentPolicyConfig


@dataclass
class GRegionGraphConfig:
    cluster_radius: float = 2.5


@dataclass
class GStallConfig:
    window_size: int = 30
    coverage_epsilon: float = 0.01
    progress_epsilon: float = 0.5
    threshold: float = 2.0


@dataclass
class GCandidateConfig:
    merge_radius: float = 1.0
    likely_threshold: float = 0.55
    verified_threshold: float = 0.75


@dataclass
class GLLMConfig:
    enabled: bool = True
    min_interval_steps: int = 200
    floor_gap_threshold: float = 0.10
    region_gap_threshold: float = 0.05


@dataclass
class GLoggingConfig:
    log_dir_template: str = "results/g_ascent/hm3d/{run_id}"


@dataclass
class GAscentConfig:
    coordination_mode: str = "region_graph"
    region_graph: GRegionGraphConfig = field(default_factory=GRegionGraphConfig)
    stall: GStallConfig = field(default_factory=GStallConfig)
    candidate: GCandidateConfig = field(default_factory=GCandidateConfig)
    llm_scheduler: GLLMConfig = field(default_factory=GLLMConfig)
    logging: GLoggingConfig = field(default_factory=GLoggingConfig)


@dataclass
class GAscentPolicyConfig(AscentPolicyConfig):
    name: str = "GAscentPolicy"
    g_ascent: GAscentConfig = field(default_factory=GAscentConfig)


cs.store(
    group="habitat_baselines/rl/policy",
    name="g_ascent_policy",
    node={"main_agent": GAscentPolicyConfig},
)
