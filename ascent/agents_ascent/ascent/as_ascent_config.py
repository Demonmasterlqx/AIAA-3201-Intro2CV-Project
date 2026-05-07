from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from vlfm.policy.habitat_policies import cs

from ascent.utils import AscentPolicyConfig


@dataclass
class FrontierUtilityWeightsConfig:
    alpha: float = 1.0
    beta: float = 0.75
    gamma: float = 0.40
    eta: float = 0.20
    lambda_path: float = 0.60
    mu_overlap: float = 0.50
    nu_blacklist: float = 0.75
    xi_stale: float = 0.50

    use_semantic: bool = True
    use_geometry: bool = True
    use_floor_prior: bool = True
    use_area_prior: bool = True
    use_overlap_penalty: bool = True
    use_blacklist_penalty: bool = True
    use_stale_penalty: bool = True


@dataclass
class FineGrainedTriggerConfig:
    distance_threshold: float = 3.0
    min_frontiers: int = 3
    min_cluster_count: int = 2
    top2_gap_threshold: float = 0.10


@dataclass
class SpawnConfig:
    min_remaining_steps: int = 120
    spawn_margin: float = 0.15
    max_overlap: float = 0.35
    max_spawn_overhead_ratio: float = 0.25
    reservation_ttl: int = 30
    sub_fail_limit: int = 2


@dataclass
class CommitWindowConfig:
    commit_window_steps: int = 40
    progress_patience: int = 15


@dataclass
class EvidenceFusionConfig:
    likely_threshold: float = 0.60
    verified_threshold: float = 0.80
    stability_window: int = 3
    candidate_merge_radius: float = 0.75
    min_verified_distance: float = 1.0


@dataclass
class SmokeEvalConfig:
    enabled: bool = True
    default_episode_count: int = 1
    dataset_split: str = "val"
    run_dir_template: str = "results/agents_ascent_naive/hm3d/{run_id}"


@dataclass
class LoggingConfig:
    log_dir_template: str = "results/agents_ascent_naive/hm3d/{run_id}"
    save_episode_jsonl: bool = True
    save_summary_jsonl: bool = True
    save_shared_memory_snapshots: bool = True


@dataclass
class ASAscentConfig:
    coordination_mode: str = "naive_shared"
    real_dual_agent: bool = True
    enable_shared_memory: bool = True
    enable_target_fusion: bool = True
    enable_soft_no_return: bool = True
    trigger: FineGrainedTriggerConfig = field(default_factory=FineGrainedTriggerConfig)
    spawn: SpawnConfig = field(default_factory=SpawnConfig)
    commit: CommitWindowConfig = field(default_factory=CommitWindowConfig)
    evidence: EvidenceFusionConfig = field(default_factory=EvidenceFusionConfig)
    utility: FrontierUtilityWeightsConfig = field(default_factory=FrontierUtilityWeightsConfig)
    smoke: SmokeEvalConfig = field(default_factory=SmokeEvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


@dataclass
class AscentMultiAgentPolicyConfig(AscentPolicyConfig):
    name: str = "AscentMultiAgentPolicy"
    as_ascent: ASAscentConfig = field(default_factory=ASAscentConfig)


cs.store(
    group="habitat_baselines/rl/policy",
    name="as_ascent_policy",
    node={"main_agent": AscentMultiAgentPolicyConfig},
)

cs.store(
    package="habitat_baselines.rl.as_ascent",
    group="habitat_baselines/rl",
    name="as_ascent_defaults",
    node=ASAscentConfig,
)
