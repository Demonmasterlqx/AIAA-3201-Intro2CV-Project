from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple


class _StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class CoordinatorState(_StrEnum):
    SINGLE_AGENT_NORMAL = "S0_single_agent_normal"
    FINE_GRAINED_TRIGGERED = "S1_fine_grained_triggered"
    SPAWN_EVALUATION = "S2_spawn_evaluation"
    DUAL_AGENT_ACTIVE = "S3_dual_agent_active"
    VERIFICATION_PRIORITY = "S4_verification_priority"
    RECLAIM_SUBAGENT = "S5_reclaim_subagent"
    TERMINATE = "S6_terminate"


class SubagentState(_StrEnum):
    DORMANT = "dormant"
    SPAWNED = "spawned"
    COMMITTED = "committed"
    EXPLORING = "exploring"
    VERIFYING = "verifying"
    REASSIGNABLE = "reassignable"
    RECLAIMED = "reclaimed"


class FrontierStatus(_StrEnum):
    FREE = "free"
    RESERVED = "reserved"
    EXPLORING = "exploring"
    EXPLORED = "explored"
    UNREACHABLE = "unreachable"
    STALE = "stale"


class ConfirmationState(_StrEnum):
    SUSPECT = "suspect"
    LIKELY = "likely"
    VERIFIED = "verified"
    REJECTED = "rejected"


class IntentMode(_StrEnum):
    IDLE = "idle"
    EXPLORING = "exploring"
    TRANSITION = "transition"
    VERIFYING = "verifying"
    RETURNING = "returning"
    RECLAIMED = "reclaimed"


@dataclass
class UtilityBreakdown:
    semantic_score: float = 0.0
    geometry_gain: float = 0.0
    floor_prior: float = 0.0
    area_prior: float = 0.0
    path_cost: float = 0.0
    overlap_penalty: float = 0.0
    blacklist_penalty: float = 0.0
    stale_penalty: float = 0.0
    total_utility: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class FloorMemory:
    floor_id: int
    visited_ratio: float = 0.0
    frontier_count: int = 0
    stairs_up: bool = False
    stairs_down: bool = False
    cross_floor_edges: List[Tuple[int, int]] = field(default_factory=list)
    floor_prior_for_goal: float = 0.0
    last_updated_step: int = 0
    agent_presence_state: Dict[str, str] = field(default_factory=dict)
    obs_map_ref: Optional[str] = None
    val_map_ref: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FrontierRecord:
    frontier_id: str
    floor_id: int
    centroid: Tuple[float, float]
    cluster_extent: float
    semantic_score: float
    geometry_gain: float
    path_cost_from_main: float
    path_cost_from_sub: float
    assigned_agent: Optional[str] = None
    reserved_by: Optional[str] = None
    reservation_start_step: int = -1
    reservation_expiry_step: int = -1
    status: FrontierStatus = FrontierStatus.FREE
    cluster_id: Optional[str] = None
    blacklist_hits: int = 0
    last_seen_step: int = 0
    stale_generation: int = 0
    utility: UtilityBreakdown = field(default_factory=UtilityBreakdown)

    def refresh_costs(
        self,
        main_position: Sequence[float],
        sub_position: Optional[Sequence[float]],
    ) -> None:
        self.path_cost_from_main = _l2(self.centroid, main_position)
        if sub_position is not None:
            self.path_cost_from_sub = _l2(self.centroid, sub_position)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class IntentRecord:
    agent_id: str
    role: str
    current_mode: IntentMode = IntentMode.IDLE
    target_floor: Optional[int] = None
    target_frontier: Optional[str] = None
    target_region: Optional[str] = None
    commit_until_step: int = -1
    last_progress_step: int = -1
    allow_reassign: bool = True
    fail_count: int = 0
    current_path_summary: str = ""
    current_position: Optional[Tuple[float, float]] = None
    state: SubagentState = SubagentState.DORMANT

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["current_mode"] = self.current_mode.value
        data["state"] = self.state.value
        return data


@dataclass
class TargetEvidenceRecord:
    candidate_id: str
    floor_id: int
    pose_estimate: Tuple[float, float, float]
    supporting_frames: int = 0
    supporting_agents: Set[str] = field(default_factory=set)
    fused_score: float = 0.0
    score_history: List[float] = field(default_factory=list)
    confirmation_state: ConfirmationState = ConfirmationState.SUSPECT
    last_seen_step: int = 0
    rejection_reason: str = ""
    viewpoint_clusters: Set[str] = field(default_factory=set)

    def update_score(self, score: float) -> None:
        self.score_history.append(score)
        if self.score_history:
            self.fused_score = sum(self.score_history) / len(self.score_history)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["supporting_agents"] = sorted(self.supporting_agents)
        data["viewpoint_clusters"] = sorted(self.viewpoint_clusters)
        data["confirmation_state"] = self.confirmation_state.value
        return data


@dataclass
class SharedMemorySnapshot:
    floors: Dict[int, FloorMemory] = field(default_factory=dict)
    frontiers: Dict[str, FrontierRecord] = field(default_factory=dict)
    intents: Dict[str, IntentRecord] = field(default_factory=dict)
    target_evidence: Dict[str, TargetEvidenceRecord] = field(default_factory=dict)
    blacklist: Dict[str, str] = field(default_factory=dict)
    failure_records: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "floors": {k: v.to_dict() for k, v in self.floors.items()},
            "frontiers": {k: v.to_dict() for k, v in self.frontiers.items()},
            "intents": {k: v.to_dict() for k, v in self.intents.items()},
            "target_evidence": {
                k: v.to_dict() for k, v in self.target_evidence.items()
            },
            "blacklist": dict(self.blacklist),
            "failure_records": list(self.failure_records),
        }


def _l2(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)
