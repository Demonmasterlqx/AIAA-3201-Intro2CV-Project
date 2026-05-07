from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple


class _StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class AgentRole(_StrEnum):
    EXPLORER = "explorer"
    HUNTER = "hunter"
    VERIFIER = "verifier"
    STAIR_SCOUT = "stair_scout"


class CandidateStatus(_StrEnum):
    SUSPECT = "suspect"
    LIKELY = "likely"
    VERIFIED = "verified"
    REJECTED = "rejected"


class SchedulerTrigger(_StrEnum):
    NONE = "none"
    VALUE_AMBIGUITY = "value_ambiguity"
    FLOOR_AMBIGUITY = "floor_ambiguity"
    GLOBAL_STALL = "global_stall"
    CANDIDATE_CONFLICT = "candidate_conflict"
    FLOOR_SATURATED = "floor_saturated"


@dataclass
class FloorNode:
    floor_id: int
    explored_ratio: float = 0.0
    stairs: List[str] = field(default_factory=list)
    frontier_clusters: List[str] = field(default_factory=list)
    goal_prior: float = 0.0
    region_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RegionNode:
    region_id: str
    floor_id: int
    centroid: Tuple[float, float]
    semantic_score: float = 0.0
    unknown_gain: float = 0.0
    reachable: bool = True
    owner: Optional[str] = None
    frontier_points: List[Tuple[float, float]] = field(default_factory=list)
    source_agents: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentState:
    agent_id: str
    pose: Tuple[float, float]
    floor_id: int
    assigned_region: Optional[str] = None
    coverage_gain: float = 0.0
    stall_score: float = 0.0
    role: AgentRole = AgentRole.EXPLORER
    last_nav_goal: Optional[Tuple[float, float]] = None
    target_detected: bool = False
    stop_called: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["role"] = self.role.value
        return data


@dataclass
class TaskAssignment:
    agent_id: str
    region_id: Optional[str]
    priority: float
    issued_step: int
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateState:
    candidate_id: str
    floor_id: int
    centroid: Tuple[float, float]
    fused_score: float = 0.0
    supporters: List[str] = field(default_factory=list)
    verifier_needed: bool = False
    rejected_reason: str = ""
    status: CandidateStatus = CandidateStatus.SUSPECT

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class RegionGraphSnapshot:
    floors: Dict[int, FloorNode] = field(default_factory=dict)
    regions: Dict[str, RegionNode] = field(default_factory=dict)
    assignments: Dict[str, TaskAssignment] = field(default_factory=dict)
    candidates: Dict[str, CandidateState] = field(default_factory=dict)
    trigger: SchedulerTrigger = SchedulerTrigger.NONE
    step: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "floors": {floor_id: node.to_dict() for floor_id, node in self.floors.items()},
            "regions": {region_id: node.to_dict() for region_id, node in self.regions.items()},
            "assignments": {agent_id: assignment.to_dict() for agent_id, assignment in self.assignments.items()},
            "candidates": {candidate_id: candidate.to_dict() for candidate_id, candidate in self.candidates.items()},
            "trigger": self.trigger.value,
            "step": self.step,
        }
