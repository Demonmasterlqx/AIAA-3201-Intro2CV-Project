from __future__ import annotations

from collections import deque
from typing import Deque
from typing import Dict
from typing import List

from ascent.g_ascent_structures import AgentRole
from ascent.g_ascent_structures import AgentState
from ascent.g_ascent_structures import CandidateState


class StallDetector:
    def __init__(
        self,
        *,
        window_size: int = 30,
        coverage_epsilon: float = 0.01,
        progress_epsilon: float = 0.5,
        threshold: float = 2.0,
    ) -> None:
        self._window_size = window_size
        self._coverage_epsilon = coverage_epsilon
        self._progress_epsilon = progress_epsilon
        self._threshold = threshold
        self._history: Dict[str, Deque[float]] = {}

    def update(self, agent_state: AgentState) -> float:
        history = self._history.setdefault(
            agent_state.agent_id,
            deque(maxlen=self._window_size),
        )
        history.append(agent_state.coverage_gain)
        if len(history) < max(5, self._window_size // 3):
            return 0.0
        coverage_delta = max(history) - min(history)
        stall_score = 0.0
        if coverage_delta < self._coverage_epsilon:
            stall_score += 1.0
        if agent_state.assigned_region is None:
            stall_score += 0.5
        if not agent_state.target_detected and coverage_delta < self._progress_epsilon:
            stall_score += 1.0
        return stall_score

    def is_stalled(self, agent_state: AgentState) -> bool:
        return agent_state.stall_score >= self._threshold


class RoleAssigner:
    def assign(
        self,
        agent_states: List[AgentState],
        candidates: Dict[str, CandidateState],
    ) -> Dict[str, AgentRole]:
        roles: Dict[str, AgentRole] = {}
        verifier_target = next(
            (candidate for candidate in candidates.values() if candidate.verifier_needed),
            None,
        )
        if verifier_target is not None and agent_states:
            nearest = min(
                agent_states,
                key=lambda state: (state.pose[0] - verifier_target.centroid[0]) ** 2
                + (state.pose[1] - verifier_target.centroid[1]) ** 2,
            )
            roles[nearest.agent_id] = AgentRole.VERIFIER

        remaining = [state for state in agent_states if state.agent_id not in roles]
        if remaining:
            roles[remaining[0].agent_id] = AgentRole.EXPLORER
        if len(remaining) > 1:
            roles[remaining[1].agent_id] = AgentRole.HUNTER
        for state in remaining[2:]:
            roles[state.agent_id] = AgentRole.STAIR_SCOUT
        return roles
