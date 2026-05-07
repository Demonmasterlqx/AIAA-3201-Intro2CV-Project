from __future__ import annotations

from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional

from ascent.g_ascent_structures import AgentRole
from ascent.g_ascent_structures import AgentState
from ascent.g_ascent_structures import CandidateState
from ascent.g_ascent_structures import RegionGraphSnapshot
from ascent.g_ascent_structures import SchedulerTrigger
from ascent.g_ascent_structures import TaskAssignment
from ascent.g_llm_scheduler import LLMScheduler
from ascent.g_role_stall import RoleAssigner
from ascent.g_role_stall import StallDetector


class GlobalScheduler:
    def __init__(
        self,
        *,
        llm_scheduler: Optional[LLMScheduler],
        distance_weight: float = 1.0,
        overlap_penalty: float = 0.5,
        assignment_bonus: float = 0.35,
    ) -> None:
        self._llm_scheduler = llm_scheduler
        self._distance_weight = distance_weight
        self._overlap_penalty = overlap_penalty
        self._assignment_bonus = assignment_bonus
        self._role_assigner = RoleAssigner()
        self._stall_detector = StallDetector()
        self._released_agents: Dict[str, str] = {}
        self._last_assignments: Dict[str, Optional[str]] = {}

    def update(
        self,
        *,
        snapshot: RegionGraphSnapshot,
        agent_states: List[AgentState],
        candidates: Dict[str, CandidateState],
    ) -> RegionGraphSnapshot:
        roles = self._role_assigner.assign(agent_states, candidates)
        for state in agent_states:
            state.role = roles.get(state.agent_id, AgentRole.EXPLORER)
            state.stall_score = self._stall_detector.update(state)
        region_scores = sorted(
            [region.semantic_score + region.unknown_gain for region in snapshot.regions.values()],
            reverse=True,
        )
        floor_scores = sorted([floor.goal_prior for floor in snapshot.floors.values()], reverse=True)
        stalled_agents = sum(1 for state in agent_states if self._stall_detector.is_stalled(state))
        trigger = SchedulerTrigger.NONE
        if self._llm_scheduler is not None:
            trigger = self._llm_scheduler.should_trigger(snapshot, region_scores, floor_scores, stalled_agents)
        snapshot.trigger = trigger
        return snapshot

    def assign(
        self,
        *,
        snapshot: RegionGraphSnapshot,
        agent_states: Iterable[AgentState],
    ) -> List[TaskAssignment]:
        available_regions = list(snapshot.regions.values())
        assignments: List[TaskAssignment] = []
        used_regions: set[str] = set()
        for state in agent_states:
            if self._stall_detector.is_stalled(state):
                self.release(state.agent_id, "stalled")
                self._last_assignments.pop(state.agent_id, None)
            best_region = None
            best_score = float("-inf")
            last_region_id = self._last_assignments.get(state.agent_id)
            for region in available_regions:
                if region.region_id in used_regions:
                    continue
                distance = ((state.pose[0] - region.centroid[0]) ** 2 + (state.pose[1] - region.centroid[1]) ** 2) ** 0.5
                score = region.semantic_score + region.unknown_gain - self._distance_weight * distance
                if region.owner not in (None, state.agent_id):
                    score -= self._overlap_penalty
                if region.region_id == last_region_id:
                    score += self._assignment_bonus
                if state.role == AgentRole.HUNTER:
                    score += region.semantic_score
                elif state.role == AgentRole.EXPLORER:
                    score += region.unknown_gain
                elif state.role == AgentRole.STAIR_SCOUT and "stairs" in snapshot.floors[region.floor_id].stairs:
                    score += 0.5
                if score > best_score:
                    best_region = region
                    best_score = score
            assignment = TaskAssignment(
                agent_id=state.agent_id,
                region_id=best_region.region_id if best_region is not None else None,
                priority=best_score if best_region is not None else 0.0,
                issued_step=snapshot.step,
                reason="greedy_region_score",
            )
            assignments.append(assignment)
            snapshot.assignments[state.agent_id] = assignment
            self._last_assignments[state.agent_id] = assignment.region_id
            if best_region is not None:
                best_region.owner = state.agent_id
                used_regions.add(best_region.region_id)
        return assignments

    def release(self, agent_id: str, reason: str) -> None:
        self._released_agents[agent_id] = reason
        self._last_assignments.pop(agent_id, None)

    def replan(self, trigger: SchedulerTrigger) -> Dict[str, str]:
        return {"trigger": trigger.value, "released_agents": dict(self._released_agents)}

    def get_last_assignment(self, agent_id: str) -> Optional[str]:
        return self._last_assignments.get(agent_id)
