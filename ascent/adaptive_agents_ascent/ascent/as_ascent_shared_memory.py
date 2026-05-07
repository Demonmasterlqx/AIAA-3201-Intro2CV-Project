from __future__ import annotations

from dataclasses import replace
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

from ascent.as_ascent_structures import ConfirmationState
from ascent.as_ascent_structures import FloorMemory
from ascent.as_ascent_structures import FrontierRecord
from ascent.as_ascent_structures import FrontierStatus
from ascent.as_ascent_structures import IntentRecord
from ascent.as_ascent_structures import SharedMemorySnapshot
from ascent.as_ascent_structures import SubagentState
from ascent.as_ascent_structures import TargetEvidenceRecord
from ascent.as_ascent_structures import UtilityBreakdown


class SharedMemoryManager:
    def __init__(self, reservation_ttl: int, candidate_merge_radius: float) -> None:
        self._reservation_ttl = reservation_ttl
        self._candidate_merge_radius = candidate_merge_radius
        self.reset()

    def reset(self) -> None:
        self._snapshot = SharedMemorySnapshot()
        self._frontier_generation = 0

    @property
    def snapshot(self) -> SharedMemorySnapshot:
        return self._snapshot

    def ensure_floor(self, floor_id: int) -> FloorMemory:
        if floor_id not in self._snapshot.floors:
            self._snapshot.floors[floor_id] = FloorMemory(floor_id=floor_id)
        return self._snapshot.floors[floor_id]

    def update_floor(
        self,
        floor_id: int,
        *,
        visited_ratio: float,
        frontier_count: int,
        stairs_up: bool,
        stairs_down: bool,
        floor_prior_for_goal: float,
        last_updated_step: int,
        agent_presence_state: Optional[Dict[str, str]] = None,
        obs_map_ref: Optional[str] = None,
        val_map_ref: Optional[str] = None,
    ) -> None:
        floor = self.ensure_floor(floor_id)
        floor.visited_ratio = visited_ratio
        floor.frontier_count = frontier_count
        floor.stairs_up = stairs_up
        floor.stairs_down = stairs_down
        floor.floor_prior_for_goal = floor_prior_for_goal
        floor.last_updated_step = last_updated_step
        floor.obs_map_ref = obs_map_ref
        floor.val_map_ref = val_map_ref
        if agent_presence_state is not None:
            floor.agent_presence_state = dict(agent_presence_state)

    def refresh_frontiers(
        self,
        floor_id: int,
        frontiers: Iterable[FrontierRecord],
        step: int,
    ) -> None:
        self._frontier_generation += 1
        seen: Dict[str, FrontierRecord] = {}
        for frontier in frontiers:
            frontier.last_seen_step = step
            frontier.stale_generation = self._frontier_generation
            seen[frontier.frontier_id] = frontier
            if frontier.frontier_id in self._snapshot.frontiers:
                previous = self._snapshot.frontiers[frontier.frontier_id]
                frontier.assigned_agent = previous.assigned_agent
                frontier.reserved_by = previous.reserved_by
                frontier.reservation_start_step = previous.reservation_start_step
                frontier.reservation_expiry_step = previous.reservation_expiry_step
                frontier.status = previous.status
                frontier.blacklist_hits = previous.blacklist_hits
            self._snapshot.frontiers[frontier.frontier_id] = frontier

        for frontier_id, record in list(self._snapshot.frontiers.items()):
            if record.floor_id != floor_id:
                continue
            if frontier_id not in seen and record.status not in (
                FrontierStatus.EXPLORED,
                FrontierStatus.UNREACHABLE,
            ):
                record.status = FrontierStatus.STALE

    def reserve_frontier(
        self,
        frontier_id: str,
        agent_id: str,
        step: int,
        ttl: Optional[int] = None,
    ) -> None:
        frontier = self._snapshot.frontiers[frontier_id]
        frontier.reserved_by = agent_id
        frontier.assigned_agent = agent_id
        frontier.reservation_start_step = step
        frontier.reservation_expiry_step = step + (ttl or self._reservation_ttl)
        frontier.status = FrontierStatus.RESERVED

    def release_frontier(self, frontier_id: str, status: FrontierStatus) -> None:
        frontier = self._snapshot.frontiers[frontier_id]
        frontier.reserved_by = None
        frontier.assigned_agent = None
        frontier.reservation_start_step = -1
        frontier.reservation_expiry_step = -1
        frontier.status = status

    def release_expired_reservations(self, step: int) -> None:
        for frontier in self._snapshot.frontiers.values():
            if (
                frontier.status == FrontierStatus.RESERVED
                and frontier.reservation_expiry_step >= 0
                and step > frontier.reservation_expiry_step
            ):
                frontier.status = FrontierStatus.FREE
                frontier.reserved_by = None
                frontier.assigned_agent = None

    def blacklist_frontier(self, frontier_id: str, reason: str) -> None:
        frontier = self._snapshot.frontiers.get(frontier_id)
        if frontier is not None:
            frontier.blacklist_hits += 1
            frontier.status = FrontierStatus.UNREACHABLE
        self._snapshot.blacklist[frontier_id] = reason

    def is_blacklisted(self, frontier_id: str) -> bool:
        return frontier_id in self._snapshot.blacklist

    def set_intent(self, record: IntentRecord) -> None:
        self._snapshot.intents[record.agent_id] = record

    def clear_intent(self, agent_id: str) -> None:
        if agent_id in self._snapshot.intents:
            current = self._snapshot.intents[agent_id]
            current.target_frontier = None
            current.target_region = None
            current.commit_until_step = -1
            current.current_path_summary = ""
            current.state = SubagentState.RECLAIMED

    def get_intent(self, agent_id: str) -> Optional[IntentRecord]:
        return self._snapshot.intents.get(agent_id)

    def record_failure(self, payload: Dict[str, Any]) -> None:
        self._snapshot.failure_records.append(dict(payload))

    def upsert_candidate(
        self,
        floor_id: int,
        pose_estimate: Sequence[float],
        agent_id: str,
        score: float,
        step: int,
        viewpoint_cluster: str,
        likely_threshold: float,
        verified_threshold: float,
    ) -> TargetEvidenceRecord:
        candidate = self._find_candidate(floor_id, pose_estimate)
        if candidate is None:
            candidate_id = f"cand_{len(self._snapshot.target_evidence):04d}"
            candidate = TargetEvidenceRecord(
                candidate_id=candidate_id,
                floor_id=floor_id,
                pose_estimate=(float(pose_estimate[0]), float(pose_estimate[1]), float(pose_estimate[2])),
            )
            self._snapshot.target_evidence[candidate_id] = candidate
        candidate.supporting_frames += 1
        candidate.supporting_agents.add(agent_id)
        candidate.viewpoint_clusters.add(viewpoint_cluster)
        candidate.last_seen_step = step
        candidate.update_score(score)
        if candidate.fused_score >= verified_threshold:
            candidate.confirmation_state = ConfirmationState.VERIFIED
        elif candidate.fused_score >= likely_threshold or (
            candidate.supporting_frames >= 2 and len(candidate.supporting_agents) >= 1
        ):
            candidate.confirmation_state = ConfirmationState.LIKELY
        else:
            candidate.confirmation_state = ConfirmationState.SUSPECT
        return candidate

    def reject_candidate(self, candidate_id: str, reason: str) -> None:
        if candidate_id not in self._snapshot.target_evidence:
            return
        candidate = self._snapshot.target_evidence[candidate_id]
        candidate.confirmation_state = ConfirmationState.REJECTED
        candidate.rejection_reason = reason

    def best_candidate(self) -> Optional[TargetEvidenceRecord]:
        valid = [
            candidate
            for candidate in self._snapshot.target_evidence.values()
            if candidate.confirmation_state != ConfirmationState.REJECTED
        ]
        if not valid:
            return None
        return max(valid, key=lambda item: item.fused_score)

    def set_frontier_utility(
        self,
        frontier_id: str,
        breakdown: UtilityBreakdown,
    ) -> None:
        if frontier_id in self._snapshot.frontiers:
            self._snapshot.frontiers[frontier_id].utility = breakdown

    def clone_intent(self, agent_id: str) -> Optional[IntentRecord]:
        if agent_id not in self._snapshot.intents:
            return None
        return replace(self._snapshot.intents[agent_id])

    def _find_candidate(
        self, floor_id: int, pose_estimate: Sequence[float]
    ) -> Optional[TargetEvidenceRecord]:
        for candidate in self._snapshot.target_evidence.values():
            if candidate.floor_id != floor_id:
                continue
            dx = candidate.pose_estimate[0] - pose_estimate[0]
            dz = candidate.pose_estimate[2] - pose_estimate[2]
            if (dx * dx + dz * dz) ** 0.5 <= self._candidate_merge_radius:
                return candidate
        return None
