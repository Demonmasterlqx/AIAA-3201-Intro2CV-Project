from __future__ import annotations

from typing import Dict
from typing import Optional
from typing import Sequence

from ascent.g_ascent_structures import CandidateState
from ascent.g_ascent_structures import CandidateStatus


class CandidateFusion:
    def __init__(
        self,
        *,
        merge_radius: float = 1.0,
        likely_threshold: float = 0.55,
        verified_threshold: float = 0.75,
    ) -> None:
        self._merge_radius = merge_radius
        self._likely_threshold = likely_threshold
        self._verified_threshold = verified_threshold
        self._candidates: Dict[str, CandidateState] = {}

    @property
    def candidates(self) -> Dict[str, CandidateState]:
        return self._candidates

    def observe(
        self,
        *,
        agent_id: str,
        floor_id: int,
        centroid: Sequence[float],
        score: float,
    ) -> CandidateState:
        candidate = self._find_candidate(floor_id, centroid)
        if candidate is None:
            candidate = CandidateState(
                candidate_id=f"candidate_{len(self._candidates):04d}",
                floor_id=floor_id,
                centroid=(float(centroid[0]), float(centroid[1])),
            )
            self._candidates[candidate.candidate_id] = candidate
        supporters = set(candidate.supporters)
        supporters.add(agent_id)
        candidate.supporters = sorted(supporters)
        candidate.fused_score = max(candidate.fused_score, score) if len(candidate.supporters) == 1 else (candidate.fused_score + score) * 0.5
        if candidate.fused_score >= self._verified_threshold and len(candidate.supporters) >= 2:
            candidate.status = CandidateStatus.VERIFIED
            candidate.verifier_needed = False
        elif candidate.fused_score >= self._likely_threshold:
            candidate.status = CandidateStatus.LIKELY
            candidate.verifier_needed = True
        else:
            candidate.status = CandidateStatus.SUSPECT
            candidate.verifier_needed = False
        return candidate

    def reject(self, candidate_id: str, reason: str) -> None:
        candidate = self._candidates.get(candidate_id)
        if candidate is None:
            return
        candidate.status = CandidateStatus.REJECTED
        candidate.verifier_needed = False
        candidate.rejected_reason = reason

    def best(self) -> Optional[CandidateState]:
        valid = [
            candidate
            for candidate in self._candidates.values()
            if candidate.status != CandidateStatus.REJECTED
        ]
        if not valid:
            return None
        return max(valid, key=lambda item: item.fused_score)

    def _find_candidate(
        self,
        floor_id: int,
        centroid: Sequence[float],
    ) -> Optional[CandidateState]:
        for candidate in self._candidates.values():
            if candidate.floor_id != floor_id:
                continue
            dx = candidate.centroid[0] - float(centroid[0])
            dy = candidate.centroid[1] - float(centroid[1])
            if (dx * dx + dy * dy) ** 0.5 <= self._merge_radius:
                return candidate
        return None
