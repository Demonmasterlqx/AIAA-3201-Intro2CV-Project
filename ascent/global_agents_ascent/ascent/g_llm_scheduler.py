from __future__ import annotations

import json
from typing import Any
from typing import Dict
from typing import List

from model_api.llm_backends import create_llm_client

from ascent.g_ascent_structures import RegionGraphSnapshot
from ascent.g_ascent_structures import SchedulerTrigger


class LLMScheduler:
    def __init__(
        self,
        *,
        llm_config: Any,
        min_interval_steps: int = 200,
        floor_gap_threshold: float = 0.10,
        region_gap_threshold: float = 0.05,
    ) -> None:
        self._client = create_llm_client(llm_config)
        self._min_interval_steps = min_interval_steps
        self._floor_gap_threshold = floor_gap_threshold
        self._region_gap_threshold = region_gap_threshold
        self._last_step = -min_interval_steps

    def should_trigger(
        self,
        snapshot: RegionGraphSnapshot,
        region_scores: List[float],
        floor_scores: List[float],
        stalled_agents: int,
    ) -> SchedulerTrigger:
        if snapshot.step - self._last_step < self._min_interval_steps:
            return SchedulerTrigger.NONE
        if len(floor_scores) >= 2 and abs(floor_scores[0] - floor_scores[1]) < self._floor_gap_threshold:
            return SchedulerTrigger.FLOOR_AMBIGUITY
        if len(region_scores) >= 2 and abs(region_scores[0] - region_scores[1]) < self._region_gap_threshold:
            return SchedulerTrigger.VALUE_AMBIGUITY
        if stalled_agents >= max(2, len(snapshot.assignments)):
            return SchedulerTrigger.GLOBAL_STALL
        candidate_conflict = sum(
            1 for candidate in snapshot.candidates.values() if candidate.verifier_needed
        )
        if candidate_conflict > 1:
            return SchedulerTrigger.CANDIDATE_CONFLICT
        saturated_floors = sum(1 for floor in snapshot.floors.values() if floor.explored_ratio >= 0.85)
        if saturated_floors > 0:
            return SchedulerTrigger.FLOOR_SATURATED
        return SchedulerTrigger.NONE

    def query(self, snapshot: RegionGraphSnapshot) -> Dict[str, Any]:
        self._last_step = snapshot.step
        prompt = self._build_prompt(snapshot)
        response = self._client.chat(prompt)
        try:
            return json.loads(response)
        except Exception:
            return {"fallback_policy": "greedy_region_score", "raw_response": response}

    def _build_prompt(self, snapshot: RegionGraphSnapshot) -> str:
        payload = {
            "step": snapshot.step,
            "floors": [floor.to_dict() for floor in snapshot.floors.values()],
            "regions": [region.to_dict() for region in list(snapshot.regions.values())[:12]],
            "assignments": [assignment.to_dict() for assignment in snapshot.assignments.values()],
            "candidates": [candidate.to_dict() for candidate in snapshot.candidates.values()],
            "required_fields": [
                "floor_priority",
                "region_priority",
                "assign_role",
                "assign_agent",
                "verification_target",
                "fallback_policy",
            ],
        }
        return (
            "You are the global scheduler for G-ASCENT.\n"
            "Return strict JSON with the required fields and concise values.\n"
            + json.dumps(payload, ensure_ascii=False)
        )
