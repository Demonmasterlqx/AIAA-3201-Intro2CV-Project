from __future__ import annotations

import os
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from gym import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.rl.ppo.policy import PolicyActionData
from vlfm.policy.habitat_policies import VLFMPolicyConfig

from ascent.ascent_policy import Ascent_Policy
from ascent.g_ascent_config import GAscentPolicyConfig
from ascent.g_ascent_logging import GEpisodeLogger
from ascent.g_ascent_structures import AgentState
from ascent.g_candidate_fusion import CandidateFusion
from ascent.g_llm_scheduler import LLMScheduler
from ascent.g_region_graph import RegionGraphBuilder
from ascent.g_scheduler import GlobalScheduler
from ascent.utils import infer_dataset_type_from_path


def _extract_prefixed_obs(observations: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    extracted: Dict[str, Any] = {}
    for key, value in observations.items():
        if key.startswith(prefix):
            extracted[key[len(prefix) :]] = value
    return extracted


@baseline_registry.register_policy
class GAscentPolicy(Policy):
    @classmethod
    def from_config(
        cls,
        config,
        *args_unused: Any,
        **kwargs_unused: Any,
    ) -> "GAscentPolicy":
        rl_policy_config = config.habitat_baselines.rl.policy
        policy_config: GAscentPolicyConfig = rl_policy_config.main_agent
        kwargs = {key: policy_config[key] for key in VLFMPolicyConfig.kwaarg_names}
        sim_sensors_cfg = config.habitat.simulator.agents.agent_0.sim_sensors
        kwargs["camera_height"] = sim_sensors_cfg.rgb_sensor.position[1]
        kwargs["min_depth"] = sim_sensors_cfg.depth_sensor.min_depth
        kwargs["max_depth"] = sim_sensors_cfg.depth_sensor.max_depth
        kwargs["camera_fov"] = sim_sensors_cfg.depth_sensor.hfov
        kwargs["image_width"] = sim_sensors_cfg.depth_sensor.width
        kwargs["image_height"] = sim_sensors_cfg.rgb_sensor.height
        kwargs["visualize"] = len(config.habitat_baselines.eval.video_option) > 0
        kwargs["action_space"] = args_unused[-1]
        kwargs["dataset_type"] = infer_dataset_type_from_path(config.habitat.dataset.data_path)
        kwargs["full_config"] = config
        kwargs["num_envs"] = config.habitat_baselines.num_environments
        kwargs["g_ascent_config"] = policy_config.g_ascent
        return cls(**kwargs)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._g_cfg = kwargs.pop("g_ascent_config")
        self._llm_policy_cfg = kwargs["full_config"].habitat_baselines.rl.policy.main_agent.llm
        executor_kwargs = dict(kwargs)
        super().__init__(kwargs["action_space"])
        self._main_policy = Ascent_Policy(*args, **executor_kwargs)
        self._sub_policy = Ascent_Policy(*args, **executor_kwargs)
        self._num_envs = kwargs["num_envs"]
        self._action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([6.0, 6.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._region_builder = RegionGraphBuilder(
            cluster_radius=self._g_cfg.region_graph.cluster_radius,
        )
        self._enable_assignment_hints = os.environ.get("G_ASCENT_ENABLE_HINTS", "1") != "0"
        self._schedulers: List[GlobalScheduler] = [
            self._build_scheduler() for _ in range(self._num_envs)
        ]
        self._candidate_fusions: List[CandidateFusion] = [
            self._build_candidate_fusion() for _ in range(self._num_envs)
        ]
        self._episode_loggers: List[Optional[GEpisodeLogger]] = [None] * self._num_envs
        self._last_policy_info: List[Dict[str, Any]] = [{} for _ in range(self._num_envs)]
        self._last_explored_ratio: Dict[Tuple[str, int], float] = {}

    def _build_scheduler(self) -> GlobalScheduler:
        llm_scheduler = None
        if self._g_cfg.llm_scheduler.enabled:
            llm_scheduler = LLMScheduler(
                llm_config=self._llm_policy_cfg,
                min_interval_steps=self._g_cfg.llm_scheduler.min_interval_steps,
                floor_gap_threshold=self._g_cfg.llm_scheduler.floor_gap_threshold,
                region_gap_threshold=self._g_cfg.llm_scheduler.region_gap_threshold,
            )
        return GlobalScheduler(llm_scheduler=llm_scheduler)

    def _build_candidate_fusion(self) -> CandidateFusion:
        return CandidateFusion(
            merge_radius=self._g_cfg.candidate.merge_radius,
            likely_threshold=self._g_cfg.candidate.likely_threshold,
            verified_threshold=self._g_cfg.candidate.verified_threshold,
        )

    @property
    def should_load_agent_state(self) -> bool:
        return False

    def eval(self) -> None:
        self._main_policy.eval()
        self._sub_policy.eval()

    def train(self) -> None:
        self._main_policy.train()
        self._sub_policy.train()

    def act(
        self,
        observations: TensorDict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: torch.Tensor,
        deterministic: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> PolicyActionData:
        current_episodes_info = kwargs.get("current_episodes_info")
        main_obs = observations.to_tree()
        agent_0_obs = TensorDict.from_tree(_extract_prefixed_obs(main_obs, "agent_0_"))
        agent_1_obs = TensorDict.from_tree(_extract_prefixed_obs(main_obs, "agent_1_"))

        main_data = self._main_policy.act(
            agent_0_obs,
            rnn_hidden_states,
            prev_actions,
            masks,
            deterministic=deterministic,
            *args,
            **kwargs,
        )
        sub_data = self._sub_policy.act(
            agent_1_obs,
            rnn_hidden_states,
            prev_actions,
            masks,
            deterministic=deterministic,
            *args,
            **kwargs,
        )

        main_infos = main_data.policy_info or [{} for _ in range(self._num_envs)]
        sub_infos = sub_data.policy_info or [{} for _ in range(self._num_envs)]
        main_actions = main_data.actions.to(masks.device).float()
        sub_actions = sub_data.actions.to(masks.device).float()

        joint_actions: List[torch.Tensor] = []
        merged_infos: List[Dict[str, Any]] = []
        for env in range(self._num_envs):
            if masks[env][0] == 0:
                self._reset_env_state(env, current_episodes_info)
            agent_states = self._build_agent_states(env, agent_0_obs, agent_1_obs, main_infos[env], sub_infos[env])
            scheduler = self._schedulers[env]
            for state in agent_states:
                state.assigned_region = scheduler.get_last_assignment(state.agent_id)

            frontier_sets = self._collect_frontiers(env)
            explored_ratios = self._collect_explored_ratios(env)
            floor_priors = {floor_id: 0.0 for floor_id in explored_ratios.keys()}
            snapshot = self._region_builder.build(
                agent_states=agent_states,
                frontier_sets=frontier_sets,
                explored_ratios=explored_ratios,
                floor_priors=floor_priors,
                step=int(main_infos[env].get("num_steps", 0)),
            )
            self._update_candidates(env, agent_states, main_infos[env], sub_infos[env], snapshot)
            snapshot = scheduler.update(
                snapshot=snapshot,
                agent_states=agent_states,
                candidates=snapshot.candidates,
            )
            assignments = scheduler.assign(snapshot=snapshot, agent_states=agent_states)
            if self._enable_assignment_hints:
                self._apply_assignment_hints(
                    env=env,
                    snapshot=snapshot,
                    assignments=assignments,
                    agent_states=agent_states,
                    main_info=main_infos[env],
                    sub_info=sub_infos[env],
                )
            else:
                self._clear_forced_frontier(self._main_policy, env)
                self._clear_forced_frontier(self._sub_policy, env)
            self._log_snapshot(env, snapshot)

            main_action = float(main_actions[env].view(-1)[0].item())
            sub_action = float(sub_actions[env].view(-1)[0].item())
            main_action, sub_action = self._guard_joint_stops(
                env=env,
                agent_0_obs=agent_0_obs,
                agent_1_obs=agent_1_obs,
                main_info=main_infos[env],
                sub_info=sub_infos[env],
                main_action=main_action,
                sub_action=sub_action,
            )
            joint_actions.append(
                torch.tensor(
                    [[main_action, sub_action, 0.0, 0.0]],
                    dtype=torch.float32,
                    device=masks.device,
                )
            )
            merged_infos.append(
                self._merge_policy_info(
                    main_info=main_infos[env],
                    sub_info=sub_infos[env],
                    snapshot=snapshot,
                    assignments=assignments,
                    agent_states=agent_states,
                )
            )

        action_tensor = torch.cat(joint_actions, dim=0)
        self._last_policy_info = merged_infos
        return PolicyActionData(
            actions=action_tensor,
            take_actions=action_tensor,
            rnn_hidden_states=rnn_hidden_states,
            policy_info=merged_infos,
        )

    def get_extra(self, action_data: PolicyActionData, infos, dones) -> List[Dict[str, float]]:
        del infos
        del dones
        return action_data.policy_info or []

    def _reset_env_state(self, env: int, current_episodes_info) -> None:
        self._episode_loggers[env] = self._build_logger(env, current_episodes_info)
        self._candidate_fusions[env] = self._build_candidate_fusion()
        self._schedulers[env] = self._build_scheduler()
        for agent_id in ("agent_0", "agent_1"):
            self._last_explored_ratio.pop((agent_id, env), None)
        self._clear_forced_frontier(self._main_policy, env)
        self._clear_forced_frontier(self._sub_policy, env)

    def _build_logger(self, env: int, current_episodes_info) -> GEpisodeLogger:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        if current_episodes_info is not None and env < len(current_episodes_info):
            run_id = f"{run_id}_{current_episodes_info[env].episode_id}"
        run_dir = self._g_cfg.logging.log_dir_template.format(run_id=run_id)
        return GEpisodeLogger(run_dir)

    def _build_agent_states(
        self,
        env: int,
        agent_0_obs: TensorDict,
        agent_1_obs: TensorDict,
        main_info: Dict[str, Any],
        sub_info: Dict[str, Any],
    ) -> List[AgentState]:
        states: List[AgentState] = []
        for agent_id, obs, info, policy in [
            ("agent_0", agent_0_obs, main_info, self._main_policy),
            ("agent_1", agent_1_obs, sub_info, self._sub_policy),
        ]:
            gps = obs["gps"][env].cpu().numpy()
            pose = (float(gps[0]), float(-gps[1]))
            floor_id = int(policy._map_controller._cur_floor_index[env])
            explored_ratio = float(policy._map_controller._obstacle_map[env].explored_area.mean())
            ratio_key = (agent_id, env)
            last_ratio = self._last_explored_ratio.get(ratio_key, explored_ratio)
            coverage_gain = max(0.0, explored_ratio - last_ratio)
            self._last_explored_ratio[ratio_key] = explored_ratio
            nav_goal = info.get("nav_goal")
            nav_goal_tuple = None
            if isinstance(nav_goal, np.ndarray) and nav_goal.size >= 2:
                nav_goal_tuple = (float(nav_goal[0]), float(nav_goal[1]))
            states.append(
                AgentState(
                    agent_id=agent_id,
                    pose=pose,
                    floor_id=floor_id,
                    assigned_region=None,
                    coverage_gain=coverage_gain,
                    stall_score=0.0,
                    last_nav_goal=nav_goal_tuple,
                    target_detected=bool(info.get("target_detected", False)),
                    stop_called=bool(info.get("stop_called", False)),
                )
            )
        return states

    def _collect_frontiers(self, env: int) -> Dict[str, List[Tuple[Tuple[float, float], float]]]:
        frontier_sets: Dict[str, List[Tuple[Tuple[float, float], float]]] = {}
        for agent_id, policy in [("agent_0", self._main_policy), ("agent_1", self._sub_policy)]:
            frontiers = getattr(policy._map_controller._obstacle_map[env], "frontiers", [])
            frontier_array = np.asarray(frontiers)
            points: List[Tuple[Tuple[float, float], float]] = []
            score_lookup: Dict[Tuple[float, float], float] = {}
            if frontier_array.size > 0 and not np.array_equal(frontier_array, np.zeros((1, 2))):
                sorted_pts, sorted_values = policy._map_controller._value_map[env].sort_waypoints(frontier_array, 0.5)
                for point, value in zip(sorted_pts, sorted_values):
                    score_lookup[(float(point[0]), float(point[1]))] = float(value)
            for frontier in frontier_array:
                if len(frontier) >= 2:
                    key = (float(frontier[0]), float(frontier[1]))
                    points.append((key, score_lookup.get(key, 0.0)))
            frontier_sets[agent_id] = points
        return frontier_sets

    def _collect_explored_ratios(self, env: int) -> Dict[int, float]:
        explored_ratios: Dict[int, float] = {}
        for policy in (self._main_policy, self._sub_policy):
            floor_id = int(policy._map_controller._cur_floor_index[env])
            explored_ratio = float(policy._map_controller._obstacle_map[env].explored_area.mean())
            explored_ratios[floor_id] = max(explored_ratios.get(floor_id, 0.0), explored_ratio)
        return explored_ratios

    def _update_candidates(
        self,
        env: int,
        agent_states: List[AgentState],
        main_info: Dict[str, Any],
        sub_info: Dict[str, Any],
        snapshot,
    ) -> None:
        fusion = self._candidate_fusions[env]
        for state, info, policy in [
            (agent_states[0], main_info, self._main_policy),
            (agent_states[1], sub_info, self._sub_policy),
        ]:
            if not state.target_detected:
                continue
            nav_goal = info.get("nav_goal")
            centroid = state.pose
            if isinstance(nav_goal, np.ndarray) and nav_goal.size >= 2:
                centroid = (float(nav_goal[0]), float(nav_goal[1]))
            candidate = fusion.observe(
                agent_id=state.agent_id,
                floor_id=state.floor_id,
                centroid=centroid,
                score=max(0.5, float(policy._map_controller._blip_cosine[env])),
            )
            snapshot.candidates[candidate.candidate_id] = candidate

    def _apply_assignment_hints(
        self,
        *,
        env: int,
        snapshot,
        assignments,
        agent_states: List[AgentState],
        main_info: Dict[str, Any],
        sub_info: Dict[str, Any],
    ) -> None:
        del main_info
        self._clear_forced_frontier(self._main_policy, env)

        assignment_map = {assignment.agent_id: assignment.region_id for assignment in assignments}
        sub_region_id = assignment_map.get("agent_1")
        sub_region = snapshot.regions.get(sub_region_id) if sub_region_id is not None else None
        if (
            sub_region is None
            or sub_region.floor_id != agent_states[1].floor_id
            or agent_states[1].target_detected
            or sub_info.get("stop_called", False)
        ):
            self._clear_forced_frontier(self._sub_policy, env)
            return

        hint_frontier = self._select_hint_frontier(
            region=sub_region,
            agent_state=agent_states[1],
            other_state=agent_states[0],
            nav_goal=sub_info.get("nav_goal"),
        )
        if hint_frontier is None:
            self._clear_forced_frontier(self._sub_policy, env)
            return
        self._sub_policy.llm_planner._force_frontier[env] = hint_frontier

    def _select_hint_frontier(
        self,
        *,
        region,
        agent_state: AgentState,
        other_state: AgentState,
        nav_goal: Any,
    ) -> Optional[np.ndarray]:
        frontier_points = [np.asarray(point, dtype=np.float32) for point in region.frontier_points]
        if not frontier_points:
            return None
        centroid = np.asarray(region.centroid, dtype=np.float32)
        other_pose = np.asarray(other_state.pose, dtype=np.float32)
        own_pose = np.asarray(agent_state.pose, dtype=np.float32)
        current_goal = None
        if isinstance(nav_goal, np.ndarray) and nav_goal.size >= 2:
            current_goal = np.asarray(nav_goal[:2], dtype=np.float32)

        best_frontier = None
        best_score = float("-inf")
        for point in frontier_points:
            score = 1.0 * float(np.linalg.norm(point - other_pose))
            score -= 0.35 * float(np.linalg.norm(point - own_pose))
            score -= 0.25 * float(np.linalg.norm(point - centroid))
            if current_goal is not None:
                score += 0.15 * max(0.0, 2.0 - float(np.linalg.norm(point - current_goal)))
            if score > best_score:
                best_frontier = point
                best_score = score
        return best_frontier

    @staticmethod
    def _clear_forced_frontier(policy: Ascent_Policy, env: int) -> None:
        policy.llm_planner._force_frontier[env] = np.zeros(2, dtype=np.float32)

    def _guard_joint_stops(
        self,
        *,
        env: int,
        agent_0_obs: TensorDict,
        agent_1_obs: TensorDict,
        main_info: Dict[str, Any],
        sub_info: Dict[str, Any],
        main_action: float,
        sub_action: float,
    ) -> tuple[float, float]:
        del agent_0_obs
        del agent_1_obs
        if int(sub_action) != 0:
            sub_info["sub_stop_suppressed"] = False
            return main_action, sub_action
        candidate = self._candidate_fusions[env].best()
        allow_stop = bool(main_info.get("target_detected", False))
        if candidate is not None and len(candidate.supporters) >= 2 and str(candidate.status) == "verified":
            allow_stop = True
        if allow_stop:
            sub_info["sub_stop_suppressed"] = False
            return main_action, sub_action
        sub_info["stop_called"] = False
        sub_info["sub_stop_suppressed"] = True
        return main_action, 1.0

    def _merge_policy_info(
        self,
        *,
        main_info: Dict[str, Any],
        sub_info: Dict[str, Any],
        snapshot,
        assignments,
        agent_states: List[AgentState],
    ) -> Dict[str, Any]:
        merged = dict(main_info)
        merged["coordination_mode"] = self._g_cfg.coordination_mode
        merged["region_graph_region_count"] = float(len(snapshot.regions))
        merged["region_graph_floor_count"] = float(len(snapshot.floors))
        merged["scheduler_trigger"] = str(snapshot.trigger)
        merged["candidate_count"] = float(len(snapshot.candidates))
        merged["target_detected"] = bool(main_info.get("target_detected", False) or sub_info.get("target_detected", False))
        merged["subagent_active"] = 1.0
        merged["sub_target_detected"] = bool(sub_info.get("target_detected", False))
        assignment_map = {assignment.agent_id: assignment.region_id for assignment in assignments}
        merged["assigned_region_main"] = assignment_map.get("agent_0") or ""
        merged["assigned_region_sub"] = assignment_map.get("agent_1") or ""
        merged["main_role"] = str(agent_states[0].role)
        merged["sub_role"] = str(agent_states[1].role)
        merged["verifier_needed"] = any(candidate.verifier_needed for candidate in snapshot.candidates.values())
        merged["sub_stop_suppressed"] = bool(sub_info.get("sub_stop_suppressed", False))
        return merged

    def _log_snapshot(self, env: int, snapshot) -> None:
        logger = self._episode_loggers[env]
        if logger is None:
            return
        logger.save_snapshot(snapshot.to_dict())
        logger.log_event(
            "scheduler_step",
            {
                "step": snapshot.step,
                "trigger": str(snapshot.trigger),
                "region_count": len(snapshot.regions),
                "assignment_count": len(snapshot.assignments),
                "candidate_count": len(snapshot.candidates),
            },
        )
