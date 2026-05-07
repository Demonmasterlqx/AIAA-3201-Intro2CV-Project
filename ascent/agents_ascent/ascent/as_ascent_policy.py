from __future__ import annotations

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
from habitat_baselines.rl.ppo.policy import PolicyActionData
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from torch import Tensor
from vlfm.obs_transformers.utils import image_resize
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.policy.habitat_policies import VLFMPolicyConfig
from vlfm.utils.geometry_utils import get_fov
from vlfm.utils.geometry_utils import rho_theta

from ascent.as_ascent_config import AscentMultiAgentPolicyConfig
from ascent.as_ascent_coordinator import compute_frontier_utility
from ascent.as_ascent_coordinator import frontier_overlap_penalty
from ascent.as_ascent_coordinator import should_trigger_fine_grained
from ascent.as_ascent_logging import ASEpisodeLogger
from ascent.as_ascent_shared_memory import SharedMemoryManager
from ascent.as_ascent_structures import ConfirmationState
from ascent.as_ascent_structures import CoordinatorState
from ascent.as_ascent_structures import FrontierRecord
from ascent.as_ascent_structures import FrontierStatus
from ascent.as_ascent_structures import IntentMode
from ascent.as_ascent_structures import IntentRecord
from ascent.as_ascent_structures import SubagentState
from ascent.ascent_policy import Ascent_Policy
from ascent.pointnav_policy import WrappedPointNavResNetPolicy
from ascent.utils import build_non_coco_caption
from ascent.utils import get_action_tensor
from ascent.utils import infer_dataset_type_from_path
from ascent.utils import resolve_object_goal_names
from constants import LOOK_DOWN
from constants import LOOK_UP
from constants import MOVE_FORWARD
from constants import STOP
from constants import TURN_LEFT
from constants import TURN_RIGHT


NOOP_ACTION_ID = 6


@baseline_registry.register_policy
class AscentMultiAgentPolicy(Ascent_Policy):
    @classmethod
    def from_config(
        cls,
        config,
        *args_unused: Any,
        **kwargs_unused: Any,
    ) -> "AscentMultiAgentPolicy":
        rl_policy_config = config.habitat_baselines.rl.policy
        policy_config: AscentMultiAgentPolicyConfig = rl_policy_config.main_agent
        kwargs = {k: policy_config[k] for k in VLFMPolicyConfig.kwaarg_names}
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
        kwargs["as_ascent_config"] = policy_config.as_ascent
        return cls(**kwargs)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._as_ascent_cfg = kwargs.pop("as_ascent_config")
        sub_policy_kwargs = dict(kwargs)
        super().__init__(*args, **kwargs)
        self._sub_policy = Ascent_Policy(*args, **sub_policy_kwargs)
        self._joint_action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([6.0, 6.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._sub_pointnav_policy: List[WrappedPointNavResNetPolicy] = [
            WrappedPointNavResNetPolicy(
                kwargs["pointnav_policy_path"],
                original_config=kwargs["full_config"],
                agent_name="agent_1",
                device=self.device,
            )
            for _ in range(self._num_envs)
        ]
        self._sub_last_goal: List[np.ndarray] = [np.zeros(2) for _ in range(self._num_envs)]
        self._sub_called_stop: List[bool] = [False] * self._num_envs
        self._sub_history_action: List[List[int]] = [[] for _ in range(self._num_envs)]
        self._sub_pitch_angle: List[int] = [0] * self._num_envs
        self._sub_observations_cache: List[dict] = [{} for _ in range(self._num_envs)]
        self._sub_active: List[bool] = [
            self._is_naive_shared_mode() for _ in range(self._num_envs)
        ]
        self._sub_frontier: List[np.ndarray] = [np.array([]) for _ in range(self._num_envs)]
        self._shared_memory = SharedMemoryManager(
            reservation_ttl=self._as_ascent_cfg.spawn.reservation_ttl,
            candidate_merge_radius=self._as_ascent_cfg.evidence.candidate_merge_radius,
        )
        self._coordinator_state: List[CoordinatorState] = [
            CoordinatorState.SINGLE_AGENT_NORMAL for _ in range(self._num_envs)
        ]
        self._episode_loggers: List[Optional[ASEpisodeLogger]] = [None] * self._num_envs
        self._spawn_count: List[int] = [0] * self._num_envs
        self._reclaim_count: List[int] = [0] * self._num_envs
        self._cross_floor_transition_count: List[int] = [0] * self._num_envs
        self._cross_floor_success_count: List[int] = [0] * self._num_envs
        self._first_detection_step: List[int] = [-1] * self._num_envs
        self._verification_step: List[int] = [-1] * self._num_envs
        self._false_positive_abort_count: List[int] = [0] * self._num_envs
        self._overlap_ratio: List[float] = [0.0] * self._num_envs
        self._last_floor_index: List[int] = [0] * self._num_envs
        self._verification_source: List[str] = ["" for _ in range(self._num_envs)]
        self._pending_spawn: List[bool] = [False] * self._num_envs
        self._pending_reclaim: List[bool] = [False] * self._num_envs
        self._sub_last_progress_step: List[int] = [0] * self._num_envs
        self._sub_commit_until_step: List[int] = [-1] * self._num_envs
        self._sub_fail_count: List[int] = [0] * self._num_envs
        self._sub_state: List[SubagentState] = [SubagentState.DORMANT] * self._num_envs
        self._last_policy_info: List[Dict[str, Any]] = [{} for _ in range(self._num_envs)]
        self._sub_target_detections: List[Optional[Any]] = [None] * self._num_envs

    @property
    def should_load_agent_state(self) -> bool:
        return False

    @property
    def policy_action_space(self):
        return self._joint_action_space

    def eval(self) -> None:
        return None

    def train(self) -> None:
        return None

    def act(
        self,
        observations: TensorDict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> PolicyActionData:
        if self._is_naive_shared_mode():
            return self._act_naive_independent(
                observations,
                rnn_hidden_states,
                prev_actions,
                masks,
                deterministic,
                *args,
                **kwargs,
            )
        del deterministic
        current_episodes_info = kwargs.get("current_episodes_info")
        main_obs = observations.to_tree()
        agent_0_obs = _extract_prefixed_obs(main_obs, "agent_0_")
        agent_1_obs = _extract_prefixed_obs(main_obs, "agent_1_")

        object_ids = agent_0_obs[ObjectGoalSensor.cls_uuid]
        goal_names = resolve_object_goal_names(
            self._dataset_type,
            object_ids,
            current_episodes_info=current_episodes_info,
        )
        agent_0_obs[ObjectGoalSensor.cls_uuid] = goal_names
        agent_1_obs[ObjectGoalSensor.cls_uuid] = goal_names
        non_coco_caption = build_non_coco_caption(self._dataset_type, goal_names)
        self._non_coco_caption = non_coco_caption or ""

        self._pre_step(agent_0_obs, masks)
        for env in range(self._num_envs):
            self._maybe_reset_multi_agent(env, masks, current_episodes_info, goal_names)
            self._cache_secondary_observations(agent_1_obs, env)

        img_height, img_width = agent_0_obs["rgb"].shape[1:3]
        self._map_controller._update_object_map_with_stair_and_person(
            img_height,
            img_width,
            self._observations_cache,
            self._non_coco_caption,
            self._num_steps,
            self._try_to_navigate,
        )
        for env in range(self._num_envs):
            self._sub_target_detections[env] = self._update_secondary_target_map(env)
        self.red_semantic_pred_list = []
        self.seg_map_color_list = []
        for env in range(self._num_envs):
            rgb = agent_0_obs["rgb"][env : env + 1].float()
            depth = agent_0_obs["depth"][env : env + 1].float()
            with torch.no_grad():
                red_semantic_pred = self.red_sem_pred(rgb, depth).squeeze().cpu().numpy().astype(np.uint8)
            self.red_semantic_pred_list.append(red_semantic_pred)
            self.seg_map_color_list.append(np.zeros((*red_semantic_pred.shape, 3), dtype=np.uint8))

        self._map_controller._update_obstacle_map(self._observations_cache, self.red_semantic_pred_list, self._pitch_angle)
        self._map_controller._update_value_map(self._observations_cache)
        self._map_controller._update_distance_on_object_map(self._observations_cache)

        joint_actions: List[Tensor] = []
        for env in range(self._num_envs):
            self._ingest_shared_memory(env)
            self._update_target_evidence(env, "main", self._observations_cache[env], self._map_controller.target_detection_list[env])
            if self._sub_active[env] or self._pending_spawn[env]:
                self._update_target_evidence(
                    env,
                    "sub",
                    self._sub_observations_cache[env],
                    None,
                )

            spawn_signal, reclaim_signal = self._update_subagent_state(env)
            main_action = self._compute_main_action(agent_0_obs, env, masks)
            sub_action = self._compute_sub_action(env, masks)
            joint_actions.append(
                torch.tensor(
                    [[float(main_action), float(sub_action), spawn_signal, reclaim_signal]],
                    dtype=torch.float32,
                    device=masks.device,
                )
            )
            self._last_policy_info[env] = self._build_policy_info(env)
            self._observations_cache[env] = {}
            self._sub_observations_cache[env] = {}
            self._did_reset[env] = False

        action_tensor = torch.cat(joint_actions, dim=0)
        return PolicyActionData(
            actions=action_tensor,
            take_actions=action_tensor,
            rnn_hidden_states=rnn_hidden_states,
            policy_info=self._last_policy_info,
        )

    def _act_naive_independent(
        self,
        observations: TensorDict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool,
        *args: Any,
        **kwargs: Any,
    ) -> PolicyActionData:
        main_obs = observations.to_tree()
        agent_0_obs = TensorDict.from_tree(_extract_prefixed_obs(main_obs, "agent_0_"))
        agent_1_obs = TensorDict.from_tree(_extract_prefixed_obs(main_obs, "agent_1_"))

        main_data = super().act(
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

        main_actions = main_data.actions.to(masks.device).float()
        sub_actions = sub_data.actions.to(masks.device).float()
        joint_actions: List[Tensor] = []
        merged_infos: List[Dict[str, Any]] = []
        main_infos = main_data.policy_info or [{} for _ in range(main_actions.shape[0])]
        sub_infos = sub_data.policy_info or [{} for _ in range(sub_actions.shape[0])]

        for env in range(main_actions.shape[0]):
            main_action = float(main_actions[env].view(-1)[0].item())
            sub_action = float(sub_actions[env].view(-1)[0].item())
            if masks[env][0] == 0 and int(main_action) == int(sub_action):
                sub_action = float(TURN_RIGHT if int(main_action) != TURN_RIGHT else TURN_LEFT)

            joint_actions.append(
                torch.tensor(
                    [[main_action, sub_action, 0.0, 0.0]],
                    dtype=torch.float32,
                    device=masks.device,
                )
            )
            merged_infos.append(self._merge_naive_policy_info(main_infos[env], sub_infos[env]))

        action_tensor = torch.cat(joint_actions, dim=0)
        self._last_policy_info = merged_infos
        return PolicyActionData(
            actions=action_tensor,
            take_actions=action_tensor,
            rnn_hidden_states=rnn_hidden_states,
            policy_info=merged_infos,
        )

    def _merge_naive_policy_info(
        self,
        main_info: Dict[str, Any],
        sub_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = dict(main_info)
        merged["coordination_mode"] = self._as_ascent_cfg.coordination_mode
        merged["subagent_active"] = 1.0
        merged["sub_stop_called"] = bool(sub_info.get("stop_called", False))
        merged["sub_target_detected"] = bool(sub_info.get("target_detected", False))
        merged["stop_called"] = bool(main_info.get("stop_called", False) or sub_info.get("stop_called", False))
        merged["target_detected"] = bool(
            main_info.get("target_detected", False) or sub_info.get("target_detected", False)
        )
        if bool(sub_info.get("stop_called", False)) or (
            bool(sub_info.get("target_detected", False)) and not bool(main_info.get("target_detected", False))
        ):
            if "nav_goal" in sub_info:
                merged["nav_goal"] = sub_info["nav_goal"]
            if "rho_theta" in sub_info:
                merged["rho_theta"] = sub_info["rho_theta"]
            if "goal_detected" in sub_info:
                merged["goal_detected"] = sub_info["goal_detected"]
        return merged

    def get_extra(self, action_data: PolicyActionData, infos, dones) -> List[Dict[str, float]]:
        del infos
        del dones
        return action_data.policy_info or []

    def _maybe_reset_multi_agent(
        self,
        env: int,
        masks: Tensor,
        current_episodes_info,
        goal_names: List[str],
    ) -> None:
        if masks[env][0] != 0:
            return
        self._reset(env)
        self._map_controller._target_object[env] = goal_names[env]
        self._sub_pointnav_policy[env].reset()
        self._sub_last_goal[env] = np.zeros(2)
        self._sub_called_stop[env] = False
        self._sub_history_action[env].clear()
        self._sub_pitch_angle[env] = 0
        self._sub_active[env] = self._is_naive_shared_mode()
        self._shared_memory.reset()
        self._coordinator_state[env] = CoordinatorState.SINGLE_AGENT_NORMAL
        self._spawn_count[env] = 0
        self._reclaim_count[env] = 0
        self._cross_floor_transition_count[env] = 0
        self._cross_floor_success_count[env] = 0
        self._first_detection_step[env] = -1
        self._verification_step[env] = -1
        self._false_positive_abort_count[env] = 0
        self._overlap_ratio[env] = 0.0
        self._last_floor_index[env] = 0
        self._verification_source[env] = ""
        self._pending_spawn[env] = False
        self._pending_reclaim[env] = False
        self._sub_last_progress_step[env] = 0
        self._sub_commit_until_step[env] = -1
        self._sub_fail_count[env] = 0
        self._sub_state[env] = SubagentState.DORMANT
        self._episode_loggers[env] = self._build_logger(env, current_episodes_info)

    def _build_logger(self, env: int, current_episodes_info) -> ASEpisodeLogger:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        if current_episodes_info is not None and env < len(current_episodes_info):
            episode = current_episodes_info[env]
            run_id = f"{run_id}_{episode.episode_id}"
        run_dir = self._as_ascent_cfg.logging.log_dir_template.format(run_id=run_id)
        return ASEpisodeLogger(run_dir)

    def _cache_secondary_observations(self, observations: Dict[str, Tensor], env: int) -> None:
        rgb = observations["rgb"][env].cpu().numpy()
        depth = observations["depth"][env].cpu().numpy()
        x, y = observations["gps"][env].cpu().numpy()
        camera_yaw = observations["compass"][env].cpu().item()
        camera_position = np.array([x, -y, self._camera_height])
        robot_xy = camera_position[:2]
        camera_pitch = np.radians(-self._sub_pitch_angle[env])
        tf_camera_to_episodic = self._build_tf(camera_position, camera_yaw, camera_pitch)
        self._sub_observations_cache[env] = {
            "robot_xy": robot_xy,
            "robot_heading": camera_yaw,
            "tf_camera_to_episodic": tf_camera_to_episodic,
            "rgb": rgb,
            "depth": depth.reshape(depth.shape[:2]),
            "min_depth": self._min_depth,
            "max_depth": self._max_depth,
            "fx": self._fx,
            "fy": self._fy,
            "camera_fov": self._camera_fov,
            "habitat_start_yaw": observations["heading"][env].item(),
            "nav_rgb": torch.unsqueeze(observations["rgb"][env], dim=0),
            "nav_depth": torch.unsqueeze(observations["depth"][env], dim=0),
        }

    def _build_tf(self, camera_position: np.ndarray, camera_yaw: float, camera_pitch: float):
        from ascent.utils import xyz_yaw_pitch_roll_to_tf_matrix

        return xyz_yaw_pitch_roll_to_tf_matrix(camera_position, camera_yaw, camera_pitch, 0.0)

    def _ingest_shared_memory(self, env: int) -> None:
        current_floor = int(self._map_controller._cur_floor_index[env])
        floor_prior = 0.0
        if hasattr(self.llm_planner, "get_floor_probabilities"):
            try:
                probs = self.llm_planner.get_floor_probabilities(
                    self.floor_probabilities_df,
                    self._map_controller._target_object[env].split("|")[0],
                    max(1, self._map_controller.floor_num[env]),
                )
                floor_prior = float(probs.get(current_floor + 1, 0.0)) / 100.0
            except Exception:
                floor_prior = 0.0

        self._shared_memory.update_floor(
            current_floor,
            visited_ratio=float(self._map_controller._obstacle_map[env].explored_area.mean()),
            frontier_count=int(len(self._map_controller._obstacle_map[env].frontiers)),
            stairs_up=bool(self._map_controller._obstacle_map[env]._has_up_stair),
            stairs_down=bool(self._map_controller._obstacle_map[env]._has_down_stair),
            floor_prior_for_goal=floor_prior,
            last_updated_step=int(self._num_steps[env]),
            agent_presence_state={
                "main": f"floor_{current_floor}",
                "sub": f"floor_{current_floor}" if self._sub_active[env] else "dormant",
            },
        )

        frontiers = np.array(self._map_controller._obstacle_map[env].frontiers)
        frontier_records = self._build_frontier_records(env, frontiers, current_floor, floor_prior)
        self._shared_memory.refresh_frontiers(current_floor, frontier_records, int(self._num_steps[env]))

    def _build_frontier_records(
        self, env: int, frontiers: np.ndarray, current_floor: int, floor_prior: float
    ) -> List[FrontierRecord]:
        if frontiers.size == 0 or np.array_equal(frontiers, np.zeros((1, 2))):
            return []
        sorted_pts, sorted_values = self._map_controller._value_map[env].sort_waypoints(frontiers, 0.5)
        value_lookup = {tuple(point.tolist()): float(value) for point, value in zip(sorted_pts, sorted_values)}
        records: List[FrontierRecord] = []
        main_xy = self._observations_cache[env]["robot_xy"]
        sub_xy = self._sub_observations_cache[env].get("robot_xy") if self._sub_observations_cache[env] else None
        assigned_main = None
        if self.cur_frontier[env].size == 2:
            assigned_main = self.cur_frontier[env]
        for idx, frontier in enumerate(frontiers):
            centroid = (float(frontier[0]), float(frontier[1]))
            record = FrontierRecord(
                frontier_id=f"f_{env}_{current_floor}_{idx}",
                floor_id=current_floor,
                centroid=centroid,
                cluster_extent=0.5,
                semantic_score=value_lookup.get(tuple(frontier.tolist()), 0.0),
                geometry_gain=float(np.linalg.norm(frontier - main_xy)),
                path_cost_from_main=float(np.linalg.norm(frontier - main_xy)),
                path_cost_from_sub=float(np.linalg.norm(frontier - sub_xy)) if sub_xy is not None else float(np.linalg.norm(frontier - main_xy)),
            )
            breakdown = compute_frontier_utility(
                record,
                floor_prior=floor_prior,
                area_prior=0.0,
                overlap_penalty=frontier_overlap_penalty(record.centroid, assigned_main),
                blacklist_penalty=1.0 if self._shared_memory.is_blacklisted(record.frontier_id) else 0.0,
                stale_penalty=0.0,
                weights=self._as_ascent_cfg.utility,
                for_subagent=False,
            )
            record.utility = breakdown
            records.append(record)
        return records

    def _update_target_evidence(
        self,
        env: int,
        agent_id: str,
        observation_cache: Dict[str, Any],
        target_detections: Any,
    ) -> None:
        if target_detections is None:
            return
        if getattr(target_detections, "num_detections", 0) <= 0:
            return
        if self._first_detection_step[env] < 0:
            self._first_detection_step[env] = int(self._num_steps[env])
        if len(target_detections.logits) > 0:
            logits = target_detections.logits
            if hasattr(logits, "detach"):
                score = float(logits.max().detach().cpu().item())
            else:
                score = float(np.max(logits))
        else:
            score = 0.5
        score = max(score, float(self._map_controller._blip_cosine[env]))
        pose = observation_cache["tf_camera_to_episodic"][:3, 3]
        candidate = self._shared_memory.upsert_candidate(
            floor_id=int(self._map_controller._cur_floor_index[env]),
            pose_estimate=pose,
            agent_id=agent_id,
            score=score,
            step=int(self._num_steps[env]),
            viewpoint_cluster=f"{agent_id}_{round(float(pose[0]), 1)}_{round(float(pose[1]), 1)}",
            likely_threshold=self._as_ascent_cfg.evidence.likely_threshold,
            verified_threshold=self._as_ascent_cfg.evidence.verified_threshold,
        )
        if (
            candidate.confirmation_state == ConfirmationState.VERIFIED
            and self._passes_stop_guard(candidate)
        ):
            if self._verification_step[env] < 0:
                self._verification_step[env] = int(self._num_steps[env])
                self._verification_source[env] = agent_id
        elif candidate.confirmation_state == ConfirmationState.VERIFIED:
            self._shared_memory.reject_candidate(candidate.candidate_id, "stop_guard_failed")
            self._false_positive_abort_count[env] += 1

    def _passes_stop_guard(self, candidate) -> bool:
        if candidate.supporting_frames < self._as_ascent_cfg.evidence.stability_window:
            return False
        if len(candidate.supporting_agents) == 0:
            return False
        return True

    def _update_secondary_target_map(self, env: int):
        if not self._sub_active[env]:
            return None
        if not self._sub_observations_cache[env]:
            return None
        cache = self._sub_observations_cache[env]
        rgb = cache["rgb"]
        depth = cache["depth"]
        try:
            detections, _, _ = self._map_controller._get_object_detections_with_stair_and_person(
                rgb,
                self._non_coco_caption,
                env,
            )
            if getattr(detections, "num_detections", 0) > 0:
                height, width = rgb.shape[:2]
                for idx in range(len(detections.logits)):
                    target_bbox_denorm = detections.boxes[idx] * np.array([width, height, width, height])
                    target_object_mask = self._map_controller._mobile_sam.segment_bbox(
                        rgb,
                        target_bbox_denorm.tolist(),
                    )
                    self._map_controller._object_map[env].update_map(
                        self._map_controller._target_object[env],
                        depth,
                        target_object_mask,
                        cache["tf_camera_to_episodic"],
                        cache["min_depth"],
                        cache["max_depth"],
                        cache["fx"],
                        cache["fy"],
                    )
            self._map_controller._object_map[env].update_explored(
                cache["tf_camera_to_episodic"],
                cache["max_depth"],
                get_fov(cache["fx"], depth.shape[1]),
            )
            return detections
        except Exception as exc:
            self._log_event(
                env,
                "sub_detection_error",
                {
                    "step": int(self._num_steps[env]),
                    "error": str(exc),
                },
            )
            return None

    def _update_subagent_state(self, env: int) -> Tuple[float, float]:
        frontiers = list(self._shared_memory.snapshot.frontiers.values())
        if self._map_controller._cur_floor_index[env] != self._last_floor_index[env]:
            self._cross_floor_transition_count[env] += 1
            self._cross_floor_success_count[env] += 1
            self._last_floor_index[env] = self._map_controller._cur_floor_index[env]

        if self._as_ascent_cfg.coordination_mode == "single_agent":
            self._sub_active[env] = False
            return 0.0, 0.0

        if self._is_naive_shared_mode():
            if self._num_steps[env] == 0:
                self._sub_active[env] = True
                self._sub_state[env] = SubagentState.SPAWNED
                self._spawn_count[env] += 1
                self._coordinator_state[env] = CoordinatorState.DUAL_AGENT_ACTIVE
                self._log_event(
                    env,
                    "spawn",
                    {
                        "mode": self._as_ascent_cfg.coordination_mode,
                        "step": self._num_steps[env],
                    },
                )
                return 1.0, 0.0
            return 0.0, 0.0

        triggered, _ = should_trigger_fine_grained(
            frontiers,
            self._observations_cache[env]["robot_xy"],
            self._as_ascent_cfg.trigger.distance_threshold,
            self._as_ascent_cfg.trigger.min_frontiers,
            self._as_ascent_cfg.trigger.min_cluster_count,
            self._as_ascent_cfg.trigger.top2_gap_threshold,
        )
        if (
            triggered
            and not self._sub_active[env]
            and (self.max_episode_steps - self._num_steps[env]) >= self._as_ascent_cfg.spawn.min_remaining_steps
            and self._verification_step[env] < 0
        ):
            self._sub_active[env] = True
            self._sub_state[env] = SubagentState.SPAWNED
            self._sub_commit_until_step[env] = self._num_steps[env] + self._as_ascent_cfg.commit.commit_window_steps
            self._spawn_count[env] += 1
            self._coordinator_state[env] = CoordinatorState.DUAL_AGENT_ACTIVE
            self._log_event(env, "spawn", {"step": self._num_steps[env]})
            return 1.0, 0.0

        if self._sub_active[env]:
            best_candidate = self._shared_memory.best_candidate()
            if best_candidate is not None and best_candidate.confirmation_state == ConfirmationState.VERIFIED:
                self._sub_active[env] = False
                self._sub_state[env] = SubagentState.RECLAIMED
                self._reclaim_count[env] += 1
                self._coordinator_state[env] = CoordinatorState.RECLAIM_SUBAGENT
                self._log_event(env, "reclaim", {"reason": "verified_target"})
                return 0.0, 1.0
        return 0.0, 0.0

    def _compute_main_action(self, observations: Dict[str, Tensor], env: int, masks: Tensor) -> int:
        robot_xy = self._observations_cache[env]["robot_xy"]
        goal = self._get_target_object_location(robot_xy, env)
        if goal is None:
            if not self._map_controller._done_initializing[env]:
                action = self._initialize(env, masks)
                return int(action.detach().cpu().numpy()[0].item())
            action = self._explore(observations, env, masks)
        else:
            self._try_to_navigate[env] = True
            action = self._navigate(observations, goal[:2], stop=True, env=env, ori_masks=masks)
        action_value = int(action.detach().cpu().numpy()[0].item())
        self._num_steps[env] += 1
        self._map_controller._obstacle_map[env]._floor_num_steps += 1
        if (
            action_value == 0
            and (
                self._verification_step[env] < 0
                or not self._map_controller._double_check_goal[env]
            )
        ):
            action_value = MOVE_FORWARD
        return action_value

    def _compute_sub_action(self, env: int, masks: Tensor) -> int:
        if not self._sub_active[env]:
            return NOOP_ACTION_ID
        best_candidate = self._shared_memory.best_candidate()
        if best_candidate is not None and best_candidate.confirmation_state in (
            ConfirmationState.LIKELY,
            ConfirmationState.VERIFIED,
        ):
            self._sub_state[env] = SubagentState.VERIFYING
            self._sub_frontier[env] = np.array(best_candidate.pose_estimate[:2], dtype=np.float32)
            robot_xy = self._sub_observations_cache[env]["robot_xy"]
            if (
                best_candidate.confirmation_state == ConfirmationState.VERIFIED
                and getattr(self._sub_target_detections[env], "num_detections", 0) > 0
                and float(np.linalg.norm(robot_xy - self._sub_frontier[env])) <= 0.75
            ):
                self._sub_called_stop[env] = True
                return STOP
        else:
            frontier = self._choose_sub_frontier(env)
            if frontier is None:
                return NOOP_ACTION_ID
            self._sub_frontier[env] = np.array(frontier.centroid, dtype=np.float32)
            self._sub_state[env] = SubagentState.COMMITTED if self._num_steps[env] <= self._sub_commit_until_step[env] else SubagentState.EXPLORING
        action = self._sub_pointnav(env, self._sub_frontier[env], masks)
        action_value = int(action.detach().cpu().numpy()[0].item())
        self._sub_history_action[env].append(action_value)
        return action_value

    def _choose_sub_frontier(self, env: int) -> Optional[FrontierRecord]:
        frontiers = [
            frontier
            for frontier in self._shared_memory.snapshot.frontiers.values()
            if frontier.status in (FrontierStatus.FREE, FrontierStatus.RESERVED)
        ]
        if not frontiers:
            return None
        main_assignment = self.cur_frontier[env] if self.cur_frontier[env].size == 2 else None
        if self._is_naive_shared_mode():
            anchor = (
                np.array(main_assignment, dtype=np.float32)
                if main_assignment is not None
                else np.array(self._observations_cache[env]["robot_xy"], dtype=np.float32)
            )
            best_frontier = None
            best_score = -1e9
            for frontier in frontiers:
                frontier_xy = np.array(frontier.centroid, dtype=np.float32)
                separation = float(np.linalg.norm(frontier_xy - anchor))
                overlap = frontier_overlap_penalty(frontier.centroid, main_assignment)
                self._overlap_ratio[env] = max(self._overlap_ratio[env], overlap)
                travel_cost = float(frontier.path_cost_from_sub)
                score = separation + 0.25 * frontier.geometry_gain - 0.15 * travel_cost
                if frontier.status == FrontierStatus.STALE:
                    score -= 0.5
                if score > best_score:
                    best_score = score
                    best_frontier = frontier
            if best_frontier is not None:
                self._shared_memory.reserve_frontier(
                    best_frontier.frontier_id,
                    "sub",
                    self._num_steps[env],
                )
            return best_frontier
        best_frontier = None
        best_score = -1e9
        for frontier in frontiers:
            overlap = frontier_overlap_penalty(frontier.centroid, main_assignment)
            self._overlap_ratio[env] = max(self._overlap_ratio[env], overlap)
            breakdown = compute_frontier_utility(
                frontier,
                floor_prior=self._shared_memory.snapshot.floors[frontier.floor_id].floor_prior_for_goal,
                area_prior=0.0,
                overlap_penalty=overlap,
                blacklist_penalty=1.0 if self._shared_memory.is_blacklisted(frontier.frontier_id) else 0.0,
                stale_penalty=1.0 if frontier.status == FrontierStatus.STALE else 0.0,
                weights=self._as_ascent_cfg.utility,
                for_subagent=True,
            )
            self._shared_memory.set_frontier_utility(frontier.frontier_id, breakdown)
            if breakdown.total_utility > best_score:
                best_score = breakdown.total_utility
                best_frontier = frontier
        if best_frontier is not None:
            self._shared_memory.reserve_frontier(
                best_frontier.frontier_id,
                "sub",
                self._num_steps[env],
            )
        return best_frontier

    def _sub_pointnav(self, env: int, goal: np.ndarray, masks: Tensor) -> Tensor:
        del masks
        cache = self._sub_observations_cache[env]
        pointnav_policy = self._sub_pointnav_policy[env]
        pointnav_device = pointnav_policy.device
        reset_mask = torch.tensor([self._num_steps[env] != 0], dtype=torch.bool, device=pointnav_device)
        if not np.array_equal(goal, self._sub_last_goal[env]):
            if np.linalg.norm(goal - self._sub_last_goal[env]) > 0.1:
                pointnav_policy.reset()
                reset_mask = torch.zeros_like(reset_mask)
            self._sub_last_goal[env] = goal
        rho, theta = rho_theta(cache["robot_xy"], cache["robot_heading"], goal)
        obs_pointnav = {
            "depth": image_resize(
                cache["nav_depth"],
                (self._depth_image_shape[0], self._depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": torch.tensor(
                [[rho, theta]], device=pointnav_device, dtype=torch.float32
            ),
        }
        return pointnav_policy.act(obs_pointnav, reset_mask, deterministic=True)

    def _build_policy_info(self, env: int) -> Dict[str, Any]:
        info = self._get_policy_info(self._map_controller.target_detection_list[env], env)
        verified_delta = -1
        if self._first_detection_step[env] >= 0 and self._verification_step[env] >= 0:
            verified_delta = self._verification_step[env] - self._first_detection_step[env]
        info["stop_called"] = bool(info.get("stop_called", False) or self._sub_called_stop[env])
        info["target_detected"] = bool(
            info.get("target_detected", False)
            or getattr(self._sub_target_detections[env], "num_detections", 0) > 0
        )
        info.update(
            {
                "spawn_count": float(self._spawn_count[env]),
                "reclaim_count": float(self._reclaim_count[env]),
                "first_detection_step": float(self._first_detection_step[env]),
                "verification_step": float(self._verification_step[env]),
                "overlap_ratio": float(self._overlap_ratio[env]),
                "cross_floor_transition_count": float(self._cross_floor_transition_count[env]),
                "cross_floor_success_rate": float(
                    self._cross_floor_success_count[env] / max(1, self._cross_floor_transition_count[env])
                ),
                "suspect_to_verified_time": float(verified_delta),
                "false_positive_abort_count": float(self._false_positive_abort_count[env]),
                "coordination_mode": self._as_ascent_cfg.coordination_mode,
                "subagent_active": float(self._sub_active[env]),
                "subagent_state": str(self._sub_state[env]),
                "coordinator_state": str(self._coordinator_state[env]),
                "verification_source": self._verification_source[env],
            }
        )
        return info

    def _is_naive_shared_mode(self) -> bool:
        return self._as_ascent_cfg.coordination_mode in {
            "naive_shared",
            "naive_always_on",
        }

    def _log_event(self, env: int, event_type: str, payload: Dict[str, Any]) -> None:
        logger = self._episode_loggers[env]
        if logger is None:
            return
        logger.log_event(event_type, payload)


def _extract_prefixed_obs(observations: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    extracted: Dict[str, Any] = {}
    for key, value in observations.items():
        if key.startswith(prefix):
            extracted[key[len(prefix) :]] = value
    return extracted
