from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import quaternion
from gym import spaces

from habitat.core.embodied_task import Measure
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.registry import registry
from habitat.core.simulator import Sensor
from habitat.core.simulator import SensorTypes
from habitat.tasks.nav.nav import DistanceToGoal
from habitat.tasks.nav.nav import EpisodicCompassSensor
from habitat.tasks.nav.nav import EpisodicGPSSensor
from habitat.tasks.nav.nav import HeadingSensor
from habitat.tasks.nav.nav import NavigationTask
from habitat.tasks.nav.nav import Success
from habitat.tasks.nav.nav import cartesian_to_polar
from habitat.tasks.nav.object_nav_task import ObjectNavigationTask
from habitat.tasks.nav.object_nav_task import ObjectGoal
from habitat.tasks.nav.object_nav_task import ObjectGoalNavEpisode
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat.utils.geometry_utils import quaternion_from_coeff
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.config.default_structured_configs import CompassSensorConfig
from habitat.config.default_structured_configs import GPSSensorConfig
from habitat.config.default_structured_configs import HeadingSensorConfig
from habitat.config.default_structured_configs import ActionConfig
from habitat.config.default_structured_configs import MeasurementConfig
from habitat.config.default_structured_configs import ObjectGoalSensorConfig
from hydra.core.config_store import ConfigStore


STOP_ACTION_ID = 0
MOVE_FORWARD_ACTION_ID = 1
TURN_LEFT_ACTION_ID = 2
TURN_RIGHT_ACTION_ID = 3
LOOK_UP_ACTION_ID = 4
LOOK_DOWN_ACTION_ID = 5
NOOP_ACTION_ID = 6


@registry.register_task(name="MultiAgentObjectNav-v0")
class MultiAgentObjectNavigationTask(ObjectNavigationTask):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.runtime_stats: Dict[str, Any] = {}
        self.is_stop_called = False
        self.stop_agent_id = 0

    def reset(self, episode) -> None:
        self.runtime_stats = {}
        self.is_stop_called = False
        self.stop_agent_id = 0
        return super().reset(episode)


@registry.register_sensor(name="AgentHeadingSensor")
class AgentHeadingSensor(HeadingSensor):
    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self.agent_id = getattr(config, "agent_id", 0)
        super().__init__(sim, config, *args, **kwargs)

    def get_observation(self, observations, episode, *args: Any, **kwargs: Any):
        agent_state = self._sim.get_agent_state(self.agent_id)
        rotation_world_agent = agent_state.rotation
        if isinstance(rotation_world_agent, quaternion.quaternion):
            return self._quat_to_xy_heading(rotation_world_agent.inverse())
        raise ValueError("Agent's rotation was not a quaternion")


@registry.register_sensor(name="AgentCompassSensor")
class AgentCompassSensor(AgentHeadingSensor):
    cls_uuid: str = "compass"

    def get_observation(self, observations, episode, *args: Any, **kwargs: Any):
        agent_state = self._sim.get_agent_state(self.agent_id)
        rotation_world_agent = agent_state.rotation
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)
        if isinstance(rotation_world_agent, quaternion.quaternion):
            return self._quat_to_xy_heading(
                rotation_world_agent.inverse() * rotation_world_start
            )
        raise ValueError("Agent's rotation was not a quaternion")


@registry.register_sensor(name="AgentGPSSensor")
class AgentGPSSensor(EpisodicGPSSensor):
    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self.agent_id = getattr(config, "agent_id", 0)
        super().__init__(sim, config, *args, **kwargs)

    def get_observation(self, observations, episode, *args: Any, **kwargs: Any):
        agent_state = self._sim.get_agent_state(self.agent_id)
        origin = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)
        agent_position = quaternion_rotate_vector(
            rotation_world_start.inverse(), agent_state.position - origin
        )
        if self._dimensionality == 2:
            return np.array([-agent_position[2], agent_position[0]], dtype=np.float32)
        return agent_position.astype(np.float32)


@registry.register_sensor(name="AgentObjectGoalSensor")
class AgentObjectGoalSensor(ObjectGoalSensor):
    def __init__(self, sim, config, dataset, *args: Any, **kwargs: Any):
        self.agent_id = getattr(config, "agent_id", 0)
        super().__init__(sim, config, dataset, *args, **kwargs)

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: ObjectGoalNavEpisode,
        **kwargs: Any,
    ) -> Optional[np.ndarray]:
        if len(episode.goals) == 0:
            return None
        if not isinstance(episode.goals[0], ObjectGoal):
            return None
        category_name = episode.object_category
        if self.config.goal_spec == "TASK_CATEGORY_ID":
            return np.array(
                [self._dataset.category_to_task_category_id[category_name]],
                dtype=np.int64,
            )
        obj_goal = episode.goals[0]
        return np.array([obj_goal.object_name_id], dtype=np.int64)


@registry.register_task_action
class JointNavAction(SimulatorTaskAction):
    def __init__(self, *args: Any, config, sim, **kwargs: Any) -> None:
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._turn_angle = getattr(sim.habitat_config, "turn_angle", 30.0)
        self._forward_step = getattr(sim.habitat_config, "forward_step_size", 0.25)
        self._allow_sliding = getattr(sim.habitat_config.habitat_sim_v0, "allow_sliding", False)
        self._tilt_angle = getattr(config, "tilt_angle", 30.0)
        self._agent1_reset_yaw_offset = float(os.environ.get("AS_NAIVE_AGENT1_YAW_OFFSET_DEG", "0.0"))

    @property
    def action_space(self):
        return spaces.Dict(
            {
                "agent_0_action": spaces.Box(low=0.0, high=6.0, shape=(1,), dtype=np.float32),
                "agent_1_action": spaces.Box(low=0.0, high=6.0, shape=(1,), dtype=np.float32),
                "spawn_signal": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "reclaim_signal": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

    def reset(self, task: NavigationTask, *args: Any, **kwargs: Any) -> None:
        task.is_stop_called = False  # type: ignore[attr-defined]
        task.stop_agent_id = 0  # type: ignore[attr-defined]
        if abs(self._agent1_reset_yaw_offset) > 1e-3:
            self._turn_agent(1, self._agent1_reset_yaw_offset)

    def step(
        self,
        task: NavigationTask,
        *args: Any,
        agent_0_action: np.ndarray,
        agent_1_action: np.ndarray,
        spawn_signal: np.ndarray,
        reclaim_signal: np.ndarray,
        **kwargs: Any,
    ):
        task.is_stop_called = False  # type: ignore[attr-defined]
        task.stop_agent_id = 0  # type: ignore[attr-defined]
        if float(spawn_signal[0]) >= 0.5:
            self._copy_agent_pose(agent_src=0, agent_dst=1)
        if float(reclaim_signal[0]) >= 0.5:
            self._copy_agent_pose(agent_src=0, agent_dst=1)

        agent_actions = [int(np.clip(round(float(agent_0_action[0])), 0, 6))]
        agent_actions.append(int(np.clip(round(float(agent_1_action[0])), 0, 6)))

        collided = False
        collided = self._apply_agent_action(0, agent_actions[0], task) or collided
        collided = self._apply_agent_action(1, agent_actions[1], task) or collided
        if hasattr(self._sim, "_prev_sim_obs") and isinstance(self._sim._prev_sim_obs, dict):  # type: ignore[attr-defined]
            self._sim._prev_sim_obs["collided"] = collided  # type: ignore[attr-defined]

    def _apply_agent_action(
        self, agent_id: int, action_id: int, task: NavigationTask
    ) -> bool:
        if action_id == NOOP_ACTION_ID:
            return False
        if action_id == STOP_ACTION_ID:
            task.is_stop_called = True  # type: ignore[attr-defined]
            task.stop_agent_id = agent_id  # type: ignore[attr-defined]
            return False
        if action_id == LOOK_UP_ACTION_ID:
            self._move_camera_vertical(agent_id, self._tilt_angle)
            return False
        if action_id == LOOK_DOWN_ACTION_ID:
            self._move_camera_vertical(agent_id, -self._tilt_angle)
            return False
        if action_id == TURN_LEFT_ACTION_ID:
            self._turn_agent(agent_id, self._turn_angle)
            return False
        if action_id == TURN_RIGHT_ACTION_ID:
            self._turn_agent(agent_id, -self._turn_angle)
            return False
        if action_id == MOVE_FORWARD_ACTION_ID:
            return self._move_forward(agent_id)
        return False

    def _move_camera_vertical(self, agent_id: int, amount: float) -> None:
        sensor_names = list(self._sim.agents[agent_id]._sensors.keys())  # type: ignore[attr-defined]
        for sensor_name in sensor_names:
            sensor = self._sim.agents[agent_id]._sensors[sensor_name].node  # type: ignore[attr-defined]
            sensor.rotation = sensor.rotation * quaternion_to_magnum_x(amount)

    def _turn_agent(self, agent_id: int, amount_degrees: float) -> None:
        state = self._sim.get_agent_state(agent_id)
        yaw_delta = np.deg2rad(amount_degrees)
        delta = quaternion.from_rotation_vector([0.0, yaw_delta, 0.0])
        new_rotation = delta * state.rotation
        rotation = [
            float(new_rotation.imag[0]),
            float(new_rotation.imag[1]),
            float(new_rotation.imag[2]),
            float(new_rotation.real),
        ]
        self._sim.set_agent_state(np.array(state.position).tolist(), rotation, agent_id=agent_id, reset_sensors=False)

    def _move_forward(self, agent_id: int) -> bool:
        state = self._sim.get_agent_state(agent_id)
        forward = quaternion_rotate_vector(state.rotation, self._sim.forward_vector)
        target_position = state.position + forward * self._forward_step
        if self._allow_sliding:
            step_fn = self._sim.pathfinder.try_step
        else:
            step_fn = self._sim.pathfinder.try_step_no_sliding
        final_position = step_fn(state.position, target_position)
        collided = np.linalg.norm(final_position - state.position) + 1e-5 < np.linalg.norm(target_position - state.position)
        rotation = [
            float(state.rotation.imag[0]),
            float(state.rotation.imag[1]),
            float(state.rotation.imag[2]),
            float(state.rotation.real),
        ]
        self._sim.set_agent_state(np.array(final_position).tolist(), rotation, agent_id=agent_id, reset_sensors=False)
        return bool(collided)

    def _copy_agent_pose(self, agent_src: int, agent_dst: int) -> None:
        src_state = self._sim.get_agent_state(agent_src)
        rotation = [
            float(src_state.rotation.imag[0]),
            float(src_state.rotation.imag[1]),
            float(src_state.rotation.imag[2]),
            float(src_state.rotation.real),
        ]
        self._sim.set_agent_state(np.array(src_state.position).tolist(), rotation, agent_id=agent_dst, reset_sensors=False)


def quaternion_to_magnum_x(amount_degrees: float):
    import magnum as mn

    return mn.Quaternion.rotation(mn.Deg(amount_degrees), mn.Vector3.x_axis())


@registry.register_measure
class JointSPL(Measure):
    def __init__(self, sim, config, *args: Any, **kwargs: Any) -> None:
        self._sim = sim
        self._config = config
        self._previous_positions: Optional[List[np.ndarray]] = None
        self._start_end_episode_distance: Optional[float] = None
        self._agent_episode_distance: float = 0.0
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "spl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any) -> None:
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid, Success.cls_uuid]
        )
        self._previous_positions = [
            self._sim.get_agent_state(agent_id).position.copy()
            for agent_id in range(len(self._sim.habitat_config.agents_order))
        ]
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(self, episode, task, *args: Any, **kwargs: Any) -> None:
        assert self._previous_positions is not None
        ep_success = float(task.measurements.measures[Success.cls_uuid].get_metric())
        total_delta = 0.0
        for agent_id in range(len(self._sim.habitat_config.agents_order)):
            current_position = self._sim.get_agent_state(agent_id).position
            total_delta += float(
                np.linalg.norm(current_position - self._previous_positions[agent_id], ord=2)
            )
            self._previous_positions[agent_id] = current_position.copy()
        self._agent_episode_distance += total_delta
        shortest_path = float(self._start_end_episode_distance or 0.0)
        if not np.isfinite(shortest_path) or shortest_path <= 0.0:
            self._metric = ep_success
            return
        path_length = max(shortest_path, float(self._agent_episode_distance))
        if not np.isfinite(path_length) or path_length <= 0.0:
            self._metric = ep_success
            return
        self._metric = ep_success * (shortest_path / path_length)


@registry.register_measure
class JointDistanceToGoal(Measure):
    cls_uuid: str = "distance_to_goal"

    def __init__(self, sim, config, *args: Any, **kwargs: Any) -> None:
        self._sim = sim
        self._config = config
        self._distance_to = self._config.distance_to
        self._episode_view_points: Optional[List[Any]] = None
        self._previous_positions: Optional[List[np.ndarray]] = None
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any) -> None:
        self._previous_positions = None
        if self._distance_to == "VIEW_POINTS":
            self._episode_view_points = [
                view_point.agent_state.position
                for goal in episode.goals
                for view_point in goal.view_points
            ]
        self.update_metric(episode=episode, *args, **kwargs)

    def update_metric(self, episode, *args: Any, **kwargs: Any) -> None:
        current_positions = [
            self._sim.get_agent_state(agent_id).position
            for agent_id in range(len(self._sim.habitat_config.agents_order))
        ]
        if self._previous_positions is None or any(
            not np.allclose(prev, curr, atol=1e-4)
            for prev, curr in zip(self._previous_positions, current_positions)
        ):
            distances: List[float] = []
            for current_position in current_positions:
                if self._distance_to == "POINT":
                    distance_to_target = self._sim.geodesic_distance(
                        current_position,
                        [goal.position for goal in episode.goals],
                        episode,
                    )
                else:
                    distance_to_target = self._sim.geodesic_distance(
                        current_position,
                        self._episode_view_points,
                        episode,
                    )
                if distance_to_target is not None:
                    distances.append(float(distance_to_target))
            self._previous_positions = [position.copy() for position in current_positions]
            self._metric = min(distances) if distances else float("inf")


@registry.register_measure
class JointSuccess(Measure):
    cls_uuid: str = "success"

    def __init__(self, sim, config, *args: Any, **kwargs: Any) -> None:
        self._sim = sim
        self._config = config
        self._success_distance = self._config.success_distance
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any) -> None:
        task.measurements.check_measure_dependencies(
            self.uuid, [JointDistanceToGoal.cls_uuid]
        )
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(self, episode, task: NavigationTask, *args: Any, **kwargs: Any) -> None:
        distance_to_target = task.measurements.measures[
            JointDistanceToGoal.cls_uuid
        ].get_metric()
        if (
            hasattr(task, "is_stop_called")
            and task.is_stop_called  # type: ignore[attr-defined]
            and distance_to_target < self._success_distance
        ):
            self._metric = 1.0
        else:
            self._metric = 0.0


@dataclass
class JointNavActionConfig(ActionConfig):
    type: str = "JointNavAction"
    tilt_angle: float = 30.0


@dataclass
class AgentHeadingSensorCfg(HeadingSensorConfig):
    type: str = "AgentHeadingSensor"
    uuid: str = "agent_heading"
    agent_id: int = 0


@dataclass
class AgentCompassSensorCfg(CompassSensorConfig):
    type: str = "AgentCompassSensor"
    uuid: str = "agent_compass"
    agent_id: int = 0


@dataclass
class AgentGPSSensorCfg(GPSSensorConfig):
    type: str = "AgentGPSSensor"
    uuid: str = "agent_gps"
    agent_id: int = 0


@dataclass
class AgentObjectGoalSensorCfg(ObjectGoalSensorConfig):
    type: str = "AgentObjectGoalSensor"
    uuid: str = "agent_objectgoal"
    agent_id: int = 0


@dataclass
class JointSPLMeasurementConfig(MeasurementConfig):
    type: str = "JointSPL"


@dataclass
class JointDistanceToGoalMeasurementConfig(MeasurementConfig):
    type: str = "JointDistanceToGoal"
    distance_to: str = "VIEW_POINTS"


@dataclass
class JointSuccessMeasurementConfig(MeasurementConfig):
    type: str = "JointSuccess"
    success_distance: float = 0.1


cs = ConfigStore.instance()
cs.store(
    package="habitat.task.actions.joint_nav_action",
    group="habitat/task/actions",
    name="joint_nav_action",
    node=JointNavActionConfig,
)
cs.store(
    package="habitat.task.lab_sensors.agent_0_heading_sensor",
    group="habitat/task/lab_sensors",
    name="agent_0_heading_sensor",
    node=AgentHeadingSensorCfg(uuid="agent_0_heading", agent_id=0),
)
cs.store(
    package="habitat.task.lab_sensors.agent_0_compass_sensor",
    group="habitat/task/lab_sensors",
    name="agent_0_compass_sensor",
    node=AgentCompassSensorCfg(uuid="agent_0_compass", agent_id=0),
)
cs.store(
    package="habitat.task.lab_sensors.agent_0_gps_sensor",
    group="habitat/task/lab_sensors",
    name="agent_0_gps_sensor",
    node=AgentGPSSensorCfg(uuid="agent_0_gps", agent_id=0),
)
cs.store(
    package="habitat.task.lab_sensors.agent_0_objectgoal_sensor",
    group="habitat/task/lab_sensors",
    name="agent_0_objectgoal_sensor",
    node=AgentObjectGoalSensorCfg(uuid="agent_0_objectgoal", agent_id=0),
)
cs.store(
    package="habitat.task.lab_sensors.agent_1_heading_sensor",
    group="habitat/task/lab_sensors",
    name="agent_1_heading_sensor",
    node=AgentHeadingSensorCfg(uuid="agent_1_heading", agent_id=1),
)
cs.store(
    package="habitat.task.lab_sensors.agent_1_compass_sensor",
    group="habitat/task/lab_sensors",
    name="agent_1_compass_sensor",
    node=AgentCompassSensorCfg(uuid="agent_1_compass", agent_id=1),
)
cs.store(
    package="habitat.task.lab_sensors.agent_1_gps_sensor",
    group="habitat/task/lab_sensors",
    name="agent_1_gps_sensor",
    node=AgentGPSSensorCfg(uuid="agent_1_gps", agent_id=1),
)
cs.store(
    package="habitat.task.lab_sensors.agent_1_objectgoal_sensor",
    group="habitat/task/lab_sensors",
    name="agent_1_objectgoal_sensor",
    node=AgentObjectGoalSensorCfg(uuid="agent_1_objectgoal", agent_id=1),
)
cs.store(
    package="habitat.task.measurements.spl",
    group="habitat/task/measurements",
    name="joint_spl",
    node=JointSPLMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.distance_to_goal",
    group="habitat/task/measurements",
    name="joint_distance_to_goal",
    node=JointDistanceToGoalMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.success",
    group="habitat/task/measurements",
    name="joint_success",
    node=JointSuccessMeasurementConfig,
)
