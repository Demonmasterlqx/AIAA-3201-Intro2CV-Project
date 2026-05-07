from __future__ import annotations

from typing import Any
from typing import List

from habitat.config.default import get_agent_config
from habitat.config.read_write import read_write
from habitat.core.registry import registry
from habitat.core.simulator import Observations
from habitat.core.simulator import SensorSuite
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSimSensor
from habitat.sims.habitat_simulator.habitat_simulator import overwrite_config
from omegaconf import DictConfig
from omegaconf import open_dict

import gym.spaces as spaces
import habitat_sim


@registry.register_simulator(name="GlobalAscentMultiAgentSim-v0")
class GlobalAscentMultiAgentSim(HabitatSim):
    def __init__(self, config: DictConfig) -> None:
        if len(config.agents) > 1:
            with read_write(config):
                for agent_name, agent_cfg in config.agents.items():
                    sensor_keys = list(agent_cfg.sim_sensors.keys())
                    for sensor_key in sensor_keys:
                        sensor_cfg = agent_cfg.sim_sensors.pop(sensor_key)
                        base_uuid = getattr(sensor_cfg, "uuid", sensor_key.replace("_sensor", ""))
                        with open_dict(sensor_cfg):
                            sensor_cfg.uuid = f"{agent_name}_{base_uuid}"
                        agent_cfg.sim_sensors[f"{agent_name}_{sensor_key}"] = sensor_cfg
        super().__init__(config)

    def create_sim_config(
        self, _sensor_suite: SensorSuite
    ) -> habitat_sim.Configuration:
        sim_config = habitat_sim.SimulatorConfiguration()
        overwrite_config(
            config_from=self.habitat_config.habitat_sim_v0,
            config_to=sim_config,
            ignore_keys={"gpu_gpu"},
        )
        sim_config.scene_dataset_config_file = self.habitat_config.scene_dataset
        sim_config.scene_id = self.habitat_config.scene

        agent_configs: List[habitat_sim.AgentConfiguration] = []
        for agent_id, agent_name in enumerate(self.habitat_config.agents_order):
            lab_agent_config = get_agent_config(self.habitat_config, agent_id)
            agent_config = habitat_sim.AgentConfiguration()
            overwrite_config(
                config_from=lab_agent_config,
                config_to=agent_config,
                ignore_keys={
                    "is_set_start_state",
                    "sensors",
                    "sim_sensors",
                    "start_position",
                    "start_rotation",
                    "articulated_agent_urdf",
                    "articulated_agent_type",
                    "joint_start_noise",
                    "joint_that_can_control",
                    "motion_data_path",
                    "ik_arm_urdf",
                    "grasp_managers",
                    "max_climb",
                    "max_slope",
                    "joint_start_override",
                },
            )
            sensor_specifications = []
            for sensor in _sensor_suite.sensors.values():
                if not sensor.uuid.startswith(f"{agent_name}_"):
                    continue
                assert isinstance(sensor, HabitatSimSensor)
                sim_sensor_cfg = sensor._get_default_spec()  # type: ignore[operator]
                overwrite_config(
                    config_from=sensor.config,
                    config_to=sim_sensor_cfg,
                    ignore_keys=sensor._config_ignore_keys,
                    trans_dict={
                        "sensor_model_type": lambda v: getattr(
                            habitat_sim.FisheyeSensorModelType, v
                        ),
                        "sensor_subtype": lambda v: getattr(
                            habitat_sim.SensorSubType, v
                        ),
                    },
                )
                sim_sensor_cfg.uuid = sensor.uuid
                sim_sensor_cfg.resolution = list(sensor.observation_space.shape[:2])
                sim_sensor_cfg.sensor_type = sensor.sim_sensor_type
                sim_sensor_cfg.gpu2gpu_transfer = self.habitat_config.habitat_sim_v0.gpu_gpu
                sensor_specifications.append(sim_sensor_cfg)
            agent_config.sensor_specifications = sensor_specifications
            agent_config.action_space = {
                0: habitat_sim.ActionSpec("stop"),
                1: habitat_sim.ActionSpec(
                    "move_forward",
                    habitat_sim.ActuationSpec(
                        amount=self.habitat_config.forward_step_size
                    ),
                ),
                2: habitat_sim.ActionSpec(
                    "turn_left",
                    habitat_sim.ActuationSpec(
                        amount=self.habitat_config.turn_angle
                    ),
                ),
                3: habitat_sim.ActionSpec(
                    "turn_right",
                    habitat_sim.ActuationSpec(
                        amount=self.habitat_config.turn_angle
                    ),
                ),
            }
            agent_configs.append(agent_config)

        output = habitat_sim.Configuration(sim_config, agent_configs)
        output.enable_batch_renderer = self.habitat_config.renderer.enable_batch_renderer
        return output

    @property
    def action_space(self):
        if len(self.sim_config.agents) == 0:
            return spaces.Discrete(4)
        return spaces.Discrete(len(self.sim_config.agents[self.habitat_config.default_agent_id].action_space))

    def reset(self) -> Observations:
        sim_obs = habitat_sim.Simulator.reset(self)
        if len(self.habitat_config.agents_order) > 1:
            sim_obs = self._collect_all_sensor_observations()
        elif self._update_agents_state():
            sim_obs = self.get_sensor_observations()
        self._prev_sim_obs = sim_obs
        if self.config.enable_batch_renderer:
            self.add_keyframe_to_observations(sim_obs)
            return sim_obs
        return self._sensor_suite.get_observations(sim_obs)

    def step(self, action=None) -> Observations:
        if action is None:
            sim_obs = self._collect_all_sensor_observations() if len(self.habitat_config.agents_order) > 1 else self.get_sensor_observations()
        else:
            sim_obs = habitat_sim.Simulator.step(self, action)
            if len(self.habitat_config.agents_order) > 1:
                sim_obs = self._collect_all_sensor_observations()
        self._prev_sim_obs = sim_obs
        if self.config.enable_batch_renderer:
            self.add_keyframe_to_observations(sim_obs)
            return sim_obs
        return self._sensor_suite.get_observations(sim_obs)

    def get_observations_at(
        self,
        position=None,
        rotation=None,
        keep_agent_at_new_pose: bool = False,
    ):
        if len(self.habitat_config.agents_order) <= 1:
            return super().get_observations_at(position, rotation, keep_agent_at_new_pose)
        current_state = self.get_agent_state()
        if position is None or rotation is None:
            success = True
        else:
            success = self.set_agent_state(position, rotation, reset_sensors=False)
        if not success:
            return None
        sim_obs = self._collect_all_sensor_observations()
        self._prev_sim_obs = sim_obs
        observations = self._sensor_suite.get_observations(sim_obs)
        if not keep_agent_at_new_pose:
            self.set_agent_state(current_state.position, current_state.rotation, reset_sensors=False)
        return observations

    def _collect_all_sensor_observations(self):
        agent_obs = self.get_sensor_observations(agent_ids=list(range(len(self.habitat_config.agents_order))))
        if not isinstance(agent_obs, dict):
            return agent_obs
        if all(isinstance(key, int) for key in agent_obs.keys()):
            merged = {}
            collided = False
            for obs in agent_obs.values():
                if "collided" in obs:
                    collided = collided or bool(obs["collided"])
                merged.update(obs)
            merged["collided"] = collided
            return merged
        return agent_obs
