from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat_baselines.rl.ppo.agent_access_mgr import AgentAccessMgr


@baseline_registry.register_agent_access_mgr(name="GAscentCentralAccessMgr")
class GAscentCentralAccessMgr(AgentAccessMgr):
    def __init__(
        self,
        config,
        env_spec: EnvironmentSpec,
        is_distrib: bool,
        device,
        resume_state: Optional[Dict[str, Any]],
        num_envs: int,
        percent_done_fn: Callable[[], float],
        lr_schedule_fn: Optional[Callable[[float], float]] = None,
    ) -> None:
        del is_distrib
        del resume_state
        del num_envs
        del percent_done_fn
        del lr_schedule_fn
        self._config = config
        self._env_spec = env_spec
        self._device = device
        self._ppo_cfg = config.habitat_baselines.rl.ppo
        policy_cls = baseline_registry.get_policy(
            self._config.habitat_baselines.rl.policy.main_agent.name
        )
        if policy_cls is None:
            raise ValueError(
                f"Unknown central policy {self._config.habitat_baselines.rl.policy.main_agent.name}"
            )
        self._actor_critic = policy_cls.from_config(
            self._config,
            self._env_spec.observation_space,
            self._env_spec.action_space,
            orig_action_space=self._env_spec.orig_action_space,
        )
        self._rollouts = None
        self._updater = None

    @property
    def nbuffers(self) -> int:
        return 1

    @property
    def masks_shape(self) -> Tuple:
        return (1,)

    def post_init(self, create_rollouts_fn: Optional[Callable] = None) -> None:
        del create_rollouts_fn
        return None

    @property
    def rollouts(self):
        return self._rollouts

    @property
    def actor_critic(self):
        return self._actor_critic

    @property
    def updater(self):
        return self._updater

    def get_resume_state(self) -> Dict[str, Any]:
        return {}

    def get_save_state(self) -> Dict[str, Any]:
        return {}

    def eval(self) -> None:
        if hasattr(self._actor_critic, "eval"):
            self._actor_critic.eval()

    def train(self) -> None:
        if hasattr(self._actor_critic, "train"):
            self._actor_critic.train()

    def load_ckpt_state_dict(self, ckpt: Dict) -> None:
        if getattr(self._actor_critic, "should_load_agent_state", True):
            self.load_state_dict(ckpt)

    def load_state_dict(self, state: Dict) -> None:
        if hasattr(self._actor_critic, "load_state_dict"):
            actor_state = state.get("state_dict", state)
            try:
                self._actor_critic.load_state_dict(actor_state)
            except Exception:
                return None

    def after_update(self) -> None:
        return None

    def pre_rollout(self) -> None:
        return None
