"""Adapter for this lab's PPO checkpoints (not arbitrary Hub robot models)."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .base import validate_action


class PPOPolicy:
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        from stable_baselines3 import PPO
        from gymnasium import spaces

        self.model = PPO.load(model_path, device=device)
        expected = {
            "q_rad": (6,),
            "qd_rad_s": (6,),
            "tcp_m": (3,),
            "target_m": (3,),
            "delta_m": (3,),
            "cube_velocity_m_s": (3,),
        }
        observations = self.model.observation_space
        if not isinstance(observations, spaces.Dict) or set(observations.spaces) != set(expected):
            raise ValueError("This PPO model does not use the lab's reaching observation contract.")
        if any(observations[key].shape != shape for key, shape in expected.items()):
            raise ValueError("PPO observation shapes do not match the six-joint lab robot.")
        actions = self.model.action_space
        if not isinstance(actions, spaces.Box) or actions.shape != (6,):
            raise ValueError("The lab PPO adapter requires a continuous six-action model.")
        if not np.allclose(actions.low, -1) or not np.allclose(actions.high, 1):
            raise ValueError("The lab PPO adapter requires actions normalized to [-1, 1].")

    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        action, _ = self.model.predict(dict(observation), deterministic=True)
        return validate_action(action)
