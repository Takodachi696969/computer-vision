"""Run any Policy against a local headless physics world for bounded episodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .base import Policy, load_policy


def run_policy(
    policy: Policy | str,
    *,
    scene_path: str | Path | None = None,
    steps: int = 200,
    seed: int = 0,
    policy_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run simulation locally; the live HTTP transport lives in the root CLI."""
    from humaned_lab.training.env import ReachCubeEnv

    if steps < 1:
        raise ValueError("steps must be positive.")
    controller = load_policy(policy, policy_kwargs) if isinstance(policy, str) else policy
    env = ReachCubeEnv(scene_path, max_episode_steps=steps)
    try:
        observation, info = env.reset(seed=seed)
        total_reward = 0.0
        for index in range(steps):
            action = controller.act(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return {
            "steps": index + 1,
            "reward": float(total_reward),
            "success": info["is_success"],
            "final_distance_m": info["distance_m"],
            "final_state": env.world.state(),
        }
    finally:
        env.close()


class RandomPolicy:
    """A seeded action baseline, not a learned controller."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, observation):
        return self.rng.uniform(-1, 1, size=6).astype(np.float32)


class IKReachPolicy:
    """Position-only IK baseline using this world's robot model.

    This demonstrates reachability and motor response, not collision-aware path
    planning. A failure to solve is recorded and returns a hold action.
    """

    def __init__(self, world, action_delta_rad: float = 0.04) -> None:
        self.world = world
        self.action_delta_rad = action_delta_rad
        self._last_target = None
        self._goal_q = None
        self.ik_failures = 0

    def reset(self) -> None:
        self._last_target = self._goal_q = None
        self.ik_failures = 0

    def act(self, observation):
        target = np.asarray(observation["target_m"], dtype=float)
        q = np.asarray(observation["q_rad"], dtype=float)
        if self._last_target is None or np.linalg.norm(target - self._last_target) > 0.002:
            try:
                self._goal_q = np.asarray(
                    self.world.inverse_kinematics(target, q_start=q), dtype=float
                )
            except ValueError:
                self.ik_failures += 1
                self._goal_q = None
            self._last_target = target.copy()
        if self._goal_q is None:
            return np.zeros(6, dtype=np.float32)
        return np.clip((self._goal_q - q) / self.action_delta_rad, -1, 1).astype(np.float32)
