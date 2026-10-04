"""Physics-backed reaching task with a bounded normalized joint action space."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from humaned_lab.policies.base import (
    ACTION_DELTA_RAD,
    CONTROL_INTERVAL_S,
    TARGET_OFFSET_M,
    observation_from_state,
    validate_action,
)
from humaned_lab.simulation.world import PhysicsWorld


class ReachCubeEnv(gym.Env):
    """Reach 6 cm above a cube centre; no grasping/contact success is claimed.

    Observation fields use metres, radians, seconds and float32 arrays. Reward is
    -distance in metres plus progress shaping, a small action penalty, and a
    success bonus. A task ends after three consecutive controls within 3.5 cm of
    the target, or truncates after 200 controls (10 simulated seconds).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scene_path: str | Path | None = None,
        *,
        cube_name: str | None = None,
        max_episode_steps: int = 200,
        success_radius_m: float = 0.035,
        target_offset_m: float = TARGET_OFFSET_M,
        target_jitter_m: float = 0.025,
    ) -> None:
        super().__init__()
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive.")
        if success_radius_m <= 0 or target_jitter_m < 0:
            raise ValueError("The success radius must be positive and jitter nonnegative.")
        self.world = PhysicsWorld(scene_path=scene_path)
        self.scene_path = Path(scene_path).resolve() if scene_path else None
        self.cube_name = cube_name
        self.max_episode_steps = int(max_episode_steps)
        self.success_radius_m = float(success_radius_m)
        self.target_offset_m = float(target_offset_m)
        self.target_jitter_m = float(target_jitter_m)
        self.action_delta_rad = ACTION_DELTA_RAD
        self.physics_steps = max(1, round(CONTROL_INTERVAL_S / self.world.timestep))
        self.control_interval_s = self.physics_steps * self.world.timestep
        if not np.isclose(self.control_interval_s, CONTROL_INTERVAL_S, atol=1e-8):
            raise ValueError("Scene timestep must divide the 0.05 s policy control interval.")
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        # Unbounded velocity avoids lying about physical impulses in user scenes.
        self.observation_space = spaces.Dict(
            {
                "q_rad": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                "qd_rad_s": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                "tcp_m": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "target_m": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "delta_m": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "cube_velocity_m_s": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            }
        )
        self._steps = 0
        self._hold_steps = 0
        self._distance = 0.0
        self._finished = False
        self.last_joint_targets = np.zeros(6)

    def _observation(self) -> dict[str, np.ndarray]:
        observation = observation_from_state(
            self.world.state(), self.cube_name, self.target_offset_m
        )
        self.world.set_goal(observation["target_m"])
        return observation

    def _info(self, observation: dict[str, np.ndarray]) -> dict[str, Any]:
        return {
            "is_success": bool(self._hold_steps >= 3),
            "distance_m": float(np.linalg.norm(observation["delta_m"])),
            "elapsed_s": self._steps * self.control_interval_s,
            "target_m": observation["target_m"].tolist(),
            "joint_targets_rad": self.last_joint_targets.tolist(),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.world.reset(seed=episode_seed)
        state = self.world.state()
        # Optional physics helper enables held-out cube locations, never fake TCP.
        if self.target_jitter_m and callable(getattr(self.world, "set_cube_pose", None)):
            cubes = state["cubes"]
            cube = next((item for item in cubes if item["name"] == self.cube_name), None)
            if cube is None and cubes and self.cube_name is None:
                cube = next((item for item in cubes if item["name"] == "target"), None)
                if cube is None:
                    cube = sorted(cubes, key=lambda item: item["name"])[0]
            if cube is not None:
                position = np.asarray(cube["position_m"], dtype=float).copy()
                position[:2] += self.np_random.uniform(
                    -self.target_jitter_m, self.target_jitter_m, 2
                )
                self.world.set_cube_pose(
                    cube["name"],
                    position,
                    cube.get("quaternion_wxyz", [1, 0, 0, 0]),
                    velocity_m_s=cube.get("velocity_m_s", [0, 0, 0]),
                    angular_velocity_rad_s=cube.get("angular_velocity_rad_s", [0, 0, 0]),
                )
        self._steps = self._hold_steps = 0
        self._finished = False
        observation = self._observation()
        self.last_joint_targets = observation["q_rad"].astype(float)
        self._distance = float(np.linalg.norm(observation["delta_m"]))
        return observation, self._info(observation)

    def step(self, action):
        if self._finished:
            raise RuntimeError("The episode has ended; call reset() before another action.")
        bounded = validate_action(action)
        current_q = np.asarray(self.world.state()["q_rad"], dtype=float)
        limits = self.world.joint_limits
        self.last_joint_targets = np.clip(
            current_q + self.action_delta_rad * bounded, limits[:, 0], limits[:, 1]
        )
        self.world.set_joint_targets(self.last_joint_targets)
        self.world.step(self.physics_steps)
        self._steps += 1
        observation = self._observation()
        distance = float(np.linalg.norm(observation["delta_m"]))
        self._hold_steps = self._hold_steps + 1 if distance < self.success_radius_m else 0
        terminated = bool(self._hold_steps >= 3)
        truncated = bool(self._steps >= self.max_episode_steps and not terminated)
        reward = (
            -distance + 5.0 * (self._distance - distance) - 0.005 * float(np.square(bounded).mean())
        )
        if terminated:
            reward += 5.0
        self._distance = distance
        self._finished = terminated or truncated
        return observation, float(reward), terminated, truncated, self._info(observation)

    def close(self) -> None:
        close = getattr(self.world, "close", None)
        if callable(close):
            close()
