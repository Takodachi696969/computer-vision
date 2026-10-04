"""Explicit episode evaluation with held-out seeds and comparison baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from humaned_lab.policies.runner import IKReachPolicy, RandomPolicy
from humaned_lab.policies.sb3 import PPOPolicy
from .env import ReachCubeEnv
from .provenance import environment_provenance


def _evaluate_controller(env: ReachCubeEnv, controller, episodes: int, seed: int) -> dict[str, Any]:
    records = []
    for episode in range(episodes):
        episode_seed = seed + episode
        # Same seeded reset for PPO, random, IK means the same cube locations.
        observation, info = env.reset(seed=episode_seed)
        reset = getattr(controller, "reset", None)
        if callable(reset):
            reset()
        if isinstance(controller, RandomPolicy):
            controller.rng = np.random.default_rng(episode_seed)
        reward_sum = 0.0
        min_distance = float(info["distance_m"])
        for step in range(env.max_episode_steps):
            observation, reward, terminated, truncated, info = env.step(controller.act(observation))
            reward_sum += reward
            min_distance = min(min_distance, info["distance_m"])
            if terminated or truncated:
                break
        record = {
            "seed": episode_seed,
            "steps": step + 1,
            "return": float(reward_sum),
            "success": bool(info["is_success"]),
            "final_distance_m": float(info["distance_m"]),
            "min_distance_m": min_distance,
            "target_m": info["target_m"],
        }
        if isinstance(controller, IKReachPolicy):
            record["ik_failures"] = controller.ik_failures
        records.append(record)
    returns = np.asarray([record["return"] for record in records])
    return {
        "episodes": records,
        "success_rate": float(np.mean([record["success"] for record in records])),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "mean_final_distance_m": float(np.mean([record["final_distance_m"] for record in records])),
    }


def evaluate_policy(
    model_path: str | Path,
    *,
    episodes: int = 10,
    seed: int = 100,
    scene_path: str | Path | None = None,
    output_path: str | Path | None = None,
    max_episode_steps: int = 200,
    include_baselines: bool = True,
) -> dict[str, Any]:
    """Evaluate a trusted lab PPO checkpoint using deterministic actions.

    A model directory resolves to model.zip and its saved scene.json. A zip path
    uses adjacent scene.json when available. Training is never run here.
    """
    if episodes < 1:
        raise ValueError("episodes must be positive.")
    model_path = Path(model_path).resolve()
    run_directory = model_path if model_path.is_dir() else model_path.parent
    checkpoint = model_path / "model.zip" if model_path.is_dir() else model_path
    if scene_path is None and (run_directory / "scene.json").exists():
        scene_path = run_directory / "scene.json"
    env = ReachCubeEnv(scene_path, max_episode_steps=max_episode_steps)
    try:
        result = {
            "task": "reach_cube_above_v1",
            "claim": "TCP reaches above cube; no grasping or physical deployment validation",
            "model": str(checkpoint),
            "deterministic_model_actions": True,
            "seed_start": seed,
            "episode_count": episodes,
            "success_radius_m": env.success_radius_m,
            "success_hold_steps": 3,
            "target_offset_from_cube_centre_m": env.target_offset_m,
            "control_interval_s": env.control_interval_s,
            "max_joint_delta_rad": env.action_delta_rad,
            "environment_provenance": environment_provenance(env.world),
            "model_result": _evaluate_controller(env, PPOPolicy(str(checkpoint)), episodes, seed),
        }
        if include_baselines:
            result["random_baseline"] = _evaluate_controller(
                env, RandomPolicy(seed), episodes, seed
            )
            result["scripted_ik_baseline"] = _evaluate_controller(
                env, IKReachPolicy(env.world), episodes, seed
            )
        if output_path:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        env.close()
