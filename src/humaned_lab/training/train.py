"""Small CPU PPO starter with reproducible configuration and honest evaluation."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
from pathlib import Path
from typing import Any

from .env import ReachCubeEnv
from .evaluate import evaluate_policy
from .provenance import environment_provenance


def train_policy(
    output_dir: str | Path,
    *,
    total_timesteps: int = 10_000,
    scene_path: str | Path | None = None,
    seed: int = 0,
    evaluate_episodes: int = 10,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train PPO and save model.zip, scene.json, metadata.json, evaluation.json.

    ``total_timesteps`` is a lower bound rounded up by PPO rollout collection.
    The default CPU MLP is suitable for setup verification. A short run proves
    the pipeline, not learned competence; inspect held-out metrics before use.
    """
    if total_timesteps < 1 or evaluate_episodes < 1:
        raise ValueError("total_timesteps and evaluate_episodes must be positive.")
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor
    import torch

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "model.zip").exists():
        raise FileExistsError(
            f"A model already exists in {output_dir}; choose a new output directory."
        )
    raw_env = ReachCubeEnv(scene_path)
    provenance = environment_provenance(raw_env.world)
    env = Monitor(
        raw_env, filename=str(output_dir / "train.monitor.csv"), info_keywords=("is_success",)
    )
    previous_threads = torch.get_num_threads()
    # Small MLP minibatches usually slow down when many CPU threads are spawned.
    torch.set_num_threads(1)
    try:
        check_env(raw_env, warn=True)
        env.reset(seed=seed)
        rollout_steps = 256
        model = PPO(
            "MultiInputPolicy",
            env,
            seed=seed,
            device=device,
            verbose=1,
            n_steps=rollout_steps,
            batch_size=64,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        )
        callback = CheckpointCallback(
            save_freq=max(rollout_steps, total_timesteps // 5),
            save_path=str(output_dir / "checkpoints"),
            name_prefix="ppo",
        )
        model.learn(total_timesteps=total_timesteps, callback=callback)
        model.save(str(output_dir / "model"))
        # Save JSON scene source, including custom object properties, for replay.
        scene_source = getattr(raw_env.world, "scene_path", None)
        if scene_path is not None:
            scene_source = Path(scene_path)
        save_scene = getattr(raw_env.world, "save_scene", None)
        if callable(save_scene):
            save_scene(output_dir / "scene.json")
        elif scene_source is not None:
            shutil.copyfile(scene_source, output_dir / "scene.json")
        else:
            raise RuntimeError(
                "PhysicsWorld must expose scene_path or save_scene for reproducible training."
            )
        versions = {}
        for package in ("mujoco", "gymnasium", "stable-baselines3", "torch", "numpy"):
            versions[package] = importlib.metadata.version(package)
        metadata = {
            "task": "reach_cube_above_v1",
            "algorithm": "PPO",
            "policy": "MultiInputPolicy",
            "python": platform.python_version(),
            "package_versions": versions,
            "environment_provenance": provenance,
            "seed": seed,
            "requested_timesteps": total_timesteps,
            "actual_timesteps": model.num_timesteps,
            "device": device,
            "observation_keys": sorted(raw_env.observation_space.spaces),
            "observation_units": "metres, radians, seconds; float32 arrays",
            "action": "six values in [-1,1]; measured_q + action * 0.04 rad, clipped to joint limits",
            "control_interval_s": raw_env.control_interval_s,
            "max_episode_steps": raw_env.max_episode_steps,
            "success_radius_m": raw_env.success_radius_m,
            "success_hold_steps": 3,
            "target_offset_from_cube_centre_m": raw_env.target_offset_m,
            "target_xy_jitter_m": raw_env.target_jitter_m,
            "reward": "-distance_m + 5*(previous_distance_m-distance_m) - 0.005*mean(action^2) + 5 on success",
            "wrappers": ["Monitor"],
            "observation_normalization": None,
            "evaluation_seed_start": seed + 10_000,
            "notes": "Simulation reaching only; short training is setup validation, not grasping competence.",
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        env.close()
        torch.set_num_threads(previous_threads)
    evaluation = evaluate_policy(
        output_dir,
        episodes=evaluate_episodes,
        seed=seed + 10_000,
        output_path=output_dir / "evaluation.json",
    )
    return {"output_dir": str(output_dir), "metadata": metadata, "evaluation": evaluation}
