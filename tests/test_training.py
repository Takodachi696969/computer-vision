"""Contract checks for policies and the real MuJoCo-backed Gym environment."""

import copy
import json
import subprocess
import sys

import numpy as np
import pytest

from humaned_lab.policies.base import load_policy, validate_action


def test_user_policy_factory_and_invalid_actions():
    policy = load_policy("examples.policies.my_policy:MyPolicy")
    np.testing.assert_array_equal(policy.act({}), np.zeros(6))
    np.testing.assert_array_equal(
        validate_action([10, -10, 0, 0.5, -0.5, 1]), [1, -1, 0, 0.5, -0.5, 1]
    )
    with pytest.raises(ValueError, match="shape"):
        validate_action([1, 2])
    with pytest.raises(ValueError, match="finite"):
        validate_action([0, 0, np.nan, 0, 0, 0])
    with pytest.raises(ValueError, match="policy import"):
        load_policy("no_colon")


def test_local_policy_loads_when_python_excludes_cwd():
    # -I matches the relevant console-launcher behaviour: cwd is absent from
    # sys.path. The installed package remains available through site-packages.
    command = (
        "from humaned_lab.policies.base import load_policy; "
        "policy=load_policy('examples.policies.my_policy:MyPolicy'); "
        "assert policy.act({}).shape == (6,)"
    )
    subprocess.run([sys.executable, "-I", "-c", command], check=True)


@pytest.fixture
def env():
    pytest.importorskip("gymnasium")
    pytest.importorskip("mujoco")
    from humaned_lab.training.env import ReachCubeEnv

    instance = ReachCubeEnv(max_episode_steps=10)
    yield instance
    instance.close()


def test_gymnasium_and_sb3_checkers(env):
    from gymnasium.utils.env_checker import check_env

    check_env(env, skip_render_check=True)
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env as sb3_check_env

    sb3_check_env(env, warn=True)


def test_reset_and_transition_are_seeded(env):
    action = np.asarray([0.2, -0.4, 0.1, 0.0, 0.1, -0.2], dtype=np.float32)
    observation1, info1 = env.reset(seed=42)
    transition1 = env.step(action)
    observation2, info2 = env.reset(seed=42)
    transition2 = env.step(action)
    for name in observation1:
        np.testing.assert_array_equal(observation1[name], observation2[name])
        np.testing.assert_array_equal(transition1[0][name], transition2[0][name])
    assert info1 == info2
    assert transition1[1:] == transition2[1:]
    np.testing.assert_array_equal(env.world.state()["goal_m"], transition2[0]["target_m"])
    observation3, _ = env.reset(seed=43)
    assert not np.array_equal(observation1["target_m"], observation3["target_m"])


def test_actions_have_bounded_radian_increment_and_joint_limits(env):
    observation, _ = env.reset(seed=10)
    initial_q = observation["q_rad"].astype(float)
    env.step(np.asarray([100, -100, 100, -100, 100, -100], dtype=np.float32))
    assert np.max(np.abs(env.last_joint_targets - initial_q)) <= 0.040001
    assert np.all(env.last_joint_targets >= env.world.joint_limits[:, 0])
    assert np.all(env.last_joint_targets <= env.world.joint_limits[:, 1])
    assert env.control_interval_s == pytest.approx(0.05)


def test_reset_jitter_preserves_moving_cube_velocity(env, tmp_path):
    from humaned_lab.training.env import ReachCubeEnv

    scene = copy.deepcopy(env.world.scene)
    cube = next(item for item in scene["cubes"] if item["name"] == "target")
    cube["velocity_m_s"] = [0.13, 0.02, 0.0]
    cube["angular_velocity_rad_s"] = [0.0, 0.1, 0.2]
    scene_path = tmp_path / "moving.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    moving = ReachCubeEnv(scene_path)
    try:
        observation, _ = moving.reset(seed=1)
        np.testing.assert_allclose(observation["cube_velocity_m_s"], cube["velocity_m_s"])
        state_cube = next(
            item for item in moving.world.state()["cubes"] if item["name"] == "target"
        )
        np.testing.assert_allclose(
            state_cube["angular_velocity_rad_s"], cube["angular_velocity_rad_s"]
        )
    finally:
        moving.close()


def test_time_limit_requires_reset(env):
    env.reset(seed=3)
    for _ in range(env.max_episode_steps):
        _, _, terminated, truncated, info = env.step(np.zeros(6))
        if terminated:
            break
    assert terminated or truncated
    assert info["elapsed_s"] <= env.max_episode_steps * 0.05
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(6))


def test_success_requires_three_consecutive_controls():
    pytest.importorskip("mujoco")
    from humaned_lab.training.env import ReachCubeEnv

    instance = ReachCubeEnv(success_radius_m=10, max_episode_steps=5)
    try:
        instance.reset(seed=1)
        for _ in range(2):
            _, _, terminated, truncated, info = instance.step(np.zeros(6))
            assert not terminated and not truncated and not info["is_success"]
        _, _, terminated, truncated, info = instance.step(np.zeros(6))
        assert terminated and not truncated and info["is_success"]
    finally:
        instance.close()


def test_train_save_reload_and_held_out_evaluation(tmp_path):
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("mujoco")
    from humaned_lab.training.train import train_policy
    from humaned_lab.training.evaluate import evaluate_policy

    output = tmp_path / "smoke"
    result = train_policy(output, total_timesteps=256, evaluate_episodes=1, seed=12)
    assert result["metadata"]["actual_timesteps"] >= 256
    assert result["metadata"]["evaluation_seed_start"] == 10012
    training_provenance = result["metadata"]["environment_provenance"]
    evaluation_provenance = result["evaluation"]["environment_provenance"]
    assert training_provenance["scene_sha256"] == evaluation_provenance["scene_sha256"]
    assert training_provenance["physics_xml_sha256"] == evaluation_provenance["physics_xml_sha256"]
    assert training_provenance["source_sha256"] == evaluation_provenance["source_sha256"]
    for name in (
        "model.zip",
        "scene.json",
        "metadata.json",
        "evaluation.json",
        "train.monitor.csv",
    ):
        assert (output / name).is_file()
    repeat = evaluate_policy(output, episodes=1, seed=10012)
    for key in ("model_result", "random_baseline", "scripted_ik_baseline"):
        assert result["evaluation"][key] == repeat[key]
    with pytest.raises(FileExistsError):
        train_policy(output, total_timesteps=1)
