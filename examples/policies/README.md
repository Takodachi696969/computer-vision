# Write a policy

The policy entry point is intentionally small: a Python class with
`act(observation)` returns six normalized joint actions. Start with
[`my_policy.py`](my_policy.py). You can use NumPy, a neural network, a state
machine, or your own controller inside that method.

Run the physics server in a first terminal, then run your policy in a second:

```powershell
humaned-lab serve --port 8765
humaned-lab policy examples.policies.my_policy:MyPolicy --steps 200
```

Run commands from the repository root so Python can import `examples.policies`.
The starter returns zero actions, so the robot holds near its measured joint
positions. Replace that line with your controller.

## Observation and action contract

Each observation value is a NumPy float32 array. All coordinates use the robot
world frame and SI units. The default target is the cube named `target`; if
there is none, the first name in alphabetical order is used.

| Key | Shape | Meaning |
| --- | --- | --- |
| `q_rad` | `(6,)` | Measured native URDF joint angles, radians |
| `qd_rad_s` | `(6,)` | Joint angular velocities, radians/second |
| `tcp_m` | `(3,)` | Bare-flange tool centre position, metres |
| `target_m` | `(3,)` | Selected cube centre plus 0.06 m vertically |
| `delta_m` | `(3,)` | `target_m - tcp_m` |
| `cube_velocity_m_s` | `(3,)` | Target cube linear velocity |

An action `[1, 0, 0, 0, 0, 0]` requests a joint target 0.04 radians above the
current measured first joint angle. Each control advances 0.05 seconds of
physics. Values saturate to `[-1, 1]`, then resulting targets are clipped to
the original joint limits. A malformed shape or NaN/Inf causes an error.
These actions are joint position increments, not degrees, torque or Cartesian
velocity. The same convention is used in PPO training and the HTTP runner.

The simulator models contact and actuator response. A policy can move the cube
by contact; the target follows it. Bare-flange reaching does not demonstrate a
grasp, and the IK baseline does not perform collision-free path planning.

## Run a trained PPO model

```powershell
humaned-lab train --steps 100000 --output outputs/ppo-run-1 --eval-episodes 20 --seed 0
humaned-lab evaluate outputs/ppo-run-1 --episodes 20 --seed 20000 --output outputs/ppo-run-1/test.json
humaned-lab policy humaned_lab.policies.sb3:PPOPolicy --model outputs/ppo-run-1/model.zip --steps 200
```

Training records its exact scene, algorithm, dependencies, action contract,
reward, seed and actual collected steps beside the checkpoint. New runs also
save SHA-256 fingerprints for the canonical scene, effective physics XML,
robot URDF and task source, so you can identify the exact environment snapshot.
PPO collects
complete rollouts, so actual steps can exceed the requested budget. Use a new
output directory for each run. The live server must use the same scene as the
model when you want a comparable rollout; start it with
`--scene outputs/ppo-run-1/scene.json`.

`evaluation.json` includes deterministic PPO episodes, a seeded random-action
baseline and a scripted position-IK baseline. All three start from the same
seeded scenes. Success means three consecutive controls within 0.035 metres
of the moving target. Episodes otherwise end after 200 controls (10 simulated
seconds). Inspect success rate and final distance, not only training reward.
Repeat training with several seeds before comparing algorithm changes.

## Extend the task

For local Python experiments you can use the environment directly:

```python
from humaned_lab.training.env import ReachCubeEnv

env = ReachCubeEnv("scenes/my_scene.json", cube_name="target")
observation, info = env.reset(seed=7)
try:
    for _ in range(200):
        action = my_policy.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
finally:
    env.close()
```

Customize the JSON scene for cube masses, poses, initial velocities, gravity,
obstacles and initial joint angles. Extend `training/env.py` to change the task,
observations, reward or terminal condition. Changing an observation/action
contract requires retraining existing models. Run both Gymnasium and SB3
environment checkers before starting a long experiment. The checkers validate
the API; they do not establish physical fidelity or that the reward teaches
the behaviour you intended.

The current observation uses simulator ground truth. A RealSense image is not
automatically a cube pose: implement camera calibration and pose estimation,
transform the result into the robot frame, then add a policy adapter. Vision
policies should include comparable sensor noise/latency during training.

## Hugging Face

Hub sharing is optional and never happens during training. Install the `hub`
extra, sign in with `hf auth login`, then explicitly call:

```python
from humaned_lab.training.hub import upload_run, download_run

# Publishes only when you execute this line; repositories default to private.
upload_run("YOUR_ACCOUNT/parol6-reach", "outputs/ppo-run-1", private=True)
download_run("YOUR_ACCOUNT/parol6-reach", "outputs/downloaded", revision="COMMIT_SHA")
```

The helper transfers `model.zip`, `scene.json`, `metadata.json`,
`evaluation.json` and an optional model-card README. Use a pinned revision and
review the producer and metadata before loading a downloaded model. SB3 uses
Python serialization, so load model files only from sources you trust.

General Hugging Face vision models can supply perception features, but a
pretrained action model must match this robot's joint order, units, timing,
observation/action spaces and tool configuration. LeRobot checkpoints use
their own datasets, policy adapters and training pipeline; they are not PPO
checkpoints. Add an explicit adapter and robot demonstration dataset before
trying those policies here. No arbitrary pretrained model is assumed to
control PAROL6 out of the box.

## References

- [Stable Baselines3 custom environment interface](https://stable-baselines3.readthedocs.io/en/v2.9.0/guide/custom_env.html)
- [Gymnasium custom environment guide](https://gymnasium.farama.org/main/introduction/create_custom_env/)
- [Hugging Face Hub file uploads](https://huggingface.co/docs/huggingface_hub/en/guides/upload)
- [K-Dense Stable Baselines3 skill](https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/stable-baselines3/SKILL.md), consulted for the reproducible training workflow.
- Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
  [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065).
