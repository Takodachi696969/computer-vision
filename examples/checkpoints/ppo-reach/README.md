---
library_name: stable-baselines3
tags:
  - robotics
  - reinforcement-learning
  - mujoco
  - parol6
---

# PAROL6 cube reaching starter

A ready-to-run PPO checkpoint trained locally on this repository's MuJoCo
PAROL6 model. The task is reaching above a falling cube using simulator ground
truth. It is a useful starting point for policy experiments, with one training
seed and a small randomized target region.

Start a server using the checkpoint's saved scene in one terminal:

```powershell
humaned-lab serve --scene examples/checkpoints/ppo-reach/scene.json
```

Run the policy in another terminal:

```powershell
humaned-lab policy humaned_lab.policies.sb3:PPOPolicy --model examples/checkpoints/ppo-reach/model.zip --steps 200
```

Repeat the independent evaluation:

```powershell
humaned-lab evaluate examples/checkpoints/ppo-reach --episodes 100 --seed 20000 --output outputs/recheck.json
```

## Training and held-out results

Training used PPO `MultiInputPolicy`, seed 0, 100,096 collected controls, CPU,
and the versions/configuration recorded in `metadata.json`. Evaluation used
deterministic PPO actions in 100 complete episodes, seeds 20,000–20,099, with
the same seeded initial scenes for all controllers.

| Controller | Success | Mean final TCP distance |
| --- | --- | --- |
| PPO | 96/100 | 0.02379 m |
| Random actions | 0/100 | 0.22600 m |
| Scripted position IK | 100/100 | 0.01776 m |

Full episode records are in `evaluation-held-out.json`. `evaluation.json`
contains the separate ten episodes run automatically at the end of training.
The final 100-episode evaluation was repeated after the simulator's cube-contact
priority and rolling-friction correction, using PyTorch `2.14.1+cpu` from the
portable dependency lock. The checkpoint was trained before that correction;
the final evaluation above is against the corrected physics. Its success rate
remained 96/100.

The held-out report records canonical scene, effective physics XML, robot URDF
and task source SHA-256 fingerprints, plus evaluation dependency versions.
`metadata.json` retains the original training dependency versions and adds the
final evaluation provenance. The source fingerprints identify the evaluated
snapshot; they do not claim that the pre-correction training source was saved.

Success requires three consecutive 50 ms controls within 0.035 m of the target,
which is 0.06 m vertically above the cube centre. Target position resets with
uniform XY jitter of up to 0.025 m, and follows the cube as it moves.

The action is a six-vector in `[-1,1]`. Each value requests up to 0.04 radians
of joint position increment relative to the measured joints, clipped to joint
limits. See [the policy guide](../../policies/README.md) for observation fields.

## Scope

The robot has a bare flange and coarse contact geometry. There is no grasping
mechanism in this checkpoint. Cube contact can shift the target; the success
definition measures the updated cube position. This evaluation does not test
camera-based observations, wide workspaces, obstacle avoidance, a real robot,
or robustness across multiple independently trained seeds.

Load this trusted repository checkpoint with the lab PPO adapter. Other robot
models need an explicit observation/action adapter and their own validation.
