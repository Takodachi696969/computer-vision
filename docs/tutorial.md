# PAROL6 physics lab: from first movement to your own policy

This guide is for the package in this branch. It provides a PAROL6-shaped MuJoCo physics world with gravity, moving rigid cubes, an HTTP port, manual joint controls and an editable policy interface. You can train a reaching policy locally. Physical robot commissioning is an optional, explicitly separate operation. The implementation does not include a gripper or claim pick-and-place ability.

Commands below use Windows PowerShell from the repository root. `humaned-lab --help` and each subcommand's `--help` are the final authority for your installed version. For a concise map, read [architecture.md](architecture.md); for concrete upstream issues and model limits, read [problems.md](problems.md).

## 1. Reproduce the installation

Clone this branch on the other computer rather than copying an existing virtual environment:

```powershell
git clone --branch feat/parol6-simulation-lab https://github.com/Takodachi696969/computer-vision.git
Set-Location computer-vision
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
.\.venv\Scripts\humaned-lab.exe doctor
```

The fork carries the lab branch based on HumanED's repository. A Python virtual environment contains paths to its original interpreter and is not a portable installation artifact. The bootstrap and lock files are the portable artifact. The core uses CPython 3.12; optional robot dependencies have compiled ABI requirements.

Windows can block direct `.ps1` launches with `PSSecurityException` / "running scripts is disabled." The explicit `powershell.exe -NoProfile -ExecutionPolicy Bypass -File ...` form permits these helpers in the launched process only, without changing saved user or machine settings. Use the same form for start/stop/setup helpers, with script arguments after the filename. Group Policy can take precedence; inspect `Get-ExecutionPolicy -List` if this form is still blocked. See [Microsoft's execution policy documentation](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-5.1).

Bootstrap requires Git and `uv`. If `uv` is missing, install it with `winget install --id astral-sh.uv -e`, reopen PowerShell and rerun bootstrap. The tested version is in `.uv-version`; the script warns if yours differs. `bootstrap.ps1` defaults to core + training + Hub + developer tools. Add `-Camera` for RealSense, or use `-CoreOnly` for core + developer tools without the training/Hub extras. The script obtains Python 3.12 through `uv`.

On Linux/macOS, with `uv` installed, use `bash scripts/bootstrap.sh` and `.venv/bin/humaned-lab doctor` / `.venv/bin/humaned-lab serve`. The optional script arguments are `--core-only` or `--camera`. Desktop validation was performed on Windows; the complete Linux suite and EGL rendering additionally passed GitHub Actions. macOS execution has not been tested. Rendering requires an appropriate display/OpenGL backend; on headless Linux with EGL/Mesa available, use `MUJOCO_GL=egl .venv/bin/humaned-lab serve`. Use `serve --no-render` for physics/API work without an image backend.

For an explicit installation with `uv` already available:

```powershell
uv sync --frozen --extra train --extra camera --extra hub --group dev
.\.venv\Scripts\humaned-lab.exe doctor
```

The MuJoCo lab does not require WSL or a source build of librealsense. RealSense's Python wheel is an optional device acquisition layer. PAROL6's native command/controller stack has additional compiled dependencies; install it through the optional upstream workflow described in section 8. It is not needed to train or control the independent physics world.

Keep commands tied to the environment. Use the explicit `.\.venv\Scripts\humaned-lab.exe …` path; activation is not required, and this executable also works when PowerShell scripts are blocked. A plain `uv run` can synchronize away optional extras that are not selected in that invocation; if using it, supply the same desired `--extra` flags.

`doctor` reports package versions, an actual physics step and optional PyTorch CUDA status. Use `camera list` for actual camera discovery. A CUDA device name or a successful package import alone does not prove that a training kernel executes on that GPU. This lab's small state-vector PPO defaults to CPU; measure GPU benefit separately.

## 2. Launch and manually control the robot

In the first PowerShell window:

```powershell
.\.venv\Scripts\humaned-lab.exe serve --host 127.0.0.1 --port 8765
```

Leave this process running. Open [http://127.0.0.1:8765](http://127.0.0.1:8765). The service renders MuJoCo's scene into a live image and reads its state for the dashboard. All dashboard controls affect the simulated arm and cubes.

1. Watch the initial scene for a few seconds. Cubes fall and contact the ground under gravity; an initial velocity can make one slide or rotate.
2. Read the initial six joint slider values and the tool marker's XYZ position. Sliders show target values in degrees; the API exposes measured positions in radians.
3. Change one joint slightly, then click **Move to targets**. This requests a position-servo target, so the measured joint angle moves towards it over physics steps. A joint target is not an instantaneous robot teleport.
4. Pause physics when you want to inspect a pose or issue deterministic policy steps. Resume to let gravity and servos run in wall-clock-paced simulation.
5. Reset to return joints, cubes and velocities to the configured initial conditions. Resetting the physics lab does not send a physical robot homing command.
6. Use a cube reset/launch operation, then watch position and contact changes. The cube's pose operation places it at a new state; its later movement is simulated physics.

The server runs a shared world. A browser, notebook and policy script connected to the same port operate on that same state. Avoid multiple independent controllers fighting over the arm. The policy-step endpoint pauses continuous integration and advances one control interval per request, making it useful for repeatable experiments.

Keep the terminal visible enough to notice errors. Stop with `Ctrl+C`; launch again to reopen the port. For machines without rendering support, `serve --no-render` still supports state/control; automatic rendering failures are also exposed through `/health`.

Alternatively, start a background server and stop it later with these helpers:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-lab.ps1
```

To stop it later:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\stop-lab.ps1
```

Stop before changing the scene, then append `-Scene .\configs\scenes\moving_cubes.json` to the start command. The start helper reports an already-running lab rather than starting a competing controller.

## 3. Control it from PowerShell and Python

### Read and reset state

In a second PowerShell window:

```powershell
$labUrl = 'http://127.0.0.1:8765'
Invoke-RestMethod "$labUrl/health"
$state = Invoke-RestMethod "$labUrl/api/state"
$state.q_rad
$state.tcp_m
$state.cubes

Invoke-RestMethod -Method Post "$labUrl/api/reset" `
  -ContentType 'application/json' -Body '{"seed":42}'
```

Useful state fields are:

| Field | Meaning |
|---|---|
| `time_s`, `timestep_s` | Simulated time and integration step in seconds |
| `joint_names` | Six native URDF joint names in action order |
| `q_rad`, `qd_rad_s` | Measured joint angles and velocities |
| `joint_targets_rad` | Servo targets; may differ from measured angles |
| `joint_limits_rad` | Six `[lower, upper]` bounds |
| `tcp_m` | Bare flange marker position in robot-base coordinates |
| `cubes` | Named object poses, linear/angular velocities |
| `contact_count` | Number of current contact constraints |
| `running` | Whether the server continuously integrates physics |
| `render_error`, `physics_error` | Diagnostic information |

### Set absolute joint targets

Read the current targets, add two degrees to J1, and send all six radians:

```powershell
$state = Invoke-RestMethod "$labUrl/api/state"
$jointTargets = @($state.joint_targets_rad)
$jointTargets[0] += 2 * [Math]::PI / 180
$jointBody = @{q_rad = $jointTargets} | ConvertTo-Json -Compress
Invoke-RestMethod -Method Post "$labUrl/api/joints" `
  -ContentType 'application/json' -Body $jointBody
```

Targets outside the native URDF limits are rejected. Do not send `[0,0,0,0,0,0]` as a general home position: PAROL6's native joint ranges/zero references do not put all six zeros in range. Use reset or read the initial scene's `initial_q_rad`.

### Pause and apply a normalized policy action

```powershell
Invoke-RestMethod -Method Post "$labUrl/api/running" `
  -ContentType 'application/json' -Body '{"running":false}'

Invoke-RestMethod -Method Post "$labUrl/api/step" `
  -ContentType 'application/json' -Body '{"action":[0.1,0,0,0,0,0]}'
```

An action is six numbers in `[-1,1]`. The server computes:

```text
target_q = clip(measured_q + 0.04 * action, lower_joint_limits, upper_joint_limits)
integrate physics for 0.05 seconds
return the resulting state
```

Therefore `0.1` means a requested J1 change of `0.004` rad, about `0.229°`. It is neither `0.1°` nor a torque. A zero action requests a hold near the measured pose for that interval; disturbances and servo dynamics can still move the arm. The API rejects malformed/out-of-range actions, while the local policy adapter saturates valid finite outputs.

### Launch a cube

Use a name from `$state.cubes`, then set position and velocity:

```powershell
Invoke-RestMethod -Method Post "$labUrl/api/cubes/target" `
  -ContentType 'application/json' `
  -Body '{"position_m":[0.22,0.18,0.4],"velocity_m_s":[0.15,0,0.1]}'

Invoke-RestMethod -Method Post "$labUrl/api/running" `
  -ContentType 'application/json' -Body '{"running":true}'
```

This API operation reinitializes that existing cube's pose, clears angular velocity, and uses the supplied linear velocity. Add or remove objects by editing the scene and restarting; the current endpoint does not dynamically create new MuJoCo bodies.

### Use the reusable Python client

Save this as `examples/my_first_control.py` and run it with `.\.venv\Scripts\python.exe examples\my_first_control.py` while the server is running:

```python
import numpy as np
from humaned_lab.control.client import SimulationClient

with SimulationClient() as lab:
    state = lab.reset(seed=42)
    lab.running(False)
    print("TCP in metres:", state["tcp_m"])
    for _ in range(20):
        # One second of simulated time in total.
        state = lab.step([0.1, 0, 0, 0, 0, 0])
    print("Joint degrees for display:", np.rad2deg(state["q_rad"]))
    print("Elapsed simulated seconds:", state["time_s"])
```

HTTP reference:

| Method and route | JSON request |
|---|---|
| `GET /health` | — |
| `GET /api/state` | — |
| `GET /api/frame.jpg` | —; returns rendered JPEG |
| `POST /api/reset` | `{"seed":0}` |
| `POST /api/running` | `{"running":true}` |
| `POST /api/joints` | `{"q_rad":[six radians]}` |
| `POST /api/step` | `{"action":[six normalized numbers]}` |
| `POST /api/cubes/{name}` | `{"position_m":[x,y,z],"velocity_m_s":[vx,vy,vz]}` |

FastAPI also supplies the interactive endpoint specification at [http://127.0.0.1:8765/docs](http://127.0.0.1:8765/docs).

## 4. Write physical scenes and simulations

Start by copying a scene from `configs/scenes/` into your own JSON file. Select it with `humaned-lab serve --scene configs/scenes/my_scene.json`. The engine validates the file and converts it into MuJoCo bodies, joints, contact shapes and actuators at startup.

Here is a complete small scene you can save as `configs/scenes/my_scene.json`:

```json
{
  "schema_version": 1,
  "name": "My falling and sliding cubes",
  "timestep_s": 0.002,
  "gravity_m_s2": [0, 0, -9.81],
  "initial_q_rad": [1.5707963267948966, -1.5707963267948966, 3.141592653589793, 0, 0, 3.141592653589793],
  "goal_m": [0.22, 0.18, 0.08],
  "cubes": [
    {
      "name": "target",
      "size_m": [0.04, 0.04, 0.04],
      "mass_kg": 0.08,
      "position_m": [0.22, 0.18, 0.30],
      "quaternion_wxyz": [1, 0, 0, 0],
      "velocity_m_s": [0.1, 0, 0],
      "angular_velocity_rad_s": [0, 0, 2],
      "friction": [0.8, 0.02, 0.002],
      "position_jitter_m": [0.01, 0.01, 0],
      "rgba": [0.95, 0.46, 0.18, 1]
    }
  ],
  "obstacles": [
    {
      "name": "wall",
      "size_m": [0.02, 0.2, 0.10],
      "position_m": [0.40, 0.18, 0.05],
      "rgba": [0.5, 0.6, 0.7, 1]
    }
  ]
}
```

`size_m` is the full XYZ extent, not a half-extent. A 4 cm cube resting on the floor has its centre near `z=0.02`. Quaternions are `w,x,y,z`; they are normalized by the loader. Angular velocity is rad/s. The three friction entries describe sliding, torsional and rolling contact coefficients in MuJoCo's model. Mass is kg; gravity is m/s². [MuJoCo reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom).

`position_jitter_m` defines a uniform ±range per axis applied on a seeded reset. Keep initial objects above the floor and away from overlapping robot links. Arbitrary large penetrations can produce large impulses and poor learning data. The current scene loader accepts up to 32 cubes and 32 fixed box obstacles, rejects unknown fields and duplicate/reserved names, and bounds basic physical parameters. It does not prove that your arrangement is solvable.

`goal_m` controls a visible reference marker. The reaching-policy target is separately derived from a selected cube; editing `goal_m` alone does not change the PPO task's target.

For objects floating freely, set gravity to `[0,0,0]` and supply velocities. This is a zero-gravity physical world. For cubes that continuously follow a prescribed path or receive forces, write a simulation loop or extend `PhysicsWorld`; an initial velocity alone decays under contact/friction and does not imply a trajectory controller.

### Use MuJoCo directly without the HTTP server

The same `PhysicsWorld` can run offline, much faster than wall time and without rendering:

```python
from pathlib import Path
from PIL import Image
from humaned_lab.simulation.world import PhysicsWorld

world = PhysicsWorld("configs/scenes/my_scene.json")
try:
    initial = world.reset(seed=42)
    world.set_cube("target", [0.22, 0.18, 0.4], [0.1, 0, 0])
    # 500 integration steps at 0.002 s = 1 simulated second.
    world.step(500)
    print(world.state()["cubes"])
    Path("outputs").mkdir(exist_ok=True)
    Image.fromarray(world.render(960, 640)).save("outputs/my_scene.png")
finally:
    world.close()
```

The world exposes `model` (`mujoco.MjModel`) and `data` (`mujoco.MjData`) for advanced extensions. For example, within a single-threaded offline loop, `data.xfrc_applied[body_id]` applies a world-frame six-vector of force and torque. Reset forces explicitly between interventions. If extending the live service, acquire `world.lock`; a render context must be created, used and closed on the same thread. [MuJoCo Python interface](https://mujoco.readthedocs.io/en/stable/python.html).

Here is a force-based cube experiment. It applies 0.05 N along X for the first half second, then lets the cube coast/contact naturally:

```python
import mujoco
from humaned_lab.simulation.world import PhysicsWorld

with PhysicsWorld("configs/scenes/my_scene.json") as world:
    body = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, "target")
    for _ in range(round(2.0 / world.timestep)):
        world.data.xfrc_applied[:] = 0
        if world.data.time < 0.5:
            world.data.xfrc_applied[body, :3] = [0.05, 0, 0]
        world.step()
    print(world.state()["cubes"])
```

For a free 0.08 kg cube that force corresponds to `F/m = 0.625 m/s²` before contact effects. The integrated trajectory depends on gravity and contact, so it need not be a straight line. Periodically resetting a cube's position is a prescribed animation/intervention rather than a force-driven trajectory; use forces, constraints or an explicitly modeled motor when the physical response matters.

To add a conveyor, articulated gripper, spring constraint, moving kinematic obstacle, or mesh contact, extend `simulation/model.py` with explicit MJCF bodies/actuators and extend scene validation in `simulation/world.py`. Keep the JSON source, generated model and experiment metadata together. The root package deliberately exposes this engine instead of hiding it behind an opaque viewer.

## 5. Implement your own policy

Edit `examples/policies/my_policy.py`. The only required method is `act(observation)`; return six finite normalized actions. The current observation contract is:

| Key | Shape | Units/meaning |
|---|---|---|
| `q_rad` | `(6,)` | Joint angles in native URDF order |
| `qd_rad_s` | `(6,)` | Joint velocity |
| `tcp_m` | `(3,)` | Bare flange marker XYZ |
| `target_m` | `(3,)` | Cube centre plus 6 cm in vertical Z |
| `delta_m` | `(3,)` | `target_m - tcp_m` |
| `cube_velocity_m_s` | `(3,)` | Selected cube's linear velocity |

Arrays are NumPy `float32`. They describe the current measured simulation state. The starter returns zeros so you can establish a valid interface before writing a controller:

```python
import numpy as np

class MyPolicy:
    def act(self, observation):
        # Replace with your own model/controller.
        return np.zeros(6, dtype=np.float32)
```

Run `humaned-lab policy examples.policies.my_policy:MyPolicy --steps 200` from the repository root with the HTTP server open. Module imports execute your Python code, so use your own modules or trusted local code. This CLI drives the dashboard world through the same six-action endpoint. For an offline physics episode, use `humaned_lab.policies.runner.run_policy`; the two runners share the observation/action contract.

For a minimal live controller without relying on CLI choices:

```python
from humaned_lab.control.client import SimulationClient
from humaned_lab.policies.base import load_policy, observation_from_state, validate_action

controller = load_policy("examples.policies.my_policy:MyPolicy")
with SimulationClient() as lab:
    state = lab.reset(seed=7)
    for step in range(200):
        observation = observation_from_state(state, cube_name="target")
        action = validate_action(controller.act(observation))
        state = lab.step(action.tolist())
```

This advances 10 simulated seconds at a 20 Hz action rate. To watch in real time, insert a wall-clock delay between steps; it does not change simulated action duration. Add your own stopping criterion such as distance to target or a maximum number of actions.

Cartesian error components cannot simply be assigned to the first three joint actions: a Cartesian metre and a joint radian are different quantities, and the Jacobian depends on the pose. For a geometric baseline, use the existing position-only IK helper:

```python
import numpy as np
from humaned_lab.simulation.world import PhysicsWorld

world = PhysicsWorld()
try:
    target_q = world.solve_ik([0.22, 0.18, 0.08])
    world.set_joint_targets(target_q)
    world.step(500)
    print(world.tcp_position())
finally:
    world.close()
```

This IK solves flange position within joint bounds. It does not constrain wrist orientation or guarantee a collision-free path. The scripted IK evaluation baseline is a reachability/control comparison, not a learned policy.

If you change the observation keys, action scale, tool marker, joint zero reference or control interval, record a new contract version and retrain checkpoints. A policy checkpoint's numerical input shape is insufficient evidence that its semantics match.

## 6. Train and evaluate PPO in the physics world

The shipped task is `ReachCubeEnv`: approach a point 6 cm above the selected cube's centre. An episode succeeds when the flange marker stays within 3.5 cm for three consecutive controls. It truncates after 200 steps, or 10 simulated seconds. The cube itself remains a physical body; its current position determines the moving target.

By default, the observation helper selects the cube named `target`; if it is absent, it chooses the first name alphabetically. A requested explicit `cube_name` must exist. Use the same selection rule in training, offline evaluation and live rollout.

The reward is defined in `training/env.py`:

```text
-distance_m
+5 * (previous_distance_m - distance_m)
-0.005 * mean(action²)
+5 when the hold-success condition is reached
```

Default reset randomization jitters the training cube's XY position by up to 2.5 cm, on top of scene jitter. Reward shaping and this limited reset distribution make a useful starter problem, not a manipulation benchmark. For training around obstacles, add collision costs and a path feasibility objective; the current reward does not encode them.

### Start with a short pipeline run

```powershell
.\.venv\Scripts\humaned-lab.exe train --help
.\.venv\Scripts\humaned-lab.exe train --steps 4096 --seed 0 --output outputs/ppo-smoke
```

A short run validates environment stepping, PyTorch optimization, saving and inference. It can have a poor success rate; that is a learning result, not an installation failure. PPO collects full rollouts, so the actual step count may exceed the requested minimum. [SB3 2.9.0 PPO documentation](https://stable-baselines3.readthedocs.io/en/v2.9.0/modules/ppo.html).

Each run saves the final `model.zip`, resolved `scene.json`, `metadata.json`, `evaluation.json`, Monitor CSV and intermediate checkpoints. Metadata records requested/actual timesteps, seed, package versions, task parameters, action meaning and evaluation seeds. Choose a new output directory for a new experiment; the training helper refuses to overwrite an existing final model.

### Train for longer and compare held-out performance

```powershell
.\.venv\Scripts\humaned-lab.exe train --steps 200000 --seed 1 --output outputs/ppo-seed1
.\.venv\Scripts\humaned-lab.exe evaluate outputs/ppo-seed1 --episodes 50 --seed 10000
```

Inspect success rate, mean/final distance, return and per-episode seeds. The evaluation helper runs deterministic model actions and evaluates random and scripted IK controllers on the same seeded resets. A higher training return alone does not establish generalization. Repeat training with several seeds, then evaluate scenes whose object locations, friction and velocities differ from the training set.

Use the Python API when you want complete control over the command arguments:

```python
from humaned_lab.training.train import train_policy
from humaned_lab.training.evaluate import evaluate_policy

result = train_policy(
    "outputs/my_experiment", total_timesteps=200_000,
    scene_path="configs/scenes/my_scene.json", seed=3,
    evaluate_episodes=20, device="cpu",
)
comparison = evaluate_policy(
    "outputs/my_experiment", episodes=50, seed=20_000,
    output_path="outputs/my_experiment/held_out.json",
)
print(comparison["model_result"]["success_rate"])
```

### Run a checkpoint in your controller loop

```python
from humaned_lab.policies.sb3 import PPOPolicy
from humaned_lab.policies.base import observation_from_state
from humaned_lab.control.client import SimulationClient

controller = PPOPolicy("outputs/my_experiment/model.zip")
with SimulationClient() as lab:
    state = lab.reset(seed=10001)
    for _ in range(200):
        observation = observation_from_state(state, cube_name="target")
        state = lab.step(controller.act(observation).tolist())
```

Launch the server with the experiment's saved scene if you want the same environment: `humaned-lab serve --scene outputs/my_experiment/scene.json`. Checkpoint loading expects a trusted checkpoint from this exact observation/action contract. It does not support arbitrary downloaded robot models.

The equivalent live CLI is `humaned-lab policy humaned_lab.policies.sb3:PPOPolicy --model outputs/my_experiment/model.zip --steps 200`. It resets the open server's world before rollout; select the saved scene when starting that server.

You can also try the already trained starter: run the server with `--scene examples/checkpoints/ppo-reach/scene.json`, then `humaned-lab policy humaned_lab.policies.sb3:PPOPolicy --model examples/checkpoints/ppo-reach/model.zip --steps 200`. Its local held-out reaching results are recorded in [verification.md](verification.md); its model card explains the narrow task/fidelity assumptions.

### Change the task deliberately

For a new task, create an environment next to `training/env.py` instead of silently altering an old checkpoint's semantics. Implement Gymnasium's `reset(seed, options) → (observation, info)` and `step(action) → (observation, reward, terminated, truncated, info)`, then run SB3's `check_env`. The shipped trainer uses `Monitor` for episode logging, seeded resets and separate evaluation. `check_env` checks API consistency; it does not establish task solvability. [SB3 2.9.0 custom environments](https://stable-baselines3.readthedocs.io/en/v2.9.0/guide/custom_env.html).

A grasp task additionally needs finger geometry/joints, contact parameters, a grasp/open action and a success definition such as sustained object lift followed by release into a goal region. Then add object/drop/collision failure conditions, randomize relevant dynamics, train, and evaluate. The present six-action contract controls only the arm joints.

## 7. RealSense: acquire observations and connect them to the robot frame

Install the optional camera extra with bootstrap or `uv sync --frozen --extra camera`. Device discovery works even when no camera is plugged in:

```powershell
.\.venv\Scripts\humaned-lab.exe camera list
.\.venv\Scripts\humaned-lab.exe camera capture --output outputs/realsense
```

For multiple cameras, select `--serial SERIAL_NUMBER`. An unplugged system reports that no camera is detected. A successful import verifies the SDK wheel; a successful hardware capture requires the actual device.

Capture produces `color.png`, `depth.npy` and `metadata.json`. Depth is raw `uint16`; convert to metres using the recorded depth scale. The depth frame is aligned to color and metadata records its aligned intrinsics and device timestamp. `camera_to_robot_transform` is deliberately unset.

```python
import json
import numpy as np

raw = np.load("outputs/realsense/depth.npy")
meta = json.load(open("outputs/realsense/metadata.json", encoding="utf-8"))
depth_metres = raw * meta["depth_scale_metres_per_unit"]
print("Centre depth, m:", depth_metres[raw.shape[0] // 2, raw.shape[1] // 2])
```

A zero depth sample is invalid/missing, not a surface at the camera origin. Use `rs.rs2_deproject_pixel_to_point(intrinsics, [u,v], depth_m)` when creating XYZ observations; it handles the SDK's camera projection conventions. Do not mix raw-depth intrinsics with aligned pixels. The SDK also offers point-cloud generation and filtering. [Official alignment example](https://github.com/realsenseai/librealsense/blob/e15c5d6bb1563e778d116f682aeefffbae2daedc/wrappers/python/examples/align-depth2color.py), [API architecture](https://github.com/realsenseai/librealsense/blob/e15c5d6bb1563e778d116f682aeefffbae2daedc/doc/api_arch.md).

For cube tracking/control, the pipeline is: detect object pixels → valid depth in metres → camera XYZ → calibrated camera-to-base transform → robot-base target → your controller. Estimate camera extrinsics from a calibration object or hand-eye procedure appropriate to a fixed or wrist-mounted camera. Save its convention, units and timestamp. Neither camera factory calibration nor having a URDF establishes that camera-to-base transform.

The current PPO uses privileged simulation state, not RealSense images. To train an image policy, render/record an image observation with a consistent camera pose, change the environment observation schema and retrain. For deployment, validate latency, missing-depth handling, timestamp alignment and calibration error explicitly.

## 8. PAROL6 native command stack and physical commissioning

### Understand the two ports

| Port | Protocol | Purpose |
|---|---|---|
| `127.0.0.1:8765` | HTTP | This lab's MuJoCo physics and policy interface |
| `127.0.0.1:5001` | MessagePack/UDP | Upstream PAROL6 controller commands |

The upstream mock stack checks command planning, queueing and simulated telemetry. It does not provide a cube-contact physics environment. The optional read-only bridge reads upstream joint angles, converts degrees to radians, and sends targets into the MuJoCo service. It never sends physics targets back to the upstream controller.

Install the upstream stack into the separate `.upstream-venv`:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-upstream.ps1
```

The script clones a pinned checkout into ignored `vendor/`, verifies its exact revision/clean state, installs PAROL6 and this lab into the optional environment, and runs the mock smoke check. `requirements/upstream-constraints.txt` records the dependency versions from the working native Windows stack; these optional constraints are separate from the core `uv.lock`. To reuse the existing clean pinned source checkout on this computer, use `-SourcePath ../PAROL6-python-API`.

The Linux route is `bash scripts/setup-upstream.sh`, or `bash scripts/setup-upstream.sh ../PAROL6-python-API` for a clean existing pinned checkout. It uses `.upstream-venv/bin/humaned-lab`; use `.upstream-venv/bin/python` for Python examples. This path is provided but was not executed on the Windows development machine.

On native Windows, TOPPRA can require Microsoft C++ Build Tools with the Desktop C++ workload. The previous one-machine compiler/wheel workaround in `D:/HumanEd/wheels` is not shipped as a portable dependency. The optional script reports a source-build failure explicitly; the locked core physics package remains usable. Alternatively, run the optional native stack on a compatible Linux/WSL setup with its compiler/dependencies. Follow upstream platform wheel restrictions, particularly Pinokin's glibc requirement.

After installation, run the upstream smoke command through its own environment:

```powershell
.\.upstream-venv\Scripts\humaned-lab.exe upstream-smoke
```

For explicit mock startup:

```powershell
$env:PAROL6_FAKE_SERIAL = '1'
$env:PAROL6_NOAUTOHOME = '1'
.\.upstream-venv\Scripts\parol6-server.exe --host 127.0.0.1 --port 5001 --log-level INFO
```

In a separate terminal using the environment that contains PAROL6:

```python
from parol6 import RobotClient

with RobotClient(host="127.0.0.1", port=5001) as arm:
    if not arm.wait_ready(timeout=30):
        raise TimeoutError("Controller startup timed out")
    arm.simulator(True)
    arm.home(wait=True, timeout=60)
    print("Ping:", arm.ping())
    print("Angles, degrees:", arm.angles())
    print("Pose, native mm/angle interface:", arm.pose())
    try:
        arm.move_j([1, 0, 0, 0, 0, 0], rel=True,
                   speed=0.05, accel=0.05, wait=True, timeout=30)
    finally:
        arm.stop()
```

At the audited revision the methods are `angles`, `pose`, `select_tool`, `select_profile` and `simulator(bool)`. Some upstream README snippets use older names; see the audit. `move_j` joint vectors are degrees. `move_l` uses native Cartesian units and conventions. For precise details use the pinned source's method docstrings, not an old web snippet. [Pinned client source](https://github.com/PCrnjak/PAROL6-python-API/blob/b741d505ae9d1e3f28bd4ff4d0227b58b4921ed4/parol6/client/async_client.py).

With both servers running, mirror the upstream mock motion into the physics view:

```powershell
.\.upstream-venv\Scripts\humaned-lab.exe bridge --host 127.0.0.1 --port 5001 `
  --url http://127.0.0.1:8765 --seconds 30
```

This is useful for checking that a native controller command corresponds to the expected geometry. MuJoCo servo tracking need not be identical to firmware tracking, and cube contacts do not feed back into the upstream motion planner.

### When the actual robot is present

The package has an explicit commissioning adapter, not automatic PPO-to-hardware deployment. Verify the correct serial device, physical workspace, emergency stop and installed tool with the team before enabling the real controller. First homing can sweep joints to switches.

Start upstream on the known robot COM port; `COM3` is an example to replace:

```powershell
Remove-Item Env:PAROL6_FAKE_SERIAL -ErrorAction SilentlyContinue
$env:PAROL6_NOAUTOHOME = '1'
.\.upstream-venv\Scripts\parol6-server.exe --host 127.0.0.1 --port 5001 --serial COM3 --log-level INFO
```

Then inspect telemetry using the opt-in commissioning example. For a bare flange:

```powershell
.\.upstream-venv\Scripts\python.exe examples/hardware/joint_move.py --allow-hardware `
  --tool NONE --tcp-offset-mm 0 0 0
```

It requires an actual hardware-connected controller with simulator disabled. It selects the named tool, waits for that queued operation, applies the explicit translation offset in mm and waits again. The offset is added to the tool's configured transform; it is not a replacement for tool geometry calibration. Use the correct `--tool` and `--tool-variant` for an installed gripper.

Only after the workspace is cleared, opt into homing with `--home-cleared-workspace`. For a small, intended commissioning movement, add `--joint 1 --delta-deg 1`. The wrapper converts public radians to native degrees, checks upstream joint limits, bounds each move to 5° from measured pose, and limits speed/acceleration fractions to 10%. A telemetry watchdog requests a software protective stop if readback becomes stale. Physical E-stop remains necessary because a software stop cannot traverse a disconnected link.

Your controller can import `Parol6Adapter` for these discrete small moves. A continuous trained policy needs additional measured dynamics, calibration, action-rate limits, collision handling and a validated hardware runner. The browser and shipped policy CLI do not implicitly activate that adapter.

## 9. Hugging Face models and LeRobot

Two different uses are possible. A checkpoint trained with this lab's SB3 contract can be stored on the Hub and downloaded elsewhere. A vision-language-action model such as SmolVLA is a different architecture and training/data contract; it needs an adapter and robot-specific data before use.

### Download a matching lab checkpoint

The optional Hub extra supplies the current `hf` CLI. Inspect before downloading and pin a repository revision:

```powershell
hf --help
hf models info YOUR_ORG/YOUR_PAROL6_MODEL
hf download YOUR_ORG/YOUR_PAROL6_MODEL --revision EXACT_COMMIT `
  --local-dir outputs/hub-model
```

Replace the placeholders with a model whose card explicitly says it implements `reach_cube_above_v1`, this six-key observation dictionary, native PAROL6 joints, 0.04 rad normalized actions and a 50 ms interval. Keep its scene, package versions and task metadata. Then load `model.zip` through `PPOPolicy` or the evaluation command. Only run trusted model artifacts and code. [Official Hub CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

No matching universal PAROL6 manipulation checkpoint is bundled or claimed. An SB3 model trained on another robot/environment can have a six-dimensional action vector and still be incompatible.

### Explore a VLA base model without claiming robot compatibility

For example, inspect the official [SmolVLA model card](https://huggingface.co/lerobot/smolvla_base) and [SmolVLA training guide](https://huggingface.co/docs/lerobot/main/en/smolvla). If you choose to experiment, isolate LeRobot dependencies from the core environment and pin the LeRobot revision. You can download model configuration first:

```powershell
hf download lerobot/smolvla_base --revision d9f33c94a60fb382c90dea2164c96845bd955e28 `
  --include config.json --local-dir outputs/smolvla-config
```

Downloading a configuration proves access to the model artifact, not local inference, GPU compatibility or PAROL6 control. The source/dataset/action/camera assumptions must be reconciled before fine-tuning or rollout.

That configuration download was executed during this setup at revision `d9f33c94a60fb382c90dea2164c96845bd955e28`. The [inspected config](https://huggingface.co/lerobot/smolvla_base/blob/d9f33c94a60fb382c90dea2164c96845bd955e28/config.json) declares `smolvla`, a six-element state, three RGB camera feature keys of shape `(3,256,256)`, a six-element action, 50-action chunks and a CUDA device default. Six actions do not establish six PAROL6 joint angles: a source embodiment can include different joint/tool/gripper meanings and normalization. No model weights, inference or VLA training were attempted; the downloaded config is an inspection artifact under ignored `outputs/`.

### A concrete LeRobot integration plan

1. Define a versioned record schema: six joint positions/velocities in explicit units, calibrated TCP/tool state, fixed camera image keys, timestamps, task text, and actions. Decide whether actions are absolute joint targets, deltas or velocities. Do not silently mix this lab's normalized deltas with a pretrained model's absolute actions.
2. Record successful demonstrations in simulation through a scripted controller or manual teleoperation. Pair the action at time `t` with observations at the same step, and retain episode boundaries, success labels and seeds. Add an articulated gripper and its action before collecting grasp demonstrations.
3. Export through the selected LeRobot release's dataset API, specifying feature names, image size, frame rate and normalization statistics. This repository does not yet include that exporter. The [LeRobot dataset documentation](https://huggingface.co/docs/lerobot/main/en/lerobot-dataset-v3) describes the data, video and metadata organization; use a release-matched API instead of hand-writing an approximate dataset.
4. Implement the LeRobot `Robot` interface against this lab's `SimulationClient`: connection state, observation/action feature declarations, `get_observation`, and `send_action`. Convert model output units into bounded joint targets or the normalized six-action contract. Keep a simulation adapter separate from the physical adapter. [Bring Your Own Hardware](https://huggingface.co/docs/lerobot/main/en/integrate_hardware).
5. Fine-tune an appropriate base policy on your demonstrations using the official release's training command. Supply the dataset's feature mapping and normalization; inspect how images/state/action chunks and language are expected. Use the installed `lerobot-train --help` and the model's specific documentation to generate the final command. The package's `humaned-lab train` command is SB3 PPO, not LeRobot fine-tuning.
6. Evaluate in held-out scenes with changed cube positions, lighting/camera views, masses and friction. Record success, collisions, distance, inference latency and action saturation. Compare with scripted IK/behavior cloning and fresh PPO baselines.
7. Only after simulation evaluation and physical calibration, implement a separately reviewed hardware rollout runner with measured action-rate limits and explicit arming. A generic base model's familiarity with robot images does not establish safe actions for this particular arm.

This plan identifies the missing engineering pieces explicitly. The currently working extension point is the local `act(observation)` interface and HTTP port, so you can implement policies now without first adopting LeRobot.

## 10. Troubleshooting and a practical learning sequence

| Symptom | Check |
|---|---|
| `PSSecurityException` / running scripts is disabled | Use `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-lab.ps1`; its policy lasts only for the launched process. The foreground `.\.venv\Scripts\humaned-lab.exe serve` requires no `.ps1`. |
| `humaned-lab` not found | Use `.\.venv\Scripts\humaned-lab.exe` or `uv run`; confirm bootstrap completed. |
| Port already in use | Stop the existing server with `Ctrl+C`, or choose another port and matching client URL. |
| Browser image missing but state works | Inspect `/health` and `render_error`; check OpenGL/display support. Hosted Windows runners can lack an OpenGL-capable WGL driver; use `serve --no-render` there. Linux rendering was verified with EGL. |
| HTTP 422 | Check six-vector length, finite values, radians and joint limits; unknown scene/request keys are rejected. |
| Slider targets do not move while paused | Resume physics, or call `/api/step`; targets need integration time. |
| Cube stays frozen | The server may be paused by a policy-step request. Resume or continue policy steps. |
| Training imports fail | Install the `train` extra in the same environment you are using. |
| PPO success is low | Compare scripted IK on the same seeds; check reachability, reward, reset distribution and training length. |
| Model observation mismatch | Match scene/task metadata and dictionary keys; retrain after a contract change. |
| RealSense detects no devices | Plug in a supported device, inspect USB/SDK viewer and avoid a competing pipeline. |
| PAROL6 startup timeout | Allow cold JIT startup time and inspect controller logs. Use matching client/controller versions. |
| Native controller runs but motion is refused | Check simulator/hardware connection, homed state, protective stop, joint limits and tool selection. |

A useful first session is: start the browser → move one joint → launch a cube → send the same action through Python → save a custom scene → run scripted IK → do a short PPO train/evaluate → implement your own policy → collect synchronized observations. For contact manipulation, extend the gripper/contact task before increasing training time. For physical control, use the separate commissioning workflow and validate measurements before transferring policies.

The [verification record](verification.md) distinguishes what was executed on this computer from instructions for optional hardware or future learning integrations. [Skill provenance](skills.md) explains the public skills used to guide the repository analysis and training workflow.
