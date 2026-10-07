# Set up the robot lab on another machine

The public fork contains the simulator, robot model assets, browser controls, scenes, policy examples and a small trained reaching checkpoint. Clone **`feat/parol6-simulation-lab`**: the default branch does not contain this lab.

This guide installs the same code into a new local environment. The [physics walkthrough](physics-controls.md) explains joint controls, tip/waypoints, arrow jogging, cube materials and motor loads. The [full tutorial](tutorial.md) covers custom policies and training.

## Windows: install once

Use a Windows desktop with an OpenGL-capable graphics driver. Simulation does not require WSL, a robot, a camera, a C++ compiler or CUDA. Git and `uv` must be on your PATH; installation links are in the [README](../README.md).

Open PowerShell in the parent folder where you want the project:

```powershell
git clone --branch feat/parol6-simulation-lab --single-branch https://github.com/Takodachi696969/computer-vision.git
Set-Location computer-vision
git rev-parse HEAD
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
.\.venv\Scripts\humaned-lab.exe doctor
```

Bootstrap obtains Python 3.12 and installs from `uv.lock` with `--frozen`. Its default includes simulation, CPU PPO training, Hub tooling and developer tools. Record the source commit printed above alongside experiments. The validated `uv` version is recorded in `.uv-version`; Windows bootstrap warns if yours differs.

For a smaller simulation-only installation, append `-CoreOnly` to the bootstrap command. Add training later with:

```powershell
uv sync --frozen --python 3.12 --extra train --extra hub --group dev
```

Create the virtual environment on each machine. Copying another computer's `.venv` is not an installation method. The PAROL6 model is bundled; separate PAROL6/librealsense checkouts and `setup-upstream.ps1` are unnecessary for simulation.

## Windows: start and stop each session

From the cloned repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-lab.ps1
```

Open [the dashboard](http://127.0.0.1:8765/). The helper starts a background process and records its PID/logs in ignored `outputs/`. An already-running healthy lab is reused. To stop it:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\stop-lab.ps1
```

The explicit PowerShell invocation permits the helper in that process without changing saved execution-policy settings. If organizational Group Policy still blocks it, use the executable directly in a terminal:

```powershell
.\.venv\Scripts\humaned-lab.exe serve --host 127.0.0.1 --port 8765
```

Keep that terminal open; stop with `Ctrl+C`. If another application occupies 8765, choose an unused port, for example `--port 8766` or helper `-Port 8766`. Use that port in your browser and clients too. The included `examples/control/tip_waypoints.py` uses the default 8765 client.

## Check that setup really worked

From another terminal in the repository root:

```powershell
Invoke-RestMethod 'http://127.0.0.1:8765/health'
$labState = Invoke-RestMethod 'http://127.0.0.1:8765/api/state'
$labState.physics_error
$labState.render_error
$labState.viewer
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
# If you kept the default training extra, verify checkpoint loading/inference:
.\.venv\Scripts\humaned-lab.exe evaluate examples/checkpoints/ppo-reach --episodes 1 --seed 20000 --output outputs/checkpoint-smoke.json
```

Expect `doctor` to report `physics_finite: true`, health `ok: true`, and state errors to be null. Health alone does not certify rendering: confirm the browser displays changing frames and the renderer reports a positive FPS. The rendering tests require working graphics; on Windows without a graphics backend run `.\.venv\Scripts\python.exe -m pytest -q -m "not render"` and record that rendering remains unverified. A single evaluation episode verifies inference, not policy quality. See [verification evidence](verification.md) for the tested environments.

For a short first session:

1. In **Viewer settings**, try 60 FPS at 960×640 and inspect actual FPS and simulation speed. Reduce FPS/resolution if the new computer struggles; the default is 30 FPS.
2. Move one joint a few degrees in **Joint targets**, then click **Move to targets**.
3. In **Tip / waypoints**, try a small offset from the current measured tip position. This is position-only IK followed by joint interpolation; orientation and collision-free planning are not included.
4. In **Arrow keys**, enable keyboard control, select a joint with Left/Right or 1–6, and hold Up/Down. Release to stop.
5. Pause, edit a cube's dimensions/mass/COM/friction/bounce, apply, deliberately place it, then resume. Use the [load comparison](physics-controls.md#a-controlled-stall-experiment) to see contacts, deflection and torque saturation.

To verify a programmed sequence while watching the dashboard:

```powershell
.\.venv\Scripts\python.exe examples/control/tip_waypoints.py
```

The API specification is at [localhost `/docs`](http://127.0.0.1:8765/docs). Read `examples/policies/my_policy.py` and the [policy contract](../examples/policies/README.md) before replacing its zero-action starter. Policy actions are six normalized joint increments, up to 0.04 rad per 0.05 simulated seconds. They are not motor torques. The CLI policy runner pauses continuous playback; resume in the dashboard afterwards.

## Linux and macOS

With Git and `uv` installed:

```bash
git clone --branch feat/parol6-simulation-lab --single-branch https://github.com/Takodachi696969/computer-vision.git
cd computer-vision
git rev-parse HEAD
bash scripts/bootstrap.sh
.venv/bin/humaned-lab doctor
.venv/bin/humaned-lab serve --host 127.0.0.1 --port 8765
```

Use `.venv/bin/python -m pytest -q` and `.venv/bin/ruff check .` for verification. The start/stop PowerShell helpers are Windows-specific; keep this server in a terminal or use your environment's process manager. Linux full-suite/EGL rendering has passed CI. macOS has not been tested; do not treat this command path as evidence of macOS compatibility.

On headless Linux, install suitable EGL/Mesa libraries through that system's package manager, then launch with:

```bash
MUJOCO_GL=egl .venv/bin/humaned-lab serve --host 127.0.0.1 --port 8765
```

Apply the backend to verification processes too, in a second shell from the repository root:

```bash
MUJOCO_GL=egl .venv/bin/python -m pytest -q
```

If rendering cannot be made available, `serve --no-render` supports physical integration and API control, but does not provide the graphical view. Report that limitation rather than declaring graphical setup complete. Display the HTTP URL through your environment's local browser/port preview when available. A remote environment's `127.0.0.1` belongs to that environment, not automatically to your own PC.

## Bring your scenes and policies

- Clone or pull the source branch on the other computer. Preserve local edits before switching branches or updating.
- Use **Save scene JSON** for material/actuator settings and initial conditions; it is not a live-state replay. Move the downloaded file into `configs/scenes/` or your own experiment folder.
- Copy your custom Python policy files and any checkpoint directory you want to reuse, including `model.zip`, `scene.json` and `metadata.json`. Generated `outputs/` directories are ignored by Git and are not included by cloning this public fork.
- Stop the old server and launch with the saved scene: Windows helper `-Scene .\configs\scenes\my_scene.json`, or CLI `serve --scene configs/scenes/my_scene.json`.
- Reevaluate checkpoints after changing dynamics. The included checkpoint demonstrates reaching above a cube with a bare flange; the lab has no gripper or fracture model. Robot motor limits and collision shapes are approximate.

## Copy this prompt into an AI coding agent

Give the agent a local folder in which it can run commands, then paste:

```text
Set up and run the HumanED PAROL6 MuJoCo simulation lab on this machine.

Repository: https://github.com/Takodachi696969/computer-vision.git
Branch: feat/parol6-simulation-lab (the lab is on this branch).

Detect the OS and shell. Clone into a suitable local folder, or reuse the
correct checkout while preserving my edits. Read AGENTS.md, README.md,
docs/new-machine.md, docs/physics-controls.md and docs/tutorial.md. Record
the checked-out commit. Install Git/uv if needed, use Python 3.12 and the
frozen uv.lock, and create a fresh local .venv. Use the repository bootstrap
with its default training/Hub extras; keep CPU PyTorch unless I ask for GPU.
On Windows invoke .ps1 helpers with powershell.exe -NoProfile
-ExecutionPolicy Bypass -File; do not change permanent execution policy.

Run doctor, pytest and Ruff, plus a one-episode evaluation of the bundled
checkpoint to verify inference. Start the service on 127.0.0.1:8765, or choose
an unused local port and update clients accordingly. Check health/state,
finite physics, changing rendered frames and actual FPS. Configure 60 FPS
at 960x640 if it runs smoothly. Open the dashboard when possible and leave
the server running. If the environment is remote/headless, explain how I
reach its browser preview and any graphics limitation; --no-render alone
does not complete my request for a visible simulation.

Verify a small joint move and the tip-waypoint example, then reset. Give me
exact start/stop commands, a short walkthrough of joint targets, tip/IK
waypoints and arrow jogging, cube mass/size/COM/friction/bounce, and where
to edit scenes and my act(observation) policy. Explain how to run a custom
policy, train/evaluate the bundled PPO task, and extend the task/learner;
the train command does not optimize arbitrary act() classes. Do not start
a long training run. Use simulation only: no robot
or camera setup, physical motor commands or public network binding. Do not
rewrite the simulator or loosen dependency pins to hide installation errors.
Report what actually passed and resolve setup issues where possible.
```

This setup prompt needs no GitHub login: the fork is public. Authentication is only needed for later writes to GitHub or optional private Hub operations.
