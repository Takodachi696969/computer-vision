# What was actually verified

Recorded on **4 October 2026**, using this Windows computer and the repository's GitHub Actions runners. This record distinguishes physical simulation, native-controller mock operation and equipment-dependent work.

## Physics controls update — 5 October 2026

- Full local suite: **57 passed**, including owner-thread rendered model rebuilds, independent contact/gravity/friction layers, torque caps and load response, approximate restitution, live Cartesian motion, jog watchdog and policy compatibility. Ruff and JavaScript syntax checks passed. The two Gymnasium unbounded-observation warnings remain expected.
- Fresh MJPEG frames received over two six-second 960×640 tests: **30.15 FPS** at target 30 (182 unique frames), **59.92 FPS** at target 60 (361 unique frames). Renderer metrics were 30.05/59.96 FPS, physics speed approximately 1×, and recorded dropped wall time was zero. These are local measurements, not guarantees for other hardware or browser presentation.
- Browser controls exercised: viewer 60 FPS, arrow joint selection/key release, lower-torque preset application, cube dimensions/mass/local COM/restitution edits, friction-layer toggle, and Cartesian waypoint editor. The live three-waypoint Python example reached its final target with about 4.0 mm measured error. All movement uses simulated servos and contacts.
- `examples/scenes/load_comparison.py` uses independent 0.3 s worlds with caps `[12,12,8,1,1,1]` Nm and drops a box onto the wrist. Light 0.01 kg: peak arm normal load 0.629 N, peak tip deflection 3.954 mm, no saturation. Heavy 5 kg: 73.037 N, 50.553 mm peak deflection, two saturated joints; final deflection 12.511 mm after the cube moves/slips. These are model outcomes, not physical payload limits.
- The configured base model's bundled PPO was reevaluated on 20 episodes starting at seed 20000: **19/20** reached, random baseline 0/20. This small regression check does not establish quality for edited cube materials/actuators/scenes. The previous 100-episode evaluation remains historical evidence for its recorded physics model.
- Wheel build succeeded and contains the split HTML/CSS/JavaScript dashboard assets. No new runtime dependencies were needed. Source robot assets remain unchanged.

The service was left running on loopback port 8765 with baseline physical properties and a 60 FPS viewer setting. Reopening it on another session defaults to 30 FPS.

## Installed and executed locally

| Component | Observed result |
|---|---|
| Core environment | CPython 3.12.14, MuJoCo 3.14.0, NumPy 2.4.6, Gymnasium 1.3.0 |
| Learning environment | Stable-Baselines3 2.9.0, final portable PyTorch 2.14.1+cpu; CPU execution verified |
| CUDA | `torch.cuda.is_available()` was false in the installed environment. GPU training was not verified. |
| RealSense | `pyrealsense2` 2.58.4.10922 imports and enumerates; zero connected devices were detected. |
| CLI | `doctor` steps a finite physics world; `demo` settles the cube, reaches 6 cm above its centre, integrates and produces PNG/JSON output. Final demo TCP error was 0.005262 m. |
| Rendering | MuJoCo image rendered in the live browser dashboard. |
| Manual controls | J1 slider changed from 90° to 65.4°, **Move to targets** clicked, and changed TCP telemetry observed (about X 0.086 m, Y 0.225 m, Z 0.327 m). |
| Live policy | CLI loaded the bundled PPO checkpoint and drove 100 action requests against port 8765. |
| HTTP controls | State, joint commands, deterministic step/reset and cube operations exercised through the service/client tests. |
| Robot asset parity | Original URDF/STL hashes recorded; flange forward kinematics compared with upstream Pinokin at three poses, with position differences below `1e-15 m`. |
| Native PAROL6 controller | Existing `D:/HumanEd/.venv` executed this lab's `upstream-smoke` with forced mock serial. Home and a +2° J1 move passed. Reported joints were about `[92.0039, -90, 180.00074, 0, 0, 180]` degrees and ping reported `hardware_connected=False`. |
| Native-to-physics bridge | An owned mock controller on UDP 5011 homed and moved J1 by +3°. Six live bridge samples reached the HTTP world on 8765; physics joint targets matched degree-to-radian telemetry within `1e-8 rad`. The controller was stopped afterwards and the dashboard reset. |
| Hardware adapter contract | Three fake-client tests passed: explicit opt-in, refusal of a simulator connection, queued tool/TCP completion including index zero, radian/degree conversion, and small-move bound. No real hardware moved. |
| Hugging Face access | Official `hf` CLI downloaded and inspected `lerobot/smolvla_base` configuration only, revision `d9f33c94a60fb382c90dea2164c96845bd955e28`; model weights/inference/fine-tuning were not attempted. |

Forward-kinematics parity validates coordinate interpretation; it does not validate contact proxies, actuator dynamics or physical inertial measurements. The native mock result validates command/controller behavior independently of the MuJoCo cube-contact world.

## Trained policy and held-out evaluation

The included trusted demonstration checkpoint is in [`examples/checkpoints/ppo-reach/`](../examples/checkpoints/ppo-reach/README.md). It was trained locally for **100,096 actual physics control steps** (100,000 requested), with PPO seed 0, CPU, the saved default scene and `reach_cube_above_v1` task. Metadata, scene and evaluation results are stored beside `model.zip`.

The checkpoint was trained before the cube-contact material-priority/rolling-friction correction. The final 100-episode evaluation was repeated against the corrected physics and CPU lock; success remained 96/100. Evaluation scene/model/task source hashes and dependency versions are recorded. The original pre-correction training source was not archived, so those final hashes identify the evaluated snapshot rather than claiming an exact saved training-source snapshot.

Evaluation used 100 held-out episode seeds **20,000–20,099**, deterministic model actions, and the same seeded resets for all three controllers:

| Controller | Successes | Mean final distance |
|---|---:|---:|
| Trained PPO | 96 / 100 | 0.02379 m |
| Random actions | 0 / 100 | 0.22600 m |
| Scripted position IK | 100 / 100 | 0.01776 m |

Success means the **bare flange marker reaches within 3.5 cm of a point 6 cm above the cube centre for three consecutive 50 ms controls**, within 200 controls. It does not mean contact, grasp, object lift or pick-and-place. Reset cube XY jitter was within ±2.5 cm; these are held-out seeds from that task distribution, not arbitrary scenes or hardware trials. The scripted IK baseline outperformed this PPO on the tested distribution.

Use the saved [`evaluation-held-out.json`](../examples/checkpoints/ppo-reach/evaluation-held-out.json) for per-episode success, return, minimum/final distances, seeds and target positions. Re-run on new scenes before drawing broader performance conclusions.

## Final package checks

| Check | Result |
|---|---|
| Full project suite | 27 passed in 4.26 s; two expected Gymnasium warnings describe intentionally unbounded state/velocity observation boxes. |
| Lint and diff formatting | `ruff check .` and `git diff --check` passed. |
| Fresh locked environment | `uv sync --frozen --extra train --extra camera --extra hub --group dev`, with `UV_PROJECT_ENVIRONMENT=outputs/repro-venv`, succeeded independently of the existing virtual environment. `doctor` passed and reported CPU PyTorch 2.14.1+cpu. |
| Fresh-environment suite | 27 passed in 10.60 s. |
| Wheel build | `uv build --wheel` succeeded; the self-contained package wheel was 3,108,430 bytes. |
| Installed-wheel assets | Wheel installed into fresh `outputs/wheel-venv` with core dependencies, then `doctor` and rendered `demo` ran with working directory changed to `outputs`. Packaged robot/scene assets work without the source checkout as the current directory. |
| Dependency portability | The explicit CPU PyTorch source in the lock avoids pulling large Linux CUDA dependencies for this default CPU workflow. |

The fresh-environment and wheel checks above were performed on Windows. Linux execution was separately verified in GitHub Actions; macOS execution remains untested. `.gitattributes` preserves the copied URDF's original bytes for portable asset checksums and normalizes new source text to LF.

## Published CI evidence

The [initial published run](https://github.com/Takodachi696969/computer-vision/actions/runs/37201399827), at commit `92c802f`, verified the complete Linux suite, EGL rendering, `doctor` and lint. Its Linux job `111433665231` passed.

The Windows job `111433665358` passed 26 tests and failed only the rendering test: MuJoCo reported `gladLoadGL`, with WGL unable to obtain an OpenGL-capable driver on the hosted runner. This is a runner rendering limitation; the full 27-test suite and rendered dashboard/demo passed on the Windows desktop. Physics integration, state/control APIs and learning do not require a display.

The [final code verification run](https://github.com/Takodachi696969/computer-vision/actions/runs/37201738786), at commit `b3c7acd`, passed both jobs: Windows `111434663367` and Linux `111434663517`. Hosted Windows runs the 26 tests that do not render; Linux runs all 27 including EGL rendering. Both jobs also pass lint and `doctor`, and load/evaluate the bundled Windows-trained PPO checkpoint. Windows desktop image generation is independently verified locally. The final documentation commit changes this record only.

## Equipment and integrations not verified

- Real PAROL6 serial connection, limit-switch homing, motion, payload or tool calibration.
- Actual RealSense RGB/depth capture and camera-to-base extrinsic calibration.
- RTX GPU training/inference kernels.
- Gripper dynamics or successful manipulation.
- LeRobot dataset export, a PAROL6 LeRobot adapter or SmolVLA fine-tuning/inference.
- Continuous learned-policy deployment onto the physical robot.

Those remain optional/future work described in the tutorial. The shipped browser and policy runner affect the MuJoCo world; they do not activate physical hardware.
