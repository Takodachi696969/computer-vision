# Audited issues and limits

Audit date: 4 October 2026. Findings below refer to the pinned revisions in [architecture.md](architecture.md), not an unbounded claim about every upstream release. No upstream source was patched by this package.

## Verified upstream mismatches

| Finding | Source evidence | Effect and lab response |
|---|---|---|
| README examples use outdated client method names | PAROL6 `README.md:65,99` calls `get_pose`; `:286` calls `set_profile`; `:414–417` calls `set_tool`; `:210,515` calls `simulator_on/off`. Pinned `client/sync_client.py` defines `pose`, `select_profile`, `select_tool`, `simulator(bool)` instead. | Copying those snippets raises `AttributeError`. The tutorial uses the inspected methods and runnable `examples/sync_client_quickstart.py`. |
| README advertises a missing script | PAROL6 `README.md` examples list includes `pick_and_place.py`; it is absent from `examples/` at the audited revision. | There is no demonstrated upstream pick-and-place example to run at that path. This lab labels its training task reaching. |
| README simulator architecture is stale | PAROL6 README describes shared-memory simulator subprocesses; `server/transports/mock_serial_transport.py` contains `_simulate_motion_jit` and its state buffers directly, with no subprocess/shared-memory setup. | The README diagram is not a precise map of the pinned mock implementation. Inspect source when extending it. |
| Python support is wider than the wheel ABI/platform list | PAROL6 `pyproject.toml` says Python `>=3.11`, but `pinokin` direct-wheel entries enumerate CPython 3.11–3.14 and specific OS/architecture combinations, including glibc `manylinux_2_39`. | A generic new Python, old Linux distribution, or unsupported architecture can satisfy the package metadata yet lack its compiled dependency. This lab pins Python 3.12 and keeps PAROL6 an optional environment. |
| Native Windows compilation is needed for dependencies without wheels | Local prior installation reported a `toppra` build failure without a C compiler. Upstream `pyproject.toml` requires `toppra>=0.6.3`; its Windows CI has a configured compiler environment. | The earlier locally built wheel is one-machine evidence, not a portable installer. Core MuJoCo/policy work does not depend on TOPPRA. Use the optional upstream bootstrap/compiler route for PAROL6. |
| Ruckig source-build caveat exists upstream | PAROL6 `.github/workflows/tests.yml` preinstalls the fixed Ruckig commit `2249d57ffaa19ecdadeaab62daf97857813629ff`, explaining the `0.17.3` sdist/scikit-build-core issue. | A plain floating dependency install can differ from CI. Optional upstream install should use a recorded working wheel/version. |
| Cold startup can exceed caller timeout | `robot.py` defaults to a 10 s readiness wait; `server/cli.py` runs `warmup_jit()` before controller construction. The prior local first run timed out and a later run succeeded. | Startup/JIT is a plausible explanation, not proven causality. Lab upstream smoke uses a longer readiness timeout and distinguishes startup failure from control failure. |
| RealSense repository location changed | The local librealsense `README.md` migration notice points from `IntelRealSense` to `realsenseai`. | Use the canonical URL in new setup instructions. |

Exact PAROL6 source locations: [client](https://github.com/PCrnjak/PAROL6-python-API/blob/b741d505ae9d1e3f28bd4ff4d0227b58b4921ed4/parol6/client/sync_client.py), [requirements](https://github.com/PCrnjak/PAROL6-python-API/blob/b741d505ae9d1e3f28bd4ff4d0227b58b4921ed4/pyproject.toml), [CI](https://github.com/PCrnjak/PAROL6-python-API/blob/b741d505ae9d1e3f28bd4ff4d0227b58b4921ed4/.github/workflows/tests.yml), [startup](https://github.com/PCrnjak/PAROL6-python-API/blob/b741d505ae9d1e3f28bd4ff4d0227b58b4921ed4/parol6/server/cli.py).

## Interface traps that matter for control

PAROL6's Python command interface uses **degrees** for joint angles and **millimetres** for TCP translations. The physics/policy interface uses **radians** and **metres**. The hardware adapter does this conversion in one place. A six-element vector does not by itself prove compatible units, zero reference, axis order or tool calibration.

`select_tool` and `set_tcp_offset` return queued command indices. Index zero can be valid. Confirm `wait_command(index)` before subsequent motion; a truthiness check is incorrect. Tool selection resets the offset. The adapter selects the named tool, waits, applies the explicit mm offset, and waits again. [Pinned async API](https://github.com/PCrnjak/PAROL6-python-API/blob/b741d505ae9d1e3f28bd4ff4d0227b58b4921ed4/parol6/client/async_client.py).

A blocking-motion timeout means the client stopped waiting; it does not by itself cancel a queued robot move. Catch errors and call `stop()`. First homing may seek hardware switches, bypassing the collision world. `stop()` leaves the controller enabled; `estop()` latches the software protective stop until `reset()`. The commissioning example requires an explicit cleared-workspace flag for homing. Its watchdog can request a stop on lost telemetry, but cannot send a stop through a broken connection or replace the physical E-stop.

Upstream client/controller protocol versions must match. Its default CLI UDP host is `0.0.0.0`, and the motion protocol supplies no authentication. The tutorial binds the upstream controller to `127.0.0.1`; this lab binds its independent HTTP simulation service to loopback. Do not expose either service directly to an untrusted network. This is a concrete boundary of these control interfaces, not a hardware deployment test.

## Physics and learning limitations

| Implemented | Limit / next engineering step |
|---|---|
| Imported URDF frames, axes, limits, masses and inertia | URDF values are not system identification of the physical arm. Some source centres of mass lie outside the corresponding visual link bounds (L1/L4 examples documented in asset provenance); retain traceability but validate CAD/measurements before accurate-dynamics claims. |
| Rigid cubes, gravity, floor/obstacle contact and angular motion | Primitive link collision proxies approximate the robot; robot self-collision is disabled. Calibrate proxies or convex collision meshes for contact-dependent work. |
| Position actuators with limited force | Effort bounds come from generic URDF entries; gains/damping are hand-set. No electrical stepper driver, gearbox backlash or controller-latency model. |
| A marker at the bare wrist flange | No finger joints/contact pads, gripper articulation or gripping policy. Add these and a release/lift success metric for pick-and-place. |
| PPO reaching environment and a policy extension point | Smoke training checks that optimization and checkpoint inference execute. Learned competence requires held-out seeded evaluation and enough samples. |
| Simulated image rendering | PPO currently consumes state vectors, not RGB/depth observations. Image policies need a new observation schema and training pipeline. |
| RealSense enumerate/capture adapter | Connected-device frames and camera-to-base calibration require actual equipment. A no-device result is expected on an unplugged computer. |
| Explicit small-move hardware bridge | Physical robot operation has not been validated here. Dashboard and policy runners control simulation only. |

## Follow-up priorities

1. Establish measured robot/real-camera calibration and reproducible physical commissioning.
2. Improve contact geometry and enable validated self-collision before planning around tight obstacles.
3. Add an articulated gripper, object lift/release task and failure criteria before training manipulation.
4. Add controller/motor delays, actuator identification and domain randomization before a sim-to-real policy experiment.
5. Record synchronized demonstrations and implement a versioned image/state/action contract before fine-tuning a LeRobot model.

Use [verification.md](verification.md) for actual test outcomes; this audit intentionally does not infer physical success from a passing simulator test.
