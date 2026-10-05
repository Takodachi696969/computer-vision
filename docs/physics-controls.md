# Physics controls and Cartesian motion

Use this guide to experiment with cube materials, limited joint effort and arm movement in the local MuJoCo world. All examples control simulation. The physical PAROL6 controller remains a separate, explicitly enabled adapter.

Start from the repository root:

```powershell
.\.venv\Scripts\humaned-lab.exe serve --host 127.0.0.1 --port 8765
```

Open [the dashboard](http://127.0.0.1:8765) and [the live API specification](http://127.0.0.1:8765/docs). The [main tutorial](tutorial.md) covers installation, policies and training; this page explains the adjustable physics and motion controls.

## A smoother view

The image uses a continuous MJPEG stream at `/api/stream.mjpg`, independently of the dashboard's 250 ms telemetry polling. Open **Viewer settings**, select target 15/30/60 FPS, choose 640×426, 960×640 or 1280×854, then click **Apply viewer settings**. Defaults are 30 FPS, 960×640 and JPEG quality 85. **Show robot contact proxies** overlays the approximate shapes used for collision; the CAD visuals alone do not show that geometry.

Watch **Actual renderer FPS**, **Simulation speed / wall time** and render/encode time after the setting settles. A requested 60 FPS is a scheduling target, not a guaranteed frame rate. Lower resolution/quality can reduce render/encoding/network cost. Renderer FPS is measured where frames are produced; browser presentation can be slower. A simulation speed near 1× means simulated time advances near wall time. Paused physics can still render new frames. These viewer settings do not change cube mass, contacts or the fixed 50 ms policy interval.

## Three ways to move the arm

**Joint targets:** edit the six joint sliders and submit them. The sliders display degrees; the servo/API contract uses radians. The measured angle can lag the target when motors are weak or the arm is obstructed.

**Cartesian position:** enter the bare flange marker's desired X/Y/Z in metres. Position-only inverse kinematics finds a bounded joint configuration, then the controller moves joint targets towards it. Orientation is unconstrained. Solving IK proves a kinematic target can be approximated; it does not establish that the simulated servos can reach it under gravity/contact.

In **Tip / waypoints**, click **Use measured tip** before making a small coordinate change, set **Segment time / s**, and click **Solve IK + move**. For a sequence, enter each position and click **Add to sequence**, then **Run sequence**. **Loop sequence** repeats the points; **Clear** only clears the editor. **Stop / hold** cancels the running sequence. Changing tabs alone does not cancel a sequence; submitting another motion command does.

Every segment smoothly interpolates joint targets over its nominal simulation duration. It advances only after the measured tip is within the selected tolerance. A missed target gets a settling interval and then stops with `status="stalled"`; this means the waypoint was not reached, without identifying the cause. Check actual tip error, saturated joints and enabled contacts. The dashboard starts with **Arrival tolerance / mm = 10** and **Settling timeout / s = 3**; the API's default tolerance when omitted is 0.02 m. Enable **Stop the sequence if arm contact exceeds this limit** to use the **Contact stop / N** field. Optional `force_limit_n` stops on excessive moving-arm normal contact load. This is a simulation guard, not a physical safety rating.

**Arrow keys:** open the **Arrow keys** tab and click **Enable arrow controls**. Select a joint with Left/Right or keys 1–6, then hold Up/Down to change that joint's target at the selected degrees per second. Release the key to stop jogging. Losing browser focus, pausing, or switching controllers also stops it; the server expires a missing jog heartbeat within its configured maximum 0.35-second timeout. This rate controls the requested target, while force limits govern the actual joint response.

The server has one shared world and one active motion mode. Submitting another controller command cancels the previous live motion. Cartesian/jog API requests resume integration; a policy step cancels live motion and advances one deterministic 50 ms interval with playback paused. Pausing freezes Cartesian progress; resume continues it. Stop holds near the current measured pose through the simulated servos; it is not a physical E-stop or a guarantee of a motionless arm under load.

## Cube properties

In **Cube Properties & Launch**, select an existing cube, edit its properties and click **Apply physical properties**. **Tune bounce** enables the requested restitution setting; unchecking it restores the legacy contact setting. **Place / launch cube** sets position and velocity separately. Dimensions, mass and local centre of mass are separate quantities.

| Field | Units and interpretation | Accepted range |
|---|---|---|
| `size_m` | Full X/Y/Z dimensions; scalar means equal dimensions | Each dimension 0.002–2 m |
| `mass_kg` | Total body mass | 0.001–100 kg |
| `com_offset_m` | Centre of mass relative to the cube's geometric centre, along its **local** axes | Each component within 95% of the corresponding half-extent |
| `friction` | `[sliding, torsional, rolling]` contact parameters | Each component 0–5 |
| `restitution` | Requested bounce approximation; `null` preserves legacy contact settings | 0–0.95, or `null` |
| `rgba` | Display colour and opacity | Four components 0–1 |

A 0.06 m tall box rests with its geometric centre near Z=0.03 m on this floor. A local COM offset `[0.01,0,0]` rotates with the box; it is not a permanent world-X offset. Mass changes both inertia and gravitational loading. The lab uses uniform-box principal inertia moments about the displaced COM as an approximation; specifying a displaced COM does not identify a real internal mass distribution.

Sliding friction is dimensionless. Torsional and rolling parameters have length units in MuJoCo's model, so copying the same number into all three has no physical justification. Contact material mixing also matters: the configured cube material governs cube-floor contact; two object materials can combine. [MuJoCo contact semantics](https://mujoco.readthedocs.io/en/stable/computation/index.html#contact).

### What the bounce slider means

MuJoCo uses soft contacts. The lab maps requested restitution `e` to direct `solref=(-20000,-b)`, using `b=2*sqrt(20000)*zeta` and `zeta=-log(e)/sqrt(pi²+log(e)²)`; zero uses critical damping. This oscillator approximation is a starting point, not an exact material coefficient. The stiffness is a normalized constraint parameter, **not N/m**. Measured bounce depends on integration, contact geometry and solver settings. `null` restores the legacy `solref=(0.008,1)`. [MuJoCo solver parameters](https://mujoco.readthedocs.io/en/stable/modeling.html#reference), [restitution example](https://mujoco.readthedocs.io/en/stable/modeling.html#restitution).

For a calibration experiment, reset the same cube above a clear floor with zero linear/angular velocity, change one material parameter, and repeat. Measure vertical speed immediately before/after the isolated collision, or compare centre-of-mass rise heights above the settled resting height. Tumbling cubes and multiple contacts invalidate a simple height-ratio interpretation. Repeat at several timesteps and drop heights before labeling a parameter as measured.

## Finite actuator effort and stalls

The robot has six position servos. `kp` and `kv` set their proportional/damping response; `torque_limits_nm` bounds each actuator's output. With one unit-gear hinge actuator per joint, the clamp is a joint torque in Nm. Raising the target error raises requested effort until this cap is reached. A capped motor may settle with tracking error, sag under gravity or stop against an obstacle. [MuJoCo actuator output limits](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-general).

The default six caps are **300 Nm**, copied from six identical `effort="300"` entries in the pinned PAROL6 URDF. They are nominal model values, not measured stall torques. Servo gains are lab parameters. This model does not simulate stepper current, torque-speed curves, missed steps, gearbox backlash, heating or firmware delay. Use smaller caps for a controlled experiment; identify hardware parameters separately before drawing conclusions about the real arm.

In **Physics Layers & Actuators**, **URDF baseline** and **Load experiment preset** fill the editor; neither applies a model until you click **Apply actuator model**. The lower-torque preset is illustrative. Watch **Joint loads** for applied torque, measured/target angle and saturation. A contact-blocked arm or insufficient torque can leave a large target error even though the command was accepted.

`target_velocity_limits_rad_s` optionally limits how fast servo targets change. It does not guarantee the measured joints obey that speed under external forces. The legacy default is `null`, meaning this extra target slew limiter is disabled.

Each actuator field accepts a scalar for all six joints or a six-element vector. `kp` accepts 0–2000, `kv` 0–100, `torque_limits_nm` 0.001–300 Nm, and an enabled target velocity limit 0.001–10 rad/s. Defaults are `kp=[180,180,140,50,45,30]`, `kv=[10,10,8,3,3,2]`, six 300 Nm caps and no extra target slew limit.

Stall telemetry is a diagnostic condition based on tracking error, effort saturation and low simulated velocity. `joint_load_limited` becomes true when effort is at least 99.9% of the cap, speed is below 0.05 rad/s and target error exceeds 0.025 rad. It is an instantaneous heuristic, not a persistent hardware fault diagnosis. Inspect target error, actual torque and velocity together, and allow several simulation steps after changing a target. Weak servos can also fail to hold the arm without contact.

| State field | Interpretation |
|---|---|
| `joint_targets_rad` | Final requested joint setpoints |
| `actuator_targets_rad` | Setpoints actually applied after any target slew limiter |
| `joint_target_error_rad` | Requested minus simulated joint position |
| `joint_commanded_torque_nm` | Servo effort requested before force clamping |
| `joint_measured_torque_nm` | Last-step effort applied by MuJoCo's actuator model; no hardware sensor |
| `joint_torque_saturated`, `joint_load_limited` | Per-joint saturation / load-limited diagnostic flags |
| `contacts` | Named contact pairs, position, penetration/distance, normal load and force on `geom2` |
| `robot_contact_force_n` | Sum of positive normal loads involving moving robot proxies |
| `total_contact_force_n` | Sum of positive normal loads across all contacts |
| `cubes[*].contact_force_n` | Sum of positive normal loads involving that cube |
| `cubes[*].net_contact_force_world_n` | Vector sum of cube contact forces in world axes; excludes gravity |

Normal-load sums do not cancel opposing forces and are not equivalent to a net force vector. The fixed base-floor contact is excluded from the robot metric. Individual forces are extracted with `mj_contactForce` in the contact frame and transformed into world axes. [Official API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-contactforce).

The cubes and links remain rigid. Increasing force cannot crack a cube, bend a link, break a gearbox or identify damage. Modeling fracture or structural failure requires a separate material/failure model. There are also no articulated fingers or grasp attachment in this package.

### A controlled stall experiment

For an arm/object load comparison, run:

```powershell
.\.venv\Scripts\python.exe examples/scenes/load_comparison.py
```

It drops a 10 g or 5 kg cube just above the home-position wrist, with caps `[12,12,8,1,1,1]` Nm. The cube contacts, deflects and can slip away from the bare arm. It saves results and collision-overlay images in `outputs/load-comparison`. In the local 0.3 s experiment, the light case had 0.63 N peak moving-arm normal load and 3.95 mm peak tip deflection; the heavy case had 73.04 N, 50.55 mm peak deflection and two saturated joints. These are experimental outcomes for this uncalibrated model, not payload ratings.

To try it in the GUI: apply those caps, leave target speed limiting off, pause and reset, set the target cube to 0.04 m sides and either 0.01 or 5 kg, COM zero and bounce off. Place it near `[0,0.23677,0.372]` m with zero velocity, then resume. Reset between trials. Watch the contact and joint-load displays; brief impact peaks can fall between telemetry samples, so the script records each physics step for comparison.

### Isolate the effort cap

This isolates the servo cap from gravity/contact. Run it offline, then compare the same command with `torque_limits_nm=300`:

```python
from humaned_lab.simulation.world import PhysicsWorld

with PhysicsWorld() as world:
    world.configure_physics({
        "gravity_enabled": False,
        "robot_object_contacts_enabled": False,
        "robot_floor_contacts_enabled": False,
    })
    world.configure_actuators({"torque_limits_nm": 0.001})
    target = world.joint_targets_rad
    target[0] += 0.4
    world.set_joint_targets(target)
    world.step(round(1.0 / world.timestep))
    state = world.state()
    print("J1 requested error, rad:", state["joint_target_error_rad"][0])
    print("J1 applied torque, Nm:", state["joint_measured_torque_nm"][0])
    print("J1 load-limited:", state["joint_load_limited"][0])
```

The low cap should leave substantial tracking error after one second; the torque clamp is observable even though a valid target was accepted. This intentionally tiny cap illustrates the diagnostic, without estimating real PAROL6 capability. Restore normal settings before trying the reaching checkpoint.

## Model layers

These checkboxes include or omit mechanisms for comparison. They do not constitute a ranking of physical fidelity.

| Scene flag | Effect when disabled |
|---|---|
| `gravity_enabled` | Omits configured gravity |
| `robot_object_contacts_enabled` | Robot contact proxies pass through free cubes |
| `object_floor_contacts_enabled` | Free cubes pass through the floor and fixed obstacles |
| `object_object_contacts_enabled` | Objects pass through one another |
| `robot_floor_contacts_enabled` | Robot proxies pass through the floor and fixed obstacles |
| `friction_enabled` | Removes contact friction while preserving enabled normal contacts |
| `joint_damping_enabled` | Removes passive joint damping |
| `actuation_enabled` | Removes servo effort; gravity/contact/passive dynamics still act |
| `self_collision_enabled` | Only `false` is supported; current robot proxies are unsuitable for validated self-contact |

All supported mechanisms default to enabled, with self-collision disabled. Deselecting passive joint damping does not remove the servo's `kv` damping. Deselecting actuation does not pause integration. Floor and pair-contact toggles do not remove gravity, so an unsupported object keeps falling.

Numerical knobs such as timestep, solver iterations and contact softness trade stability, convergence and cost. Their values are not monotone measures of realism. A visually plausible scene can still have inaccurate geometry, inertia or actuator behavior. Compare simulated trajectories/contact forces against measurements and document the fitted parameters.

## Preserving an experiment and its policy contract

Property/model changes are applied atomically on the physics owner thread and preserve live pose, velocity and simulation time. A world rebuild preserves targets, but the live service cancels an active sequence/jog and requests a hold after configuration changes. Edits can create an overlap or unstable load, so pause, edit one variable and reset to a deliberate initial state when comparing experiments. Settings persist through reset; reset restores the scene's initial poses and velocities, not the previous material/actuator defaults.

Save the resolved scene before training or publishing results. A scene export records initial conditions and physical settings, not a full live-state replay or active trajectory.

Click **Save scene JSON** to download `physics-scene.json`, put it in `configs/scenes/` or an experiment directory, and restart with `--scene PATH`. Set initial cube positions/velocities in that file when the experiment must begin with your launch, rather than the original defaults.

State includes `model_revision`, incremented on actual configuration rebuilds, and `physics_fingerprint`, a hash of the dynamic configuration. Retain the scene and source revision too: this hash is not a complete recording of initial conditions, rendering settings or controller code.

The six policy actions remain normalized measured-joint deltas: `q_target=clip(q_measured+0.04*action)` followed by 50 ms of integration. Adding telemetry leaves this action/observation contract unchanged. Changing mass, friction, effort or contacts still changes the dynamics seen by a checkpoint. Preserve `examples/checkpoints/ppo-reach` as the reference experiment; reevaluate a copied policy against the new scene and retrain when necessary. Its historical reaching score does not certify performance on your modified physics, Cartesian path following or grasping.

## Send commands from your own program

With the server running, this client edits existing objects, exports a scene and submits a Cartesian sequence. `preview_tcp` solves IK without starting a move. Small offsets from the measured position are convenient examples; an unreachable point still raises an HTTP validation error.

```python
import time
from humaned_lab.control.client import SimulationClient

with SimulationClient() as lab:
    lab.reset(seed=42)
    lab.cube_properties(
        "target", size_m=[0.06, 0.04, 0.04], mass_kg=0.12,
        com_offset_m=[0.01, 0, 0], friction=[0.6, 0.01, 0.0005],
        restitution=0.35,
    )
    lab.physics(robot_object_contacts_enabled=True)
    lab.viewer(target_fps=30, width=960, height=640,
               jpeg_quality=85, show_collisions=True)
    lab.save_scene("outputs/my_physics/scene.json")

    start = lab.state()["tcp_m"]
    point = [start[0] + 0.03, start[1], start[2] + 0.01]
    print("IK preview:", lab.preview_tcp(point))
    lab.trajectory([point, start], segment_duration_s=2,
                   tolerance_m=0.02, settle_timeout_s=3)
    try:
        while True:
            state = lab.state()
            motion = state["trajectory"]
            print(motion["status"], motion["distance_m"],
                  state["joint_torque_saturated"])
            if not motion["active"]:
                break
            time.sleep(0.1)
    finally:
        lab.stop_trajectory()
```

The server owns integration for a live sequence. Polling state does not itself advance time, and calling `step()` here would replace the sequence with a policy action. `trajectory.reached` is the final physical-arrival result; `planned_complete` only means the current segment's joint-target interpolation finished. Check `status` for `force_limit`, `stalled`, `stopped`, `configuration_changed` or a replacing controller.

For scripted jogging, use zero-based joint index and radians per second, refresh every 100 ms or faster, and stop in `finally`:

```python
import math
import time
from humaned_lab.control.client import SimulationClient

with SimulationClient() as lab:
    try:
        for _ in range(10):
            lab.jog(joint_index=0, velocity_rad_s=math.radians(5), timeout_s=0.3)
            time.sleep(0.05)
    finally:
        lab.stop_jog()
```

Public jog speed is bounded to ±1 rad/s and heartbeat timeout to 0.05–0.35 s. Neither an HTTP timeout nor leaving your own loop guarantees an immediate hold; the explicit stop and server heartbeat expiry cover that case.

### PowerShell example

```powershell
$labUrl = 'http://127.0.0.1:8765'
Invoke-RestMethod -Method Post "$labUrl/api/cubes/target/properties" `
  -ContentType 'application/json' `
  -Body '{"mass_kg":0.12,"com_offset_m":[0.01,0,0],"restitution":0.35}'

Invoke-RestMethod -Method Post "$labUrl/api/actuators" `
  -ContentType 'application/json' `
  -Body '{"torque_limits_nm":[12,12,8,3,3,2],"target_velocity_limits_rad_s":0.5}'

$state = Invoke-RestMethod "$labUrl/api/state"
$tip = @($state.tcp_m)
$tip[0] += 0.03
$moveBody = @{position_m=$tip; duration_s=2; tolerance_m=0.02} | ConvertTo-Json -Compress
Invoke-RestMethod -Method Post "$labUrl/api/tcp" `
  -ContentType 'application/json' -Body $moveBody

Invoke-RestMethod -Method Post "$labUrl/api/trajectory/stop"
```

### Added HTTP routes

| Route | JSON body or response |
|---|---|
| `GET /api/stream.mjpg` | Continuous latest rendered frames; no request body |
| `POST /api/viewer` | Optional `target_fps`, `width`, `height`, `jpeg_quality`, `show_collisions` |
| `POST /api/tcp/preview` | `position_m`; returns `q_rad` and `ik_residual_m` without motion |
| `POST /api/tcp` | `position_m`, optional `duration_s`, `tolerance_m`, `settle_timeout_s`, `force_limit_n` |
| `POST /api/trajectory` | `waypoints_m`, optional `segment_duration_s`, `loop`, and the same arrival/force options |
| `POST /api/trajectory/stop` | No body required |
| `POST /api/jog` | `joint_index` 0–5, `velocity_rad_s`, optional `timeout_s` |
| `POST /api/jog/stop` | No body required |
| `POST /api/cubes/{name}/properties` | Any supported cube property subset |
| `POST /api/physics` | Any supported mechanism-flag subset |
| `POST /api/actuators` | Any supported actuator-field subset |
| `GET /api/scene` | Resolved initial scene and persistent physical configuration |

Omitting a property leaves it unchanged. `restitution:null` restores legacy bounce; `target_velocity_limits_rad_s:null` removes the extra target slew limiter. Other material/actuator values must be finite and within their documented ranges. Waypoint lists accept 1–64 positions; the API supports 0.1–60 s segment times, 0.001–0.1 m arrival tolerances and 0.1–30 s extra settling time.

## Write a physical experiment directly

Save this as `examples/my_material_experiment.py` and run it with `.\.venv\Scripts\python.exe examples\my_material_experiment.py` from the repository root. It uses its own offline world and needs no running server:

```python
import numpy as np
from humaned_lab.simulation.world import PhysicsWorld

with PhysicsWorld() as world:
    world.configure_cube("target", {
        "size_m": [0.06, 0.04, 0.04],
        "mass_kg": 0.12,
        "com_offset_m": [0.01, 0, 0],
        "friction": [0.6, 0.01, 0.0005],
        "restitution": 0.35,
    })
    world.configure_actuators({"torque_limits_nm": [2, 2, 2, 1, 1, 1]})
    world.configure_physics({"robot_object_contacts_enabled": False})
    world.save_scene("outputs/material_experiment/scene.json")
    # Launch intervention is separate from the scene's initial conditions.
    world.set_cube_pose(
        "target", [0.22, 0.18, 0.4],
        velocity_m_s=[0.1, 0, 0],
        angular_velocity_rad_s=[0, 0, 2],
    )
    world.step(round(2.0 / world.timestep))
    state = world.state()
    target = next(c for c in state["cubes"] if c["name"] == "target")
    print("Cube geometric centre:", target["position_m"])
    print("Cube centre of mass:", target["com_position_m"])
    print("Applied joint effort:", state["joint_measured_torque_nm"])
    print("Load-limited joints:", state["joint_load_limited"])
    assert np.isfinite(state["q_rad"]).all()
```

To make the launch itself reproducible on reset/training, put its `position_m`, `velocity_m_s` and `angular_velocity_rad_s` in the saved scene JSON. The configuration methods change existing model bodies; create/remove bodies by editing a scene and restarting.

After a model rebuild, resolve body/site/geom IDs again rather than retaining references to an older `MjModel` or `MjData`. Run offline edits, stepping and rendering on the same thread. The live service queues model edits to its owner thread; an arbitrary worker should not mutate its `world.model` directly.
