# Writing a scene

The JSON is loaded when you create `PhysicsWorld(scene_path)` or start the lab
with `--scene`. All lengths are metres; velocities use metres/second; joint
positions use native URDF radians; quaternions use **w, x, y, z** order. The
world is right handed, with +Z pointing up and gravity normally `[0,0,-9.81]`.
Reset restores the initial scene. Free cubes then fall, rotate, slide, and
collide under MuJoCo contact physics.

`default.json` is the editable copy of the default packaged scene.
`moving_cubes.json` adds initial velocities, rotation, lower friction, and a
fixed backstop. Each scene can contain up to 32 cubes and 32 static boxes.

```json
{
  "schema_version": 1,
  "name": "My bench",
  "timestep_s": 0.002,
  "gravity_m_s2": [0, 0, -9.81],
  "goal_m": [0.22, 0.18, 0.08],
  "cubes": [
    {"name": "target", "size_m": 0.04, "mass_kg": 0.08,
     "position_m": [0.22, 0.18, 0.25], "velocity_m_s": [0.1, 0, 0]}
  ],
  "obstacles": []
}
```

Cube optional fields:

| Field | Meaning / default |
|---|---|
| `size_m` | Full XYZ extents, or one scalar for equal sides; `[.04,.04,.04]` |
| `mass_kg` | Mass; `.08` |
| `quaternion_wxyz` | Initial orientation, normalized on load; `[1,0,0,0]` |
| `velocity_m_s` | Initial linear velocity in the world frame; `[0,0,0]` |
| `angular_velocity_rad_s` | Initial free-joint angular velocity; `[0,0,0]` |
| `friction` | Sliding, torsional, rolling coefficients; `[.8,.02,.002]` |
| `rgba` | RGBA colour values from 0 to 1 |
| `position_jitter_m` | Per-axis uniform initial-position randomization half-width, seeded by `reset(seed)` |

All three friction components are active (`condim=6`). Cube contact parameters
have higher priority than the floor, robot and obstacles, so a cube's lower
friction coefficient actually changes its sliding. Cube/cube contact uses the
elementwise maximum of the two cubes' coefficients, following MuJoCo's rule.

An obstacle is an immovable, axis-aligned box with `name`, `position_m`,
`size_m` and optional `rgba`. Changing the timestep, gravity, robot initial
configuration, or bodies requires recreating the world with the edited file.

At runtime use `set_cube("target", [x,y,z], [vx,vy,vz])` to reposition and
launch an existing free cube. It resets orientation and angular velocity;
`set_cube_pose` accepts a quaternion. These are explicit scene mutations,
not contact-generated motions. They do not add an invisible attachment.

The goal is a visual marker, not a physical body. `set_goal([x,y,z])` moves it.
The training environment manages the marker separately from its cube target.
To add shapes beyond boxes, extend `simulation/model.py` and scene validation
in `simulation/world.py`; the generated model is available as `world.model_xml`.
For contact debugging, `world.render(show_collision=True)` overlays the coarse
robot contact proxies in translucent cyan. Rendering does not move the bodies.

Native home angles are `[90,-90,180,0,0,180]` degrees. The joints are not
zero-centered: keep the same native coordinates when comparing the SDK.
