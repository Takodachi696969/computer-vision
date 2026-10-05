"""Compare light/heavy cube contact with force-limited robot servos.

Independent worlds, no dashboard required. Results are model experiments,
not calibrated PAROL6 payload ratings. The box falls onto the bare wrist,
deflects it, and can slip away; there is no attachment or gripper.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from humaned_lab.simulation import PhysicsWorld

output = Path("outputs/load-comparison")
output.mkdir(parents=True, exist_ok=True)
results = []
for label, mass in (("light", 0.01), ("heavy", 5.0)):
    with PhysicsWorld() as world:
        world.configure_actuators({"torque_limits_nm": [12, 12, 8, 1, 1, 1]})
        world.configure_cube("target", {"mass_kg": mass, "size_m": [0.04, 0.04, 0.04]})
        initial_tip = world.tcp_position()
        world.set_cube("target", initial_tip + [0, 0, 0.038])
        peak_force, peak_deflection, peak_saturated = 0.0, 0.0, 0
        for _ in range(round(0.3 / world.timestep)):
            state = world.step()
            peak_force = max(peak_force, state["robot_contact_force_n"])
            peak_deflection = max(peak_deflection, float(np.linalg.norm(world.tcp_position()-initial_tip)))
            peak_saturated = max(peak_saturated, sum(state["joint_torque_saturated"]))
        Image.fromarray(world.render(show_collision=True)).save(output / f"{label}.png")
        results.append({
            "case": label, "mass_kg": mass,
            "peak_robot_contact_force_n": peak_force,
            "peak_tip_deflection_m": peak_deflection,
            "final_tip_deflection_m": float(np.linalg.norm(world.tcp_position()-initial_tip)),
            "max_saturated_joints": peak_saturated,
            "physics_fingerprint": state["physics_fingerprint"],
        })
(output / "comparison.json").write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
print(json.dumps(results, indent=2))
