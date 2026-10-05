"""Run a position-only tip sequence in the live dashboard's physical world.

Start the lab first, then run this file from the repository root. Edit the
points below; trajectories are joint interpolation between IK solutions.
"""

import time

from humaned_lab.control.client import SimulationClient

with SimulationClient() as lab:
    lab.trajectory(
        [[0.22, 0.18, 0.20], [0.16, 0.24, 0.18], [0.10, 0.24, 0.25]],
        segment_duration_s=2,
        tolerance_m=0.02,
        settle_timeout_s=4,
        force_limit_n=100,
    )
    while True:
        state = lab.state()
        path = state["trajectory"]
        print(
            "Waypoint:", path["waypoint_index"] + 1,
            "status:", path["status"], "tip:", state["tcp_m"],
            "contact N:", round(state["robot_contact_force_n"], 2),
        )
        if not path["active"]:
            break
        time.sleep(0.2)
    if not path["reached"]:
        raise RuntimeError(f"Sequence stopped without reaching the final target: {path['status']}")
