"""Run with the dashboard running: python examples/control/joint_commands.py."""

import numpy as np

from humaned_lab.control.client import SimulationClient

with SimulationClient() as sim:
    sim.reset(seed=0)
    sim.running(True)
    state = sim.state()
    target = np.asarray(state["q_rad"])
    target[0] += np.deg2rad(10)
    sim.joints(target)
    print("Requested joint target in radians:", target)
    # Cubes are free rigid bodies. This sets initial position and velocity.
    sim.cube("target", position_m=[0.22, 0.18, 0.30], velocity_m_s=[0.10, 0, 0])
