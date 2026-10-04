"""Starter policy: replace act() with your own six-action controller.

Run through the lab policy CLI or import with
``load_policy('examples.policies.my_policy:MyPolicy')`` from the repository root.
"""

import numpy as np


class MyPolicy:
    def act(self, observation):
        # observation includes q_rad, qd_rad_s, tcp_m, target_m, delta_m,
        # cube_velocity_m_s. Return six floats; 1 means +0.04 rad this control.
        return np.zeros(6, dtype=np.float32)
