"""Small, explicit policy contract; actions never mean motor torque or degrees."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

ACTION_SIZE = 6
ACTION_DELTA_RAD = 0.04
CONTROL_INTERVAL_S = 0.05
TARGET_OFFSET_M = 0.06


@runtime_checkable
class Policy(Protocol):
    """Receive SI-unit observation arrays and return six numbers in [-1, 1].

    The runner adds ``0.04 * action`` radians to the measured six joint angles,
    clips the result to joint limits, and advances simulation by 50 ms. This
    interface is deliberately simulation-only; it is not a hardware safety layer.
    """

    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray: ...


def validate_action(action: Any) -> np.ndarray:
    """Reject malformed/nonfinite output, then saturate valid normalized actions."""
    array = np.asarray(action, dtype=np.float64)
    if array.shape != (ACTION_SIZE,):
        raise ValueError(f"A policy action must have shape (6,), got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("A policy action must contain only finite numbers.")
    return np.clip(array, -1.0, 1.0).astype(np.float32)


def observation_from_state(
    state: Mapping[str, Any],
    cube_name: str | None = None,
    target_offset_m: float = TARGET_OFFSET_M,
) -> dict[str, np.ndarray]:
    """Map the HTTP/world state to the exact observation used for PPO training.

    Targets follow the selected cube. ``target_offset_m`` is measured vertically
    from the cube's centre, not its top surface; the default task is reaching.
    """
    cubes = state.get("cubes", [])
    if isinstance(cubes, Mapping):
        candidates = [dict(value, name=name) for name, value in cubes.items()]
    else:
        candidates = list(cubes)
    if not candidates:
        raise ValueError("The reaching task requires at least one cube in the scene.")
    cube = next((item for item in candidates if item["name"] == cube_name), None)
    if cube is None:
        if cube_name is not None:
            raise ValueError(f"Cube {cube_name!r} was not found in the scene.")
        cube = next((item for item in candidates if item["name"] == "target"), None)
        if cube is None:
            cube = sorted(candidates, key=lambda item: item["name"])[0]
    tcp = np.asarray(state["tcp_m"], dtype=np.float32)
    target = np.asarray(cube["position_m"], dtype=np.float32).copy()
    target[2] += target_offset_m
    return {
        "q_rad": np.asarray(state["q_rad"], dtype=np.float32).copy(),
        "qd_rad_s": np.asarray(state["qd_rad_s"], dtype=np.float32).copy(),
        "tcp_m": tcp.copy(),
        "target_m": target,
        "delta_m": target - tcp,
        "cube_velocity_m_s": np.asarray(cube.get("velocity_m_s", [0, 0, 0]), dtype=np.float32),
    }


def load_policy(spec: str, kwargs: Mapping[str, Any] | None = None) -> Policy:
    """Instantiate a trusted local class/factory addressed as ``module:Name``.

    Example: ``examples.policies.my_policy:MyPolicy``. Importing user Python
    executes its code; only use policy modules you intend to run locally.
    """
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute or ":" in attribute:
        raise ValueError("Use a policy import such as 'examples.policies.my_policy:MyPolicy'.")
    # Console launchers put their Scripts directory on sys.path, unlike
    # `python -m`. Make repo-local examples importable in either entry point.
    # The caller explicitly chooses to execute trusted Python from this cwd.
    directory = str(Path.cwd())
    add_local_directory = directory not in sys.path
    if add_local_directory:
        sys.path.insert(0, directory)
    try:
        factory = getattr(importlib.import_module(module_name), attribute)
        policy = factory(**dict(kwargs or {}))
    finally:
        if add_local_directory:
            sys.path.remove(directory)
    if not callable(getattr(policy, "act", None)):
        raise TypeError(f"{spec!r} must create an object with an act(observation) method.")
    return policy
