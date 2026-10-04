"""Thread-safe MuJoCo world with one native-radian robot control interface."""

from __future__ import annotations

import copy
import json
import re
import threading
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

from .model import ASSET_DIR, HOME_Q_RAD, JOINT_NAMES, build_model_xml

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")


def _vector(value: Sequence[float], count: int, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain {count} finite numbers") from exc
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain {count} finite numbers")
    return result


def _unknown_keys(value: dict, known: set[str], label: str) -> None:
    extras = set(value) - known
    if extras:
        raise ValueError(f"Unknown {label} fields: {', '.join(sorted(extras))}")


def _scene_config(path: Path) -> dict:
    source = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(source, dict):
        raise ValueError("Scene must be a JSON object")
    _unknown_keys(
        source,
        {
            "schema_version",
            "name",
            "timestep_s",
            "gravity_m_s2",
            "initial_q_rad",
            "goal_m",
            "cubes",
            "obstacles",
        },
        "scene",
    )
    if source.get("schema_version", 1) != 1:
        raise ValueError("Only scene schema_version=1 is supported")
    scene = dict(
        schema_version=1,
        name=source.get("name", path.stem),
        timestep_s=float(source.get("timestep_s", 0.002)),
        gravity_m_s2=_vector(source.get("gravity_m_s2", [0, 0, -9.81]), 3, "gravity_m_s2").tolist(),
        initial_q_rad=_vector(source.get("initial_q_rad", HOME_Q_RAD), 6, "initial_q_rad").tolist(),
        goal_m=_vector(source.get("goal_m", [0.22, 0.18, 0.08]), 3, "goal_m").tolist(),
        cubes=[],
        obstacles=[],
    )
    if not np.isfinite(scene["timestep_s"]) or not 0.0002 <= scene["timestep_s"] <= 0.01:
        raise ValueError("timestep_s must be between 0.0002 and 0.01")
    names = {"world", "base_link", "floor", "goal", "tcp", *JOINT_NAMES}
    for category in ("cubes", "obstacles"):
        items = source.get(category, [])
        if not isinstance(items, list) or len(items) > 32:
            raise ValueError(f"{category} must be a list with at most 32 entries")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"Each {category} entry must be an object")
            keys = {"name", "size_m", "position_m", "rgba"}
            if category == "cubes":
                keys |= {
                    "mass_kg",
                    "quaternion_wxyz",
                    "velocity_m_s",
                    "angular_velocity_rad_s",
                    "friction",
                    "position_jitter_m",
                }
            _unknown_keys(item, keys, category)
            name = item.get("name")
            if (
                not isinstance(name, str)
                or not _NAME.fullmatch(name)
                or name in names
                or name.startswith(("free_", "geom_", "visual_", "collision_", "servo_", "mesh_"))
            ):
                raise ValueError(f"Invalid, reserved, or duplicate object name: {name!r}")
            names.add(name)
            if "position_m" not in item:
                raise ValueError(f"{name} requires position_m")
            size = item.get("size_m", [0.04, 0.04, 0.04])
            if isinstance(size, (int, float)):
                size = [size] * 3
            size = _vector(size, 3, f"{name}.size_m")
            if np.any(size < 0.002) or np.any(size > 2.0):
                raise ValueError(f"{name}.size_m must be between 0.002 and 2m")
            color = _vector(item.get("rgba", [0.95, 0.46, 0.18, 1]), 4, f"{name}.rgba")
            if np.any(color < 0) or np.any(color > 1):
                raise ValueError(f"{name}.rgba values must be between 0 and 1")
            obj = dict(
                name=name,
                size_m=size.tolist(),
                position_m=_vector(item["position_m"], 3, f"{name}.position_m").tolist(),
                rgba=color.tolist(),
            )
            if category == "cubes":
                mass = float(item.get("mass_kg", 0.08))
                if not np.isfinite(mass) or not 0.001 <= mass <= 100:
                    raise ValueError(f"{name}.mass_kg must be between 0.001 and 100")
                quat = _vector(
                    item.get("quaternion_wxyz", [1, 0, 0, 0]), 4, f"{name}.quaternion_wxyz"
                )
                if np.linalg.norm(quat) < 1e-9:
                    raise ValueError(f"{name}.quaternion_wxyz cannot be zero")
                friction = _vector(item.get("friction", [0.8, 0.02, 0.002]), 3, f"{name}.friction")
                if np.any(friction < 0):
                    raise ValueError(f"{name}.friction cannot be negative")
                jitter = _vector(
                    item.get("position_jitter_m", [0, 0, 0]), 3, f"{name}.position_jitter_m"
                )
                if np.any(jitter < 0):
                    raise ValueError(f"{name}.position_jitter_m cannot be negative")
                obj.update(
                    mass_kg=mass,
                    quaternion_wxyz=(quat / np.linalg.norm(quat)).tolist(),
                    velocity_m_s=_vector(
                        item.get("velocity_m_s", [0, 0, 0]), 3, f"{name}.velocity_m_s"
                    ).tolist(),
                    angular_velocity_rad_s=_vector(
                        item.get("angular_velocity_rad_s", [0, 0, 0]),
                        3,
                        f"{name}.angular_velocity_rad_s",
                    ).tolist(),
                    friction=friction.tolist(),
                    position_jitter_m=jitter.tolist(),
                )
            scene[category].append(obj)
    return scene


class PhysicsWorld:
    """Physical robot/cube world; metres, seconds, radians, wxyz quaternions.

    The robot has a bare flange and coarse contact primitives. There is no
    hidden grasp attachment. Rendering must always be called from the same
    thread because OpenGL contexts are thread-owned. Other methods use RLock.
    """

    def __init__(self, scene_path: str | Path | None = None):
        self.lock = threading.RLock()
        self.scene_path = (
            Path(scene_path).resolve() if scene_path else ASSET_DIR / "scenes" / "default.json"
        )
        self.scene = _scene_config(self.scene_path)
        self.model_xml, assets = build_model_xml(self.scene)
        self.model = mujoco.MjModel.from_xml_string(self.model_xml, assets)
        self.data = mujoco.MjData(self.model)
        self.joint_names = list(JOINT_NAMES)
        self.joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names]
        )
        self.joint_qpos_addresses = self.model.jnt_qposadr[self.joint_ids].copy()
        self.joint_dof_addresses = self.model.jnt_dofadr[self.joint_ids].copy()
        self.joint_limits = self.model.jnt_range[self.joint_ids].copy()
        self.tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal")
        self.timestep = float(self.model.opt.timestep)
        self._cubes = {}
        for cube in self.scene["cubes"]:
            joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"free_{cube['name']}")
            self._cubes[cube["name"]] = (
                int(self.model.jnt_qposadr[joint]),
                int(self.model.jnt_dofadr[joint]),
            )
        self._renderer = None
        self._render_shape = None
        self._render_thread_id = None
        self._closed = False
        self._targets = HOME_Q_RAD.copy()
        self.reset()

    @property
    def joint_targets_rad(self) -> np.ndarray:
        with self.lock:
            return self._targets.copy()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("PhysicsWorld is closed")

    def _valid_joints(self, values: Sequence[float]) -> np.ndarray:
        q = _vector(values, 6, "joint targets (radians)")
        # Upstream's native degree limits differ from its rounded URDF radians
        # by <5e-9 rad. Accept only that numerical boundary discrepancy so a
        # read-only telemetry bridge also works at the hardware endpoints.
        if np.any(q < self.joint_limits[:, 0] - 1e-8) or np.any(q > self.joint_limits[:, 1] + 1e-8):
            raise ValueError(
                "Joint targets exceed the native URDF limits; see state().joint_limits_rad"
            )
        return np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

    def reset(self, seed: int = 0) -> dict:
        with self.lock:
            self._ensure_open()
            rng = np.random.default_rng(seed)
            mujoco.mj_resetData(self.model, self.data)
            self._targets = self._valid_joints(self.scene["initial_q_rad"])
            self.data.qpos[self.joint_qpos_addresses] = self._targets
            self.data.ctrl[:] = self._targets
            self.model.site_pos[self.goal_site_id] = self.scene["goal_m"]
            for cube in self.scene["cubes"]:
                qa, da = self._cubes[cube["name"]]
                jitter = np.asarray(cube["position_jitter_m"])
                self.data.qpos[qa : qa + 3] = np.asarray(cube["position_m"]) + rng.uniform(
                    -jitter, jitter
                )
                self.data.qpos[qa + 3 : qa + 7] = cube["quaternion_wxyz"]
                self.data.qvel[da : da + 3] = cube["velocity_m_s"]
                self.data.qvel[da + 3 : da + 6] = cube["angular_velocity_rad_s"]
            mujoco.mj_forward(self.model, self.data)
            return self.state()

    def step(self, n: int = 1) -> dict:
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 1 <= n <= 100000:
            raise ValueError("step n must be an integer between 1 and 100000")
        with self.lock:
            self._ensure_open()
            mujoco.mj_step(self.model, self.data, nstep=int(n))
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise RuntimeError(
                    "Physics became non-finite; reset the world and inspect the scene"
                )
            return self.state()

    def set_joint_targets(self, q_rad: Sequence[float]) -> None:
        with self.lock:
            self._ensure_open()
            self._targets = self._valid_joints(q_rad)
            self.data.ctrl[:] = self._targets

    def state(self) -> dict:
        with self.lock:
            self._ensure_open()
            cubes = []
            for name, (qa, da) in self._cubes.items():
                cubes.append(
                    dict(
                        name=name,
                        position_m=self.data.qpos[qa : qa + 3].tolist(),
                        quaternion_wxyz=self.data.qpos[qa + 3 : qa + 7].tolist(),
                        velocity_m_s=self.data.qvel[da : da + 3].tolist(),
                        angular_velocity_rad_s=self.data.qvel[da + 3 : da + 6].tolist(),
                    )
                )
            return dict(
                time_s=float(self.data.time),
                timestep_s=self.timestep,
                joint_names=self.joint_names.copy(),
                q_rad=self.data.qpos[self.joint_qpos_addresses].tolist(),
                qd_rad_s=self.data.qvel[self.joint_dof_addresses].tolist(),
                joint_targets_rad=self._targets.tolist(),
                joint_limits_rad=self.joint_limits.tolist(),
                tcp_m=self.data.site_xpos[self.tcp_site_id].tolist(),
                goal_m=self.model.site_pos[self.goal_site_id].tolist(),
                cubes=cubes,
                contact_count=int(self.data.ncon),
            )

    def tcp_position(self) -> np.ndarray:
        with self.lock:
            self._ensure_open()
            return self.data.site_xpos[self.tcp_site_id].copy()

    def set_cube(
        self, name: str, position_m: Sequence[float], velocity_m_s: Sequence[float] | None = None
    ) -> None:
        self.set_cube_pose(name, position_m, velocity_m_s=velocity_m_s)

    def set_cube_pose(
        self,
        name: str,
        position_m: Sequence[float],
        quaternion_wxyz: Sequence[float] = (1, 0, 0, 0),
        velocity_m_s: Sequence[float] | None = None,
        angular_velocity_rad_s: Sequence[float] | None = None,
    ) -> None:
        with self.lock:
            self._ensure_open()
            if name not in self._cubes:
                raise ValueError(f"Unknown cube {name!r}; available: {', '.join(self._cubes)}")
            pos = _vector(position_m, 3, "position_m")
            quat = _vector(quaternion_wxyz, 4, "quaternion_wxyz")
            if np.linalg.norm(quat) < 1e-9:
                raise ValueError("quaternion_wxyz cannot be zero")
            vel = _vector(
                velocity_m_s if velocity_m_s is not None else (0, 0, 0), 3, "velocity_m_s"
            )
            angular = _vector(
                angular_velocity_rad_s if angular_velocity_rad_s is not None else (0, 0, 0),
                3,
                "angular_velocity_rad_s",
            )
            qa, da = self._cubes[name]
            self.data.qpos[qa : qa + 3] = pos
            self.data.qpos[qa + 3 : qa + 7] = quat / np.linalg.norm(quat)
            self.data.qvel[da : da + 6] = 0
            self.data.qvel[da : da + 3] = vel
            self.data.qvel[da + 3 : da + 6] = angular
            mujoco.mj_forward(self.model, self.data)

    def set_goal(self, position_m: Sequence[float]) -> None:
        with self.lock:
            self._ensure_open()
            self.model.site_pos[self.goal_site_id] = _vector(position_m, 3, "goal position_m")
            mujoco.mj_forward(self.model, self.data)

    def save_scene(self, output_path: str | Path) -> Path:
        """Save resolved initial scene parameters for reproducible training."""
        with self.lock:
            self._ensure_open()
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(copy.deepcopy(self.scene), indent=2) + "\n", encoding="utf-8"
            )
            return path

    def forward_kinematics(self, q_rad: Sequence[float]) -> np.ndarray:
        with self.lock:
            self._ensure_open()
            scratch = mujoco.MjData(self.model)
            scratch.qpos[:] = self.data.qpos
            scratch.qpos[self.joint_qpos_addresses] = self._valid_joints(q_rad)
            mujoco.mj_forward(self.model, scratch)
            return scratch.site_xpos[self.tcp_site_id].copy()

    def solve_ik(
        self,
        position_m: Sequence[float],
        q_seed: Sequence[float] | None = None,
        tolerance_m: float = 0.002,
        max_iterations: int = 180,
    ) -> np.ndarray:
        """Position-only damped Jacobian IK, not a collision-free motion planner.

        Uses independent MjData, leaving physical positions and time unchanged.
        The returned target is bounded by the original joint limits. Tool
        orientation is unconstrained, so this is a reaching baseline only.
        """
        target = _vector(position_m, 3, "IK target position_m")
        if not np.isfinite(tolerance_m) or tolerance_m <= 0:
            raise ValueError("IK tolerance_m must be positive and finite")
        if not isinstance(max_iterations, int) or not 1 <= max_iterations <= 2000:
            raise ValueError("IK max_iterations must be between 1 and 2000")
        with self.lock:
            self._ensure_open()
            seed = (
                self._valid_joints(q_seed)
                if q_seed is not None
                else np.clip(
                    self.data.qpos[self.joint_qpos_addresses],
                    self.joint_limits[:, 0],
                    self.joint_limits[:, 1],
                )
            )
            scratch = mujoco.MjData(self.model)
            scratch.qpos[:] = self.data.qpos
            jac = np.zeros((3, self.model.nv))
            rng = np.random.default_rng(0)
            starts = [seed, HOME_Q_RAD]
            starts.extend(
                rng.uniform(self.joint_limits[:, 0], self.joint_limits[:, 1]) for _ in range(4)
            )
            best_error = float("inf")
            for initial in starts:
                q = np.asarray(initial).copy()
                for _ in range(max_iterations):
                    scratch.qpos[self.joint_qpos_addresses] = q
                    mujoco.mj_forward(self.model, scratch)
                    error = target - scratch.site_xpos[self.tcp_site_id]
                    distance = float(np.linalg.norm(error))
                    best_error = min(best_error, distance)
                    if distance <= tolerance_m:
                        return q
                    mujoco.mj_jacSite(self.model, scratch, jac, None, self.tcp_site_id)
                    j = jac[:, self.joint_dof_addresses]
                    dq = j.T @ np.linalg.solve(j @ j.T + 0.003**2 * np.eye(3), error)
                    q = np.clip(
                        q + np.clip(dq, -0.18, 0.18),
                        self.joint_limits[:, 0],
                        self.joint_limits[:, 1],
                    )
            raise ValueError(
                f"No position IK solution within {tolerance_m:g}m; best error={best_error:.4f}m"
            )

    def inverse_kinematics(
        self, target_m: Sequence[float], q_start: Sequence[float] | None = None
    ) -> np.ndarray:
        return self.solve_ik(target_m, q_seed=q_start)

    def render(
        self, width: int = 960, height: int = 640, *, show_collision: bool = False
    ) -> np.ndarray:
        if (
            not isinstance(width, int)
            or not isinstance(height, int)
            or not 32 <= width <= 1920
            or not 32 <= height <= 1080
        ):
            raise ValueError("Render dimensions must be integers between 32x32 and 1920x1080")
        with self.lock:
            self._ensure_open()
            thread_id = threading.get_ident()
            if self._render_thread_id is not None and self._render_thread_id != thread_id:
                raise RuntimeError(
                    "Render must stay on one thread because OpenGL contexts are thread-owned"
                )
            if self._renderer is None or self._render_shape != (width, height):
                if self._renderer is not None:
                    self._renderer.close()
                self._renderer = mujoco.Renderer(self.model, height=height, width=width)
                self._render_shape = (width, height)
                self._render_thread_id = thread_id
            options = mujoco.MjvOption()
            options.geomgroup[3] = int(show_collision)  # contact proxies are hidden by default
            self._renderer.update_scene(self.data, camera="overview", scene_option=options)
            return self._renderer.render().copy()

    def close(self) -> None:
        with self.lock:
            if self._renderer is not None:
                self._renderer.close()
                self._renderer = None
            self._closed = True

    def __enter__(self) -> "PhysicsWorld":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
