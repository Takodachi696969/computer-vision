"""Thread-safe MuJoCo world with one native-radian robot control interface."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

from .model import (
    ASSET_DIR,
    DEFAULT_ACTUATORS,
    DEFAULT_PHYSICS,
    HOME_Q_RAD,
    JOINT_NAMES,
    build_model_xml,
    restitution_solref,
)

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


def _validated_physics(source: dict) -> dict:
    if not isinstance(source, dict):
        raise ValueError("physics settings must be an object")
    _unknown_keys(source, set(DEFAULT_PHYSICS), "physics")
    result = copy.deepcopy(DEFAULT_PHYSICS)
    for name, value in source.items():
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
        result[name] = value
    if result["self_collision_enabled"]:
        raise ValueError(
            "Self-collision is unavailable with the overlapping approximate robot proxies"
        )
    return result


def _validated_actuators(source: dict) -> dict:
    if not isinstance(source, dict):
        raise ValueError("actuator settings must be an object")
    _unknown_keys(source, set(DEFAULT_ACTUATORS), "actuator")
    result = copy.deepcopy(DEFAULT_ACTUATORS)
    result.update(source)
    for name, low, high in (
        ("kp", 0, 2000),
        ("kv", 0, 100),
        ("torque_limits_nm", 0.001, 300),
        ("target_velocity_limits_rad_s", 0.001, 10),
    ):
        value = result[name]
        if value is None and name == "target_velocity_limits_rad_s":
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = [value] * 6
        vector = _vector(value, 6, name)
        if np.any(vector < low) or np.any(vector > high):
            raise ValueError(f"{name} values must be between {low} and {high}")
        result[name] = vector.tolist()
    return result


def _scene_config(path: Path) -> dict:
    return _validated_scene(json.loads(path.read_text(encoding="utf-8-sig")), path.stem)


def _validated_scene(source: dict, default_name: str = "Scene") -> dict:
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
            "physics",
            "actuators",
        },
        "scene",
    )
    if source.get("schema_version", 1) != 1:
        raise ValueError("Only scene schema_version=1 is supported")
    scene = dict(
        schema_version=1,
        name=source.get("name", default_name),
        timestep_s=float(source.get("timestep_s", 0.002)),
        gravity_m_s2=_vector(source.get("gravity_m_s2", [0, 0, -9.81]), 3, "gravity_m_s2").tolist(),
        initial_q_rad=_vector(source.get("initial_q_rad", HOME_Q_RAD), 6, "initial_q_rad").tolist(),
        goal_m=_vector(source.get("goal_m", [0.22, 0.18, 0.08]), 3, "goal_m").tolist(),
        cubes=[],
        obstacles=[],
        physics=_validated_physics(source.get("physics", {})),
        actuators=_validated_actuators(source.get("actuators", {})),
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
                    "com_offset_m",
                    "restitution",
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
                if np.any(friction < 0) or np.any(friction > 5):
                    raise ValueError(f"{name}.friction values must be between 0 and 5")
                com = _vector(item.get("com_offset_m", [0, 0, 0]), 3, f"{name}.com_offset_m")
                if np.any(np.abs(com) > size * 0.5 * 0.95):
                    raise ValueError(
                        f"{name}.com_offset_m must be within 95% of the box half extents"
                    )
                restitution = item.get("restitution")
                if restitution is not None:
                    if (
                        isinstance(restitution, bool)
                        or not isinstance(restitution, (int, float))
                        or not np.isfinite(restitution)
                        or not 0 <= restitution <= 0.95
                    ):
                        raise ValueError(
                            f"{name}.restitution must be null or a number between 0 and 0.95"
                        )
                    restitution = float(restitution)
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
                    com_offset_m=com.tolist(),
                    restitution=restitution,
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
        self._bind_model()
        self._renderer = None
        self._render_shape = None
        self._render_thread_id = None
        self._physics_thread_id = None
        self._closed = False
        self._targets = HOME_Q_RAD.copy()
        self.model_revision = 0
        self.reset()

    def _bind_model(self) -> None:
        """Rebind all ids after an atomic model replacement."""
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
        self._cube_body_ids = {}
        self._cube_geom_ids = {}
        for cube in self.scene["cubes"]:
            joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"free_{cube['name']}")
            self._cubes[cube["name"]] = (
                int(self.model.jnt_qposadr[joint]),
                int(self.model.jnt_dofadr[joint]),
            )
            self._cube_body_ids[cube["name"]] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, cube["name"]
            )
            self._cube_geom_ids[cube["name"]] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"geom_{cube['name']}"
            )
        self.physics_settings = copy.deepcopy(self.scene["physics"])
        self.actuator_settings = copy.deepcopy(self.scene["actuators"])
        dynamic_config = {
            "model_version": "parol6-contact-controls-v2",
            "timestep_s": self.timestep,
            "gravity_m_s2": self.scene["gravity_m_s2"],
            "physics": self.physics_settings,
            "actuators": self.actuator_settings,
            "cubes": [
                {
                    key: cube[key]
                    for key in (
                        "name",
                        "size_m",
                        "mass_kg",
                        "com_offset_m",
                        "friction",
                        "restitution",
                    )
                }
                for cube in self.scene["cubes"]
            ],
            "obstacles": self.scene["obstacles"],
        }
        self.physics_fingerprint = hashlib.sha256(
            json.dumps(
                dynamic_config, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()

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
            if self._physics_thread_id is None:
                self._physics_thread_id = threading.get_ident()
            slew = self.actuator_settings["target_velocity_limits_rad_s"]
            if slew is None:
                mujoco.mj_step(self.model, self.data, nstep=int(n))
            else:
                max_change = np.asarray(slew) * self.timestep
                for _ in range(int(n)):
                    self.data.ctrl[:] += np.clip(
                        self._targets - self.data.ctrl, -max_change, max_change
                    )
                    mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise RuntimeError(
                    "Physics became non-finite; reset the world and inspect the scene"
                )
            return self.state()

    def set_joint_targets(self, q_rad: Sequence[float]) -> None:
        with self.lock:
            self._ensure_open()
            self._targets = self._valid_joints(q_rad)
            if self.actuator_settings["target_velocity_limits_rad_s"] is None:
                self.data.ctrl[:] = self._targets

    def _rebuild(self, candidate_scene: dict) -> dict:
        """Validate/compile first; replace the shared model on its owner thread.

        This preserves body origins/orientations, free-joint velocities, robot
        positions/velocities, simulated time, the goal, and requested/applied
        controls. COM edits change the body's inertial description instantly;
        they do not reconstruct an underlying material density distribution.
        """
        self._ensure_open()
        thread_id = threading.get_ident()
        owner = self._render_thread_id or self._physics_thread_id
        if owner is not None and owner != thread_id:
            raise RuntimeError("Model configuration must run on the owning physics/render thread")
        scene = _validated_scene(candidate_scene)
        if scene == self.scene:
            return self.state()
        xml, assets = build_model_xml(scene)
        model = mujoco.MjModel.from_xml_string(xml, assets)
        data = mujoco.MjData(model)
        # Property/toggle edits never add/remove joints, so layout is preserved.
        if model.nq != self.model.nq or model.nv != self.model.nv:
            raise ValueError("Live configuration cannot change the robot/body topology")
        data.qpos[:] = self.data.qpos
        data.qvel[:] = self.data.qvel
        data.qacc_warmstart[:] = self.data.qacc_warmstart
        data.qfrc_applied[:] = self.data.qfrc_applied
        data.xfrc_applied[:] = self.data.xfrc_applied
        data.ctrl[:] = self.data.ctrl
        data.time = self.data.time
        goal_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "goal")
        model.site_pos[goal_site] = self.model.site_pos[self.goal_site_id]
        mujoco.mj_forward(model, data)
        if (
            not np.isfinite(data.qpos).all()
            or not np.isfinite(data.qvel).all()
            or not np.isfinite(data.qacc).all()
        ):
            raise ValueError("Configuration produces non-finite physics")
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
            self._render_shape = None
        self.scene, self.model_xml, self.model, self.data = scene, xml, model, data
        self._physics_thread_id = thread_id
        self._bind_model()
        self.model_revision += 1
        return self.state()

    def configure_cube(self, name: str, properties: dict) -> dict:
        """Change geometry/material/inertia without resetting current motion."""
        with self.lock:
            self._ensure_open()
            if name not in self._cubes:
                raise ValueError(f"Unknown cube {name!r}")
            if not isinstance(properties, dict):
                raise ValueError("cube properties must be an object")
            _unknown_keys(
                properties,
                {"size_m", "mass_kg", "com_offset_m", "friction", "restitution", "rgba"},
                "cube property",
            )
            scene = copy.deepcopy(self.scene)
            cube = next(c for c in scene["cubes"] if c["name"] == name)
            cube.update(properties)
            return self._rebuild(scene)

    def configure_physics(self, settings: dict) -> dict:
        with self.lock:
            self._ensure_open()
            if not isinstance(settings, dict):
                raise ValueError("physics settings must be an object")
            scene = copy.deepcopy(self.scene)
            scene["physics"].update(settings)
            return self._rebuild(scene)

    def configure_actuators(self, settings: dict) -> dict:
        with self.lock:
            self._ensure_open()
            if not isinstance(settings, dict):
                raise ValueError("actuator settings must be an object")
            scene = copy.deepcopy(self.scene)
            scene["actuators"].update(settings)
            return self._rebuild(scene)

    def state(self) -> dict:
        with self.lock:
            self._ensure_open()
            contacts, cube_forces, robot_force, total_force = self._contact_telemetry()
            cubes = []
            for name, (qa, da) in self._cubes.items():
                properties = next(c for c in self.scene["cubes"] if c["name"] == name)
                body_id = self._cube_body_ids[name]
                cubes.append(
                    dict(
                        name=name,
                        position_m=self.data.qpos[qa : qa + 3].tolist(),
                        quaternion_wxyz=self.data.qpos[qa + 3 : qa + 7].tolist(),
                        velocity_m_s=self.data.qvel[da : da + 3].tolist(),
                        angular_velocity_rad_s=self.data.qvel[da + 3 : da + 6].tolist(),
                        size_m=properties["size_m"].copy(),
                        mass_kg=properties["mass_kg"],
                        com_offset_m=properties["com_offset_m"].copy(),
                        com_position_m=self.data.xipos[body_id].tolist(),
                        inertia_diagonal_kg_m2=self.model.body_inertia[body_id].tolist(),
                        friction=properties["friction"].copy(),
                        restitution=properties["restitution"],
                        contact_solref=list(restitution_solref(properties["restitution"])),
                        restitution_model="legacy_soft_contact"
                        if properties["restitution"] is None
                        else "requested_approximate_restitution",
                        rgba=properties["rgba"].copy(),
                        contact_force_n=cube_forces[name][0],
                        net_contact_force_world_n=cube_forces[name][1].tolist(),
                    )
                )
            q = self.data.qpos[self.joint_qpos_addresses]
            qd = self.data.qvel[self.joint_dof_addresses]
            error = self._targets - q
            raw = (
                np.asarray(self.actuator_settings["kp"]) * (self.data.ctrl - q)
                - np.asarray(self.actuator_settings["kv"]) * qd
            )
            if not self.physics_settings["actuation_enabled"]:
                raw[:] = 0
            caps = np.asarray(self.actuator_settings["torque_limits_nm"])
            measured = self.data.qfrc_actuator[self.joint_dof_addresses].copy()
            saturated = (np.abs(raw) >= caps * 0.999) | (np.abs(measured) >= caps * 0.999)
            low_speed_loaded = saturated & (np.abs(qd) < 0.05) & (np.abs(error) > 0.025)
            tcp_velocity = np.zeros(6)
            mujoco.mj_objectVelocity(
                self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.tcp_site_id, tcp_velocity, 0
            )
            return dict(
                time_s=float(self.data.time),
                timestep_s=self.timestep,
                joint_names=self.joint_names.copy(),
                q_rad=self.data.qpos[self.joint_qpos_addresses].tolist(),
                qd_rad_s=self.data.qvel[self.joint_dof_addresses].tolist(),
                joint_targets_rad=self._targets.tolist(),
                actuator_targets_rad=self.data.ctrl.tolist(),
                joint_commanded_torque_nm=raw.tolist(),
                joint_measured_torque_nm=measured.tolist(),
                joint_target_error_rad=error.tolist(),
                joint_torque_limits_nm=caps.tolist(),
                joint_torque_saturated=saturated.tolist(),
                joint_load_limited=low_speed_loaded.tolist(),
                joint_limits_rad=self.joint_limits.tolist(),
                tcp_m=self.data.site_xpos[self.tcp_site_id].tolist(),
                tcp_velocity_m_s=tcp_velocity[3:].tolist(),
                goal_m=self.model.site_pos[self.goal_site_id].tolist(),
                cubes=cubes,
                contact_count=int(self.data.ncon),
                contacts=contacts,
                contact_force_n=robot_force,
                robot_contact_force_n=robot_force,
                total_contact_force_n=total_force,
                physics_settings=copy.deepcopy(self.physics_settings),
                actuator_settings=copy.deepcopy(self.actuator_settings),
                physics_fingerprint=self.physics_fingerprint,
                model_revision=self.model_revision,
            )

    def _contact_telemetry(self):
        """Last physics-step contact forces; positive force acts on geom2.

        Normal loads sum magnitudes rather than cancelling opposing contacts.
        Net per-cube world forces include contact only, excluding gravity.
        The robot metric excludes the fixed base/floor pair so it can be used
        to detect loaded interaction at the moving arm's contact proxies.
        """
        cube_forces = {name: [0.0, np.zeros(3)] for name in self._cubes}
        cube_by_geom = {geom: name for name, geom in self._cube_geom_ids.items()}
        records, robot_force, total_force = [], 0.0, 0.0
        force = np.zeros(6)
        for index, contact in enumerate(self.data.contact):
            first, second = (int(g) for g in contact.geom)
            names = [
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom) or str(geom)
                for geom in (first, second)
            ]
            mujoco.mj_contactForce(self.model, self.data, index, force)
            normal = max(0.0, float(force[0]))
            world_force = contact.frame.reshape(3, 3).T @ force[:3]
            total_force += normal
            moving_robot = any(name.startswith("collision_L") for name in names)
            if moving_robot:
                robot_force += normal
            for geom, sign in ((first, -1.0), (second, 1.0)):
                if geom in cube_by_geom:
                    cube_forces[cube_by_geom[geom]][0] += normal
                    cube_forces[cube_by_geom[geom]][1] += sign * world_force
            records.append(
                dict(
                    geom1=names[0],
                    geom2=names[1],
                    position_m=contact.pos.tolist(),
                    distance_m=float(contact.dist),
                    normal_force_n=normal,
                    force_on_geom2_world_n=world_force.tolist(),
                    robot_involved=moving_robot,
                )
            )
        return records, cube_forces, robot_force, total_force

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
