"""Convert the pinned upstream URDF to an inspectable MuJoCo model.

Joint frames, axes, native limits, and link inertials come from the original
URDF. Visual STLs are unchanged. Contact primitives and actuators are the lab's
explicit approximations; see assets/parol6/PROVENANCE.md.
"""

from __future__ import annotations

import math
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

ASSET_DIR = Path(__file__).resolve().parent / "assets"
ROBOT_DIR = ASSET_DIR / "parol6"
JOINT_NAMES = tuple(f"L{i}" for i in range(1, 7))
HOME_Q_RAD = np.deg2rad([90.0, -90.0, 180.0, 0.0, 0.0, 180.0])
DEFAULT_PHYSICS = {
    "gravity_enabled": True,
    "robot_object_contacts_enabled": True,
    "object_floor_contacts_enabled": True,
    "object_object_contacts_enabled": True,
    "robot_floor_contacts_enabled": True,
    "friction_enabled": True,
    "joint_damping_enabled": True,
    "actuation_enabled": True,
    "self_collision_enabled": False,
}
DEFAULT_ACTUATORS = {
    "kp": [180.0, 180.0, 140.0, 50.0, 45.0, 30.0],
    "kv": [10.0, 10.0, 8.0, 3.0, 3.0, 2.0],
    "torque_limits_nm": [300.0] * 6,
    "target_velocity_limits_rad_s": None,
}
RESTITUTION_STIFFNESS = 20000.0


def restitution_solref(requested: float | None) -> tuple[float, float]:
    """Approximate restitution using normalized contact stiffness/damping.

    The oscillator damping-ratio relation is a tuning aid, not an exact
    coefficient of restitution in MuJoCo's discrete, soft contact solver.
    None preserves the original lab contact model for existing checkpoints.
    Direct-format stiffness has MuJoCo's normalized units, not N/m.
    """
    if requested is None:
        return (0.008, 1.0)
    ratio = (
        1.0
        if requested <= 0
        else -math.log(requested) / math.sqrt(math.pi**2 + math.log(requested) ** 2)
    )
    return (-RESTITUTION_STIFFNESS, -2.0 * math.sqrt(RESTITUTION_STIFFNESS) * ratio)


def _numbers(values: list[float] | np.ndarray) -> str:
    return " ".join(f"{float(v):.14g}" for v in values)


def _rpy_quat(text: str) -> str:
    r, p, y = (float(v) / 2.0 for v in text.split())
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(y),
        math.sin(y),
    )
    return _numbers(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


# Capsule axes / cylinder bounds in each URDF link frame, in metres.
# These are coarse contact proxies, not the original concave collision STLs.
_COLLISIONS = {
    "base_link": dict(type="box", pos="-0.045 0 0.034", size="0.092 0.055 0.034"),
    "L1": dict(type="capsule", fromto="0 -0.065 0 0 0.07 0", size="0.028"),
    "L2": dict(type="capsule", fromto="0 0 0 0.18 0 0", size="0.023"),
    "L3": dict(type="capsule", fromto="0 0 0 -0.0435 0.14 0", size="0.02"),
    "L4": dict(type="capsule", fromto="0 0 0.02 0 0 0.145", size="0.023"),
    "L5": dict(type="capsule", fromto="0 -0.02 0 0 0.03 0", size="0.022"),
    "L6": dict(type="sphere", pos="0 0 0", size="0.016"),
}


def build_model_xml(scene: dict) -> tuple[str, dict[str, bytes]]:
    """Return generated MJCF and binary assets; usable from an installed wheel."""
    urdf = ET.parse(ROBOT_DIR / "PAROL6.urdf").getroot()
    links = {element.attrib["name"]: element for element in urdf.findall("link")}
    joints = {element.find("child").attrib["link"]: element for element in urdf.findall("joint")}
    physics = scene.get("physics", DEFAULT_PHYSICS)
    servo = scene.get("actuators", DEFAULT_ACTUATORS)
    # Separate robot (1), free object (2), and static environment (4) categories
    # let one contact layer be disabled without disabling the others.
    robot_affinity = (2 if physics["robot_object_contacts_enabled"] else 0) | (
        4 if physics["robot_floor_contacts_enabled"] else 0
    )
    object_affinity = (
        (1 if physics["robot_object_contacts_enabled"] else 0)
        | (4 if physics["object_floor_contacts_enabled"] else 0)
        | (2 if physics["object_object_contacts_enabled"] else 0)
    )
    static_affinity = (1 if physics["robot_floor_contacts_enabled"] else 0) | (
        2 if physics["object_floor_contacts_enabled"] else 0
    )
    root = ET.Element("mujoco", model="humaned_parol6_lab")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", inertiafromgeom="false")
    ET.SubElement(
        root,
        "option",
        timestep=str(scene["timestep_s"]),
        gravity=_numbers(scene["gravity_m_s2"] if physics["gravity_enabled"] else [0, 0, 0]),
        integrator="implicitfast",
        iterations="60",
        cone="elliptic",
    )
    ET.SubElement(root, "size", njmax="3000", nconmax="500")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1920", offheight="1080")
    ET.SubElement(visual, "quality", shadowsize="2048")
    ET.SubElement(
        visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.7 0.7 0.7", specular="0.1 0.1 0.1"
    )
    default = ET.SubElement(root, "default")
    ET.SubElement(
        default,
        "joint",
        damping="0.06" if physics["joint_damping_enabled"] else "0",
        armature="0.008",
        limited="true",
        solreflimit="0.006 1",
    )
    ET.SubElement(
        default,
        "geom",
        friction="0.8 0.02 0.002" if physics["friction_enabled"] else "0 0 0",
        solref="0.008 1",
        solimp="0.95 0.99 0.001",
        condim="6",
    )
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        name="floor_checker",
        type="2d",
        builtin="checker",
        width="512",
        height="512",
        rgb1="0.19 0.23 0.30",
        rgb2="0.23 0.28 0.35",
    )
    ET.SubElement(
        asset,
        "material",
        name="floor_mat",
        texture="floor_checker",
        texrepeat="12 12",
        reflectance="0.08",
    )
    assets: dict[str, bytes] = {}
    for name in ("base_link", *JOINT_NAMES):
        filename = links[name].find("visual/geometry/mesh").attrib["filename"].split("/")[-1]
        ET.SubElement(asset, "mesh", name=f"mesh_{name}", file=filename)
        assets[filename] = (ROBOT_DIR / "meshes" / filename).read_bytes()
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0.1 -0.5 1.3", dir="0 0 -1", directional="true")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="1.5 1.5 0.1",
        material="floor_mat",
        contype="4",
        conaffinity=str(static_affinity),
    )
    ET.SubElement(
        world,
        "camera",
        name="overview",
        pos="0.90 -0.90 0.72",
        xyaxes="0.7071 0.7071 0 -0.34 0.34 0.876",
    )

    def add_link(parent: ET.Element, name: str) -> ET.Element:
        link, joint = links[name], joints[name]
        origin = joint.find("origin")
        body = ET.SubElement(
            parent,
            "body",
            name=name,
            pos=origin.attrib.get("xyz", "0 0 0"),
            quat=_rpy_quat(origin.attrib.get("rpy", "0 0 0")),
        )
        inertia = link.find("inertial/inertia")
        inertial_origin = link.find("inertial/origin")
        # Upstream inertials use rpy=0: preserve all six tensor coefficients.
        if inertial_origin.attrib.get("rpy", "0 0 0") != "0 0 0":
            raise ValueError("This converter requires inertial tensors in the link frame")
        ET.SubElement(
            body,
            "inertial",
            pos=inertial_origin.attrib["xyz"],
            mass=link.find("inertial/mass").attrib["value"],
            fullinertia=" ".join(
                inertia.attrib[key] for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
            ),
        )
        if joint.attrib["type"] == "revolute":
            limit = joint.find("limit")
            ET.SubElement(
                body,
                "joint",
                name=name,
                type="hinge",
                axis=joint.find("axis").attrib["xyz"],
                range=f"{limit.attrib['lower']} {limit.attrib['upper']}",
            )
        vis_origin = link.find("visual/origin")
        ET.SubElement(
            body,
            "geom",
            name=f"visual_{name}",
            type="mesh",
            mesh=f"mesh_{name}",
            pos=vis_origin.attrib["xyz"],
            quat=_rpy_quat(vis_origin.attrib["rpy"]),
            rgba="0.76 0.79 0.84 1" if name != "base_link" else "0.28 0.33 0.42 1",
            contype="0",
            conaffinity="0",
            group="2",
        )
        ET.SubElement(
            body,
            "geom",
            name=f"collision_{name}",
            **_COLLISIONS[name],
            rgba="0.1 0.7 0.9 0.35",
            contype="1",
            conaffinity=str(robot_affinity),
            group="3",
        )
        if name == "L6":
            ET.SubElement(body, "site", name="tcp", pos="0 0 0", size="0.008", rgba="0.1 0.9 0.7 1")
        for child_name, child_joint in joints.items():
            if child_joint.find("parent").attrib["link"] == name:
                add_link(body, child_name)
        return body

    add_link(world, "base_link")
    for cube in scene["cubes"]:
        body = ET.SubElement(
            world,
            "body",
            name=cube["name"],
            pos=_numbers(cube["position_m"]),
            quat=_numbers(cube["quaternion_wxyz"]),
        )
        ET.SubElement(body, "freejoint", name=f"free_{cube['name']}")
        half_size = np.asarray(cube["size_m"]) / 2.0
        mass = float(cube["mass_kg"])
        diagonal = (
            mass
            * (np.sum(np.asarray(cube["size_m"]) ** 2) - np.asarray(cube["size_m"]) ** 2)
            / 12.0
        )
        # Approximation: retain uniform-box principal moments about the displaced
        # local COM. This does not reconstruct a measured nonuniform density.
        ET.SubElement(
            body,
            "inertial",
            pos=_numbers(cube.get("com_offset_m", [0, 0, 0])),
            mass=str(mass),
            diaginertia=_numbers(diagonal),
        )
        ET.SubElement(
            body,
            "geom",
            name=f"geom_{cube['name']}",
            type="box",
            size=_numbers(half_size),
            rgba=_numbers(cube["rgba"]),
            contype="2",
            conaffinity=str(object_affinity),
            friction=_numbers(cube["friction"] if physics["friction_enabled"] else [0, 0, 0]),
            solref=_numbers(restitution_solref(cube.get("restitution"))),
            priority="1",
        )
    for obstacle in scene["obstacles"]:
        ET.SubElement(
            world,
            "geom",
            name=obstacle["name"],
            type="box",
            pos=_numbers(obstacle["position_m"]),
            size=_numbers(np.asarray(obstacle["size_m"]) / 2),
            rgba=_numbers(obstacle["rgba"]),
            contype="4",
            conaffinity=str(static_affinity),
        )
    goal = scene["goal_m"]
    ET.SubElement(
        world,
        "site",
        name="goal",
        pos=_numbers(goal),
        size="0.014",
        type="sphere",
        rgba="0.3 0.95 0.45 0.45",
    )
    actuators = ET.SubElement(root, "actuator")
    for i, name in enumerate(JOINT_NAMES):
        limit = joints[name].find("limit")
        ET.SubElement(
            actuators,
            "position",
            name=f"servo_{name}",
            joint=name,
            kp=str(servo["kp"][i] if physics["actuation_enabled"] else 0),
            kv=str(servo["kv"][i] if physics["actuation_enabled"] else 0),
            ctrlrange=f"{limit.attrib['lower']} {limit.attrib['upper']}",
            forcerange=f"-{servo['torque_limits_nm'][i]} {servo['torque_limits_nm'][i]}",
        )
    return ET.tostring(root, encoding="unicode"), assets
