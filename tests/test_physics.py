"""Physics checks independent of the HTTP server and training algorithm."""

import json
from xml.etree import ElementTree as ET

import numpy as np
import pytest
import mujoco

from humaned_lab.simulation import PhysicsWorld
from humaned_lab.simulation.model import HOME_Q_RAD, ROBOT_DIR


@pytest.fixture
def world():
    with PhysicsWorld() as value:
        yield value


def test_gravity_and_floor_contact(world):
    world.set_cube("target", [0.7, 0.7, 0.4])
    world.step(50)
    cube = next(c for c in world.state()["cubes"] if c["name"] == "target")
    assert cube["position_m"][2] == pytest.approx(0.4 - 0.5 * 9.81 * 0.1**2, abs=0.002)
    assert cube["velocity_m_s"][2] == pytest.approx(-0.981, abs=0.01)
    world.step(1000)
    cube = next(c for c in world.state()["cubes"] if c["name"] == "target")
    assert cube["position_m"][2] == pytest.approx(0.02, abs=0.001)
    assert np.linalg.norm(cube["velocity_m_s"]) < 1e-3
    assert world.state()["contact_count"] > 0


def test_cube_launch_moves_and_reset_restores_scene(world):
    original = world.state()
    world.set_cube("target", [0.7, 0.7, 0.4], [0.4, 0, 0])
    world.step(20)
    cube = next(c for c in world.state()["cubes"] if c["name"] == "target")
    assert cube["position_m"][0] > 0.71
    reset = world.reset(seed=42)
    assert reset["time_s"] == 0
    assert reset["q_rad"] == original["q_rad"]
    assert reset["cubes"] == original["cubes"]


def test_cube_contact_with_robot_is_physical(world):
    # Slight overlap at the bare flange must produce contact and an impulse;
    # the CAD mesh alone would not produce either result.
    world.set_cube("target", world.tcp_position() + [0, 0, 0.025])
    world.step(1)
    contacts = []
    for contact in world.data.contact:
        names = [
            mujoco.mj_id2name(world.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom))
            for geom in contact.geom
        ]
        contacts.append(names)
    assert any(
        "geom_target" in pair and any(name.startswith("collision_") for name in pair)
        for pair in contacts
    )
    cube = next(c for c in world.state()["cubes"] if c["name"] == "target")
    assert cube["velocity_m_s"][2] > 0  # contact impulse opposes penetration


def test_scene_cube_friction_changes_sliding_on_floor(world, tmp_path):
    # Without an explicit priority the floor's default friction would override
    # a lower per-cube coefficient under MuJoCo's maximum-combination rule.
    positions = []
    for coefficient in (0.8, 0.0):
        scene = json.loads(json.dumps(world.scene))
        scene["cubes"][0]["friction"] = [coefficient, 0.02, 0.002]
        path = tmp_path / f"friction-{coefficient}.json"
        path.write_text(json.dumps(scene))
        with PhysicsWorld(path) as candidate:
            candidate.set_cube("target", [0.7, 0.7, 0.02], [0.4, 0, 0])
            candidate.step(300)
            cube = next(c for c in candidate.state()["cubes"] if c["name"] == "target")
            positions.append(cube["position_m"][0])
    assert positions[1] - positions[0] > 0.15


def test_joint_targets_move_and_reject_invalid_inputs(world):
    target = HOME_Q_RAD.copy()
    target[0] -= 0.3
    world.set_joint_targets(target)
    world.step(1000)
    actual = np.asarray(world.state()["q_rad"])
    assert actual[0] == pytest.approx(target[0], abs=0.012)
    assert actual[0] < HOME_Q_RAD[0] - 0.25
    assert np.all(actual >= world.joint_limits[:, 0] - 0.01)
    assert np.all(actual <= world.joint_limits[:, 1] + 0.01)
    with pytest.raises(ValueError):
        world.set_joint_targets([0] * 6)  # invalid L2/L3 native angles
    with pytest.raises(ValueError):
        world.set_joint_targets([np.nan] * 6)
    with pytest.raises(ValueError):
        world.set_cube("unknown", [0, 0, 0])


def test_native_upstream_degree_endpoints_survive_urdf_rounding(world):
    # PAROL6_ROBOT.py native limits are in degrees; its URDF rounded radians
    # otherwise rejected real telemetry by only a few nanoradians.
    native_deg = np.array(
        [
            [-123.046875, 123.046875],
            [-145.0088, -3.375],
            [107.866, 287.8675],
            [-105.46975, 105.46975],
            [-90, 90],
            [0, 360],
        ]
    )
    for endpoint in (0, 1):
        world.set_joint_targets(np.deg2rad(native_deg[:, endpoint]))
        np.testing.assert_allclose(
            world.joint_targets_rad, world.joint_limits[:, endpoint], atol=1e-8
        )
    outside = world.joint_limits[:, 1] + 1e-4
    with pytest.raises(ValueError):
        world.set_joint_targets(outside)


def _rpy_matrix(rpy):
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def test_forward_kinematics_preserves_urdf_convention(world):
    urdf = ET.parse(ROBOT_DIR / "PAROL6.urdf").getroot()
    by_child = {j.find("child").attrib["link"]: j for j in urdf.findall("joint")}
    # Independent homogeneous transforms detect frame/offset/sign errors.
    q = HOME_Q_RAD + np.array([-0.2, 0.1, -0.1, 0.2, -0.2, 0.1])
    transform = np.eye(4)
    for i, name in enumerate(["base_link", *world.joint_names]):
        origin = by_child[name].find("origin")
        fixed = np.eye(4)
        fixed[:3, 3] = np.fromstring(origin.attrib["xyz"], sep=" ")
        fixed[:3, :3] = _rpy_matrix(np.fromstring(origin.attrib["rpy"], sep=" "))
        motion = np.eye(4)
        if i:
            motion[:3, :3] = _rpy_matrix([0, 0, q[i - 1]])
        transform = transform @ fixed @ motion
    np.testing.assert_allclose(world.forward_kinematics(q), transform[:3, 3], atol=1e-10)


def test_position_ik_keeps_physical_state_unchanged(world):
    goal = [0.22, 0.18, 0.08]
    before = world.state()
    q = world.solve_ik(goal)
    assert np.linalg.norm(world.forward_kinematics(q) - goal) < 0.002
    assert world.state() == before
    world.set_joint_targets(q)
    world.step(1500)
    assert np.linalg.norm(world.tcp_position() - goal) < 0.015


def test_scene_save_roundtrip_and_validation(world, tmp_path):
    saved = world.save_scene(tmp_path / "scene.json")
    with PhysicsWorld(saved) as replica:
        assert replica.state() == world.state()
    config = json.loads(saved.read_text())
    config["cubes"][0]["mass_kg"] = -1
    saved.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="mass_kg"):
        PhysicsWorld(saved)


def test_render_returns_actual_finite_rgb_image(world):
    world.step(100)
    pixels = world.render(480, 320)
    assert pixels.shape == (320, 480, 3)
    assert pixels.dtype == np.uint8
    assert np.isfinite(pixels).all()
    assert np.std(pixels) > 10  # scene is not blank
    before = world.state()
    proxies = world.render(480, 320, show_collision=True)
    assert not np.array_equal(pixels, proxies)
    assert world.state() == before
