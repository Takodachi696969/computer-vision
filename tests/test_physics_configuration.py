"""Verify live physical configuration, load response, and owner-thread rebuilds."""

import copy
import threading

import numpy as np
import pytest

from humaned_lab.simulation import PhysicsWorld


@pytest.fixture
def world():
    with PhysicsWorld() as instance:
        yield instance


def test_live_cube_rebuild_preserves_motion_controls_time_and_goal(world):
    world.set_cube("target", [0.7, 0.7, 0.4], [0.2, 0.1, 0])
    world.step(40)
    world.set_goal([0.2, 0.2, 0.3])
    qpos, qvel, ctrl = world.data.qpos.copy(), world.data.qvel.copy(), world.data.ctrl.copy()
    before = world.state()
    original_model = world.model
    after = world.configure_cube(
        "target",
        {
            "size_m": [0.06, 0.05, 0.04],
            "mass_kg": 2,
            "com_offset_m": [0.015, 0, 0],
            "friction": [0.3, 0.01, 0.001],
            "restitution": 0.5,
        },
    )
    assert world.model is not original_model
    np.testing.assert_array_equal(world.data.qpos, qpos)
    np.testing.assert_array_equal(world.data.qvel, qvel)
    np.testing.assert_array_equal(world.data.ctrl, ctrl)
    assert after["time_s"] == before["time_s"]
    assert after["goal_m"] == before["goal_m"]
    assert after["joint_targets_rad"] == before["joint_targets_rad"]
    assert after["physics_fingerprint"] != before["physics_fingerprint"]
    assert after["model_revision"] == before["model_revision"] + 1
    cube = next(c for c in after["cubes"] if c["name"] == "target")
    assert cube["mass_kg"] == 2
    assert cube["size_m"] == [0.06, 0.05, 0.04]
    np.testing.assert_allclose(world.model.body_ipos[world._cube_body_ids["target"]], [0.015, 0, 0])
    assert np.all(world.model.body_inertia[world._cube_body_ids["target"]] > 0)
    world.reset()
    assert world.state()["cubes"][0]["mass_kg"] == 2  # reset retains configured physical properties


@pytest.mark.parametrize(
    "method,args",
    [
        ("configure_cube", ("target", {"mass_kg": -1})),
        ("configure_cube", ("target", {"size_m": [0.02, 0, 0.02]})),
        ("configure_cube", ("target", {"com_offset_m": [0.1, 0, 0]})),
        ("configure_cube", ("target", {"restitution": 1.1})),
        ("configure_cube", ("target", {"friction": [-0.1, 0, 0]})),
        ("configure_physics", ({"gravity_enabled": "false"},)),
        ("configure_physics", ({"self_collision_enabled": True},)),
        ("configure_actuators", ({"kp": np.nan},)),
        ("configure_actuators", ({"torque_limits_nm": 0},)),
        ("configure_actuators", ({"target_velocity_limits_rad_s": -1},)),
    ],
)
def test_invalid_live_configuration_is_transactional(world, method, args):
    world.step(4)
    before, scene, model = world.state(), copy.deepcopy(world.scene), world.model
    with pytest.raises(ValueError):
        getattr(world, method)(*args)
    assert world.model is model
    assert world.scene == scene
    assert world.state() == before


def test_rebuild_rejects_a_different_physics_owner_thread(world):
    world.step(1)
    model = world.model
    errors = []

    def configure_elsewhere():
        try:
            world.configure_cube("target", {"mass_kg": 1})
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=configure_elsewhere)
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
    assert "owning physics/render thread" in str(errors[0])
    assert world.model is model


def test_independent_gravity_and_floor_contact_layers(world):
    world.configure_physics({"gravity_enabled": False})
    world.set_cube("target", [0.7, 0.7, 0.4], [0.2, 0, 0])
    world.step(100)
    cube = world.state()["cubes"][0]
    assert cube["position_m"][2] == pytest.approx(0.4, abs=1e-9)
    assert cube["position_m"][0] == pytest.approx(0.74, abs=1e-8)
    world.configure_physics({"gravity_enabled": True, "object_floor_contacts_enabled": False})
    world.step(200)
    assert world.state()["cubes"][0]["position_m"][2] < -0.3
    assert world.physics_settings["robot_object_contacts_enabled"]


def test_robot_object_contact_layer_removes_contact_impulse(world):
    world.configure_physics({"robot_object_contacts_enabled": False})
    world.set_cube("target", world.tcp_position() + [0, 0, 0.025])
    state = world.step()
    assert state["robot_contact_force_n"] == 0
    assert state["cubes"][0]["velocity_m_s"][2] < 0
    assert not any(
        "geom_target" in (c["geom1"], c["geom2"]) and c["robot_involved"] for c in state["contacts"]
    )


def test_local_com_position_and_resting_force_units(world):
    world.configure_cube("target", {"mass_kg": 2.0, "com_offset_m": [0.008, 0, 0]})
    world.set_cube_pose("target", [0.7, 0.7, 0.02], [np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
    cube = world.state()["cubes"][0]
    np.testing.assert_allclose(cube["com_position_m"], [0.7, 0.708, 0.02], atol=1e-10)
    world.step(1000)
    cube = world.state()["cubes"][0]
    assert cube["contact_force_n"] == pytest.approx(2 * 9.81, rel=0.01)
    assert cube["net_contact_force_world_n"][2] == pytest.approx(2 * 9.81, rel=0.01)


def test_requested_restitution_changes_empirical_rebound_height():
    heights = []
    for requested in (0, 0.2, 0.8):
        with PhysicsWorld() as world:
            world.configure_cube("target", {"restitution": requested})
            world.set_cube("target", [0.7, 0.7, 0.4])
            qa, _ = world._cubes["target"]
            geom = world._cube_geom_ids["target"]
            touched, peak = False, 0.02
            for _ in range(500):
                world.step()
                touched = touched or any(geom in c.geom for c in world.data.contact)
                if touched:
                    peak = max(peak, float(world.data.qpos[qa + 2]))
            heights.append(peak)
    assert heights[0] < 0.025
    assert heights[1] > heights[0] + 0.01
    assert heights[2] > heights[1] + 0.1
    assert heights[2] < 0.45  # no spurious energy gain from this calibrated drop scenario


def test_torque_caps_create_loaded_tracking_error_and_saturation():
    errors = []
    for cap in (300, 0.001):
        with PhysicsWorld() as world:
            world.configure_physics({"gravity_enabled": False})
            world.configure_actuators({"torque_limits_nm": cap})
            target = np.asarray(world.state()["q_rad"])
            target[0] -= 0.4
            world.set_joint_targets(target)
            state = world.step(500)
            errors.append(abs(state["joint_target_error_rad"][0]))
            assert np.max(np.abs(state["joint_measured_torque_nm"])) <= cap + 1e-8
            if cap < 1:
                assert state["joint_torque_saturated"][0]
                assert state["joint_load_limited"][0]
    assert errors[0] < 0.01
    assert errors[1] > 0.3


def test_actual_heavy_cube_contact_deflects_arm_more_than_light_cube():
    deflections, peak_forces = [], []
    for mass in (0.01, 5):
        with PhysicsWorld() as world:
            world.configure_actuators({"torque_limits_nm": [12, 12, 8, 1, 1, 1]})
            world.configure_cube("target", {"mass_kg": mass, "size_m": [0.04] * 3})
            initial_tcp = world.tcp_position().copy()
            world.set_cube("target", initial_tcp + [0, 0, 0.038])
            peak = 0.0
            for _ in range(150):
                state = world.step()
                peak = max(peak, state["robot_contact_force_n"])
            deflections.append(float(np.linalg.norm(world.tcp_position() - initial_tcp)))
            peak_forces.append(peak)
    assert peak_forces[1] > 5 * peak_forces[0] > 0
    assert deflections[1] > deflections[0] + 0.004


def test_velocity_slew_keeps_requested_and_applied_targets_separate(world):
    world.configure_physics({"gravity_enabled": False})
    world.configure_actuators({"target_velocity_limits_rad_s": 0.2})
    before = world.data.ctrl.copy()
    requested = before.copy()
    requested[0] -= 0.4
    world.set_joint_targets(requested)
    np.testing.assert_array_equal(world.data.ctrl, before)
    state = world.step(100)
    assert state["joint_targets_rad"][0] == requested[0]
    assert state["actuator_targets_rad"][0] == pytest.approx(before[0] - 0.04, abs=1e-10)


def test_friction_damping_and_actuation_layers_have_distinct_effects(world):
    world.configure_physics(
        {"friction_enabled": False, "joint_damping_enabled": False, "actuation_enabled": False}
    )
    assert np.all(world.model.geom_friction == 0)
    assert np.all(world.model.dof_damping[world.joint_dof_addresses] == 0)
    world.step(20)
    np.testing.assert_allclose(world.state()["joint_measured_torque_nm"], 0)
    world.configure_physics({"actuation_enabled": True})
    world.step()
    assert np.any(np.abs(world.state()["joint_measured_torque_nm"]) > 0.001)
    # Passive joint damping remains off; restoring actuation restores servo kv.
    assert np.all(world.model.dof_damping[world.joint_dof_addresses] == 0)


@pytest.mark.render
def test_renderer_recreated_after_geometry_change_on_same_thread(world):
    before = world.render(480, 320)
    world.configure_cube("target", {"size_m": 0.10, "rgba": [0.9, 0.1, 0.2, 1]})
    assert world._renderer is None
    after = world.render(480, 320)
    assert after.shape == before.shape
    assert np.isfinite(after).all()
    assert not np.array_equal(before, after)
