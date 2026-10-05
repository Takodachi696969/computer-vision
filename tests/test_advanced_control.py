"""Physical control, ownership, watchdog and streaming contracts."""

import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from humaned_lab.control.server import Engine, create_app


@pytest.fixture
def lab():
    app = create_app(render=False)
    with TestClient(app) as client:
        client.post("/api/running", json={"running": False}).raise_for_status()
        client.post("/api/reset", json={}).raise_for_status()
        yield client, app.state.engine


def advance_paused(engine, seconds):
    def advance():
        engine.running = False
        engine._advance(round(seconds / engine.world.timestep))
        return engine.state()

    return engine.call(advance)


def test_tcp_ik_is_not_a_position_teleport_and_reaches(lab):
    client, engine = lab
    before = client.get("/api/state").json()
    goal = (np.asarray(before["tcp_m"]) + [0.02, 0, 0.01]).tolist()
    preview = client.post("/api/tcp/preview", json={"position_m": goal})
    preview.raise_for_status()
    assert preview.json()["ik_residual_m"] <= 0.002
    np.testing.assert_array_equal(client.get("/api/state").json()["q_rad"], before["q_rad"])
    response = client.post(
        "/api/tcp", json={"position_m": goal, "duration_s": 0.3, "tolerance_m": 0.01}
    )
    response.raise_for_status()
    planned = response.json()
    np.testing.assert_array_equal(planned["q_rad"], before["q_rad"])
    assert not planned["trajectory"]["reached"]
    client.post("/api/running", json={"running": False})
    for _ in range(60):
        state = advance_paused(engine, 0.05)
        if not state["trajectory"]["active"]:
            break
    assert state["trajectory"]["status"] == "reached"
    assert state["trajectory"]["reached"]
    assert np.linalg.norm(np.asarray(state["tcp_m"]) - goal) <= 0.01
    assert not np.array_equal(state["q_rad"], before["q_rad"])


def test_unreachable_sequence_is_transactional(lab):
    client, _ = lab
    start = client.get("/api/state").json()["tcp_m"]
    response = client.post("/api/tcp", json={"position_m": start, "duration_s": 10})
    response.raise_for_status()
    client.post("/api/running", json={"running": False})
    before = client.get("/api/state").json()
    invalid = client.post("/api/trajectory", json={"waypoints_m": [start, [99, 99, 99]]})
    assert invalid.status_code == 422
    assert "Waypoint 2" in invalid.json()["detail"]
    after = client.get("/api/state").json()
    assert after["trajectory"] == before["trajectory"]
    assert after["joint_targets_rad"] == before["joint_targets_rad"]
    assert after["time_s"] == before["time_s"]


def test_waypoint_completion_uses_measured_tip_and_stall_timeout(lab):
    client, engine = lab
    client.post(
        "/api/physics",
        json={
            "gravity_enabled": False,
            "actuation_enabled": False,
            "robot_object_contacts_enabled": False,
            "robot_floor_contacts_enabled": False,
        },
    ).raise_for_status()
    goal = (np.asarray(client.get("/api/state").json()["tcp_m"]) + [0.04, 0, 0]).tolist()
    response = client.post(
        "/api/tcp",
        json={"position_m": goal, "duration_s": 0.1, "settle_timeout_s": 0.2, "tolerance_m": 0.001},
    )
    response.raise_for_status()
    client.post("/api/running", json={"running": False})
    state = advance_paused(engine, 0.15)
    assert state["trajectory"]["planned_complete"]
    assert not state["trajectory"]["reached"]
    assert state["trajectory"]["active"]
    state = advance_paused(engine, 0.3)
    assert state["trajectory"]["stalled"]
    assert state["trajectory"]["status"] == "stalled"
    assert not state["trajectory"]["reached"]


def test_sequence_visits_each_waypoint_and_loop_can_stop(lab):
    client, engine = lab
    tcp = np.asarray(client.get("/api/state").json()["tcp_m"])
    points = [(tcp + [0.015, 0, 0.01]).tolist(), (tcp + [0, 0.015, 0.01]).tolist()]
    response = client.post(
        "/api/trajectory",
        json={"waypoints_m": points, "segment_duration_s": 0.3, "loop": True, "tolerance_m": 0.015},
    )
    response.raise_for_status()
    client.post("/api/running", json={"running": False})
    for _ in range(100):
        state = advance_paused(engine, 0.05)
        if state["trajectory"]["loop_count"]:
            break
    assert state["trajectory"]["loop_count"] >= 1
    assert state["trajectory"]["completed_waypoints"] >= 2
    assert state["trajectory"]["active"]
    stop = client.post("/api/trajectory/stop")
    stop.raise_for_status()
    assert not stop.json()["trajectory"]["active"]
    np.testing.assert_allclose(stop.json()["joint_targets_rad"], stop.json()["q_rad"], atol=1e-8)


def test_jog_slew_watchdog_and_policy_cancellation(lab):
    client, engine = lab
    before = client.get("/api/state").json()
    started = client.post(
        "/api/jog", json={"joint_index": 0, "velocity_rad_s": 0.2, "timeout_s": 0.1}
    )
    started.raise_for_status()
    assert started.json()["jog"]["active"]
    time.sleep(0.17)
    stopped = client.get("/api/state").json()
    assert not stopped["jog"]["active"]
    assert stopped["jog"]["status"] == "timeout"
    assert stopped["q_rad"][0] > before["q_rad"][0]
    assert stopped["q_rad"][0] - before["q_rad"][0] < 0.05
    client.post("/api/jog", json={"joint_index": 0, "velocity_rad_s": 0.2})
    policy = client.post("/api/step", json={"action": [0] * 6})
    policy.raise_for_status()
    assert not policy.json()["jog"]["active"]
    assert not policy.json()["running"]
    assert (
        client.post("/api/jog", json={"joint_index": 6, "velocity_rad_s": 0.2}).status_code == 422
    )
    assert (
        client.post(
            "/api/jog", json={"joint_index": 0, "velocity_rad_s": 0.2, "timeout_s": 1}
        ).status_code
        == 422
    )


def test_force_guard_stops_on_measured_robot_contact(lab):
    client, engine = lab
    tip = client.get("/api/state").json()["tcp_m"]
    client.post("/api/cubes/target", json={"position_m": tip}).raise_for_status()
    response = client.post(
        "/api/tcp", json={"position_m": tip, "duration_s": 1, "force_limit_n": 1}
    )
    response.raise_for_status()
    client.post("/api/running", json={"running": False})
    state = advance_paused(engine, 0.02)
    assert state["trajectory"]["status"] == "force_limit"
    assert state["trajectory"]["force_limit_triggered"]
    assert state["trajectory"]["measured_contact_force_n"] > 1
    assert not state["trajectory"]["reached"]


def test_rebuild_configuration_runs_on_owner_thread_and_exports_scene(lab, monkeypatch):
    client, engine = lab
    original = engine.world.configure_cube
    observed = []

    def configured(name, properties):
        observed.append(threading.get_ident())
        return original(name, properties)

    monkeypatch.setattr(engine.world, "configure_cube", configured)
    before = client.get("/api/state").json()
    updated = client.post(
        "/api/cubes/target/properties",
        json={
            "mass_kg": 0.5,
            "size_m": [0.06, 0.04, 0.04],
            "com_offset_m": [0.01, 0, 0],
            "restitution": 0.3,
        },
    )
    updated.raise_for_status()
    after = updated.json()
    assert observed == [engine.thread.ident]
    assert after["time_s"] == before["time_s"]
    np.testing.assert_array_equal(after["q_rad"], before["q_rad"])
    cube = next(item for item in after["cubes"] if item["name"] == "target")
    assert cube["mass_kg"] == 0.5
    scene = client.get("/api/scene").json()
    assert next(item for item in scene["cubes"] if item["name"] == "target")["mass_kg"] == 0.5
    assert client.post("/api/cubes/target/properties", json={"mass_kg": -1}).status_code == 422
    assert client.post("/api/cubes/target/properties", json={"mass_kg": None}).status_code == 422
    assert client.post("/api/physics", json={"self_collision_enabled": True}).status_code == 422
    assert client.post("/api/actuators", json={"torque_limits_nm": 3}).status_code == 200
    assert (
        client.post("/api/actuators", json={"target_velocity_limits_rad_s": None}).status_code
        == 200
    )
    assert client.post("/api/physics", json={"gravity_enabled": False}).status_code == 200
    assert not client.get("/api/state").json()["physics_settings"]["gravity_enabled"]


def test_viewer_validation_and_only_new_frames(lab):
    client, engine = lab
    assert client.get("/api/stream.mjpg").status_code == 503
    assert client.post("/api/viewer", json={"target_fps": 1000}).status_code == 422
    configured = client.post(
        "/api/viewer",
        json={
            "target_fps": 45,
            "width": 640,
            "height": 480,
            "jpeg_quality": 75,
            "show_collisions": True,
        },
    )
    configured.raise_for_status()
    assert configured.json()["viewer"]["target_fps"] == 45
    with engine.frame_condition:
        engine.frame = b"new-frame"
        engine.frame_sequence = 7
        engine.frame_condition.notify_all()
    assert engine.wait_frame(6, timeout=0) == (7, b"new-frame")
    assert engine.wait_frame(7, timeout=0.01) is None


def test_renderer_metrics_and_closure_stay_on_engine_thread(monkeypatch):
    owners = []
    closed = []
    from humaned_lab.simulation.world import PhysicsWorld

    original_close = PhysicsWorld.close

    def fake_render(self, width, height, *, show_collision=False):
        owners.append(threading.get_ident())
        return np.zeros((height, width, 3), dtype=np.uint8)

    def close(self):
        closed.append(threading.get_ident())
        return original_close(self)

    monkeypatch.setattr(PhysicsWorld, "render", fake_render)
    monkeypatch.setattr(PhysicsWorld, "close", close)
    engine = Engine(render=True)
    engine.start()
    try:
        first = engine.wait_frame(0, timeout=2)
        assert first is not None
        time.sleep(0.22)
        state = engine.state()
        assert state["viewer"]["actual_fps"] > 20
        assert state["viewer"]["frame_sequence"] >= 4
        assert all(owner == engine.thread.ident for owner in owners)
    finally:
        engine.close()
    assert closed == [engine.thread.ident]
