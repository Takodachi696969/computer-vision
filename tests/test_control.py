"""Verify external policy control semantics, not just route wiring."""

import numpy as np
from fastapi.testclient import TestClient

from humaned_lab.control.server import create_app


def test_policy_step_advances_exactly_and_pauses():
    with TestClient(create_app(render=False)) as client:
        client.post("/api/running", json={"running": False}).raise_for_status()
        before = client.post("/api/reset", json={"seed": 4}).json()
        after = client.post("/api/step", json={"action": [0.5, 0, 0, 0, 0, 0]}).json()
        assert not after["running"]
        assert abs(after["time_s"] - before["time_s"] - 0.05) < 1e-8
        assert abs(after["joint_targets_rad"][0] - before["q_rad"][0] - 0.02) < 1e-8
        assert after["q_rad"][0] > before["q_rad"][0]
        assert np.isfinite(after["q_rad"]).all()


def test_invalid_commands_do_not_mutate_world():
    with TestClient(create_app(render=False)) as client:
        client.post("/api/running", json={"running": False})
        before = client.post("/api/reset", json={}).json()
        assert client.post("/api/step", json={"action": [2, 0, 0, 0, 0, 0]}).status_code == 422
        assert client.post("/api/joints", json={"q_rad": [99] * 6}).status_code == 422
        assert client.post("/api/cubes/missing", json={"position_m": [0, 0, 1]}).status_code == 422
        assert client.post("/api/joints", json={"q_rad": [0] * 5}).status_code == 422
        after = client.get("/api/state").json()
        assert after["joint_targets_rad"] == before["joint_targets_rad"]
        assert after["time_s"] == before["time_s"]


def test_cube_velocity_and_browser_origin_boundary():
    with TestClient(create_app(render=False)) as client:
        client.post("/api/running", json={"running": False})
        response = client.post(
            "/api/cubes/target", json={"position_m": [0.2, 0.1, 0.5], "velocity_m_s": [0.2, 0, 0]}
        )
        response.raise_for_status()
        cube = next(c for c in response.json()["cubes"] if c["name"] == "target")
        assert cube["velocity_m_s"][0] == 0.2
        assert (
            client.post(
                "/api/reset", json={}, headers={"Origin": "https://unrelated.example"}
            ).status_code
            == 403
        )
        assert client.get("/health").json()["ok"]
