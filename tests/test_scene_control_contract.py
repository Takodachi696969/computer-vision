"""Scene changes must preserve the server's documented policy timing."""

import json

import pytest
from fastapi.testclient import TestClient

from humaned_lab.control.server import Engine, create_app
from humaned_lab.simulation import PhysicsWorld


def test_api_rejects_scene_timestep_that_cannot_make_50ms(tmp_path):
    with PhysicsWorld() as world:
        scene = world.scene
    scene["timestep_s"] = 0.003
    path = tmp_path / "custom.json"
    path.write_text(json.dumps(scene))
    # The standalone physical integrator accepts this dt; the HTTP policy
    # interface must refuse it rather than silently advance 51ms per action.
    with PhysicsWorld(path) as world:
        assert world.timestep == pytest.approx(0.003)
    with pytest.raises(ValueError, match="divide.*control interval"):
        Engine(path, render=False)


def test_custom_scene_50ms_step_without_cube_bodies(tmp_path):
    with PhysicsWorld() as world:
        scene = world.scene
    scene["timestep_s"] = 0.005
    scene["cubes"] = []
    path = tmp_path / "empty-world.json"
    path.write_text(json.dumps(scene))
    with TestClient(create_app(path, render=False)) as client:
        client.post("/api/running", json={"running": False}).raise_for_status()
        before = client.post("/api/reset", json={}).json()
        result = client.post("/api/step", json={"action": [0.2, 0, 0, 0, 0, 0]})
        result.raise_for_status()
        after = result.json()
        assert after["cubes"] == []
        assert after["time_s"] - before["time_s"] == pytest.approx(0.05, abs=1e-9)
        assert after["q_rad"][0] > before["q_rad"][0]
