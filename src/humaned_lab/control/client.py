"""Small reusable client for policies, notebooks, and scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import httpx


class SimulationClient:
    def __init__(self, url: str = "http://127.0.0.1:8765", timeout: float = 10.0):
        self.http = httpx.Client(base_url=url.rstrip("/"), timeout=timeout)

    def _request(self, method: str, path: str, **kwargs) -> dict:
        response = self.http.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()

    def state(self) -> dict:
        return self._request("GET", "/api/state")

    def reset(self, seed: int = 0) -> dict:
        return self._request("POST", "/api/reset", json={"seed": seed})

    def joints(self, q_rad: Sequence[float]) -> dict:
        return self._request("POST", "/api/joints", json={"q_rad": list(q_rad)})

    def step(self, action: Sequence[float]) -> dict:
        """Apply six [-1,1] joint deltas and advance exactly 50ms, in paused mode."""
        return self._request("POST", "/api/step", json={"action": list(action)})

    def running(self, enabled: bool) -> dict:
        return self._request("POST", "/api/running", json={"running": enabled})

    def cube(self, name: str, position_m: Sequence[float], velocity_m_s=None) -> dict:
        payload = {"position_m": list(position_m)}
        if velocity_m_s is not None:
            payload["velocity_m_s"] = list(velocity_m_s)
        return self._request("POST", f"/api/cubes/{name}", json=payload)

    def tcp(self, position_m: Sequence[float], duration_s: float = 2, **options) -> dict:
        """Position-only IK and smooth joint interpolation, driven by real servos."""
        return self._request(
            "POST",
            "/api/tcp",
            json={"position_m": list(position_m), "duration_s": duration_s, **options},
        )

    def preview_tcp(self, position_m: Sequence[float]) -> dict:
        return self._request("POST", "/api/tcp/preview", json={"position_m": list(position_m)})

    def trajectory(
        self,
        waypoints_m: Sequence[Sequence[float]],
        segment_duration_s: float = 2,
        loop: bool = False,
        **options,
    ) -> dict:
        return self._request(
            "POST",
            "/api/trajectory",
            json={
                "waypoints_m": [list(point) for point in waypoints_m],
                "segment_duration_s": segment_duration_s,
                "loop": loop,
                **options,
            },
        )

    def stop_trajectory(self) -> dict:
        return self._request("POST", "/api/trajectory/stop")

    def jog(self, joint_index: int, velocity_rad_s: float, timeout_s: float = 0.3) -> dict:
        """Refresh every <=100ms while jogging; watchdog stops at <=350ms."""
        return self._request(
            "POST",
            "/api/jog",
            json={
                "joint_index": joint_index,
                "velocity_rad_s": velocity_rad_s,
                "timeout_s": timeout_s,
            },
        )

    def stop_jog(self) -> dict:
        return self._request("POST", "/api/jog/stop")

    def cube_properties(self, name: str, **properties) -> dict:
        return self._request("POST", f"/api/cubes/{name}/properties", json=properties)

    def physics(self, **settings) -> dict:
        return self._request("POST", "/api/physics", json=settings)

    def actuators(self, **settings) -> dict:
        return self._request("POST", "/api/actuators", json=settings)

    def viewer(self, **settings) -> dict:
        return self._request("POST", "/api/viewer", json=settings)

    def scene(self) -> dict:
        return self._request("GET", "/api/scene")

    def save_scene(self, path: str | Path) -> Path:
        """Export resolved initial/configured scene, not a live state checkpoint."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.scene(), indent=2) + "\n", encoding="utf-8")
        return destination

    def close(self):
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
