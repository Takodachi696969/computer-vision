"""Small reusable client for policies, notebooks, and scripts."""

from __future__ import annotations

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

    def close(self):
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
