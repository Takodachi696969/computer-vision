"""Loopback physics service. One thread owns simulation rendering contexts."""

from __future__ import annotations

import io
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from humaned_lab.simulation.world import PhysicsWorld

logger = logging.getLogger(__name__)
ACTION_SCALE_RAD = 0.04
CONTROL_DT_S = 0.05


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class JointTarget(Payload):
    q_rad: list[float] = Field(min_length=6, max_length=6)


class PolicyAction(Payload):
    action: list[float] = Field(min_length=6, max_length=6)


class Reset(Payload):
    seed: int = Field(default=0, ge=0)


class Running(Payload):
    running: bool


class CubeTarget(Payload):
    position_m: list[float] = Field(min_length=3, max_length=3)
    velocity_m_s: list[float] | None = Field(default=None, min_length=3, max_length=3)


class Engine:
    def __init__(self, scene_path=None, render: bool = True):
        self.world = PhysicsWorld(scene_path)
        if not np.isclose(
            round(CONTROL_DT_S / self.world.timestep) * self.world.timestep,
            CONTROL_DT_S,
            atol=1e-8,
            rtol=0,
        ):
            self.world.close()
            raise ValueError("Scene timestep must divide the 0.05 s policy control interval")
        self.running = True
        self.render_enabled = render
        self.frame: bytes | None = None
        self.render_error: str | None = None
        self.physics_error: str | None = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="humaned-physics", daemon=True)

    def start(self):
        self.thread.start()

    def _loop(self):
        render_at = 0.0
        try:
            while not self.stop_event.is_set():
                started = time.perf_counter()
                with self.world.lock:
                    if self.running:
                        self.world.step(max(1, round(0.02 / self.world.timestep)))
                    if self.render_enabled and started >= render_at:
                        try:
                            rgb = self.world.render(960, 640)
                            buffer = io.BytesIO()
                            Image.fromarray(rgb).save(buffer, format="JPEG", quality=88)
                            self.frame = buffer.getvalue()
                            self.render_error = None
                        except Exception as exc:
                            self.render_error = str(exc)
                            self.render_enabled = False
                            logger.exception("Rendering failed; physics and API remain available")
                        render_at = started + 0.1
                self.stop_event.wait(max(0.0, 0.02 - (time.perf_counter() - started)))
        except Exception as exc:
            self.physics_error = str(exc)
            logger.exception("Physics loop failed")
        finally:
            # The renderer is closed on its creating thread (OpenGL requirement).
            self.world.close()

    def state(self):
        with self.world.lock:
            state = self.world.state()
            state.update(
                running=self.running,
                joint_limits_rad=self.world.joint_limits.tolist(),
                action_scale_rad=ACTION_SCALE_RAD,
                control_dt_s=CONTROL_DT_S,
                render_error=self.render_error,
                physics_error=self.physics_error,
                mode="physics",
            )
            return state

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError("Physics thread did not shut down")


def create_app(scene_path=None, render: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = Engine(scene_path, render=render)
        app.state.engine.start()
        try:
            yield
        finally:
            app.state.engine.close()

    app = FastAPI(title="HumanED PAROL6 Lab", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def same_origin_mutations(request: Request, call_next):
        # Reject unrelated websites issuing browser requests to this local service.
        origin = request.headers.get("origin")
        if request.method != "GET" and origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "Use the dashboard origin or a local HTTP client"}, 403)
        return await call_next(request)

    def engine() -> Engine:
        return app.state.engine

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health():
        e = engine()
        return {
            "ok": e.thread.is_alive() and e.physics_error is None,
            "mode": "physics",
            "port_purpose": "simulation and policy control",
            "render_error": e.render_error,
        }

    @app.get("/api/state")
    def state():
        return engine().state()

    @app.post("/api/reset")
    def reset(payload: Reset):
        e = engine()
        with e.world.lock:
            e.world.reset(seed=payload.seed)
            return e.state()

    @app.post("/api/running")
    def running(payload: Running):
        e = engine()
        with e.world.lock:
            e.running = payload.running
            return e.state()

    @app.post("/api/joints")
    def joints(payload: JointTarget):
        e = engine()
        with e.world.lock:
            try:
                e.world.set_joint_targets(payload.q_rad)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            return e.state()

    @app.post("/api/step")
    def step(payload: PolicyAction):
        action = np.asarray(payload.action, dtype=np.float64)
        if np.any(np.abs(action) > 1):
            raise HTTPException(422, "Actions must be six values in [-1,1]")
        e = engine()
        with e.world.lock:
            e.running = False
            q = np.asarray(e.world.state()["q_rad"])
            limits = e.world.joint_limits
            e.world.set_joint_targets(
                np.clip(q + action * ACTION_SCALE_RAD, limits[:, 0], limits[:, 1])
            )
            e.world.step(max(1, round(CONTROL_DT_S / e.world.timestep)))
            return e.state()

    @app.post("/api/cubes/{name}")
    def cube(name: str, payload: CubeTarget):
        e = engine()
        with e.world.lock:
            try:
                e.world.set_cube(name, payload.position_m, payload.velocity_m_s)
            except (ValueError, KeyError) as exc:
                raise HTTPException(422, str(exc)) from exc
            return e.state()

    @app.get("/api/frame.jpg")
    def frame():
        e = engine()
        if e.frame is None:
            raise HTTPException(503, e.render_error or "First frame is being rendered")
        return Response(e.frame, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    return app
