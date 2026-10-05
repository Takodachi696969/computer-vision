"""Loopback physics service; the engine thread owns rendering and mutations."""

from __future__ import annotations

import asyncio
import copy
import io
import logging
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future, TimeoutError
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class ViewerSettings(Payload):
    target_fps: int | None = Field(default=None, ge=1, le=60)
    width: int | None = Field(default=None, ge=160, le=1920)
    height: int | None = Field(default=None, ge=120, le=1080)
    jpeg_quality: int | None = Field(default=None, ge=30, le=95)
    show_collisions: bool | None = None


class TCPMove(Payload):
    position_m: list[float] = Field(min_length=3, max_length=3)
    duration_s: float = Field(default=2, ge=0.1, le=60)
    tolerance_m: float = Field(default=0.02, ge=0.001, le=0.1)
    settle_timeout_s: float = Field(default=3, ge=0.1, le=30)
    force_limit_n: float | None = Field(default=None, gt=0, le=10000)


class Trajectory(Payload):
    waypoints_m: list[list[float]] = Field(min_length=1, max_length=64)
    segment_duration_s: float = Field(default=2, ge=0.1, le=60)
    loop: bool = False
    tolerance_m: float = Field(default=0.02, ge=0.001, le=0.1)
    settle_timeout_s: float = Field(default=3, ge=0.1, le=30)
    force_limit_n: float | None = Field(default=None, gt=0, le=10000)

    @field_validator("waypoints_m")
    @classmethod
    def points_are_xyz(cls, value):
        if any(len(point) != 3 for point in value):
            raise ValueError("Each waypoint must contain exactly three metre coordinates")
        return value


class Jog(Payload):
    joint_index: int = Field(ge=0, le=5)
    velocity_rad_s: float = Field(ge=-1, le=1)
    timeout_s: float = Field(default=0.3, ge=0.05, le=0.35)


class CubeProperties(Payload):
    size_m: float | list[float] | None = None
    mass_kg: float | None = None
    com_offset_m: list[float] | None = None
    friction: list[float] | None = None
    restitution: float | None = None
    rgba: list[float] | None = None

    @field_validator("size_m", "mass_kg", "com_offset_m", "friction", "rgba")
    @classmethod
    def provided_properties_are_not_null(cls, value):
        if value is None:
            raise ValueError("Only restitution can be null; omit other unchanged properties")
        return value


class PhysicsSettings(Payload):
    gravity_enabled: bool | None = None
    robot_object_contacts_enabled: bool | None = None
    object_floor_contacts_enabled: bool | None = None
    object_object_contacts_enabled: bool | None = None
    robot_floor_contacts_enabled: bool | None = None
    friction_enabled: bool | None = None
    joint_damping_enabled: bool | None = None
    actuation_enabled: bool | None = None
    self_collision_enabled: bool | None = None


class ActuatorSettings(Payload):
    kp: float | list[float] | None = None
    kv: float | list[float] | None = None
    torque_limits_nm: float | list[float] | None = None
    target_velocity_limits_rad_s: float | list[float] | None = None


class Engine:
    """An owner thread advances elapsed wall time and produces fresh JPEGs.

    HTTP workers submit commands. Rebuilds never close an OpenGL renderer on
    another thread. Physics catch-up is capped at 100 ms per tick.
    """

    def __init__(self, scene_path=None, render: bool = True):
        self.world = PhysicsWorld(scene_path)
        self._validate_timestep()
        self.running = True
        self.render_enabled = render
        self.viewer = dict(
            target_fps=30, width=960, height=640, jpeg_quality=85, show_collisions=False
        )
        self.frame: bytes | None = None
        self.frame_sequence = 0
        self.frame_condition = threading.Condition()
        self.render_error: str | None = None
        self.physics_error: str | None = None
        self.render_ms = 0.0
        self.actual_fps = 0.0
        self.physics_realtime_factor = 0.0
        self.dropped_wall_time_s = 0.0
        self._frames = deque(maxlen=121)
        self._commands = queue.Queue()
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self._trajectory = None
        self._trajectory_state = self._idle_trajectory()
        self._jog = None
        self._jog_state = dict(active=False, joint_index=None, velocity_rad_s=0, status="idle")
        self.thread = threading.Thread(target=self._loop, name="humaned-physics", daemon=True)

    def _validate_timestep(self):
        if not np.isclose(
            round(CONTROL_DT_S / self.world.timestep) * self.world.timestep,
            CONTROL_DT_S,
            atol=1e-8,
            rtol=0,
        ):
            self.world.close()
            raise ValueError("Scene timestep must divide the 0.05 s policy control interval")

    @staticmethod
    def _idle_trajectory():
        return dict(
            active=False,
            status="idle",
            waypoint_index=0,
            waypoint_count=0,
            planned_fraction=0,
            planned_complete=False,
            reached=False,
            stalled=False,
            completed_waypoints=0,
            loop_count=0,
            distance_m=None,
            ik_residual_m=[],
            interpolation="smooth joint position targets; position-only TCP IK",
        )

    def start(self):
        self.thread.start()

    def call(self, operation, timeout=20):
        if threading.current_thread() is self.thread:
            return operation()
        if self.stop_event.is_set() or not self.thread.is_alive():
            raise RuntimeError(self.physics_error or "Physics engine is unavailable")
        future = Future()
        self._commands.put((future, operation))
        self.wake_event.set()
        try:
            return future.result(timeout=timeout)
        except TimeoutError as exc:
            future.cancel()
            raise RuntimeError("Physics command timed out") from exc

    def _process_commands(self):
        for _ in range(64):
            try:
                future, operation = self._commands.get_nowait()
            except queue.Empty:
                break
            if not future.set_running_or_notify_cancel():
                continue
            try:
                with self.world.lock:
                    future.set_result(operation())
            except Exception as exc:
                future.set_exception(exc)

    def _measured_q(self):
        return np.clip(
            np.asarray(self.world.state()["q_rad"]),
            self.world.joint_limits[:, 0],
            self.world.joint_limits[:, 1],
        )

    def stop_trajectory(self, reason="stopped", hold=True):
        if self._trajectory is not None:
            self._trajectory = None
            self._trajectory_state.update(active=False, status=reason)
            if hold:
                self.world.set_joint_targets(self._measured_q())

    def stop_jog(self, reason="stopped", hold=True):
        if self._jog is not None:
            self._jog = None
            self._jog_state.update(active=False, velocity_rad_s=0, status=reason)
            if hold:
                self.world.set_joint_targets(self._measured_q())

    def cancel_controls(self, reason, hold=False):
        self.stop_trajectory(reason, hold=hold)
        self.stop_jog(reason, hold=hold)

    def set_running(self, enabled):
        self.running = enabled
        if not enabled:
            self.stop_jog("paused")
        return self.state()

    def reset(self, seed):
        self.cancel_controls("reset")
        self.world.reset(seed)
        self._trajectory_state = self._idle_trajectory()
        return self.state()

    def set_joints(self, q_rad):
        self.world.set_joint_targets(q_rad)
        self.cancel_controls("joint_command")
        return self.state()

    def policy_step(self, action):
        action = np.asarray(action)
        if np.any(np.abs(action) > 1):
            raise ValueError("Actions must be six values in [-1,1]")
        self.cancel_controls("policy")
        self.running = False
        q = np.asarray(self.world.state()["q_rad"])
        limits = self.world.joint_limits
        self.world.set_joint_targets(
            np.clip(q + action * ACTION_SCALE_RAD, limits[:, 0], limits[:, 1])
        )
        self.world.step(round(CONTROL_DT_S / self.world.timestep))
        return self.state()

    def plan_waypoints(self, points):
        q = self._measured_q()
        joints, residuals = [], []
        for index, point in enumerate(points):
            try:
                q = self.world.inverse_kinematics(point, q_start=q)
            except ValueError as exc:
                raise ValueError(f"Waypoint {index + 1}: {exc}") from exc
            error = float(np.linalg.norm(self.world.forward_kinematics(q) - np.asarray(point)))
            joints.append(np.asarray(q).copy())
            residuals.append(error)
        return joints, residuals

    def begin_trajectory(self, payload):
        points = payload.waypoints_m
        joints, residuals = self.plan_waypoints(points)
        self.cancel_controls("trajectory")
        start_time = self.world.state()["time_s"]
        self._trajectory = dict(
            points=copy.deepcopy(points),
            joints=joints,
            start_q=self._measured_q(),
            start_time=start_time,
            duration=payload.segment_duration_s,
            tolerance=payload.tolerance_m,
            settle_timeout=payload.settle_timeout_s,
            loop=payload.loop,
            force_limit=payload.force_limit_n,
            index=0,
        )
        self._trajectory_state = dict(
            self._idle_trajectory(),
            active=True,
            status="running",
            waypoint_count=len(points),
            waypoints_m=copy.deepcopy(points),
            ik_residual_m=residuals,
            target_m=points[0],
            segment_duration_s=payload.segment_duration_s,
            tolerance_m=payload.tolerance_m,
            force_limit_n=payload.force_limit_n,
        )
        self.world.set_goal(points[0])
        self.running = True
        return self.state()

    def jog(self, payload):
        if payload.velocity_rad_s == 0:
            self.stop_jog()
            return self.state()
        if self._jog is None or self._jog["index"] != payload.joint_index:
            self.cancel_controls("jog")
            target = self._measured_q()
        else:
            target = self._jog["target"]
        self._jog = dict(
            index=payload.joint_index,
            velocity=payload.velocity_rad_s,
            deadline=time.perf_counter() + payload.timeout_s,
            target=target,
        )
        self._jog_state = dict(
            active=True,
            joint_index=payload.joint_index,
            velocity_rad_s=payload.velocity_rad_s,
            timeout_s=payload.timeout_s,
            status="running",
        )
        self.running = True
        return self.state()

    def _trajectory_targets(self, state):
        path = self._trajectory
        if path is None:
            return
        status = self._trajectory_state
        force = float(state.get("robot_contact_force_n", 0))
        if path["force_limit"] is not None and force > path["force_limit"]:
            status.update(force_limit_triggered=True, measured_contact_force_n=force)
            self.stop_trajectory("force_limit")
            return
        index = path["index"]
        target = np.asarray(path["points"][index])
        distance = float(np.linalg.norm(np.asarray(state["tcp_m"]) - target))
        elapsed = max(0, state["time_s"] - path["start_time"])
        fraction = min(1.0, elapsed / path["duration"])
        status.update(
            waypoint_index=index,
            target_m=target.tolist(),
            distance_m=distance,
            planned_fraction=fraction,
            planned_complete=fraction >= 1,
            status="running" if fraction < 1 else "settling",
        )
        blend = fraction * fraction * (3 - 2 * fraction)
        desired = path["start_q"] + blend * (path["joints"][index] - path["start_q"])
        self.world.set_joint_targets(desired)
        if fraction < 1:
            return
        if distance <= path["tolerance"]:
            status["completed_waypoints"] += 1
            if index + 1 == len(path["points"]):
                if not path["loop"]:
                    status.update(reached=True, active=False, status="reached")
                    self._trajectory = None
                    return
                status["loop_count"] += 1
                index = 0
            else:
                index += 1
            path.update(index=index, start_time=state["time_s"], start_q=self._measured_q())
            self.world.set_goal(path["points"][index])
            status.update(waypoint_index=index, planned_complete=False, planned_fraction=0)
        elif elapsed >= path["duration"] + path["settle_timeout"]:
            status.update(stalled=True, reached=False)
            self.stop_trajectory("stalled")

    def _advance(self, n):
        chunk = max(1, round(0.005 / self.world.timestep))
        while n > 0:
            count = min(chunk, n)
            state = self.world.state()
            if self._jog is not None:
                jog = self._jog
                target = jog["target"].copy()
                target[jog["index"]] += jog["velocity"] * count * self.world.timestep
                target = np.clip(
                    target, self.world.joint_limits[:, 0], self.world.joint_limits[:, 1]
                )
                jog["target"] = target
                self.world.set_joint_targets(target)
            else:
                self._trajectory_targets(state)
            self.world.step(count)
            n -= count
        if self._trajectory is not None:
            self._trajectory_targets(self.world.state())

    def configure(self, method, *args):
        getattr(self.world, method)(*args)
        self.cancel_controls("configuration_changed", hold=True)
        self._validate_timestep()
        return self.state()

    def configure_viewer(self, settings):
        self.viewer.update(settings)
        if settings:
            self.render_error = None
            self._frames.clear()
            self.actual_fps = 0
        return self.state()

    def _render_frame(self, started):
        try:
            rgb = self.world.render(
                self.viewer["width"],
                self.viewer["height"],
                show_collision=self.viewer["show_collisions"],
            )
            buffer = io.BytesIO()
            Image.fromarray(rgb).save(buffer, format="JPEG", quality=self.viewer["jpeg_quality"])
            self.render_ms = (time.perf_counter() - started) * 1000
            now = time.perf_counter()
            self._frames.append(now)
            if len(self._frames) > 1:
                self.actual_fps = (len(self._frames) - 1) / (self._frames[-1] - self._frames[0])
            with self.frame_condition:
                self.frame = buffer.getvalue()
                self.frame_sequence += 1
                self.frame_condition.notify_all()
            self.render_error = None
        except Exception as exc:
            self.render_error = str(exc)
            logger.exception("Rendering failed; physics and API remain available")

    def _loop(self):
        previous = time.perf_counter()
        render_at = previous
        accumulator = 0.0
        metric_at = previous
        metric_sim = self.world.state()["time_s"]
        try:
            while not self.stop_event.is_set():
                started = time.perf_counter()
                elapsed = max(0, started - previous)
                previous = started
                self._process_commands()
                with self.world.lock:
                    if self._jog is not None and started >= self._jog["deadline"]:
                        self.stop_jog("timeout")
                    if self.running:
                        accepted = min(elapsed, 0.25)
                        self.dropped_wall_time_s += elapsed - accepted
                        accumulator += accepted
                        available = min(
                            int(accumulator / self.world.timestep),
                            max(1, round(0.1 / self.world.timestep)),
                        )
                        if available:
                            self._advance(available)
                            accumulator -= available * self.world.timestep
                    else:
                        accumulator = 0
                    if started - metric_at >= 0.5:
                        sim = self.world.state()["time_s"]
                        self.physics_realtime_factor = max(0, sim - metric_sim) / (
                            started - metric_at
                        )
                        metric_at, metric_sim = started, sim
                    if self.render_enabled and started >= render_at and self.render_error is None:
                        self._render_frame(time.perf_counter())
                        interval = 1 / self.viewer["target_fps"]
                        render_at += interval
                        completed = time.perf_counter()
                        if render_at < completed - interval:
                            # Retain the cadence instead of accumulating timer
                            # oversleep; skip old frames after a longer stall.
                            render_at += (int((completed - render_at) / interval) + 1) * interval
                self.wake_event.wait(
                    max(0, min(0.005, render_at - time.perf_counter()))
                    if self.render_enabled and self.render_error is None
                    else 0.005
                )
                self.wake_event.clear()
        except Exception as exc:
            self.physics_error = str(exc)
            logger.exception("Physics loop failed")
        finally:
            self.stop_event.set()
            with self.frame_condition:
                self.frame_condition.notify_all()
            self.world.close()
            while not self._commands.empty():
                future, _ = self._commands.get_nowait()
                if not future.done():
                    future.set_exception(
                        RuntimeError(self.physics_error or "Physics engine stopped")
                    )

    def state(self):
        with self.world.lock:
            state = self.world.state()
            state.update(
                running=self.running,
                action_scale_rad=ACTION_SCALE_RAD,
                control_dt_s=CONTROL_DT_S,
                render_error=self.render_error,
                physics_error=self.physics_error,
                mode="physics",
                viewer=dict(
                    self.viewer,
                    actual_fps=self.actual_fps,
                    render_ms=self.render_ms,
                    frame_sequence=self.frame_sequence,
                    physics_realtime_factor=self.physics_realtime_factor,
                    dropped_wall_time_s=self.dropped_wall_time_s,
                    rendering_enabled=self.render_enabled,
                ),
                trajectory=copy.deepcopy(self._trajectory_state),
                jog=dict(self._jog_state),
            )
            return state

    def wait_frame(self, after_sequence, timeout=0.5):
        with self.frame_condition:
            self.frame_condition.wait_for(
                lambda: self.frame_sequence > after_sequence or self.stop_event.is_set(), timeout
            )
            if self.frame_sequence > after_sequence:
                return self.frame_sequence, self.frame
            return None

    def close(self):
        self.stop_event.set()
        self.wake_event.set()
        with self.frame_condition:
            self.frame_condition.notify_all()
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
        origin = request.headers.get("origin")
        if request.method != "GET" and origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "Use the dashboard origin or a local HTTP client"}, 403)
        return await call_next(request)

    def engine() -> Engine:
        return app.state.engine

    def command(operation):
        try:
            return engine().call(operation)
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")

    @app.get("/assets/dashboard.css")
    def dashboard_css():
        return FileResponse(Path(__file__).with_name("dashboard.css"), media_type="text/css")

    @app.get("/assets/dashboard.js")
    def dashboard_js():
        return FileResponse(Path(__file__).with_name("dashboard.js"), media_type="text/javascript")

    @app.get("/health")
    def health():
        e = engine()
        return dict(
            ok=e.thread.is_alive() and e.physics_error is None,
            mode="physics",
            port_purpose="simulation and policy control",
            render_error=e.render_error,
        )

    @app.get("/api/state")
    def state():
        return engine().state()

    @app.post("/api/reset")
    def reset(payload: Reset):
        return command(lambda: engine().reset(payload.seed))

    @app.post("/api/running")
    def running(payload: Running):
        return command(lambda: engine().set_running(payload.running))

    @app.post("/api/joints")
    def joints(payload: JointTarget):
        return command(lambda: engine().set_joints(payload.q_rad))

    @app.post("/api/step")
    def step(payload: PolicyAction):
        return command(lambda: engine().policy_step(payload.action))

    @app.post("/api/cubes/{name}")
    def cube(name: str, payload: CubeTarget):
        def update():
            engine().world.set_cube(name, payload.position_m, payload.velocity_m_s)
            return engine().state()

        return command(update)

    @app.post("/api/viewer")
    def viewer(payload: ViewerSettings):
        return command(lambda: engine().configure_viewer(payload.model_dump(exclude_none=True)))

    @app.post("/api/tcp")
    def tcp(payload: TCPMove):
        trajectory = Trajectory(
            waypoints_m=[payload.position_m],
            segment_duration_s=payload.duration_s,
            tolerance_m=payload.tolerance_m,
            settle_timeout_s=payload.settle_timeout_s,
            force_limit_n=payload.force_limit_n,
        )
        return command(lambda: engine().begin_trajectory(trajectory))

    @app.post("/api/tcp/preview")
    def tcp_preview(payload: TCPMove):
        def preview():
            q, errors = engine().plan_waypoints([payload.position_m])
            return dict(
                position_m=payload.position_m,
                q_rad=q[0].tolist(),
                ik_residual_m=errors[0],
                note="Position-only IK; path and contacts are not validated",
            )

        return command(preview)

    @app.post("/api/trajectory")
    def trajectory(payload: Trajectory):
        return command(lambda: engine().begin_trajectory(payload))

    @app.post("/api/trajectory/stop")
    def trajectory_stop():
        def stop():
            engine().stop_trajectory()
            return engine().state()

        return command(stop)

    @app.post("/api/jog")
    def jog(payload: Jog):
        return command(lambda: engine().jog(payload))

    @app.post("/api/jog/stop")
    def jog_stop():
        def stop():
            engine().stop_jog()
            return engine().state()

        return command(stop)

    @app.post("/api/cubes/{name}/properties")
    def cube_properties(name: str, payload: CubeProperties):
        return command(
            lambda: engine().configure(
                "configure_cube", name, payload.model_dump(exclude_unset=True)
            )
        )

    @app.post("/api/physics")
    def physics(payload: PhysicsSettings):
        return command(
            lambda: engine().configure("configure_physics", payload.model_dump(exclude_unset=True))
        )

    @app.post("/api/actuators")
    def actuators(payload: ActuatorSettings):
        return command(
            lambda: engine().configure(
                "configure_actuators", payload.model_dump(exclude_unset=True)
            )
        )

    @app.get("/api/scene")
    def scene():
        return command(lambda: copy.deepcopy(engine().world.scene))

    @app.get("/api/frame.jpg")
    def frame():
        e = engine()
        with e.frame_condition:
            jpeg, sequence = e.frame, e.frame_sequence
        if jpeg is None:
            raise HTTPException(503, e.render_error or "First frame is being rendered")
        return Response(
            jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store", "X-Frame-Id": str(sequence)},
        )

    @app.get("/api/stream.mjpg")
    async def stream(request: Request):
        e = engine()
        if not e.render_enabled:
            raise HTTPException(503, "Rendering is disabled for this server")

        async def frames():
            sequence = -1
            while not e.stop_event.is_set() and not await request.is_disconnected():
                newest = await asyncio.to_thread(e.wait_frame, sequence)
                if newest is None:
                    continue
                sequence, jpeg = newest
                if jpeg is None:
                    continue
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode()
                    + b"\r\nX-Frame-Id: "
                    + str(sequence).encode()
                    + b"\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )

        return StreamingResponse(
            frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, no-cache", "X-Accel-Buffering": "no"},
        )

    return app
