"""Explicit, opt-in PAROL6 hardware adapter; public angles are radians.

Pinned upstream uses degrees for joint commands and millimetres for tool
offsets. The dashboard never imports or calls this adapter.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Sequence

import numpy as np


def _six(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (6,) or not np.isfinite(array).all():
        raise ValueError("Expected six finite joint angles in radians")
    return array


class Parol6Adapter:
    """Connect to an already running, explicitly configured PAROL6 controller.

    This is a discrete-move bridge, not a trained-policy hardware executor.
    It refuses simulator connections to prevent misleading hardware reports.
    The watchdog checks fresh telemetry and requests a protective stop on
    timeout. It is a software layer; the robot's physical E-stop is independent.
    """

    def __init__(
        self,
        *,
        allow_hardware: bool = False,
        tool_name: str,
        tcp_offset_mm: Sequence[float],
        host: str = "127.0.0.1",
        port: int = 5001,
        tool_variant: str = "",
        startup_timeout_s: float = 30.0,
        telemetry_timeout_s: float = 2.0,
        max_step_rad: float = math.radians(5),
    ) -> None:
        if not allow_hardware:
            raise PermissionError(
                "Physical control requires allow_hardware=True after configuring the actual robot."
            )
        offset = np.asarray(tcp_offset_mm, dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("tcp_offset_mm must contain three finite values in millimetres")
        if not tool_name.strip():
            raise ValueError(
                "An explicit installed tool name is required, including NONE for a bare flange"
            )
        for label, value in (
            ("startup_timeout_s", startup_timeout_s),
            ("telemetry_timeout_s", telemetry_timeout_s),
            ("max_step_rad", max_step_rad),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be finite and positive")
        self.host, self.port = host, port
        self.tool_name, self.tool_variant = tool_name, tool_variant
        self.tcp_offset_mm = offset
        self.startup_timeout_s, self.telemetry_timeout_s = startup_timeout_s, telemetry_timeout_s
        self.max_step_rad = max_step_rad
        self._client = None
        self._limits = None
        self._closing = threading.Event()
        self._watchdog = None
        self._fault: str | None = None

    def connect(self) -> "Parol6Adapter":
        """Probe readiness, verify real transport, configure tool, start watchdog."""
        if self._client is not None:
            return self
        try:
            from parol6 import RobotClient
            from parol6.config import LIMITS
        except ImportError as exc:
            raise RuntimeError(
                "Install the pinned PAROL6 optional environment first; see docs/tutorial.md."
            ) from exc
        client = RobotClient(
            host=self.host,
            port=self.port,
            timeout=min(0.5, self.telemetry_timeout_s / 2),
            retries=0,
        )
        try:
            if not client.wait_ready(timeout=self.startup_timeout_s):
                raise TimeoutError(
                    "PAROL6 controller did not become ready before the startup deadline"
                )
            ping = client.ping()
            if not ping or not ping.hardware_connected or client.is_simulator():
                raise RuntimeError(
                    "The controller must have real hardware connected and simulator disabled"
                )
            index = client.select_tool(self.tool_name, variant_key=self.tool_variant)
            if index < 0 or not client.wait_command(index, timeout=10):
                raise RuntimeError("Tool selection did not complete")
            offset_index = client.set_tcp_offset(*self.tcp_offset_mm)
            if offset_index < 0 or not client.wait_command(offset_index, timeout=10):
                raise RuntimeError("Tool offset configuration did not complete")
            self._client = client
            self._limits = np.asarray(LIMITS.joint.position.rad, dtype=float).copy()
            self._fault = None
            self._closing.clear()
            self._watchdog = threading.Thread(
                target=self._watch, name="humaned-parol6-watchdog", daemon=True
            )
            self._watchdog.start()
            return self
        except BaseException:
            client.close()
            raise

    def _watch(self) -> None:
        last_ok = time.monotonic()
        while not self._closing.wait(0.25):
            try:
                angles = self._client.angles()
                if angles is not None and np.isfinite(angles).all():
                    last_ok = time.monotonic()
            except Exception:
                pass
            if time.monotonic() - last_ok > self.telemetry_timeout_s:
                self._fault = "Telemetry watchdog expired; protective stop requested"
                try:
                    self._client.estop()
                except Exception:
                    pass
                return

    def _ready_client(self):
        if self._client is None:
            raise RuntimeError("Call connect() first")
        if self._fault:
            raise RuntimeError(self._fault)
        return self._client

    def joint_positions_rad(self) -> np.ndarray:
        angles = self._ready_client().angles()
        if angles is None:
            raise TimeoutError("Joint telemetry unavailable")
        return np.deg2rad(_six(angles))

    def move_joints_rad(
        self,
        target: Sequence[float],
        *,
        speed: float = 0.05,
        accel: float = 0.05,
        timeout_s: float = 30,
    ) -> int:
        """Small absolute joint move; converts radians to upstream degrees."""
        joints = _six(target)
        if not 0 < speed <= 0.1 or not 0 < accel <= 0.1:
            raise ValueError(
                "This commissioning adapter limits speed and accel fractions to (0, 0.1]"
            )
        client = self._ready_client()
        if np.any(joints < self._limits[:, 0]) or np.any(joints > self._limits[:, 1]):
            raise ValueError("Target is outside upstream PAROL6 joint limits")
        if np.max(np.abs(joints - self.joint_positions_rad())) > self.max_step_rad:
            raise ValueError(
                f"Move exceeds commissioning step limit of {self.max_step_rad:.4f} radians"
            )
        try:
            index = client.move_j(
                np.rad2deg(joints).tolist(), speed=speed, accel=accel, wait=True, timeout=timeout_s
            )
            if index < 0:
                raise RuntimeError("Controller rejected the move")
            return index
        except BaseException:
            client.stop()
            raise

    def home(self, *, cleared_workspace: bool = False) -> int:
        """Firmware homing sweeps joints; explicitly confirm the cleared workspace."""
        if not cleared_workspace:
            raise PermissionError(
                "Homing requires cleared_workspace=True; it can seek limit switches"
            )
        return self._ready_client().home(wait=True, timeout=90)

    def stop(self) -> None:
        if self._client is not None:
            self._client.stop()

    def close(self) -> None:
        self._closing.set()
        if self._watchdog:
            self._watchdog.join(timeout=3)
        if self._client is not None:
            try:
                self._client.stop()
            finally:
                self._client.close()
                self._client = None

    def __enter__(self) -> "Parol6Adapter":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
