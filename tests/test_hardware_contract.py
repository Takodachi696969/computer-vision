"""Check the hardware boundary without importing optional SDKs or moving equipment."""

import sys
import types

import numpy as np
import pytest

from humaned_lab.hardware.parol6 import Parol6Adapter


class FakeNativeClient:
    def __init__(self, **kwargs):
        self.q_deg = np.array([90, -90, 180, 0, 0, 180], dtype=float)
        self.calls = []
        self.simulated = False
        self.closed = False

    def wait_ready(self, **kwargs):
        return True

    def ping(self):
        return types.SimpleNamespace(hardware_connected=True)

    def is_simulator(self):
        return self.simulated

    def select_tool(self, name, variant_key):
        self.calls.append(("tool", name, variant_key))
        return 0  # Zero is a valid queued index, not false/failure.

    def set_tcp_offset(self, *values):
        self.calls.append(("offset_mm", *values))
        return 1

    def wait_command(self, index, **kwargs):
        self.calls.append(("wait", index))
        return True

    def angles(self):
        return self.q_deg.tolist()

    def move_j(self, values, **kwargs):
        self.calls.append(("move_degrees", values))
        self.q_deg[:] = values
        return 2

    def stop(self):
        self.calls.append(("stop",))

    def close(self):
        self.closed = True


@pytest.fixture
def fake_native(monkeypatch):
    client = FakeNativeClient()
    module = types.ModuleType("parol6")
    module.RobotClient = lambda **kwargs: client
    config = types.ModuleType("parol6.config")
    config.LIMITS = types.SimpleNamespace(
        joint=types.SimpleNamespace(
            position=types.SimpleNamespace(
                rad=np.deg2rad(
                    [[-123, 123], [-145, -3.375], [107, 288], [-105, 105], [-90, 90], [0, 360]]
                )
            )
        )
    )
    monkeypatch.setitem(sys.modules, "parol6", module)
    monkeypatch.setitem(sys.modules, "parol6.config", config)
    return client


def test_tool_queue_units_and_small_move(fake_native):
    with Parol6Adapter(allow_hardware=True, tool_name="NONE", tcp_offset_mm=[0, 0, 10]) as arm:
        assert fake_native.calls[:4] == [
            ("tool", "NONE", ""),
            ("wait", 0),
            ("offset_mm", 0, 0, 10),
            ("wait", 1),
        ]
        joints = arm.joint_positions_rad()
        assert joints[0] == pytest.approx(np.pi / 2)
        joints[0] += np.deg2rad(1)
        arm.move_joints_rad(joints)
        assert fake_native.calls[-1][0] == "move_degrees"
        assert fake_native.q_deg[0] == pytest.approx(91)
        with pytest.raises(ValueError, match="step limit"):
            arm.move_joints_rad(joints + np.deg2rad(6))
    assert fake_native.closed


def test_simulation_cannot_be_reported_as_hardware(fake_native):
    fake_native.simulated = True
    arm = Parol6Adapter(allow_hardware=True, tool_name="NONE", tcp_offset_mm=[0, 0, 0])
    with pytest.raises(RuntimeError, match="real hardware"):
        arm.connect()
    assert fake_native.closed


def test_explicit_hardware_opt_in_required():
    with pytest.raises(PermissionError):
        Parol6Adapter(tool_name="NONE", tcp_offset_mm=[0, 0, 0])
