"""Small RealSense interface with actionable no-device results.

The camera frame uses the SDK's optical convention. Nothing in this module
claims an extrinsic transform from camera coordinates to the robot base.
"""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path

import numpy as np


def _sdk():
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "RealSense support is optional. Run scripts/bootstrap.ps1 -Camera "
            "or uv sync --frozen --extra camera."
        ) from exc
    return rs


def enumerate_devices() -> dict:
    """Report SDK availability and USB devices without starting a pipeline."""
    try:
        rs = _sdk()
    except RuntimeError as exc:
        return {"installed": False, "available": False, "devices": [], "reason": str(exc)}
    try:
        devices = [
            {
                "name": device.get_info(rs.camera_info.name),
                "serial": device.get_info(rs.camera_info.serial_number),
                "firmware": device.get_info(rs.camera_info.firmware_version),
            }
            for device in rs.context().query_devices()
        ]
    except RuntimeError as exc:
        return {"installed": True, "available": False, "devices": [], "reason": str(exc)}
    return {
        "installed": True,
        "available": bool(devices),
        "sdk_version": metadata.version("pyrealsense2"),
        "devices": devices,
        "reason": ""
        if devices
        else "No RealSense camera detected. Connect a supported camera over USB.",
    }


def capture(output_dir: str | Path, serial: str | None = None, timeout_ms: int = 5000) -> dict:
    """Save aligned RGB pixels, raw uint16 depth, and calibration metadata.

    Uses device-selected default profiles, so devices without a color stream
    produce a clear error rather than pretending a generic profile is universal.
    Depth in metres = depth.npy * metadata['depth_scale_metres_per_unit'].
    """
    report = enumerate_devices()
    if not report["available"]:
        return report
    if timeout_ms <= 0:
        raise ValueError("timeout_ms must be positive")
    from PIL import Image

    rs = _sdk()
    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    started = False
    try:
        profile = pipeline.start(config)
        started = True
        frames = pipeline.wait_for_frames(timeout_ms)
        if not frames.get_color_frame() or not frames.get_depth_frame():
            raise RuntimeError(
                "The selected camera profile must provide both color and depth streams."
            )
        aligned = rs.align(rs.stream.color).process(frames)
        depth = aligned.get_depth_frame()
        color = aligned.get_color_frame()
        raw_depth = np.asanyarray(depth.get_data()).copy()
        color_pixels = np.asanyarray(color.get_data()).copy()
        color_format = color.profile.format()
        if color_format == rs.format.bgr8:
            color_pixels = color_pixels[..., ::-1]
        elif color_format != rs.format.rgb8:
            raise RuntimeError(f"Color profile {color_format} is unsupported; select RGB8 or BGR8.")
        intr = depth.profile.as_video_stream_profile().get_intrinsics()
        info = {
            "serial": profile.get_device().get_info(rs.camera_info.serial_number),
            "sdk_version": metadata.version("pyrealsense2"),
            "depth_scale_metres_per_unit": profile.get_device()
            .first_depth_sensor()
            .get_depth_scale(),
            "depth_frame_timestamp_ms": depth.get_timestamp(),
            "color_frame_timestamp_ms": color.get_timestamp(),
            "timestamp_domain": str(depth.get_frame_timestamp_domain()),
            "aligned_to": "color",
            "intrinsics": {
                "width": intr.width,
                "height": intr.height,
                "fx": intr.fx,
                "fy": intr.fy,
                "ppx": intr.ppx,
                "ppy": intr.ppy,
                "model": str(intr.model),
                "coeffs": list(intr.coeffs),
            },
            "camera_to_robot_transform": None,
        }
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        Image.fromarray(color_pixels).save(destination / "color.png")
        np.save(destination / "depth.npy", raw_depth)
        (destination / "metadata.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        return {"available": True, "output_dir": str(destination.resolve()), **info}
    finally:
        if started:
            pipeline.stop()
