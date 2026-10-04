"""Portable fingerprints of the task, robot assets and effective physics model."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path


def environment_provenance(world) -> dict:
    package_root = Path(__file__).resolve().parents[1]
    source_files = (
        "simulation/model.py",
        "simulation/world.py",
        "training/env.py",
        "policies/base.py",
    )
    sources = {}
    for relative in source_files:
        text = (package_root / relative).read_text(encoding="utf-8")
        sources[relative] = hashlib.sha256(text.replace("\r\n", "\n").encode()).hexdigest()
    canonical_scene = json.dumps(world.scene, sort_keys=True, separators=(",", ":"))
    urdf = package_root / "simulation" / "assets" / "parol6" / "PAROL6.urdf"
    versions = {
        package: importlib.metadata.version(package)
        for package in ("mujoco", "gymnasium", "stable-baselines3", "torch", "numpy")
    }
    return {
        "scene_sha256": hashlib.sha256(canonical_scene.encode()).hexdigest(),
        "scene_hash_encoding": "UTF-8 JSON, sorted keys, compact separators",
        "physics_xml_sha256": hashlib.sha256(world.model_xml.encode()).hexdigest(),
        "robot_urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "source_sha256": sources,
        "source_hash_encoding": "UTF-8 text with LF newlines",
        "package_versions": versions,
    }
