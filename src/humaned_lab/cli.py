"""Small, discoverable commands; optional integrations import only when used."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np


def print_json(value):
    print(json.dumps(value, indent=2, default=str, allow_nan=False))


def doctor():
    packages = [
        "humaned-robot-lab",
        "mujoco",
        "numpy",
        "gymnasium",
        "stable-baselines3",
        "torch",
        "pyrealsense2",
        "parol6",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = (
                "not installed (optional)" if package not in packages[:4] else "MISSING"
            )
    from humaned_lab.simulation.world import PhysicsWorld

    with PhysicsWorld() as world:
        world.step(500)
        state = world.state()
        finite = bool(np.isfinite(world.data.qpos).all())
    torch_status = None
    if versions["torch"] != "not installed (optional)":
        import torch

        torch_status = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_build": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        }
    print_json(
        {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
            "packages": versions,
            "torch": torch_status,
            "physics_finite": finite,
            "tcp_m": state["tcp_m"],
            "note": "Core simulation and PPO MLP training use CPU. Hardware is optional.",
        }
    )
    if not finite:
        raise RuntimeError("Physics did not remain finite")


def demo(output: Path, scene=None):
    from PIL import Image
    from humaned_lab.simulation.world import PhysicsWorld

    output.mkdir(parents=True, exist_ok=True)
    world = PhysicsWorld(scene)
    try:
        world.step(round(0.5 / world.timestep))
        before = world.state()
        if not before["cubes"]:
            raise ValueError("The reaching demo requires at least one cube")
        cube = next((c for c in before["cubes"] if c["name"] == "target"), before["cubes"][0])
        target = np.asarray(cube["position_m"]) + [0, 0, 0.06]
        world.set_goal(target)
        q = world.solve_ik(target)
        world.set_joint_targets(q)
        world.step(round(2 / world.timestep))
        after = world.state()
        Image.fromarray(world.render()).save(output / "simulation.png")
        report = {
            "before": before,
            "after": after,
            "target_m": target.tolist(),
            "tcp_error_m": float(np.linalg.norm(np.asarray(after["tcp_m"]) - target)),
            "note": "Position-only IK reach demonstration; no grasping.",
        }
        (output / "demo.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print_json(
            {
                "image": output / "simulation.png",
                "report": output / "demo.json",
                "tcp_error_m": report["tcp_error_m"],
            }
        )
    finally:
        world.close()


def run_policy(spec, url, steps, model_path=None):
    from humaned_lab.control.client import SimulationClient
    from humaned_lab.policies.base import load_policy, observation_from_state, validate_action

    kwargs = {"model_path": model_path} if model_path else None
    policy = load_policy(spec, kwargs=kwargs)
    with SimulationClient(url) as client:
        client.running(False)
        state = client.reset()
        try:
            for _ in range(steps):
                action = validate_action(policy.act(observation_from_state(state)))
                state = client.step(action.tolist())
                time.sleep(0.05)
        finally:
            client.running(False)
        print_json(state)


def upstream_smoke(port):
    # This environment applies only to the owned child controller; never starts real serial.
    from parol6 import Robot, RobotClient

    robot = Robot(host="127.0.0.1", port=port, timeout=90, normalize_logs=True)
    robot.start(extra_env={"PAROL6_FAKE_SERIAL": "1", "PAROL6_STATUS_TRANSPORT": "UNICAST"})
    try:
        with RobotClient(host="127.0.0.1", port=port, timeout=10) as client:
            if not client.wait_ready(timeout=30):
                raise RuntimeError("Upstream mock controller did not become ready")
            client.simulator(True)
            client.home(wait=True, timeout=60)
            q = client.angles()
            q[0] += 2.0
            client.move_j(q, speed=0.1, accel=0.1, wait=True, timeout=30)
            print_json(
                {
                    "ping": client.ping(),
                    "angles_deg": client.angles(),
                    "pose": client.pose(),
                    "mode": "upstream mock serial",
                }
            )
    finally:
        robot.stop()


def bridge(url, host, port, seconds):
    from parol6 import RobotClient
    from humaned_lab.control.client import SimulationClient

    with RobotClient(host=host, port=port, timeout=2) as robot, SimulationClient(url) as sim:
        if not robot.wait_ready(timeout=10):
            raise RuntimeError("Start the upstream controller before the read-only bridge")
        sim.running(True)
        deadline = time.monotonic() + seconds
        count = 0
        while time.monotonic() < deadline:
            sim.joints(np.deg2rad(robot.angles()).tolist())
            count += 1
            time.sleep(0.05)
        print_json({"samples": count, "direction": "upstream angles -> physics targets"})


def main():
    parser = argparse.ArgumentParser(description="HumanED PAROL6 physics and policy lab")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check installation, physics and optional devices")
    serve = sub.add_parser("serve", help="Open the local dashboard and physics control port")
    serve.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"])
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--scene", type=Path)
    serve.add_argument("--no-render", action="store_true")
    d = sub.add_parser("demo", help="Run IK reach and save a simulation image")
    d.add_argument("--output", type=Path, default=Path("outputs/demo"))
    d.add_argument("--scene", type=Path)
    train = sub.add_parser("train", help="Train PPO on the reach-cube Gymnasium environment")
    train.add_argument("--steps", type=int, default=10000)
    train.add_argument("--output", type=Path, default=Path("outputs/ppo-reach"))
    train.add_argument("--scene", type=Path)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--eval-episodes", type=int, default=10)
    evaluate = sub.add_parser("evaluate", help="Evaluate a saved PPO and baselines")
    evaluate.add_argument("model", type=Path)
    evaluate.add_argument("--episodes", type=int, default=10)
    evaluate.add_argument("--seed", type=int, default=100)
    evaluate.add_argument("--scene", type=Path)
    evaluate.add_argument("--output", type=Path)
    policy = sub.add_parser(
        "policy", help="Run your module:Class policy through the open HTTP port"
    )
    policy.add_argument("spec")
    policy.add_argument("--url", default="http://127.0.0.1:8765")
    policy.add_argument("--steps", type=int, default=200)
    policy.add_argument("--model", type=Path)
    camera = sub.add_parser("camera", help="Enumerate or capture optional RealSense camera")
    camera_sub = camera.add_subparsers(dest="operation", required=True)
    camera_sub.add_parser("list")
    capture = camera_sub.add_parser("capture")
    capture.add_argument("--output", type=Path, default=Path("outputs/realsense"))
    capture.add_argument("--serial")
    smoke = sub.add_parser("upstream-smoke", help="Exercise PAROL6's mock serial controller")
    smoke.add_argument("--port", type=int, default=5001)
    b = sub.add_parser(
        "bridge", help="Read upstream joint angles into physics; sends no robot commands"
    )
    b.add_argument("--url", default="http://127.0.0.1:8765")
    b.add_argument("--host", default="127.0.0.1")
    b.add_argument("--port", type=int, default=5001)
    b.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            doctor()
        elif args.command == "serve":
            import uvicorn
            from humaned_lab.control.server import create_app

            uvicorn.run(
                create_app(args.scene, render=not args.no_render),
                host=args.host,
                port=args.port,
                log_level="info",
            )
        elif args.command == "demo":
            demo(args.output, args.scene)
        elif args.command == "train":
            from humaned_lab.training.train import train_policy

            print_json(
                train_policy(
                    args.output,
                    total_timesteps=args.steps,
                    scene_path=args.scene,
                    seed=args.seed,
                    evaluate_episodes=args.eval_episodes,
                )
            )
        elif args.command == "evaluate":
            from humaned_lab.training.evaluate import evaluate_policy

            print_json(
                evaluate_policy(
                    args.model,
                    episodes=args.episodes,
                    seed=args.seed,
                    scene_path=args.scene,
                    output_path=args.output,
                )
            )
        elif args.command == "policy":
            run_policy(args.spec, args.url, args.steps, args.model)
        elif args.command == "camera":
            from humaned_lab.hardware.camera import enumerate_devices, capture

            print_json(
                enumerate_devices()
                if args.operation == "list"
                else capture(args.output, serial=args.serial)
            )
        elif args.command == "upstream-smoke":
            upstream_smoke(args.port)
        elif args.command == "bridge":
            bridge(args.url, args.host, args.port, args.seconds)
    except KeyboardInterrupt:
        pass
    except ImportError as exc:
        parser.exit(
            1, f"Missing optional dependency: {exc}\nRun the matching setup script/extra.\n"
        )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(1, f"{exc}\n")
