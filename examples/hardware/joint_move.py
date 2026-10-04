"""Explicit hardware commissioning; reads by default, moves only with --delta-deg."""

import argparse
import math

from humaned_lab.hardware.parol6 import Parol6Adapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-hardware", action="store_true", help="Explicitly allow physical robot connection"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument(
        "--tool", required=True, help="Installed tool: NONE, PNEUMATIC, SSG-48, MSG, VACUUM"
    )
    parser.add_argument("--tool-variant", default="")
    parser.add_argument("--tcp-offset-mm", nargs=3, type=float, required=True)
    parser.add_argument("--home-cleared-workspace", action="store_true")
    parser.add_argument("--joint", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--delta-deg", type=float, default=0)
    args = parser.parse_args()
    with Parol6Adapter(
        allow_hardware=args.allow_hardware,
        tool_name=args.tool,
        tool_variant=args.tool_variant,
        tcp_offset_mm=args.tcp_offset_mm,
        host=args.host,
        port=args.port,
    ) as arm:
        if args.home_cleared_workspace:
            arm.home(cleared_workspace=True)
        joints = arm.joint_positions_rad()
        print("Joint radians:", joints.tolist())
        if args.delta_deg:
            joints[args.joint - 1] += math.radians(args.delta_deg)
            arm.move_joints_rad(joints)
            print("After move, radians:", arm.joint_positions_rad().tolist())


if __name__ == "__main__":
    main()
