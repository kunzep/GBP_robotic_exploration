#!/usr/bin/env python3
"""Convert a multi_rover_setup run folder into one rosbag2 per rover for Swarm-SLAM.

Writes <run>/bags/r<id>/ with /r<id>/color/image, /r<id>/depth/image,
/r<id>/color/camera_info and, with --odom gt_drift, /r<id>/odom.

Usage (from /root/ws in the Swarm-SLAM container, or anywhere with `rosbags` installed):
    python3 swarm_slam/tools/run_to_bags.py swarm_slam/out/multi_rover_setup/latest              # for odom:=vo
    python3 swarm_slam/tools/run_to_bags.py <run> --odom gt_drift --drift-trans 0                # GT odometry
    python3 swarm_slam/tools/run_to_bags.py <run> --odom gt_drift --drift-trans 0.02 --drift-yaw 0.5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # swarm_slam/

from cslam_bridge import load_run  # noqa: E402
from cslam_bridge.bags import write_bag  # noqa: E402
from cslam_bridge.odometry import drifted_odometry  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--odom", choices=("vo", "gt_drift"), default="vo",
                        help="vo: no odometry in the bags (RTAB-Map computes it); gt_drift: ground truth + drift")
    parser.add_argument("--drift-trans", type=float, default=0.02, help="translation noise std as a fraction of the distance driven per frame [-]")
    parser.add_argument("--drift-yaw", type=float, default=0.5, help="heading bias [deg per metre driven]")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    for stream in load_run(run_dir):
        odometry = None
        if args.odom == "gt_drift":
            odometry = drifted_odometry(stream.poses, trans_sigma=args.drift_trans,
                                        yaw_bias_deg_per_m=args.drift_yaw, seed=args.seed + stream.robot_id)
        path = write_bag(stream, run_dir / "bags" / f"r{stream.robot_id}", odometry)
        print(f"{stream.label} -> {path} ({len(stream)} frames)")


if __name__ == "__main__":
    main()
