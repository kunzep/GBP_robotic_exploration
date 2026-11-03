#!/usr/bin/env python3
"""Compare Swarm-SLAM's newest merged pose graph of a run with the ground truth.

Writes into <run>/slam/: trajectories.png (GT / odometry / optimized, top-down),
loop_closures.png (image pairs of inter-robot loop closures) and ate.csv.

Usage (from /root/ws):
    python3 swarm_slam/tools/eval_slam.py swarm_slam/out/multi_rover_setup/latest
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # swarm_slam/

from cslam_bridge import load_run  # noqa: E402
from cslam_bridge.evaluate import evaluate, plot_loop_closures, plot_trajectories, write_ate_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    slam_dir = run_dir / "slam"
    streams = load_run(run_dir)
    result = evaluate(streams, slam_dir)
    print(f"graph: {result.log_dir}")
    print(f"inter-robot loop closures: {len(result.loop_closures)}")
    print(f"{'robot':>5} {'keyframes':>9} {'ATE odom [m]':>12} {'ATE opt [m]':>11}")
    for t in result.tracks:
        print(f"{'r%d' % t.robot_id:>5} {len(t.keys):>9} {t.ate_odometry:>12.3f} {t.ate_optimized:>11.3f}")

    write_ate_csv(result, slam_dir / "ate.csv")
    plot_trajectories(result, slam_dir / "trajectories.png")
    outputs = ["ate.csv", "trajectories.png"]
    if plot_loop_closures(result, streams, slam_dir / "loop_closures.png"):
        outputs.append("loop_closures.png")
    print(f"wrote {', '.join(outputs)} to {slam_dir}")


if __name__ == "__main__":
    main()
