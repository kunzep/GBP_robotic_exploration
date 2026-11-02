#!/usr/bin/env python3
"""Spawn several mobile rovers on terrain, drive them and record every camera.

Rovers come only from the YAML config (default:
swarm_slam/multi_rover_setup/presets/default.yaml); all other settings can be
overridden here. Each run writes one folder:
swarm_slam/out/multi_rover_setup/<stamp>__<terrain>__<N>rovers__<motion>__<WxH>/
(`latest` links to the newest one).

Usage (inside the container, from /root/ws):
    python3 swarm_slam/tools/multi_rover_setup.py --snapshot-only          # quick scene check
    python3 swarm_slam/tools/multi_rover_setup.py                          # 500 steps, orbit, 640x480
    python3 swarm_slam/tools/multi_rover_setup.py --res 1280x720 --overseer-res 2560x1440 --video
    python3 swarm_slam/tools/multi_rover_setup.py --heading center --motion idle --snapshot-only
    python3 swarm_slam/tools/multi_rover_setup.py --terrain apollo17 --terrain-lod high
    python3 swarm_slam/tools/multi_rover_setup.py --config swarm_slam/multi_rover_setup/presets/mars_jezero.yaml
    python3 swarm_slam/tools/multi_rover_setup.py --config swarm_slam/out/multi_rover_setup/latest/scene.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # swarm_slam/

from multi_rover_setup import MultiRoverSetup, parse_cli  # noqa: E402


def main() -> None:
    cfg = parse_cli(sys.argv[1:])
    with MultiRoverSetup(cfg) as scene:
        scene.run()


if __name__ == "__main__":
    main()
