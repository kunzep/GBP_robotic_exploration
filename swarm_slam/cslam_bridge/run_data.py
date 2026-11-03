"""Read a ``multi_rover_setup`` run folder as one RGB-D camera stream per rover."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from multi_rover_setup.cameras import quat_xyzw_to_matrix

# Added to the simulation time so that no ROS stamp is zero (zero means "unset" to many nodes).
STAMP_OFFSET_S = 1.0


@dataclass
class CameraStream:
    """Frames of one rover camera; index ``k`` refers to the same frame in every array/list."""

    robot_id: int
    """cslam robot id (0..N-1), assigned in rover order."""
    label: str
    """Camera folder name, e.g. ``rover_0__front``."""
    K: np.ndarray
    resolution: tuple[int, int]
    """``(width, height)``"""
    stamps: np.ndarray
    """``(N,)`` ROS time [s] = simulation time + :data:`STAMP_OFFSET_S`."""
    poses: np.ndarray
    """``(N, 4, 4)`` ground-truth ROS-optical camera poses in the world frame."""
    rgb: list[Path]
    depth: list[Path]

    def __len__(self) -> int:
        return len(self.stamps)

    def nearest(self, stamp: float) -> int:
        """Index of the frame closest in time to ``stamp``."""
        return int(np.argmin(np.abs(self.stamps - stamp)))


def load_run(run_dir: str | Path) -> list[CameraStream]:
    """Streams of all rover cameras that record depth, ordered by rover index.

    The overseer camera has no depth and is skipped. One depth camera per rover is expected.
    """
    cameras_dir = Path(run_dir) / "cameras"
    streams_by_rover: dict[int, Path] = {}
    for cam_dir in sorted(cameras_dir.iterdir()):
        info_path = cam_dir / "intrinsics.json"
        if not info_path.is_file() or not json.loads(info_path.read_text()).get("has_depth", False):
            continue
        rover = cam_dir.name.split("__", 1)[0]  # rover_<i>
        index = int(rover.rsplit("_", 1)[1])
        if index in streams_by_rover:
            raise ValueError(f"{rover} has several depth cameras; cslam takes one per robot")
        streams_by_rover[index] = cam_dir
    if not streams_by_rover:
        raise FileNotFoundError(f"No rover depth cameras under {cameras_dir}")
    return [_load_camera(cam_dir, robot_id) for robot_id, (_, cam_dir) in enumerate(sorted(streams_by_rover.items()))]


def _load_camera(cam_dir: Path, robot_id: int) -> CameraStream:
    info = json.loads((cam_dir / "intrinsics.json").read_text())
    stamps, poses, rgb, depth = [], [], [], []
    with open(cam_dir / "poses.csv", newline="") as f:
        for row in csv.DictReader(f):
            name = f"{int(row['frame']):06d}.png"
            rgb_path, depth_path = cam_dir / "rgb" / name, cam_dir / "depth" / name
            if not (rgb_path.is_file() and depth_path.is_file()):
                continue  # e.g. an aborted run whose last frame was never written
            pose = np.eye(4)
            pose[:3, :3] = quat_xyzw_to_matrix([float(row[k]) for k in ("qx", "qy", "qz", "qw")])
            pose[:3, 3] = [float(row[k]) for k in ("x", "y", "z")]
            stamps.append(float(row["time_s"]) + STAMP_OFFSET_S)
            poses.append(pose)
            rgb.append(rgb_path)
            depth.append(depth_path)
    return CameraStream(
        robot_id=robot_id,
        label=cam_dir.name,
        K=np.asarray(info["K"], dtype=float),
        resolution=(int(info["width"]), int(info["height"])),
        stamps=np.asarray(stamps),
        poses=np.asarray(poses).reshape(-1, 4, 4),
        rgb=rgb,
        depth=depth,
    )
