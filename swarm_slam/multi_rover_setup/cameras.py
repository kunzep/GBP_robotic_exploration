"""Overseer (3rd-person) camera and rover-to-rover visibility.

Geometry helpers are pure numpy; :func:`overseer_camera_cfg` imports SRB lazily.
"""

from __future__ import annotations

import math

import numpy as np

from .config import OverseerCfg

OVERSEER_SENSOR = "cam_overseer"
OVERSEER_LABEL = "overseer"


def overseer_camera_cfg(cfg: OverseerCfg):
    """SRB/Isaac Lab ``CameraCfg`` for a free-standing camera under the env prim."""
    from srb.core.sensor import CameraCfg, PinholeCameraCfg

    width, height = cfg.resolution
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/camera_overseer",
        offset=CameraCfg.OffsetCfg(convention="world", pos=(0.0, 0.0, 10.0)),
        spawn=PinholeCameraCfg(focal_length=cfg.focal_length, clipping_range=(0.05, 5000.0)),
        width=int(width),
        height=int(height),
        data_types=["rgb"],
    )


class OverseerRig:
    """Keeps the overseer eye behind and above the rover group, looking at it."""

    def __init__(self, cfg: OverseerCfg, *, group_center: np.ndarray, group_radius: float, dt: float):
        self.cfg = cfg
        self.dt = float(dt)
        spread = max(float(group_radius), 2.0)
        self.distance = cfg.distance_m if cfg.distance_m is not None else 2.6 * spread + 4.0
        self.height = cfg.height_m if cfg.height_m is not None else 0.9 * self.distance
        self.azimuth = math.radians(cfg.azimuth_deg)
        self.target = np.asarray(group_center, dtype=float).copy()
        self._initialized = False

    def update(self, group_center: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(eye, target)`` to apply, or ``None`` if the pose is unchanged."""
        mode = self.cfg.mode
        if mode == "off" or (mode == "fixed" and self._initialized):
            return None
        center = np.asarray(group_center, dtype=float)
        if not self._initialized:
            self.target = center.copy()
        else:
            alpha = 1.0 - math.exp(-self.dt / 1.5)  # ~1.5 s smoothing
            self.target = self.target + alpha * (center - self.target)
            if mode == "orbit":
                self.azimuth += math.radians(self.cfg.orbit_deg_per_s) * self.dt
        self._initialized = True
        return self.eye(), self.target.copy()

    def eye(self) -> np.ndarray:
        offset = np.array(
            (math.cos(self.azimuth) * self.distance, math.sin(self.azimuth) * self.distance, self.height)
        )
        return self.target + offset


################
## Visibility ##
################


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quat, dtype=float).reshape(4) / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=float)
    w = math.sqrt(max(0.0, 1.0 + rot[0, 0] + rot[1, 1] + rot[2, 2])) / 2.0
    x = math.copysign(math.sqrt(max(0.0, 1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])) / 2.0, rot[2, 1] - rot[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - rot[0, 0] + rot[1, 1] - rot[2, 2])) / 2.0, rot[0, 2] - rot[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - rot[0, 0] - rot[1, 1] + rot[2, 2])) / 2.0, rot[1, 0] - rot[0, 1])
    quat = np.array((x, y, z, w))
    return quat / np.linalg.norm(quat)


class MountedPose:
    """Fixed camera-in-root transform; turns live root poses into camera poses.

    Isaac Lab reports camera poses from USD, which goes stale while Fabric
    drives the rovers, so rover camera extrinsics are composed from the root.
    """

    def __init__(self, root_pos, root_quat, cam_pos, cam_quat):
        rot_root = quat_xyzw_to_matrix(root_quat)
        self.offset = rot_root.T @ (np.asarray(cam_pos, dtype=float) - np.asarray(root_pos, dtype=float))
        self.rot = rot_root.T @ quat_xyzw_to_matrix(cam_quat)

    def __call__(self, root_pos, root_quat) -> tuple[np.ndarray, np.ndarray]:
        rot_root = quat_xyzw_to_matrix(root_quat)
        return np.asarray(root_pos, dtype=float) + rot_root @ self.offset, matrix_to_quat_xyzw(rot_root @ self.rot)


def project_points(
    K: np.ndarray, cam_pos: np.ndarray, cam_quat_ros: np.ndarray, points_w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points into a camera with ROS optical pose (z fwd, x right, y down).

    Returns pixel coordinates ``(M, 2)`` and depth along the optical axis ``(M,)``.
    """
    rot = quat_xyzw_to_matrix(cam_quat_ros)
    pts_cam = (np.asarray(points_w, dtype=float).reshape(-1, 3) - np.asarray(cam_pos).reshape(1, 3)) @ rot
    z = pts_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (np.asarray(K, dtype=float) @ pts_cam.T).T
        uv = uv[:, :2] / uv[:, 2:3]
    return uv, z


def visibility(
    K: np.ndarray,
    cam_pos: np.ndarray,
    cam_quat_ros: np.ndarray,
    resolution: tuple[int, int],
    targets_w: np.ndarray,
    depth: np.ndarray | None = None,
    *,
    margin_m: float = 1.0,
    window: int = 2,
) -> list[tuple[float, float, float, bool, bool]]:
    """Per target ``(u, v, range, in_fov, visible)``.

    A target is visible when it projects into the image in front of the camera
    and, if depth is given, some pixel around it is not closer than the target
    minus ``margin_m`` (i.e. no terrain in between).
    """
    width, height = resolution
    uv, z = project_points(K, cam_pos, cam_quat_ros, targets_w)
    ranges = np.linalg.norm(np.asarray(targets_w, dtype=float).reshape(-1, 3) - np.asarray(cam_pos).reshape(1, 3), axis=-1)
    rows = []
    for (u, v), depth_z, rng in zip(uv, z, ranges):
        in_fov = bool(depth_z > 0.05 and 0.0 <= u < width and 0.0 <= v < height)
        visible = in_fov
        if in_fov and depth is not None:
            ui, vi = int(u), int(v)
            patch = depth[max(vi - window, 0) : vi + window + 1, max(ui - window, 0) : ui + window + 1]
            patch = patch[np.isfinite(patch) & (patch > 0)]
            visible = bool(patch.size == 0 or patch.max() >= depth_z - margin_m)
        rows.append((float(u), float(v), float(rng), in_fov, visible))
    return rows
