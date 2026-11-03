"""Ground truth + drift odometry: the fallback when visual odometry cannot track.

The odometry integrates the ground-truth frame-to-frame motion with errors that
grow with the distance driven, and starts at the identity like a real odometry
frame (so map merging across rovers is still cslam's job).
"""

from __future__ import annotations

import numpy as np


def drifted_odometry(
    poses: np.ndarray,
    *,
    trans_sigma: float = 0.0,
    yaw_bias_deg_per_m: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Odometry ``(N, 4, 4)`` from ground-truth ROS-optical camera poses ``(N, 4, 4)``.

    Per frame step of length ``d``:

    - translation gets Gaussian noise with standard deviation ``trans_sigma * d`` per axis;
    - heading drifts by ``yaw_bias_deg_per_m * d`` about the camera's vertical axis
      (optical ``y``, pointing down).

    With both set to zero the result equals the ground truth expressed relative to the first pose.
    """
    rng = np.random.default_rng(seed)
    odom = np.empty_like(poses)
    odom[0] = np.eye(4)
    for k in range(1, len(poses)):
        step = np.linalg.inv(poses[k - 1]) @ poses[k]
        dist = float(np.linalg.norm(step[:3, 3]))
        step[:3, 3] += rng.normal(0.0, trans_sigma * dist, 3) if trans_sigma > 0 else 0.0
        step[:3, :3] = _rot_y(np.deg2rad(yaw_bias_deg_per_m) * dist) @ step[:3, :3]
        odom[k] = odom[k - 1] @ step
    return odom


def _rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
