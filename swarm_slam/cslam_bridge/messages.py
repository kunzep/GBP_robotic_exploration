"""Topic/frame contract with cslam and ROS 2 message builders.

Everything a robot publishes lives in its namespace ``/r<id>`` under the
topic names cslam's RGB-D front-end subscribes to (see ``config/srb_rgbd.yaml``).
Sensor and odometry child frame is the camera's ROS optical frame, so cslam
needs no TF (``sensor_base_frame_id: ""``) and its estimates compare directly
with the recorded camera poses.

Messages are ``rosbags`` typestore objects (ROS 2 Jazzy definitions); a future
live publisher can mirror these builders with ``rclpy`` types one to one.
"""

from __future__ import annotations

import numpy as np
from rosbags.typesys import Stores, get_typestore

from multi_rover_setup.cameras import matrix_to_quat_xyzw

TYPESTORE = get_typestore(Stores.ROS2_JAZZY)
_T = TYPESTORE.types

# Relative names, resolved in the robot namespace (cslam `frontend.*_topic` parameters).
TOPICS = {
    "rgb": ("color/image", "sensor_msgs/msg/Image"),
    "depth": ("depth/image", "sensor_msgs/msg/Image"),
    "camera_info": ("color/camera_info", "sensor_msgs/msg/CameraInfo"),
    "odom": ("odom", "nav_msgs/msg/Odometry"),
}
# cslam drops frames whose odometry covariance[0] exceeds 1000 ("tracking lost").
_ODOM_VARIANCE = 1e-4


def namespace(robot_id: int) -> str:
    return f"/r{robot_id}"


def topic(robot_id: int, key: str) -> str:
    return f"{namespace(robot_id)}/{TOPICS[key][0]}"


def camera_frame(robot_id: int) -> str:
    return f"r{robot_id}_camera"


def odom_frame(robot_id: int) -> str:
    return f"r{robot_id}_odom"


def header(stamp: float, frame_id: str):
    sec = int(stamp)
    return _T["std_msgs/msg/Header"](
        stamp=_T["builtin_interfaces/msg/Time"](sec=sec, nanosec=int(round((stamp - sec) * 1e9))),
        frame_id=frame_id,
    )


def image_msg(rgb: np.ndarray, stamp: float, frame_id: str):
    """``rgb8`` image from an ``(H, W, 3)`` uint8 array."""
    rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    height, width = rgb.shape[:2]
    return _T["sensor_msgs/msg/Image"](
        header=header(stamp, frame_id), height=height, width=width, encoding="rgb8",
        is_bigendian=0, step=width * 3, data=rgb.reshape(-1),
    )


def depth_msg(depth_mm: np.ndarray, stamp: float, frame_id: str):
    """``16UC1`` depth image in millimetres (0 = invalid) from an ``(H, W)`` uint16 array."""
    depth = np.ascontiguousarray(depth_mm, dtype="<u2")
    height, width = depth.shape
    return _T["sensor_msgs/msg/Image"](
        header=header(stamp, frame_id), height=height, width=width, encoding="16UC1",
        is_bigendian=0, step=width * 2, data=depth.view(np.uint8).reshape(-1),
    )


def camera_info_msg(K: np.ndarray, resolution: tuple[int, int], stamp: float, frame_id: str):
    """Pinhole camera without distortion (the simulated camera has none)."""
    K = np.asarray(K, dtype=np.float64)
    P = np.hstack((K, np.zeros((3, 1))))
    return _T["sensor_msgs/msg/CameraInfo"](
        header=header(stamp, frame_id), width=int(resolution[0]), height=int(resolution[1]),
        distortion_model="plumb_bob", d=np.zeros(5), k=K.reshape(-1), r=np.eye(3).reshape(-1), p=P.reshape(-1),
        binning_x=0, binning_y=0,
        roi=_T["sensor_msgs/msg/RegionOfInterest"](x_offset=0, y_offset=0, height=0, width=0, do_rectify=False),
    )


def odometry_msg(pose: np.ndarray, stamp: float, frame_id: str, child_frame_id: str):
    """Odometry message carrying the ``(4, 4)`` pose of ``child_frame_id`` in ``frame_id``."""
    x, y, z = pose[:3, 3]
    qx, qy, qz, qw = (float(v) for v in matrix_to_quat_xyzw(pose[:3, :3]))
    covariance = np.diag(np.full(6, _ODOM_VARIANCE)).reshape(-1)
    return _T["nav_msgs/msg/Odometry"](
        header=header(stamp, frame_id),
        child_frame_id=child_frame_id,
        pose=_T["geometry_msgs/msg/PoseWithCovariance"](
            pose=_T["geometry_msgs/msg/Pose"](
                position=_T["geometry_msgs/msg/Point"](x=float(x), y=float(y), z=float(z)),
                orientation=_T["geometry_msgs/msg/Quaternion"](x=qx, y=qy, z=qz, w=qw),
            ),
            covariance=covariance,
        ),
        twist=_T["geometry_msgs/msg/TwistWithCovariance"](
            twist=_T["geometry_msgs/msg/Twist"](
                linear=_T["geometry_msgs/msg/Vector3"](x=0.0, y=0.0, z=0.0),
                angular=_T["geometry_msgs/msg/Vector3"](x=0.0, y=0.0, z=0.0),
            ),
            covariance=np.zeros(36),
        ),
    )
