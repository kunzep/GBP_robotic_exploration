"""Write one rosbag2 per rover from a :class:`CameraStream` (no ROS install needed)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from . import messages as msgs
from .run_data import CameraStream


def write_bag(stream: CameraStream, out_dir: str | Path, odometry: np.ndarray | None = None) -> Path:
    """Write ``stream`` to the bag folder ``out_dir`` (replaced if it exists).

    Topics are already namespaced (``/r<id>/color/image`` ...), so ``ros2 bag play``
    needs no remapping. ``odometry`` ``(N, 4, 4)`` adds ``/r<id>/odom``; leave it out
    when visual odometry runs during replay. Storage is MCAP with zstd chunks.
    """
    from PIL import Image
    from rosbags.rosbag2 import CompressionFormat, CompressionMode, StoragePlugin, Writer

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    rid = stream.robot_id
    cam, odom = msgs.camera_frame(rid), msgs.odom_frame(rid)
    keys = ["rgb", "depth", "camera_info"] + (["odom"] if odometry is not None else [])

    writer = Writer(out_dir, version=9, storage_plugin=StoragePlugin.MCAP)
    writer.set_compression(CompressionMode.STORAGE, CompressionFormat.ZSTD)
    with writer:
        conns = {key: writer.add_connection(msgs.topic(rid, key), msgs.TOPICS[key][1], typestore=msgs.TYPESTORE)
                 for key in keys}
        for k, stamp in enumerate(stream.stamps):
            frame = {
                "rgb": msgs.image_msg(np.asarray(Image.open(stream.rgb[k]).convert("RGB")), stamp, cam),
                "depth": msgs.depth_msg(np.asarray(Image.open(stream.depth[k]), dtype=np.uint16), stamp, cam),
                "camera_info": msgs.camera_info_msg(stream.K, stream.resolution, stamp, cam),
            }
            if odometry is not None:
                frame["odom"] = msgs.odometry_msg(odometry[k], stamp, odom, cam)
            t_ns = int(round(stamp * 1e9))
            for key, msg in frame.items():
                writer.write(conns[key], t_ns, msgs.TYPESTORE.serialize_cdr(msg, msgs.TOPICS[key][1]))
    return out_dir
