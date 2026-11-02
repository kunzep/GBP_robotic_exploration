"""Run folders and output writers (Isaac-free).

Layout of one run::

    <root>/<stamp>__<tag>/
      scene.yaml  run.log  overview.png  trajectories.csv  observations.csv
      cameras/<label>/{rgb,depth}/<frame>.{png,npy}  intrinsics.json  poses.csv
      videos/<label>.mp4
"""

from __future__ import annotations

import csv
import json
import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from .config import SWARM_SLAM_DIR, MultiRoverSetupCfg

STAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def run_tag(cfg: MultiRoverSetupCfg) -> str:
    """Short descriptive tag; rover names and types stay out of it (see scene.yaml)."""
    if cfg.output.name:
        return _slug(cfg.output.name)
    width, height = cfg.cameras.resolution
    parts = [cfg.terrain.kind, f"{len(cfg.rovers)}rovers", cfg.motion.kind, f"{width}x{height}"]
    if cfg.output.snapshot_only:
        parts.append("snapshot")
    return "__".join(parts)


def _slug(text: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_." else "_" for c in text.strip())
    return keep.strip("_") or "run"


def make_run_dir(root: str | Path, tag: str, *, now: datetime | None = None) -> Path:
    """Create ``<root>/<stamp>__<tag>`` (unique) and point ``<root>/latest`` at it.

    A relative ``root`` is taken relative to the repository (parent of ``swarm_slam/``).
    """
    root = Path(root)
    if not root.is_absolute():
        root = SWARM_SLAM_DIR.parent / root
    root.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime(STAMP_FORMAT)
    base = f"{stamp}__{tag}"
    run_dir = root / base
    suffix = 1
    while True:
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            suffix += 1
            run_dir = root / f"{base}-{suffix}"
    latest = root / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_dir.name, target_is_directory=True)
    except OSError:
        pass  # symlinks are a convenience only
    return run_dir


def attach_file_logging(path: Path, level: int = logging.INFO) -> logging.Handler:
    """Mirror Python logging (root, ``srb``, ``isaaclab``) into ``path``."""
    handler = logging.FileHandler(path)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    for name in ("", "srb"):
        logger = logging.getLogger(name)
        logger.addHandler(handler)
        if logger.level == logging.NOTSET or logger.level > level:
            logger.setLevel(level)
    return handler


def detach_file_logging(handler: logging.Handler) -> None:
    for name in ("", "srb"):
        logging.getLogger(name).removeHandler(handler)
    handler.close()


def write_scene_yaml(path: Path, cfg: MultiRoverSetupCfg, run_info: dict[str, Any]) -> None:
    data = cfg.to_dict()
    data["run"] = run_info
    with open(path, "w") as f:
        yaml.safe_dump(_plain(data), f, sort_keys=False)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def depth_to_png16(depth_m: np.ndarray) -> np.ndarray:
    """Metric depth [m] → uint16 millimetres; invalid/out-of-range → 0."""
    depth = np.asarray(depth_m, dtype=np.float32)
    depth = np.where(np.isfinite(depth), depth, 0.0)
    return np.clip(np.round(depth * 1000.0), 0, 65535).astype(np.uint16)


class _CsvLog:
    def __init__(self, path: Path, header: Iterable[str]):
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(list(header))
        self._lock = threading.Lock()

    def write(self, rows: Iterable[Iterable[Any]]) -> None:
        with self._lock:
            self._writer.writerows(rows)

    def close(self) -> None:
        self._file.close()


class Recorder:
    """Writes camera frames asynchronously plus pose/observation CSV logs."""

    TRAJECTORY_HEADER = ("frame", "step", "time_s", "rover", "x", "y", "z", "qx", "qy", "qz", "qw", "cmd_v", "cmd_w")
    OBSERVATION_HEADER = ("frame", "step", "observer", "target", "u", "v", "range_m", "in_fov", "visible")
    CAMERA_POSE_HEADER = ("frame", "step", "time_s", "x", "y", "z", "qx", "qy", "qz", "qw")

    def __init__(self, run_dir: Path, *, depth_format: str = "png16", workers: int = 4, max_pending: int = 64):
        self.run_dir = Path(run_dir)
        self.cameras_dir = self.run_dir / "cameras"
        self.depth_format = depth_format
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mrs-writer")
        self._slots = threading.BoundedSemaphore(max_pending)
        self._futures: list[Future] = []
        self._camera_logs: dict[str, _CsvLog] = {}
        self._camera_frames: dict[str, int] = {}
        self.trajectories = _CsvLog(self.run_dir / "trajectories.csv", self.TRAJECTORY_HEADER)
        self.observations = _CsvLog(self.run_dir / "observations.csv", self.OBSERVATION_HEADER)
        self.frame = 0

    def add_camera(self, label: str, *, intrinsics: np.ndarray, resolution: tuple[int, int], info: dict[str, Any]) -> Path:
        cam_dir = self.cameras_dir / label
        (cam_dir / "rgb").mkdir(parents=True, exist_ok=True)
        if self.depth_format != "none" and info.get("has_depth", True):
            (cam_dir / "depth").mkdir(exist_ok=True)
        with open(cam_dir / "intrinsics.json", "w") as f:
            json.dump(
                {
                    "width": int(resolution[0]),
                    "height": int(resolution[1]),
                    "K": np.asarray(intrinsics, dtype=float).round(6).tolist(),
                    "pose_convention": "position + quaternion (x, y, z, w) of the ROS optical frame "
                    "(z forward, x right, y down) in the world frame",
                    "depth": "distance to image plane; png16 = millimetres (0 = invalid), npy = float32 metres",
                    **info,
                },
                f,
                indent=2,
            )
        self._camera_logs[label] = _CsvLog(cam_dir / "poses.csv", self.CAMERA_POSE_HEADER)
        self._camera_frames[label] = 0
        return cam_dir

    def save_camera(
        self,
        label: str,
        *,
        step: int,
        time_s: float,
        rgb: np.ndarray,
        depth: np.ndarray | None,
        pos: np.ndarray,
        quat: np.ndarray,
    ) -> None:
        cam_dir = self.cameras_dir / label
        name = f"{self.frame:06d}"
        self._submit(_save_png, cam_dir / "rgb" / f"{name}.png", rgb)
        if depth is not None and self.depth_format != "none":
            if self.depth_format in ("png16", "both"):
                self._submit(_save_png, cam_dir / "depth" / f"{name}.png", depth_to_png16(depth))
            if self.depth_format in ("npy", "both"):
                self._submit(np.save, cam_dir / "depth" / f"{name}.npy", np.asarray(depth, dtype=np.float32))
        self._camera_logs[label].write([[self.frame, step, f"{time_s:.4f}", *_fmt(pos), *_fmt(quat)]])
        self._camera_frames[label] += 1

    def log_trajectories(self, rows: Iterable[Iterable[Any]]) -> None:
        self.trajectories.write(rows)

    def log_observations(self, rows: Iterable[Iterable[Any]]) -> None:
        self.observations.write(rows)

    def save_image(self, path: Path, rgb: np.ndarray) -> None:
        self._submit(_save_png, path, rgb)

    def next_frame(self) -> None:
        self.frame += 1

    def _submit(self, func, *args) -> None:
        self._slots.acquire()
        future = self._pool.submit(func, *args)
        future.add_done_callback(lambda _: self._slots.release())
        self._futures.append(future)
        if len(self._futures) > 512:
            self._futures = [f for f in self._futures if not f.done()]

    def flush(self) -> None:
        for future in self._futures:
            future.result()
        self._futures.clear()

    def close(self, *, video: bool = False, fps: float = 10.0) -> list[Path]:
        self.flush()
        self._pool.shutdown(wait=True)
        for log in (self.trajectories, self.observations, *self._camera_logs.values()):
            log.close()
        return self.encode_videos(fps) if video else []

    def encode_videos(self, fps: float) -> list[Path]:
        videos = []
        frames_by_cam = {
            label: sorted((self.cameras_dir / label / "rgb").glob("*.png")) for label in self._camera_logs
        }
        if not any(frames_by_cam.values()):
            return videos
        import imageio.v2 as imageio

        out_dir = self.run_dir / "videos"
        out_dir.mkdir(exist_ok=True)
        for label, frames in frames_by_cam.items():
            if len(frames) < 2:
                continue
            path = out_dir / f"{label}.mp4"
            with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
                for frame in frames:
                    writer.append_data(imageio.imread(frame))
            videos.append(path)
        return videos


def _fmt(values: np.ndarray) -> list[str]:
    return [f"{float(v):.6f}" for v in np.asarray(values).reshape(-1)]


def _save_png(path: Path, image: np.ndarray) -> None:
    from PIL import Image  # lazy: importing Pillow before Kit starts breaks Kit's extensions

    image = np.asarray(image)
    if image.dtype != np.uint8 and image.dtype != np.uint16:
        image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    tmp = path.with_name(f".{path.name}.tmp")
    Image.fromarray(image).save(tmp, format="PNG")
    os.replace(tmp, path)
