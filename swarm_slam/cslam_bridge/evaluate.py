"""Read cslam's logs back and compare them with the recorded ground truth.

cslam (``evaluation.enable_logs``) writes, each time a robot receives optimized
estimates, ``<log_folder>/<date>_experiment_robot_<id>/<date>/`` containing
``pose_timestamps<id>.csv`` (vertex key -> odometry stamp) and, on the robot
that ran the optimization, ``initial_global_pose_graph.g2o`` (odometry only)
and ``optimized_global_pose_graph.g2o``. Vertex keys are GTSAM
``LabeledSymbol('g', 'A' + robot_id, keyframe_id)``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from multi_rover_setup.cameras import quat_xyzw_to_matrix

from .run_data import CameraStream

_INDEX_BITS = 48


def decode_key(key: int) -> tuple[int, int]:
    """GTSAM ``LabeledSymbol`` key -> ``(robot_id, keyframe_id)``."""
    label = (key >> _INDEX_BITS) & 0xFF
    return label - ord("A"), key & ((1 << _INDEX_BITS) - 1)


def encode_key(robot_id: int, keyframe_id: int) -> int:
    return (ord("g") << 56) | ((ord("A") + robot_id) << _INDEX_BITS) | keyframe_id


@dataclass
class PoseGraph:
    vertices: dict[int, np.ndarray] = field(default_factory=dict)
    """key -> ``(4, 4)`` pose"""
    edges: list[tuple[int, int]] = field(default_factory=list)

    def inter_robot_edges(self) -> list[tuple[int, int]]:
        """Edges between keyframes of different robots, i.e. inter-robot loop closures."""
        return [(a, b) for a, b in self.edges if decode_key(a)[0] != decode_key(b)[0]]


def read_g2o(path: str | Path) -> PoseGraph:
    """Parse ``VERTEX_SE3:QUAT`` / ``EDGE_SE3:QUAT`` lines (as written by ``gtsam.writeG2o``)."""
    graph = PoseGraph()
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "VERTEX_SE3:QUAT":
            x, y, z, qx, qy, qz, qw = map(float, parts[2:9])
            pose = np.eye(4)
            pose[:3, :3] = quat_xyzw_to_matrix((qx, qy, qz, qw))
            pose[:3, 3] = (x, y, z)
            graph.vertices[int(parts[1])] = pose
        elif parts[0] == "EDGE_SE3:QUAT":
            graph.edges.append((int(parts[1]), int(parts[2])))
    return graph


def read_pose_timestamps(path: str | Path) -> dict[int, float]:
    """``pose_timestamps<id>.csv`` -> {vertex key: stamp [s]}."""
    with open(path, newline="") as f:
        return {int(row["vertice_id"]): int(row["sec"]) + int(row["nanosec"]) * 1e-9 for row in csv.DictReader(f)}


def latest_logs(slam_dir: str | Path) -> tuple[Path, dict[int, float]]:
    """Folder of the newest merged graph, plus the newest keyframe stamps of every robot."""
    slam_dir = Path(slam_dir)
    graphs = sorted(slam_dir.rglob("optimized_global_pose_graph.g2o"), key=lambda p: p.stat().st_mtime)
    if not graphs:
        raise FileNotFoundError(f"No optimized_global_pose_graph.g2o under {slam_dir} (did a loop closure merge the maps?)")
    stamps: dict[int, float] = {}
    newest: dict[str, Path] = {}
    for path in slam_dir.rglob("pose_timestamps*.csv"):
        if path.name not in newest or path.stat().st_mtime > newest[path.name].stat().st_mtime:
            newest[path.name] = path
    for path in newest.values():  # stamps accumulate over a run, so the newest file is the fullest
        stamps.update(read_pose_timestamps(path))
    return graphs[-1].parent, stamps


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rigid ``(4, 4)`` transform minimising ``|T @ src - dst|`` for points ``(N, 3)`` (no scale)."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    u, _, vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s))
    sign = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    T = np.eye(4)
    T[:3, :3] = u @ sign @ vt
    T[:3, 3] = mu_d - T[:3, :3] @ mu_s
    return T


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1)))) if len(a) else float("nan")


@dataclass
class RobotTrack:
    """Keyframes of one robot that have a stamp, in keyframe order."""

    robot_id: int
    keys: list[int]
    gt: np.ndarray
    """``(K, 3)`` ground-truth positions"""
    odometry: np.ndarray
    """``(K, 3)`` initial (odometry-only) positions, aligned to GT per robot"""
    optimized: np.ndarray
    """``(K, 3)`` optimized positions, aligned to GT jointly for all robots"""

    @property
    def ate_odometry(self) -> float:
        return rmse(self.odometry, self.gt)

    @property
    def ate_optimized(self) -> float:
        return rmse(self.optimized, self.gt)


@dataclass
class Evaluation:
    log_dir: Path
    tracks: list[RobotTrack]
    loop_closures: list[tuple[int, int]]
    """Inter-robot loop-closure edges (vertex keys) of the optimized graph."""
    stamps: dict[int, float]
    optimized: PoseGraph
    alignment: np.ndarray
    """Transform applied to all optimized positions."""


def evaluate(streams: list[CameraStream], slam_dir: str | Path) -> Evaluation:
    """Match keyframes to GT by stamp, align, and compute per-robot ATE.

    Odometry is aligned to GT robot by robot (robots share no frame before merging);
    the optimized graph is aligned with **one** transform for all robots, so a wrong
    merge shows up as a displaced robot instead of being hidden by the alignment.
    """
    log_dir, stamps = latest_logs(slam_dir)
    optimized = read_g2o(log_dir / "optimized_global_pose_graph.g2o")
    initial_path = log_dir / "initial_global_pose_graph.g2o"
    initial = read_g2o(initial_path) if initial_path.exists() else optimized

    raw = []
    for stream in streams:
        keys = sorted(
            (k for k in optimized.vertices if decode_key(k)[0] == stream.robot_id and k in stamps),
            key=lambda k: decode_key(k)[1],
        )
        gt = np.array([stream.poses[stream.nearest(stamps[k]), :3, 3] for k in keys]).reshape(-1, 3)
        odom = np.array([initial.vertices.get(k, optimized.vertices[k])[:3, 3] for k in keys]).reshape(-1, 3)
        opt = np.array([optimized.vertices[k][:3, 3] for k in keys]).reshape(-1, 3)
        if len(keys) >= 3:
            odom = transform_points(umeyama(odom, gt), odom)
        raw.append((stream.robot_id, keys, gt, odom, opt))

    all_opt = np.concatenate([r[4] for r in raw]) if raw else np.zeros((0, 3))
    all_gt = np.concatenate([r[2] for r in raw]) if raw else np.zeros((0, 3))
    alignment = umeyama(all_opt, all_gt) if len(all_opt) >= 3 else np.eye(4)
    tracks = [RobotTrack(rid, keys, gt, odom, transform_points(alignment, opt)) for rid, keys, gt, odom, opt in raw]
    return Evaluation(log_dir, tracks, optimized.inter_robot_edges(), stamps, optimized, alignment)


def write_ate_csv(result: Evaluation, path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("robot_id", "keyframes", "ate_odometry_m", "ate_optimized_m", "inter_robot_loop_closures"))
        for track in result.tracks:
            n_lc = sum(track.robot_id in (decode_key(a)[0], decode_key(b)[0]) for a, b in result.loop_closures)
            writer.writerow((track.robot_id, len(track.keys), f"{track.ate_odometry:.3f}", f"{track.ate_optimized:.3f}", n_lc))


#############
## Figures ##
#############


def plot_trajectories(result: Evaluation, path: str | Path) -> None:
    """Top-down GT / odometry / optimized panels; grey lines mark inter-robot loop closures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharex=True, sharey=True)
    panels = (("Ground truth", "gt"), ("Odometry (aligned per robot)", "odometry"), ("Swarm-SLAM optimized", "optimized"))
    for ax, (title, attr) in zip(axes, panels):
        positions = {}
        for track in result.tracks:
            xyz = getattr(track, attr)
            ax.plot(xyz[:, 0], xyz[:, 1], "-", lw=1.5, label=f"r{track.robot_id} ({len(track.keys)} kf)")
            ax.plot(xyz[:1, 0], xyz[:1, 1], "o", color=ax.lines[-1].get_color())
            positions.update(zip(track.keys, xyz))
        if attr != "odometry":  # odometry tracks live in unrelated frames
            for a, b in result.loop_closures:
                if a in positions and b in positions:
                    ax.plot(*np.stack((positions[a], positions[b]))[:, :2].T, "-", color="0.6", lw=0.8, zorder=0)
        ate = [getattr(t, f"ate_{attr}") for t in result.tracks] if attr != "gt" else []
        ax.set_title(title + (f"\nmean ATE {np.nanmean(ate):.2f} m" if ate else f"\n{len(result.loop_closures)} inter-robot loop closures"))
        ax.set_xlabel("x [m]")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("y [m]")
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_loop_closures(result: Evaluation, streams: list[CameraStream], path: str | Path, max_pairs: int = 6) -> bool:
    """Side-by-side RGB of the keyframe pairs of up to ``max_pairs`` inter-robot loop closures."""
    pairs = [(a, b) for a, b in result.loop_closures if a in result.stamps and b in result.stamps][:max_pairs]
    if not pairs:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    by_robot = {s.robot_id: s for s in streams}
    fig, axes = plt.subplots(len(pairs), 2, figsize=(8, 3 * len(pairs)), squeeze=False)
    for row, pair in zip(axes, pairs):
        for ax, key in zip(row, pair):
            robot_id, keyframe_id = decode_key(key)
            stream = by_robot[robot_id]
            k = stream.nearest(result.stamps[key])
            ax.imshow(Image.open(stream.rgb[k]))
            ax.set_title(f"r{robot_id} kf {keyframe_id}  t={stream.stamps[k]:.1f}s", fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return True
