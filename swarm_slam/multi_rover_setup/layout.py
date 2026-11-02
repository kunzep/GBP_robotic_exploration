"""Spawn layouts and rover motion controllers (pure numpy, Isaac-free).

Poses are planar ``(x, y, yaw)`` in the env frame [m, m, rad]; commands are
``(v, w)`` in [m/s, rad/s].
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from .config import LayoutCfg, MotionCfg


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def ring_radius(n: int, spacing: float) -> float:
    """Radius at which neighbouring rovers on a circle are ``spacing`` apart (chord)."""
    if n <= 1:
        return 0.0
    return spacing / (2.0 * math.sin(math.pi / n))


#############
## Layouts ##
#############


def spawn_layout(n: int, cfg: LayoutCfg, *, seed: int = 0, bound_m: float | None = None) -> np.ndarray:
    """Return ``(n, 3)`` spawn poses ``(x, y, yaw)`` centred on the origin.

    ``bound_m`` is the half-extent that every rover must stay within; layouts
    that do not fit are shrunk uniformly (with a warning).
    """
    rng = np.random.default_rng(seed)
    spacing = cfg.spacing_m
    explicit_yaw = None

    if cfg.kind == "ring":
        radius = ring_radius(n, spacing)
        # Clockwise numbering: driving counter-clockwise, rover i trails rover i-1.
        theta = -2.0 * np.pi * np.arange(n) / max(n, 1)
        xy = radius * np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    elif cfg.kind == "line":
        xy = np.stack(((np.arange(n) - (n - 1) / 2.0) * spacing, np.zeros(n)), axis=-1)
    elif cfg.kind == "grid":
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        idx = np.arange(n)
        xy = np.stack(
            ((idx % cols - (cols - 1) / 2.0) * spacing, (idx // cols - (rows - 1) / 2.0) * spacing),
            axis=-1,
        )
    elif cfg.kind == "random":
        xy = _poisson_disk(n, spacing, rng)
    elif cfg.kind == "explicit":
        positions = cfg.positions or []
        if len(positions) != n:
            raise ValueError(f"layout.positions has {len(positions)} entries for {n} rovers")
        xy = np.array([p[:2] for p in positions], dtype=float)
        if all(len(p) == 3 for p in positions):
            explicit_yaw = np.radians([p[2] for p in positions])
    else:
        raise ValueError(f"Unknown layout kind {cfg.kind!r}")

    xy = np.asarray(xy, dtype=float).reshape(n, 2)
    if bound_m is not None and n > 0:
        extent = float(np.max(np.abs(xy))) if xy.size else 0.0
        if extent > bound_m > 0.0:
            warnings.warn(
                f"Layout extent {extent:.1f} m exceeds the terrain bound {bound_m:.1f} m; "
                "shrinking it (reduce layout.spacing_m or increase terrain.size_m)",
                stacklevel=2,
            )
            xy *= bound_m / extent

    yaw = explicit_yaw if explicit_yaw is not None else headings(xy, cfg.heading, rng)
    return np.concatenate((xy, np.asarray(yaw, dtype=float).reshape(n, 1)), axis=-1)


def headings(xy: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Yaw per rover: chain (face the previous rover), counter-clockwise tangent,
    facing the centroid, or random."""
    rel = xy - xy.mean(axis=0, keepdims=True)
    theta = np.arctan2(rel[:, 1], rel[:, 0])
    at_center = np.linalg.norm(rel, axis=-1) < 1e-6
    if mode == "chain":
        yaw = theta + np.pi / 2.0
        ahead = xy[:-1] - xy[1:]
        yaw[1:] = np.arctan2(ahead[:, 1], ahead[:, 0])
        at_center[1:] = False
    elif mode == "tangent":
        yaw = theta + np.pi / 2.0
    elif mode == "center":
        yaw = theta + np.pi
    elif mode == "random":
        return rng.uniform(-np.pi, np.pi, size=len(xy))
    else:
        raise ValueError(f"Unknown heading mode {mode!r}")
    yaw = np.where(at_center, 0.0, yaw)
    return wrap_angle(yaw)


def _poisson_disk(n: int, spacing: float, rng: np.random.Generator, attempts: int = 2000) -> np.ndarray:
    half = max(spacing, spacing * math.sqrt(n) * 0.75)
    points: list[np.ndarray] = []
    while len(points) < n:
        for _ in range(attempts):
            candidate = rng.uniform(-half, half, size=2)
            if all(np.linalg.norm(candidate - p) >= spacing for p in points):
                points.append(candidate)
                break
        else:
            half *= 1.25  # region too dense, grow it
    xy = np.array(points)
    return xy - xy.mean(axis=0, keepdims=True)


########################
## Motion controllers ##
########################


class MotionController:
    """Maps current poses ``(n, 3)`` to commands ``(n, 2)`` in [m/s, rad/s]."""

    def __init__(
        self,
        cfg: MotionCfg,
        *,
        center: np.ndarray,
        radius: float,
        bound_m: float,
        gap_m: float | np.ndarray = 4.0,
        max_lin: np.ndarray,
        max_ang: np.ndarray,
        dt: float,
        seed: int = 0,
    ):
        self.cfg = cfg
        self.center = np.asarray(center, dtype=float)[:2]
        self.radius = float(radius)
        self.bound_m = float(bound_m)
        n = len(np.asarray(max_lin))
        # Follow distance between rover i and i-1 (index i-1 of this array).
        self.gaps = np.broadcast_to(np.asarray(gap_m, dtype=float), (max(n - 1, 0),)).copy()
        if cfg.kind == "follow" and n > 1:
            # Each follower trails on a smaller circle (r_i^2 = r_{i-1}^2 - gap^2),
            # so the leader's orbit must leave room for the whole chain.
            needed = 1.3 * math.sqrt(float(np.sum(self.gaps**2)))
            self.radius = min(max(self.radius, needed), 0.9 * self.bound_m)
        self.max_lin = np.asarray(max_lin, dtype=float)
        self.max_ang = np.asarray(max_ang, dtype=float)
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed)
        # Common convoy speed so rovers of different top speeds stay together;
        # followers keep headroom to close gaps.
        headroom = 0.6 if cfg.kind == "follow" else 0.8
        self.speed = float(min(cfg.speed_mps, headroom * float(np.min(self.max_lin)) if n else cfg.speed_mps))
        self._wander_state = np.zeros((n, 2))
        self._wander_state[:, 0] = self.speed

    def __call__(self, poses: np.ndarray) -> np.ndarray:
        poses = np.asarray(poses, dtype=float)
        kind = self.cfg.kind
        if kind == "idle":
            cmd = np.zeros((len(poses), 2))
        elif kind == "constant":
            cmd = np.tile(np.asarray(self.cfg.cmd, dtype=float), (len(poses), 1))
        elif kind == "orbit":
            cmd = self._orbit(poses)
        elif kind == "follow":
            cmd = self._follow(poses)
        elif kind == "wander":
            cmd = self._wander(poses)
        else:
            raise ValueError(f"Unknown motion kind {kind!r}")
        cmd[:, 0] = np.clip(cmd[:, 0], -self.max_lin, self.max_lin)
        cmd[:, 1] = np.clip(cmd[:, 1], -self.max_ang, self.max_ang)
        return cmd

    def _orbit(self, poses: np.ndarray, k_radial: float = 0.8, k_heading: float = 1.5) -> np.ndarray:
        rel = poses[:, :2] - self.center
        dist = np.linalg.norm(rel, axis=-1)
        theta = np.arctan2(rel[:, 1], rel[:, 0])
        radius = max(self.radius, 1.0)
        # CCW tangent, tilted inwards when outside the ring and outwards inside it.
        desired = theta + np.pi / 2.0 + np.arctan(k_radial * (dist - radius))
        err = wrap_angle(desired - poses[:, 2])
        v = np.full(len(poses), self.speed) * np.clip(np.cos(err), 0.2, 1.0)
        w = v / radius + k_heading * err
        return np.stack((v, w), axis=-1)

    def _follow(self, poses: np.ndarray, k_gap: float = 0.5, k_heading: float = 1.5) -> np.ndarray:
        """Rover 0 orbits; every other rover pursues the one before it, keeping it in view."""
        cmd = self._orbit(poses)
        ahead = poses[:-1, :2] - poses[1:, :2]
        dist = np.linalg.norm(ahead, axis=-1)
        err = wrap_angle(np.arctan2(ahead[:, 1], ahead[:, 0]) - poses[1:, 2])
        v = np.clip(k_gap * (dist - self.gaps) + self.speed, 0.0, 1.5 * self.speed) * np.clip(np.cos(err), 0.0, 1.0)
        cmd[1:, 0] = v
        cmd[1:, 1] = k_heading * err
        return cmd

    def _wander(self, poses: np.ndarray, theta_ou: float = 0.5, sigma_v: float = 0.1, sigma_w: float = 0.6) -> np.ndarray:
        state = self._wander_state
        noise = self.rng.standard_normal(state.shape) * math.sqrt(self.dt)
        state[:, 0] += theta_ou * (self.speed - state[:, 0]) * self.dt + sigma_v * noise[:, 0]
        state[:, 1] += theta_ou * (0.0 - state[:, 1]) * self.dt + sigma_w * noise[:, 1]
        state[:, 0] = np.clip(state[:, 0], 0.0, self.speed * 1.5)
        cmd = state.copy()
        # Soft boundary: blend in a turn back towards the centre near the edge.
        rel = poses[:, :2] - self.center
        dist = np.linalg.norm(rel, axis=-1)
        soft = 0.7 * self.bound_m
        weight = np.clip((dist - soft) / max(self.bound_m - soft, 1e-6), 0.0, 1.0)
        home = wrap_angle(np.arctan2(-rel[:, 1], -rel[:, 0]) - poses[:, 2])
        cmd[:, 1] = (1.0 - weight) * cmd[:, 1] + weight * 2.0 * home
        return cmd


def normalize_commands(cmd: np.ndarray, scale_lin: np.ndarray, scale_ang: np.ndarray) -> np.ndarray:
    """Convert [m/s, rad/s] commands to the ``cmd_vel`` action range [-1, 1]."""
    out = np.zeros_like(cmd, dtype=float)
    out[:, 0] = cmd[:, 0] / np.maximum(scale_lin, 1e-6)
    out[:, 1] = cmd[:, 1] / np.maximum(scale_ang, 1e-6)
    return np.clip(out, -1.0, 1.0)


def yaw_from_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float).reshape(-1, 4)
    x, y, z, w = quat.T
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_xyzw_from_yaw(yaw) -> np.ndarray:
    yaw = np.asarray(yaw, dtype=float).reshape(-1)
    half = yaw / 2.0
    return np.stack((np.zeros_like(half), np.zeros_like(half), np.sin(half), np.cos(half)), axis=-1)
