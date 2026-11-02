"""``MultiRoverSetup``: several mobile rovers on terrain, watched by an overseer camera.

Builds on the generic ``srb/_ground`` task: the primary rover is the scene's
``robot``, further rovers are ``extra_robots`` (``rover_1``, ``rover_2``, ...),
and the overseer is an env-level camera sensor. Everything Isaac-related is
imported only after :meth:`MultiRoverSetup.launch` has started Kit.
"""

from __future__ import annotations

import logging
import math
import sys
import traceback
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

import numpy as np

from .cameras import (
    OVERSEER_LABEL,
    OVERSEER_SENSOR,
    MountedPose,
    OverseerRig,
    overseer_camera_cfg,
    visibility,
)
from .config import MultiRoverSetupCfg
from .layout import (
    MotionController,
    normalize_commands,
    quat_xyzw_from_yaw,
    spawn_layout,
    yaw_from_quat_xyzw,
)
from .recorder import (
    Recorder,
    attach_file_logging,
    detach_file_logging,
    make_run_dir,
    run_tag,
    write_scene_yaml,
)
from .terrain import TerrainHeightField, build_scenery, terrain_domain, terrain_half_extent

logger = logging.getLogger("multi_rover_setup")

TASK_ID = "srb/_ground"
# Rover root lifted this far above its configured spawn height before it is
# placed onto the sampled terrain height (no physics step happens in between).
_PRE_PLACEMENT_LIFT_M = 1.0
# Target point on another rover used for visibility checks (above its root).
_VISIBILITY_TARGET_OFFSET_M = 0.2


@dataclass
class RoverHandle:
    index: int
    label: str
    type: str
    key: str
    asset: Any = None
    cameras: dict[str, str] = field(default_factory=dict)
    """camera label → scene sensor name"""
    drive_term: str | None = None
    drive: slice | None = None
    scale_linear: float = 1.0
    scale_angular: float = 1.0
    spawn: dict[str, float] = field(default_factory=dict)
    heading_offset: float = 0.0
    """Yaw of the rover's forward (front-camera) direction relative to its root frame [rad]."""


class MultiRoverSetup:
    """Spawn, drive and record a multi-rover scene.

    Example::

        with MultiRoverSetup(cfg) as scene:
            for _ in range(cfg.sim.steps):
                scene.step()                     # built-in motion
                # scene.step({"rover_1": (0.3, 0.1)})  # or own commands [m/s, rad/s]
    """

    def __init__(self, cfg: MultiRoverSetupCfg, *, argv: list[str] | None = None):
        cfg.validate()
        self.cfg = cfg
        self.argv = list(argv if argv is not None else sys.argv)
        self.started_at = datetime.now()
        self.run_dir = make_run_dir(cfg.output.root, run_tag(cfg), now=self.started_at)
        self._log_handler = attach_file_logging(self.run_dir / "run.log")
        logger.info("Run folder: %s", self.run_dir)

        self.bound_m = 0.8 * terrain_half_extent(cfg.terrain)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.spawn = spawn_layout(len(cfg.rovers), cfg.layout, seed=cfg.sim.seed, bound_m=self.bound_m)
        for warning in caught:
            logger.warning("%s", warning.message)

        self.rovers: list[RoverHandle] = [
            RoverHandle(index=i, label=f"rover_{i}", type=spec.type, key="robot" if i == 0 else f"rover_{i}")
            for i, spec in enumerate(cfg.rovers)
        ]
        self.app = None
        self.env = None
        self.env_cfg = None
        self.recorder: Recorder | None = None
        self.overseer: OverseerRig | None = None
        self.controller: MotionController | None = None
        self.step_count = 0
        self.height_field_points = 0
        self._camera_sensors: dict[str, str] = {}
        self._camera_mounts: dict[str, tuple[RoverHandle, MountedPose]] = {}
        self._last_cmd = np.zeros((len(self.rovers), 2))
        self._closed = False

    ###############
    ## Lifecycle ##
    ###############

    def __enter__(self) -> MultiRoverSetup:
        try:
            self.start()
        except BaseException:
            logger.error("Scene setup failed:\n%s", traceback.format_exc())
            self.close(failed=True)
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            logger.error("Run aborted:\n%s", "".join(traceback.format_exception(exc_type, exc, tb)))
        self.close(failed=exc is not None and not isinstance(exc, KeyboardInterrupt))

    def start(self) -> None:
        self.launch()
        self.build_env()
        self.place_on_terrain()
        self.settle()
        self.start_recording()

    def launch(self) -> None:
        """Start Kit with camera rendering through SRB's own launch planning."""
        from srb.cli.commands.agent_launch import plan_agent_app_launch

        kwargs: dict[str, Any] = {"enable_cameras": True, "rendering_mode": self.cfg.sim.quality}
        if self.cfg.sim.kit_args:
            kwargs["kit_args"] = self.cfg.sim.kit_args
        plan = plan_agent_app_launch(
            env_id=TASK_ID,
            headless=self.cfg.sim.headless,
            video_enable=False,
            hydra_args=[],
            kwargs=kwargs,
        )
        from srb.core.app import AppLauncher

        launcher = AppLauncher(headless=self.cfg.sim.headless, **plan.kwargs)
        self.app = launcher.app

        from srb.cli.runtime_startup import run_agent_startup_checks

        run_agent_startup_checks()

    def build_cfg(self):
        """Task config of ``srb/_ground`` with all rovers, terrain and cameras."""
        from srb.core.asset import GroundRobot
        from srb.core.domain import Domain
        from srb.core.sensor.category import SensorCategory
        from srb.hydra.resolve import resolve_mobile_robot
        from srb.tasks.mobile._ground.task import TaskCfg

        cfg = self.cfg
        robots = []
        for rover in self.rovers:
            robot = resolve_mobile_robot(rover.type)
            if not isinstance(robot, GroundRobot):
                logger.warning("%s (%s) is not a ground robot; it may not drive", rover.label, rover.type)
            robots.append(robot)

        scenery = build_scenery(cfg.terrain)
        if cfg.terrain.kind == "flat":
            lift = 0.0
        elif cfg.terrain.kind in ("moon_surface", "mars_surface"):
            lift = 0.1 * cfg.terrain.size_m + _PRE_PLACEMENT_LIFT_M
        else:
            lift = 10.0
        env_sensor_cfgs = {}
        if cfg.overseer.mode != "off":
            env_sensor_cfgs[OVERSEER_SENSOR] = overseer_camera_cfg(cfg.overseer)

        env_cfg = TaskCfg(
            num_envs=1,
            seed=cfg.sim.seed,
            domain=Domain[terrain_domain(cfg.terrain).upper()],
            stack=True,
            scenery=scenery,
            skydome=None if cfg.sim.skydome == "none" else cfg.sim.skydome,
            sunlight_rpy=tuple(float(v) for v in cfg.sim.sun_rpy),
            robot=robots[0],
            extra_robots={rover.key: robot for rover, robot in zip(self.rovers[1:], robots[1:])},
            robot_positions=tuple((float(x), float(y)) for x, y, _ in self.spawn),
            robot_spawn_height=lift,
            active_sensors={
                SensorCategory.PROPRIOCEPTION,
                SensorCategory.CONTACT,
                SensorCategory.TRANSFORM,
                SensorCategory.VISUAL,
            },
            camera_resolution=tuple(int(v) for v in cfg.cameras.resolution),
            camera_data_types=("rgb", "depth"),
            env_sensor_cfgs=env_sensor_cfgs,
            truncate_episodes=False,
            episode_length_s=1.0e6,
        )
        for rover, (_, _, yaw) in zip(self.rovers, self.spawn):
            asset_cfg = getattr(env_cfg.scene, rover.key)
            asset_cfg.init_state.rot = tuple(float(v) for v in quat_xyzw_from_yaw(yaw)[0])
        if cfg.cameras.hfov_deg is not None:
            self._set_rover_fov(env_cfg, cfg.cameras.hfov_deg)
        # Keep Fabric on: with `use_fabric=False` PhysX re-reads the USD spawn pose on the
        # first step and the terrain placement in `place_on_terrain` is lost.
        env_cfg.sim.log_dir = str(self.run_dir)

        center = np.array((*self.spawn[:, :2].mean(axis=0), 0.0))
        rig = OverseerRig(cfg.overseer, group_center=center, group_radius=self._group_radius(), dt=1.0)
        env_cfg.viewer.eye = tuple(float(v) for v in rig.eye())
        env_cfg.viewer.lookat = tuple(float(v) for v in center)
        self.env_cfg = env_cfg
        return env_cfg

    def _set_rover_fov(self, env_cfg, hfov_deg: float) -> None:
        """Widen/narrow every rover camera lens to ``hfov_deg`` (keeps the aperture)."""
        from srb.core.sensor import CameraCfg

        keys = [f"/{rover.key}/" for rover in self.rovers]
        for name, sensor_cfg in vars(env_cfg.scene).items():
            spawn = getattr(sensor_cfg, "spawn", None)
            if not isinstance(sensor_cfg, CameraCfg) or not hasattr(spawn, "horizontal_aperture"):
                continue
            if not any(key in sensor_cfg.prim_path for key in keys):
                continue
            spawn.focal_length = spawn.horizontal_aperture / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
            logger.info("%s: focal length %.2f mm for %.0f deg HFOV", name, spawn.focal_length, hfov_deg)

    def build_env(self) -> None:
        import gymnasium
        from srb.wrappers.composition import wrap_env

        env_cfg = self.env_cfg or self.build_cfg()
        self.env = wrap_env(gymnasium.make(TASK_ID, cfg=env_cfg, render_mode=None))
        self.env.reset(seed=self.cfg.sim.seed)
        self._bind_handles()
        self._write_scene_yaml()

    def _bind_handles(self) -> None:
        unwrapped = self.env.unwrapped
        scene = unwrapped.scene
        manager = unwrapped.action_manager
        slices, start = {}, 0
        for name, dim in zip(manager.active_terms, manager.action_term_dim):
            slices[name] = slice(start, start + int(dim))
            start += int(dim)
        self._action_dim = start

        cameras = {name: s for name, s in scene.sensors.items() if hasattr(s.data, "intrinsic_matrices")}
        for rover, (x, y, yaw) in zip(self.rovers, self.spawn):
            rover.asset = scene[rover.key]
            for term_name, term_slice in slices.items():
                if not term_name.startswith(f"{rover.key}/"):
                    continue
                term_cfg = manager.get_term(term_name).cfg
                if hasattr(term_cfg, "scale_linear") and term_slice.stop - term_slice.start == 2:
                    rover.drive_term, rover.drive = term_name, term_slice
                    rover.scale_linear = float(term_cfg.scale_linear)
                    rover.scale_angular = float(term_cfg.scale_angular)
                    break
            if rover.drive is None:
                logger.warning("%s (%s) has no 2-D drive action; it will stay idle", rover.label, rover.type)

            for sensor_name, sensor in cameras.items():
                if f"/{rover.key}/" not in sensor.cfg.prim_path:
                    continue
                cam = sensor_name.removesuffix(f"_{rover.key}")
                for prefix in ("cam_", "camera_"):
                    cam = cam.removeprefix(prefix)
                label = f"{rover.label}__{cam}"
                rover.cameras[label] = sensor_name
                self._camera_sensors[label] = sensor_name
                # Right after reset the USD camera and PhysX root agree on the spawn pose.
                self._camera_mounts[label] = (
                    rover,
                    MountedPose(
                        _np(rover.asset.data.root_pos_w)[0],
                        _np(rover.asset.data.root_quat_w)[0],
                        _np(sensor.data.pos_w)[0],
                        _np(sensor.data.quat_w_ros)[0],
                    ),
                )
            front = [m for r, m in self._camera_mounts.values() if r is rover]
            if front:
                # Rover assets differ in which root axis points forward (e.g. Perseverance: -y);
                # the front camera's optical axis tells.
                axis = front[0].rot[:, 2]
                rover.heading_offset = float(math.atan2(axis[1], axis[0]))
            rover.spawn = {
                "x": float(x),
                "y": float(y),
                "yaw_deg": float(np.degrees(yaw)),
                "heading_offset_deg": float(np.degrees(rover.heading_offset)),
            }
            logger.info(
                "%s: type=%s key=%s drive=%s %s cameras=%s",
                rover.label, rover.type, rover.key, rover.drive_term,
                f"[{rover.drive.start}:{rover.drive.stop}]" if rover.drive else "-",
                sorted(rover.cameras),
            )
        if OVERSEER_SENSOR in cameras:
            self._camera_sensors[OVERSEER_LABEL] = OVERSEER_SENSOR

    def place_on_terrain(self) -> None:
        """Drop every rover onto the sampled terrain height under its footprint."""
        import omni.usd
        import torch
        from pxr import Usd, UsdGeom

        unwrapped = self.env.unwrapped
        stage = omni.usd.get_context().get_stage()
        scenery_cfg = getattr(unwrapped.scene.cfg, "scenery", None)
        field = None
        if scenery_cfg is not None:
            field = TerrainHeightField.from_stage(stage, scenery_cfg.prim_path)
        if field is None:
            logger.warning("No terrain mesh found; rovers drop from their spawn height instead")
        else:
            self.height_field_points = field.num_points
            logger.info("Terrain height field from %d vertices", field.num_points)

        origin = unwrapped.scene.env_origins[0].detach().cpu().numpy().astype(float)
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        device = unwrapped.device
        for rover, (x, y, yaw) in zip(self.rovers, self.spawn):
            root_z = float(_np(rover.asset.data.root_pos_w)[0, 2])
            prim = stage.GetPrimAtPath(f"/World/envs/env_0/{rover.key}")
            box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            size = np.asarray(box.GetSize(), dtype=float)
            bottom_offset = root_z - float(box.GetMin()[2]) if not box.IsEmpty() else 0.3
            radius = 0.5 * float(np.hypot(size[0], size[1])) if not box.IsEmpty() else 0.6
            world_x, world_y = origin[0] + x, origin[1] + y
            ground = field.height_at(world_x, world_y, radius) if field is not None else None
            if ground is None:
                z = root_z
            else:
                z = ground + bottom_offset + self.cfg.layout.spawn_clearance_m
            pose = torch.tensor(
                [[world_x, world_y, z, *quat_xyzw_from_yaw(yaw - rover.heading_offset)[0]]],
                device=device,
                dtype=torch.float32,
            )
            rover.asset.write_root_pose_to_sim_index(root_pose=pose)
            rover.asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=device))
            default_pose = pose.clone()
            default_pose[:, :3] -= torch.as_tensor(origin, device=device, dtype=torch.float32)
            rover.asset.data.default_root_pose.torch[:] = default_pose
            rover.spawn.update({"z": float(z), "ground_z": None if ground is None else float(ground), "radius_m": radius})
            logger.info(
                "%s placed at (%.2f, %.2f, %.2f), ground %s, footprint r=%.2f m",
                rover.label, world_x, world_y, z, "n/a" if ground is None else f"{ground:.2f}", radius,
            )

    def settle(self) -> None:
        """Let the rovers come to rest under gravity, then warm up the renderer."""
        from srb.cli.commands.agent_runtime import _settle_robots


        if self.cfg.sim.settle_steps > 0:
            _settle_robots(self.env, self.app, steps=self.cfg.sim.settle_steps)
        dt = float(self.env.unwrapped.step_dt)
        self.overseer = OverseerRig(
            self.cfg.overseer, group_center=self._group_center(), group_radius=self._group_radius(), dt=dt
        )
        self._update_overseer()
        for _ in range(self.cfg.sim.warmup_steps):
            self._apply(np.zeros((len(self.rovers), 2)))
        center = self.spawn[:, :2].mean(axis=0)
        self.controller = MotionController(
            self.cfg.motion,
            center=center,
            radius=max(self._group_radius(), self.cfg.layout.spacing_m / 2.0),
            bound_m=self.bound_m,
            gap_m=self._follow_gaps(),
            max_lin=np.array([r.scale_linear if r.drive else np.inf for r in self.rovers]),
            max_ang=np.array([r.scale_angular if r.drive else np.inf for r in self.rovers]),
            dt=dt,
            seed=self.cfg.sim.seed,
        )

    def start_recording(self) -> None:
        self.recorder = Recorder(self.run_dir, depth_format=self.cfg.cameras.depth_format)
        scene = self.env.unwrapped.scene
        for label, sensor_name in self._camera_sensors.items():
            sensor = scene.sensors[sensor_name]
            data = sensor.data
            self.recorder.add_camera(
                label,
                intrinsics=_np(data.intrinsic_matrices)[0],
                resolution=(int(sensor.cfg.width), int(sensor.cfg.height)),
                info={
                    "sensor": sensor_name,
                    "prim_path": sensor.cfg.prim_path,
                    "has_depth": label != OVERSEER_LABEL,
                },
            )
        frames = self.camera_frames()
        if OVERSEER_LABEL in frames:
            self.recorder.save_image(self.run_dir / "overview.png", frames[OVERSEER_LABEL]["rgb"])
        else:
            from srb.cli.commands.agent_runtime import _capture_snapshot

            _capture_snapshot(self.env, self.app, str(self.run_dir / "overview.png"))
        self.record(frames)
        logger.info("Scene ready: overview.png and frame 0 written")

    def close(self, *, failed: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        videos = []
        try:
            if self.recorder is not None:
                fps = 1.0 / (self._step_dt() * self.cfg.output.record_every)
                videos = self.recorder.close(video=self.cfg.output.video, fps=fps)
            self._write_scene_yaml(finished=True, failed=failed)
            logger.info(
                "Finished %d steps, %d frames, %d videos -> %s",
                self.step_count, self.recorder.frame if self.recorder else 0, len(videos), self.run_dir,
            )
            if self.env is not None:
                self.env.close()
        finally:
            detach_file_logging(self._log_handler)
            print(f"[multi_rover_setup] outputs: {self.run_dir}", flush=True)
            if self.app is not None:
                # Kit's shutdown exits the process with this status.
                self.app.close(exit_code=1 if failed else 0)

    ##############
    ## Stepping ##
    ##############

    def run(self, steps: int | None = None) -> None:
        if self.cfg.output.snapshot_only:
            return
        for _ in range(self.cfg.sim.steps if steps is None else steps):
            if self.app is not None and not self.app.is_running():
                break
            self.step()

    def step(self, commands: Mapping[str, tuple[float, float]] | None = None):
        """Advance one agent step.

        ``commands`` maps rover labels (``rover_0`` ...) to ``(v [m/s], w [rad/s])``;
        rovers not listed stop. Without ``commands`` the configured motion drives them.
        """
        if commands is None:
            cmd = self.controller(self.planar_poses())
        else:
            unknown = set(commands) - {r.label for r in self.rovers}
            if unknown:
                raise KeyError(f"Unknown rovers {sorted(unknown)}")
            cmd = np.array([commands.get(r.label, (0.0, 0.0)) for r in self.rovers], dtype=float)
        obs = self._apply(cmd)
        self.step_count += 1
        self._update_overseer()
        if self.step_count % self.cfg.output.record_every == 0:
            self.record()
        return obs

    def _apply(self, cmd: np.ndarray):
        import torch

        unwrapped = self.env.unwrapped
        scale_lin = np.array([r.scale_linear for r in self.rovers])
        scale_ang = np.array([r.scale_angular for r in self.rovers])
        normalized = normalize_commands(np.asarray(cmd, dtype=float), scale_lin, scale_ang)
        action = torch.zeros((unwrapped.num_envs, self._action_dim), device=unwrapped.device)
        for rover, values in zip(self.rovers, normalized):
            if rover.drive is not None:
                action[:, rover.drive] = torch.as_tensor(values, device=unwrapped.device, dtype=action.dtype)
        self._last_cmd = np.where(
            np.array([[r.drive is not None] for r in self.rovers]),
            normalized * np.stack((scale_lin, scale_ang), axis=-1),
            0.0,
        )
        obs, *_ = self.env.step(action)
        return obs

    def _update_overseer(self) -> None:
        if self.overseer is None or OVERSEER_LABEL not in self._camera_sensors:
            return
        pose = self.overseer.update(self._group_center())
        if pose is None:
            return
        import torch

        eye, target = pose
        device = self.env.unwrapped.device
        camera = self.env.unwrapped.scene.sensors[OVERSEER_SENSOR]
        camera.set_world_poses_from_view(
            torch.tensor([eye], dtype=torch.float32, device=device),
            torch.tensor([target], dtype=torch.float32, device=device),
        )

    ###############
    ## Recording ##
    ###############

    def record(self, frames: dict[str, dict[str, np.ndarray]] | None = None) -> None:
        if self.recorder is None:
            return
        frames = frames if frames is not None else self.camera_frames()
        frame, step, time_s = self.recorder.frame, self.step_count, self.step_count * self._step_dt()
        poses = self.rover_poses()
        self.recorder.log_trajectories(
            [frame, step, f"{time_s:.4f}", rover.label, *_fmt(poses[rover.label][0]), *_fmt(poses[rover.label][1]),
             f"{self._last_cmd[i, 0]:.4f}", f"{self._last_cmd[i, 1]:.4f}"]
            for i, rover in enumerate(self.rovers)
        )
        rows = []
        for label, data in frames.items():
            self.recorder.save_camera(
                label, step=step, time_s=time_s, rgb=data["rgb"], depth=data.get("depth"),
                pos=data["pos"], quat=data["quat_ros"],
            )
            if label == OVERSEER_LABEL:
                continue
            observer = label.split("__", 1)[0]
            targets = [r for r in self.rovers if r.label != observer]
            if not targets:
                continue
            points = np.array([poses[r.label][0] for r in targets]) + np.array((0.0, 0.0, _VISIBILITY_TARGET_OFFSET_M))
            height, width = data["rgb"].shape[:2]
            for target, (u, v, rng, in_fov, visible) in zip(
                targets, visibility(data["K"], data["pos"], data["quat_ros"], (width, height), points, data.get("depth"))
            ):
                rows.append([frame, step, label, target.label, f"{u:.1f}", f"{v:.1f}", f"{rng:.3f}", int(in_fov), int(visible)])
        self.recorder.log_observations(rows)
        self.recorder.next_frame()

    def camera_frames(self) -> dict[str, dict[str, np.ndarray]]:
        """Latest image, metric depth, intrinsics and ROS-optical pose per camera."""
        scene = self.env.unwrapped.scene
        poses = self.rover_poses()
        frames = {}
        for label, sensor_name in self._camera_sensors.items():
            data = scene.sensors[sensor_name].data
            output = data.output
            rgb = _np(output["rgb"])[0, ..., :3]
            depth = None
            for key in ("distance_to_image_plane", "depth"):
                if key in output and label != OVERSEER_LABEL:
                    depth = _np(output[key])[0]
                    depth = depth[..., 0] if depth.ndim == 3 else depth
                    break
            if label in self._camera_mounts:
                rover, mount = self._camera_mounts[label]
                pos, quat = mount(*poses[rover.label])
            else:  # the overseer is placed through USD, so its reported pose is current
                pos, quat = _np(data.pos_w)[0], _np(data.quat_w_ros)[0]
            frames[label] = {
                "rgb": rgb.astype(np.uint8, copy=False),
                "depth": depth,
                "K": _np(data.intrinsic_matrices)[0],
                "pos": pos,
                "quat_ros": quat,
            }
        return frames

    def rover_poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """World position and ``(x, y, z, w)`` orientation of every rover root."""
        return {
            rover.label: (_np(rover.asset.data.root_pos_w)[0].copy(), _np(rover.asset.data.root_quat_w)[0].copy())
            for rover in self.rovers
        }

    def planar_poses(self) -> np.ndarray:
        """``(n, 3)`` env-local ``(x, y, heading)``; heading is the rover's forward direction."""
        origin = _np(self.env.unwrapped.scene.env_origins)[0]
        poses = self.rover_poses()
        return np.array(
            [
                (
                    *(poses[r.label][0][:2] - origin[:2]),
                    float(yaw_from_quat_xyzw(poses[r.label][1])[0]) + r.heading_offset,
                )
                for r in self.rovers
            ]
        )

    #############
    ## Helpers ##
    #############

    def _group_center(self) -> np.ndarray:
        if self.env is None or any(r.asset is None for r in self.rovers):
            return np.array((*self.spawn[:, :2].mean(axis=0), 0.0))
        return np.mean([pos for pos, _ in self.rover_poses().values()], axis=0)

    def _follow_gaps(self) -> np.ndarray:
        """Follow distance per pair: at least the spacing, plus room for both footprints."""
        radii = np.array([r.spawn.get("radius_m", 0.5) for r in self.rovers])
        return np.maximum(self.cfg.layout.spacing_m, radii[1:] + radii[:-1] + 1.5)

    def _group_radius(self) -> float:
        xy = self.spawn[:, :2]
        return float(np.max(np.linalg.norm(xy - xy.mean(axis=0), axis=-1))) if len(xy) else 0.0

    def _step_dt(self) -> float:
        return float(self.env.unwrapped.step_dt) if self.env is not None else 0.04

    def _write_scene_yaml(self, *, finished: bool = False, failed: bool = False) -> None:
        run_info = {
            "argv": self.argv,
            "started": self.started_at.isoformat(timespec="seconds"),
            "status": "failed" if failed else ("finished" if finished else "running"),
            "task": TASK_ID,
            "run_dir": str(self.run_dir),
            "step_dt_s": self._step_dt(),
            "steps_run": self.step_count,
            "frames": self.recorder.frame if self.recorder else 0,
            "terrain_height_field_points": self.height_field_points,
            "rovers": [
                {
                    "label": r.label,
                    "type": r.type,
                    "scene_key": r.key,
                    "drive_term": r.drive_term,
                    "action_slice": None if r.drive is None else [r.drive.start, r.drive.stop],
                    "scale_linear": r.scale_linear,
                    "scale_angular": r.scale_angular,
                    "cameras": r.cameras,
                    "spawn": r.spawn,
                }
                for r in self.rovers
            ],
            "cameras": dict(self._camera_sensors),
        }
        write_scene_yaml(self.run_dir / "scene.yaml", self.cfg, run_info)


def _np(value) -> np.ndarray:
    """ProxyArray / torch / array → numpy (CPU)."""
    tensor = getattr(value, "torch", value)
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _fmt(values) -> list[str]:
    return [f"{float(v):.6f}" for v in np.asarray(values).reshape(-1)]


__all__ = ("MultiRoverSetup", "RoverHandle", "TASK_ID")
