"""Isaac-free tests: python3 -m pytest swarm_slam/multi_rover_setup/tests -q"""

import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # swarm_slam/

from multi_rover_setup.cameras import MountedPose, OverseerRig, matrix_to_quat_xyzw, project_points, quat_xyzw_to_matrix, visibility  # noqa: E402
from multi_rover_setup.config import (  # noqa: E402
    DEFAULT_PRESET,
    LayoutCfg,
    MotionCfg,
    MultiRoverSetupCfg,
    OverseerCfg,
    build_arg_parser,
    parse_cli,
)
from multi_rover_setup.layout import (  # noqa: E402
    MotionController,
    normalize_commands,
    quat_xyzw_from_yaw,
    spawn_layout,
    wrap_angle,
    yaw_from_quat_xyzw,
)
from multi_rover_setup.recorder import Recorder, depth_to_png16, make_run_dir, run_tag, write_scene_yaml  # noqa: E402
from multi_rover_setup.terrain import TerrainHeightField  # noqa: E402


############
## Config ##
############


def test_default_preset_loads_and_validates():
    cfg = MultiRoverSetupCfg.from_yaml(DEFAULT_PRESET)
    cfg.validate()
    assert [r.type for r in cfg.rovers] == ["leo_rover", "perseverance", "cadre"]


def test_cli_overrides_config_but_not_rovers(tmp_path):
    preset = tmp_path / "scene.yaml"
    preset.write_text("rovers: [husky, pragyan]\ncameras: {resolution: [320, 240]}\nmotion: {kind: idle}\n")
    cfg = parse_cli(["--config", str(preset), "--res", "1280x720", "--terrain", "mars_surface", "--video"])
    assert [r.type for r in cfg.rovers] == ["husky", "pragyan"]
    assert cfg.cameras.resolution == [1280, 720]
    assert cfg.terrain.kind == "mars_surface"
    assert cfg.motion.kind == "idle"  # untouched by the CLI
    assert cfg.output.video is True
    assert cfg.sim.headless is True
    flags = {a for action in build_arg_parser()._actions for a in action.option_strings}
    assert not any("rover" in flag for flag in flags)


def test_scene_yaml_round_trip(tmp_path):
    cfg = MultiRoverSetupCfg.from_yaml(DEFAULT_PRESET)
    path = tmp_path / "scene.yaml"
    write_scene_yaml(path, cfg, {"argv": ["x"], "rovers": [{"label": "rover_0"}]})
    again = MultiRoverSetupCfg.from_yaml(path)
    assert again.to_dict() == cfg.to_dict()


def test_invalid_config_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown keys"):
        MultiRoverSetupCfg.from_dict({"rovers": ["cadre"], "terrain": {"kindd": "x"}})
    cfg = MultiRoverSetupCfg.from_dict({"rovers": []})
    with pytest.raises(ValueError, match="No rovers"):
        cfg.validate()


############
## Layout ##
############


@pytest.mark.parametrize("kind", ["ring", "line", "grid", "random"])
def test_layout_spacing(kind):
    poses = spawn_layout(5, LayoutCfg(kind=kind, spacing_m=4.0), seed=3)
    xy = poses[:, :2]
    dists = np.linalg.norm(xy[:, None] - xy[None], axis=-1) + np.eye(5) * 1e9
    assert dists.min() >= 4.0 - 1e-6
    assert np.allclose(xy.mean(axis=0), 0.0, atol=1e-6) or kind in ("grid",)


def test_ring_headings():
    poses = spawn_layout(4, LayoutCfg(kind="ring", spacing_m=4.0, heading="center"))
    to_center = np.arctan2(-poses[:, 1], -poses[:, 0])
    assert np.allclose(wrap_angle(poses[:, 2] - to_center), 0.0, atol=1e-9)
    poses = spawn_layout(4, LayoutCfg(kind="ring", spacing_m=4.0, heading="tangent"))
    radial = np.arctan2(poses[:, 1], poses[:, 0])
    assert np.allclose(wrap_angle(poses[:, 2] - radial - np.pi / 2), 0.0, atol=1e-9)


def test_chain_heading_faces_previous_rover():
    poses = spawn_layout(4, LayoutCfg(kind="ring", spacing_m=4.0, heading="chain"))
    for i in range(1, 4):
        ahead = poses[i - 1, :2] - poses[i, :2]
        assert wrap_angle(poses[i, 2] - np.arctan2(ahead[1], ahead[0])) == pytest.approx(0.0, abs=1e-9)
    # Driving counter-clockwise from its spawn, rover 0 heads away from rover 1.
    forward = np.array([np.cos(poses[0, 2]), np.sin(poses[0, 2])])
    assert np.dot(forward, poses[1, :2] - poses[0, :2]) < 0


def test_follow_keeps_predecessor_in_view():
    dt = 0.04
    cfg = LayoutCfg(kind="ring", spacing_m=4.0, heading="chain")
    poses = spawn_layout(3, cfg)
    controller = MotionController(
        MotionCfg(kind="follow", speed_mps=0.25), center=np.zeros(2), radius=2.31, bound_m=20.0, gap_m=4.0,
        max_lin=np.array([0.4, 0.7, 0.4]), max_ang=np.full(3, 1.0), dt=dt,
    )
    travelled = np.zeros(3)
    for _ in range(3000):
        before = poses[:, :2].copy()
        poses = _simulate(controller, poses, 1, dt)
        travelled += np.linalg.norm(poses[:, :2] - before, axis=-1)
        for i in (1, 2):
            ahead = poses[i - 1, :2] - poses[i, :2]
            assert abs(wrap_angle(np.arctan2(ahead[1], ahead[0]) - poses[i, 2])) < math.radians(45)
    assert travelled.min() > 5.0
    gaps = np.linalg.norm(poses[:-1, :2] - poses[1:, :2], axis=-1)
    assert np.all(gaps > 2.0) and np.all(gaps < 6.0)


def test_layout_shrinks_to_bound():
    with pytest.warns(UserWarning, match="shrinking"):
        poses = spawn_layout(6, LayoutCfg(kind="line", spacing_m=10.0), bound_m=8.0)
    assert np.abs(poses[:, :2]).max() <= 8.0 + 1e-9


def test_explicit_layout():
    cfg = LayoutCfg(kind="explicit", positions=[[1, 2, 90], [3, 4, 0]])
    poses = spawn_layout(2, cfg)
    assert np.allclose(poses, [[1, 2, math.pi / 2], [3, 4, 0]])


def test_quaternion_yaw_round_trip():
    yaw = np.array([-2.5, 0.0, 1.0, 3.0])
    assert np.allclose(yaw_from_quat_xyzw(quat_xyzw_from_yaw(yaw)), yaw)


def _simulate(controller, poses, steps, dt):
    for _ in range(steps):
        cmd = controller(poses)
        poses[:, 0] += cmd[:, 0] * np.cos(poses[:, 2]) * dt
        poses[:, 1] += cmd[:, 0] * np.sin(poses[:, 2]) * dt
        poses[:, 2] = wrap_angle(poses[:, 2] + cmd[:, 1] * dt)
    return poses


def test_orbit_converges_to_ring_and_keeps_moving():
    dt, radius = 0.04, 4.0
    controller = MotionController(
        MotionCfg(kind="orbit", speed_mps=0.3), center=np.zeros(2), radius=radius, bound_m=20.0,
        max_lin=np.array([0.4, 0.7, 0.1]), max_ang=np.full(3, 1.0), dt=dt,
    )
    assert controller.speed == pytest.approx(0.08)  # limited by the slowest rover (0.8 * 0.1)
    controller = MotionController(
        MotionCfg(kind="orbit", speed_mps=0.3), center=np.zeros(2), radius=radius, bound_m=20.0,
        max_lin=np.array([0.4, 0.7]), max_ang=np.full(2, 1.0), dt=dt,
    )
    start = np.array([[6.0, 0.0, 0.0], [-2.0, 0.5, 2.0]])
    end = _simulate(controller, start.copy(), 4000, dt)
    assert np.allclose(np.linalg.norm(end[:, :2], axis=-1), radius, atol=0.25)
    after = _simulate(controller, end.copy(), 100, dt)
    assert np.linalg.norm(after[:, :2] - end[:, :2], axis=-1).min() > 0.5


def test_wander_stays_in_bounds():
    dt = 0.04
    controller = MotionController(
        MotionCfg(kind="wander", speed_mps=0.4), center=np.zeros(2), radius=4.0, bound_m=10.0,
        max_lin=np.full(3, 0.5), max_ang=np.full(3, 1.0), dt=dt, seed=1,
    )
    poses = _simulate(controller, np.zeros((3, 3)), 10000, dt)
    assert np.linalg.norm(poses[:, :2], axis=-1).max() < 12.0


def test_normalize_commands():
    out = normalize_commands(np.array([[0.2, 0.5], [5.0, -5.0]]), np.array([0.4, 0.4]), np.array([1.0, 1.0]))
    assert np.allclose(out, [[0.5, 0.5], [1.0, -1.0]])


#############
## Terrain ##
#############


def test_height_field_on_tilted_plane():
    xs, ys = np.meshgrid(np.linspace(-10, 10, 81), np.linspace(-10, 10, 81))
    zs = 0.1 * xs + 2.0
    field = TerrainHeightField(np.stack((xs, ys, zs), axis=-1).reshape(-1, 3), cell_m=0.25)
    assert field.height_at(0.0, 0.0, radius_m=0.0) == pytest.approx(2.0, abs=0.05)
    assert field.height_at(5.0, 3.0, radius_m=1.0) == pytest.approx(2.6, abs=0.06)
    assert field.height_at(50.0, 50.0) is None
    coarse = TerrainHeightField(np.array([[0.0, 0.0, 1.0], [4.0, 0.0, 2.0], [0.0, 4.0, 3.0]]), cell_m=0.25)
    assert coarse.height_at(3.5, 0.2, radius_m=0.3) == pytest.approx(2.0)  # nearest vertex 0.5 m away


#############
## Cameras ##
#############


def _look_along_x():
    # ROS optical frame (z fwd, x right, y down) looking along world +X, level.
    rot = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)  # columns: cam axes in world
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(rot).as_quat()  # (x, y, z, w)


def test_projection_and_visibility():
    K = np.array([[320.0, 0, 320], [0, 320.0, 240], [0, 0, 1]])
    quat = _look_along_x()
    uv, z = project_points(K, np.zeros(3), quat, np.array([[5.0, 0.0, 0.0], [5.0, 1.0, 0.0], [-5.0, 0, 0]]))
    assert np.allclose(uv[0], [320, 240]) and z[0] == pytest.approx(5.0)
    assert uv[1, 0] < 320  # world +Y is image left
    rows = visibility(K, np.zeros(3), quat, (640, 480), np.array([[5.0, 0, 0], [-5.0, 0, 0]]))
    assert rows[0][3] and rows[0][4]
    assert not rows[1][3] and not rows[1][4]
    occluder = np.full((480, 640), 2.0)  # terrain 2 m in front
    rows = visibility(K, np.zeros(3), quat, (640, 480), np.array([[5.0, 0, 0]]), occluder)
    assert rows[0][3] and not rows[0][4]


def test_mounted_pose_follows_root():
    cam_quat = _look_along_x()
    mount = MountedPose(np.zeros(3), np.array([0, 0, 0, 1.0]), np.array([0.5, 0.0, 0.3]), cam_quat)
    yaw90 = quat_xyzw_from_yaw(np.pi / 2)[0]
    pos, quat = mount(np.array([1.0, 2.0, 0.0]), yaw90)
    assert np.allclose(pos, [1.0, 2.5, 0.3])
    forward = quat_xyzw_to_matrix(quat)[:, 2]  # optical z axis in world
    assert np.allclose(forward, [0, 1, 0], atol=1e-9)
    rot = quat_xyzw_to_matrix(np.array([0.3, -0.2, 0.5, 0.8]) / np.linalg.norm([0.3, -0.2, 0.5, 0.8]))
    assert np.allclose(quat_xyzw_to_matrix(matrix_to_quat_xyzw(rot)), rot)


def test_overseer_track_and_fixed():
    rig = OverseerRig(OverseerCfg(mode="track", azimuth_deg=180.0), group_center=np.zeros(3), group_radius=3.0, dt=0.04)
    eye, target = rig.update(np.zeros(3))
    assert eye[0] < 0 and eye[2] > 0 and np.allclose(target, 0)
    for _ in range(500):
        eye, target = rig.update(np.array([4.0, 0.0, 0.0]))
    assert np.allclose(target, [4, 0, 0], atol=1e-3)
    fixed = OverseerRig(OverseerCfg(mode="fixed"), group_center=np.zeros(3), group_radius=3.0, dt=0.04)
    assert fixed.update(np.zeros(3)) is not None
    assert fixed.update(np.ones(3)) is None


############
## Output ##
############


def test_run_dir_naming_and_latest(tmp_path):
    cfg = MultiRoverSetupCfg.from_yaml(DEFAULT_PRESET)
    tag = run_tag(cfg)
    assert tag == "moon_surface__3rovers__follow__640x480"
    assert not any(r.type in tag for r in cfg.rovers)
    now = datetime(2026, 11, 2, 15, 4, 11)
    first = make_run_dir(tmp_path, tag, now=now)
    second = make_run_dir(tmp_path, tag, now=now)
    assert first.name == "2026-11-02_15-04-11__moon_surface__3rovers__follow__640x480"
    assert second.name.endswith("-2")
    assert (tmp_path / "latest").resolve() == second.resolve()
    cfg.output.name = "my test/run"
    assert run_tag(cfg) == "my_test_run"


def test_recorder_writes_frames(tmp_path):
    rec = Recorder(tmp_path, depth_format="both")
    rec.add_camera("rover_0__front", intrinsics=np.eye(3), resolution=(8, 6), info={})
    depth = np.full((6, 8), 1.2345, dtype=np.float32)
    depth[0, 0] = np.inf
    rec.save_camera("rover_0__front", step=0, time_s=0.0, rgb=np.zeros((6, 8, 3), np.uint8), depth=depth,
                    pos=np.zeros(3), quat=np.array([0, 0, 0, 1.0]))
    rec.next_frame()
    rec.close()
    cam = tmp_path / "cameras" / "rover_0__front"
    from PIL import Image

    png = np.array(Image.open(cam / "depth" / "000000.png"))
    assert png[1, 1] == 1234 and png[0, 0] == 0
    assert np.load(cam / "depth" / "000000.npy")[1, 1] == pytest.approx(1.2345)
    assert (cam / "rgb" / "000000.png").exists() and (cam / "intrinsics.json").exists()
    assert len((cam / "poses.csv").read_text().splitlines()) == 2


def test_depth_png16_clips():
    assert depth_to_png16(np.array([np.nan, -1.0, 100.0]))[2] == 65535
