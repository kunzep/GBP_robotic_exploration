"""Isaac-free, ROS-free tests: python3 -m pytest swarm_slam/cslam_bridge/tests -q"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # swarm_slam/

from cslam_bridge.evaluate import (  # noqa: E402
    decode_key,
    encode_key,
    evaluate,
    read_g2o,
    rmse,
    transform_points,
    umeyama,
)
from cslam_bridge.odometry import drifted_odometry  # noqa: E402
from cslam_bridge.run_data import STAMP_OFFSET_S, load_run  # noqa: E402
from multi_rover_setup.cameras import matrix_to_quat_xyzw  # noqa: E402

K = [[10.0, 0.0, 4.0], [0.0, 10.0, 3.0], [0.0, 0.0, 1.0]]


def _pose(x, y, yaw):
    """Camera looking along world +x rotated by yaw, in ROS optical convention."""
    c, s = np.cos(yaw), np.sin(yaw)
    forward, left = np.array([c, s, 0.0]), np.array([-s, c, 0.0])
    T = np.eye(4)
    T[:3, :3] = np.column_stack((-left, (0.0, 0.0, -1.0), forward))  # x right, y down, z forward
    T[:3, 3] = (x, y, 0.5)
    return T


def _write_camera(cam_dir: Path, poses, has_depth=True):
    (cam_dir / "rgb").mkdir(parents=True)
    if has_depth:
        (cam_dir / "depth").mkdir()
    (cam_dir / "intrinsics.json").write_text(json.dumps({"width": 8, "height": 6, "K": K, "has_depth": has_depth}))
    rows = ["frame,step,time_s,x,y,z,qx,qy,qz,qw"]
    for k, T in enumerate(poses):
        Image.fromarray(np.full((6, 8, 3), k, np.uint8)).save(cam_dir / "rgb" / f"{k:06d}.png")
        if has_depth:
            Image.fromarray(np.full((6, 8), 1000 + k, np.uint16)).save(cam_dir / "depth" / f"{k:06d}.png")
        q = matrix_to_quat_xyzw(T[:3, :3])
        rows.append(",".join(map(str, (k, 2 * k, 0.08 * k, *T[:3, 3], *q))))
    (cam_dir / "poses.csv").write_text("\n".join(rows) + "\n")


@pytest.fixture
def run_dir(tmp_path):
    """Two rovers driving straight lines in opposite directions, plus an overseer without depth."""
    n = 10
    _write_camera(tmp_path / "cameras" / "rover_0__front", [_pose(0.1 * k, 0.0, 0.0) for k in range(n)])
    _write_camera(tmp_path / "cameras" / "rover_1__front", [_pose(2.0 - 0.1 * k, 1.0, np.pi) for k in range(n)])
    _write_camera(tmp_path / "cameras" / "overseer", [_pose(0.0, 0.0, 0.0)], has_depth=False)
    return tmp_path


def test_load_run_skips_overseer_and_offsets_stamps(run_dir):
    streams = load_run(run_dir)
    assert [(s.robot_id, s.label) for s in streams] == [(0, "rover_0__front"), (1, "rover_1__front")]
    s = streams[1]
    assert len(s) == 10 and s.resolution == (8, 6)
    assert s.stamps[0] == pytest.approx(STAMP_OFFSET_S)
    np.testing.assert_allclose(s.poses[3], _pose(1.7, 1.0, np.pi), atol=1e-5)
    assert s.nearest(STAMP_OFFSET_S + 0.17) == 2


def test_zero_drift_is_ground_truth_relative_to_start(run_dir):
    poses = load_run(run_dir)[0].poses
    odom = drifted_odometry(poses)
    expected = np.linalg.inv(poses[0]) @ poses
    np.testing.assert_allclose(odom, expected, atol=1e-9)


def test_drift_grows_with_distance():
    poses = np.array([_pose(0.1 * k, 0.0, 0.0) for k in range(200)])
    odom = drifted_odometry(poses, yaw_bias_deg_per_m=1.0)
    truth = np.linalg.inv(poses[0]) @ poses
    err = np.linalg.norm(odom[:, :3, 3] - truth[:, :3, 3], axis=-1)
    assert err[0] == 0.0 and err[50] > 0.0 and err[-1] > 5 * err[50]


def test_key_round_trip():
    key = encode_key(3, 1234)
    assert key == 7441072484322903250  # gtsam::LabeledSymbol('g', 'D', 1234): char << 56 | label << 48 | index
    assert decode_key(key) == (3, 1234)


def test_umeyama_recovers_rigid_transform():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(20, 3))
    T = _pose(1.0, -2.0, 0.7)
    est = umeyama(pts, transform_points(T, pts))
    np.testing.assert_allclose(est, T, atol=1e-9)
    assert rmse(pts, pts) == 0.0


def _write_slam_logs(slam_dir: Path, streams, offset: np.ndarray):
    """Fake cslam logs: every 3rd frame is a keyframe, poses = ground truth moved by ``offset``."""
    folder = slam_dir / "01-01-2026_00-00-00_experiment_robot_0" / "01-01-2026_00-01-00"
    folder.mkdir(parents=True)
    vertices, edges = [], []
    for s in streams:
        stamp_rows = ["vertice_id,sec,nanosec"]
        for kf, k in enumerate(range(0, len(s), 3)):
            key = encode_key(s.robot_id, kf)
            T = offset @ s.poses[k]
            vertices.append(f"VERTEX_SE3:QUAT {key} {' '.join(map(str, T[:3, 3]))} {' '.join(map(str, matrix_to_quat_xyzw(T[:3, :3])))}")
            if kf:
                edges.append(f"EDGE_SE3:QUAT {encode_key(s.robot_id, kf - 1)} {key} 0 0 0 0 0 0 1")
            sec = int(s.stamps[k])
            stamp_rows.append(f"{key},{sec},{int(round((s.stamps[k] - sec) * 1e9))}")
        (folder / f"pose_timestamps{s.robot_id}.csv").write_text("\n".join(stamp_rows) + "\n")
    edges.append(f"EDGE_SE3:QUAT {encode_key(0, 1)} {encode_key(1, 2)} 0 0 0 0 0 0 1")  # inter-robot
    (folder / "optimized_global_pose_graph.g2o").write_text("\n".join(vertices + edges) + "\n")
    return folder


def test_evaluate_perfect_merge_has_zero_ate(run_dir):
    streams = load_run(run_dir)
    folder = _write_slam_logs(run_dir / "slam", streams, offset=_pose(5.0, 3.0, 1.0))
    graph = read_g2o(folder / "optimized_global_pose_graph.g2o")
    assert len(graph.vertices) == 8 and graph.inter_robot_edges() == [(encode_key(0, 1), encode_key(1, 2))]

    result = evaluate(streams, run_dir / "slam")
    assert result.log_dir == folder
    assert [len(t.keys) for t in result.tracks] == [4, 4]
    for track in result.tracks:
        assert track.ate_optimized == pytest.approx(0.0, abs=1e-6)


def test_bag_round_trip(run_dir, tmp_path):
    pytest.importorskip("rosbags")
    from rosbags.rosbag2 import Reader

    from cslam_bridge.bags import write_bag
    from cslam_bridge.messages import TYPESTORE

    stream = load_run(run_dir)[1]
    bag = write_bag(stream, tmp_path / "bags" / "r1", drifted_odometry(stream.poses))
    with Reader(bag) as reader:
        counts = {c.topic: c.msgcount for c in reader.connections}
        assert counts == {t: 10 for t in ("/r1/color/image", "/r1/depth/image", "/r1/color/camera_info", "/r1/odom")}
        for conn, t_ns, raw in reader.messages():
            if conn.topic == "/r1/depth/image":
                msg = TYPESTORE.deserialize_cdr(raw, conn.msgtype)
                depth = np.frombuffer(msg.data.tobytes(), "<u2").reshape(msg.height, msg.width)
                assert msg.encoding == "16UC1" and msg.header.frame_id == "r1_camera" and depth[0, 0] == 1000
                assert t_ns == int(STAMP_OFFSET_S * 1e9)
                break
