"""Replay a multi_rover_setup run through Swarm-SLAM (cslam), one namespace /r<id> per rover.

Per rover: cslam's three nodes, optionally RTAB-Map RGB-D visual odometry, and
``ros2 bag play`` of ``<run_dir>/bags/r<id>``. cslam logs (g2o graphs, keyframe
stamps) go to ``<run_dir>/slam``. The launch shuts down ``linger_s`` after the
last bag finished, which leaves time for a final pose graph optimization.

    ros2 launch swarm_slam/cslam_bridge/launch/srb_rgbd.launch.py run_dir:=<run> [odom:=vo|bag]

odom:=vo   RTAB-Map rgbd_odometry on the replayed images (bags written with --odom vo)
odom:=bag  odometry recorded in the bags (bags written with --odom gt_drift)
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, RegisterEventHandler, Shutdown, TimerAction
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

CONFIG = Path(__file__).resolve().parent.parent / "config" / "srb_rgbd.yaml"


def robot_nodes(robot_id: int, num_robots: int, args: dict) -> list:
    ns = f"/r{robot_id}"
    params = [
        args["config"],
        {
            "robot_id": robot_id,
            "max_nb_robots": num_robots,
            "frontend.nn_checkpoint": args["nn_checkpoint"],
            "evaluation.log_folder": str(Path(args["run_dir"]) / "slam"),
        },
    ]
    nodes = [
        Node(package="cslam", executable="loop_closure_detection_node.py", name="cslam_loop_closure_detection",
             namespace=ns, parameters=params),
        Node(package="cslam", executable="map_manager", name="cslam_map_manager", namespace=ns, parameters=params),
        Node(package="cslam", executable="pose_graph_manager", name="cslam_pose_graph_manager",
             namespace=ns, parameters=params),
    ]
    if args["odom"] == "vo":
        # As cslam_experiments' rtabmap_rgbd_odometry.launch.py, but the odometry child frame is
        # the camera optical frame (no TF needed) and each rover gets its own odom frame.
        nodes.append(Node(
            package="rtabmap_odom", executable="rgbd_odometry", name="rgbd_odometry", namespace=ns, output="screen",
            parameters=[{
                "frame_id": f"r{robot_id}_camera",
                "odom_frame_id": f"r{robot_id}_odom",
                "publish_tf": False,
                "subscribe_depth": True,
                "approx_sync": False,  # bag images, depth and camera_info share exact stamps
                "Vis/MinInliers": "10",
            }],
            remappings=[("rgb/image", "color/image"), ("rgb/camera_info", "color/camera_info"),
                        ("depth/image", "depth/image"), ("odom", "odom")],
        ))
    return nodes


def launch_setup(context, *_, **__):
    args = {name: LaunchConfiguration(name).perform(context)
            for name in ("run_dir", "odom", "rate", "start_delay_s", "linger_s", "nn_checkpoint", "config")}
    if args["odom"] not in ("vo", "bag"):
        raise ValueError(f"odom must be vo or bag, got {args['odom']!r}")
    args["run_dir"] = str(Path(args["run_dir"]).resolve())  # follows the `latest` link
    bags = sorted((Path(args["run_dir"]) / "bags").glob("r[0-9]*"), key=lambda p: int(p.name[1:]))
    if not bags:
        raise FileNotFoundError(f"No bags under {args['run_dir']}/bags; run tools/run_to_bags.py first")

    actions, players = [], []
    for bag in bags:
        actions += robot_nodes(int(bag.name[1:]), len(bags), args)
        players.append(ExecuteProcess(cmd=["ros2", "bag", "play", str(bag), "--rate", args["rate"]], output="screen"))
    # All bags start together once every node had time to load its model.
    actions.append(TimerAction(period=float(args["start_delay_s"]), actions=players))

    finished = set()

    def on_bag_done(event, _context):
        finished.add(event.action)
        if len(finished) == len(players):
            return [TimerAction(period=float(args["linger_s"]), actions=[Shutdown(reason="all bags replayed")])]
        return None

    actions += [RegisterEventHandler(OnProcessExit(target_action=p, on_exit=on_bag_done)) for p in players]
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("run_dir", description="multi_rover_setup run folder containing bags/"),
        DeclareLaunchArgument("odom", default_value="vo", description="vo | bag"),
        DeclareLaunchArgument("rate", default_value="0.5", description="bag playback rate"),
        DeclareLaunchArgument("start_delay_s", default_value="15.0", description="wait before playing bags"),
        DeclareLaunchArgument("linger_s", default_value="30.0", description="wait after the bags before shutdown"),
        DeclareLaunchArgument("nn_checkpoint", default_value="/opt/cslam_models/resnet18_64.pth",
                              description="CosPlace weights (absolute path)"),
        DeclareLaunchArgument("config", default_value=str(CONFIG)),
        OpaqueFunction(function=launch_setup),
    ])
