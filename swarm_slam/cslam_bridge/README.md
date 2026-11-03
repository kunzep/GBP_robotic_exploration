# cslam_bridge: Swarm-SLAM on multi_rover_setup runs

Offline proof of concept: replay a recorded `multi_rover_setup` run through
[MISTLab Swarm-SLAM](https://github.com/MISTLab/Swarm-SLAM) (`cslam`, ROS 2 Jazzy), then
check the merged map against the ground truth.

```
SRB container                       Swarm-SLAM container (docker/)
multi_rover_setup.py ──> <run>/ ──> run_to_bags.py ──> <run>/bags/r<id>/
                                    srb_rgbd.launch.py ──> <run>/slam/ (cslam logs)
                                    eval_slam.py ──> <run>/slam/{trajectories,loop_closures}.png, ate.csv
```

## Steps

1. Record a run at 12.5 Hz in the SRB container:

   ```bash
   python3 swarm_slam/tools/multi_rover_setup.py --config swarm_slam/multi_rover_setup/presets/slam_playground.yaml
   ```

2. Run the remaining steps in the Swarm-SLAM container (`swarm_slam/cslam_bridge/docker/run.bash`). Set the run folder:

   ```bash
   RUN=swarm_slam/out/multi_rover_setup/latest
   ```

3. Choose an odometry source. Convert the bags to match, then launch with the same mode:

   | Odometry | Convert | Launch |
   |---|---|---|
   | Visual odometry (RTAB-Map) | `python3 swarm_slam/tools/run_to_bags.py $RUN` | `ros2 launch swarm_slam/cslam_bridge/launch/srb_rgbd.launch.py run_dir:=$RUN` |
   | Fallback: ground truth + drift (`--drift-trans 0` makes it exact) | `python3 swarm_slam/tools/run_to_bags.py $RUN --odom gt_drift` | `ros2 launch swarm_slam/cslam_bridge/launch/srb_rgbd.launch.py run_dir:=$RUN odom:=bag` |

4. Evaluate:

   ```bash
   python3 swarm_slam/tools/eval_slam.py $RUN
   ```

While the launch runs, `ros2 topic echo /cslam/inter_robot_loop_closure` shows the loop-closure
attempts between rovers; `success: true` marks the ones that passed. The launch ends by itself
`linger_s` after the last bag finishes.

## Contract

| What | Value |
|---|---|
| Namespace / cslam `robot_id` | `/r<id>`, rovers in order `rover_0, rover_1, ...` |
| Colour image | `/r<id>/color/image` (`rgb8`) |
| Depth image | `/r<id>/depth/image` (`16UC1`, millimetres, 0 = invalid) |
| Camera info | `/r<id>/color/camera_info` (pinhole, no distortion) |
| Odometry | `/r<id>/odom` (`nav_msgs/Odometry`, `r<id>_odom` → `r<id>_camera`) |
| Sensor frame | `r<id>_camera` = ROS optical frame (z forward, x right, y down); no TF needed |
| Stamps | simulation time + 1 s (`run_data.STAMP_OFFSET_S`) |

The topic names and frames are defined in `messages.py`, and `config/srb_rgbd.yaml` must use the
same ones. A later live mode can publish the same messages from `MultiRoverSetup.record()` with
rclpy. The Swarm-SLAM side stays unchanged; only `ros2 bag play` is dropped.

## Outputs (`<run>/slam/`)

| File | Contents |
|---|---|
| `<date>_experiment_robot_<id>/<date>/` | Raw cslam logs: `optimized_global_pose_graph.g2o`, `initial_global_pose_graph.g2o`, `pose_timestamps<id>.csv` and `log.csv` |
| `trajectories.png` | Top-down view with three panels: ground truth, odometry and optimized. Grey lines are inter-robot loop closures. |
| `loop_closures.png` | Camera images of the matched keyframe pairs, to check by eye |
| `ate.csv` | Per-rover ATE (RMSE in metres) of odometry and of optimized keyframe positions |

How the ATE is aligned to ground truth:

- **Odometry** is aligned rover by rover, because the rovers share no frame before the maps are merged.
- **Optimized** positions are aligned with a single transform for all rovers, so a wrong merge shows up as a displaced rover.

## Tests

The tests need neither Isaac nor ROS; `rosbags` must be installed for the bag round trip.

```bash
python3 -m pytest swarm_slam/cslam_bridge/tests -q
```
