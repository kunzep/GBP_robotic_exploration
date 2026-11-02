# Project: swarm SLAM experiments in Space Robotics Bench (SRB)

## Setup (Saturn, TU Delft, no sudo, rootless Docker)
- Repo: ~/space_robotics_bench (= /root/ws in container). Work dir: my_work/
- Daemon (only after reboot / if `docker ps` fails):
  export XDG_RUNTIME_DIR=/home/philipp/.docker/run
  setsid nohup dockerd-rootless.sh --exec-opt native.cgroupdriver=cgroupfs > ~/dockerd.log 2>&1 &
- Container: export SRB_DOCKER_IMAGE=space_robotics_bench:6.1.0
  PATH=~/nosudo:$PATH SRB_DOCKER_PROFILE=simulator WITH_GUI=false ./.docker/run.bash
  (use ./.docker/dev.bash when editing srb/tasks). No monitor: headless only.
- GPU: Quadro RTX 5000 16 GB. Shell in container: fish.

## What works
- srb agent zero --headless -e _ground env.domain=moon env.robot=leo_rover env.num_envs=1 'env.active_sensors=[proprioception,contact,transform,visual]'
- srb agent collect ... --collector zero --horizon 100 env.camera_resolution=[320,240] (needs num_envs=1)
- HDF5 at logs/_ground/collect/<ts>/datasets/*.hdf5: data/demo_0/obs/image_front (T,240,320,4 uint8:
  RGB + depth in channel 3), obs/vector (T,21, unlabeled), actions (T,2).
- Depth is uint8 normalized over near/far (default ~20 m); env.camera_clipping_range=[near,far].
- --snapshot PATH renders one PNG headlessly; --snapshot-settle N lets robots settle first.

## Findings
- No LiDAR in SRB's sensor catalog. Wheeled rovers have cam_front (RGB-D), imu_robot, contacts_wheels.
- env.num_envs makes isolated parallel copies (32 m apart), so rovers in different envs cannot see each other.
- Multi-rover-in-one-scene is not documented; candidates: formation_following(_decentralized), else custom task
  copied from the _ground template under srb/tasks (register in docs/data/tasks.toml, then srb cache update).
- Frames: world(GT) -> map -> odom_i -> base_i -> cam_i. Check world vs env-relative poses; check camera
  convention (ROS optical vs OpenGL).

## Plan
1. my_work/scripts: extract onboard images + snapshots from HDF5 / runs.
2. Test environments with --snapshot.
3. Custom multi-rover lunar task from the most suitable template.

## Notes
- srb needs the container. One-shot form to try: ./.docker/run.bash srb agent zero ... (verify).
- Later: swarm SLAM; maybe DelftBlue HPC for large runs.