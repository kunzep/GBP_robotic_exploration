#!/usr/bin/env bash
# Run the Swarm-SLAM (cslam) image with the repo mounted at /root/ws.
#   ./run.bash          -> interactive bash
#   ./run.bash CMD...   -> one-shot command (ROS + cslam overlay sourced)
# Rootless docker: the container's root maps to the host user, so files written to /root/ws stay yours.
REPO_DIR="$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)"
OPTS=(--rm --gpus all --net host --ipc host -v "${REPO_DIR}:/root/ws" -w /root/ws)
[[ -t 0 ]] && OPTS+=(-it)
if [[ $# -eq 0 ]]; then
    exec docker run "${OPTS[@]}" swarm_slam_cslam:jazzy bash
fi
exec docker run "${OPTS[@]}" swarm_slam_cslam:jazzy "$@"
