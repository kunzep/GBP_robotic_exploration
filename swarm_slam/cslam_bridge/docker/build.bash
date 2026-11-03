#!/usr/bin/env bash
# Build the Swarm-SLAM (cslam) image: ./build.bash [extra docker build args]
exec docker build -t swarm_slam_cslam:jazzy "$@" "$(dirname "$(readlink -f "$0")")"
