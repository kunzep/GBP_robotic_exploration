# Swarm-SLAM (cslam) for offline RGB-D runs on rosbag2 data.
# Adapted from lajoiepy/cslam_experiments docker/Dockerfile.lidar, without TEASER++ (lidar only).
FROM ros:jazzy-perception
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# Tools + RTAB-Map (cslam's RGB-D front end and the rgbd_odometry node from rtabmap_odom).
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl python3-pip python3-vcstool ros-jazzy-rtabmap-ros \
    && rm -rf /var/lib/apt/lists/*

# Swarm-SLAM sources, pinned in swarm_slam.repos.
WORKDIR /opt/cslam_ws
COPY swarm_slam.repos .
RUN mkdir src && vcs import src < swarm_slam.repos \
    && apt-get update && rosdep update --rosdistro jazzy \
    && rosdep install --from-paths src --ignore-src -r -y \
    && rm -rf /var/lib/apt/lists/*
RUN . /opt/ros/jazzy/setup.bash && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# Python deps: Swarm-SLAM requirements.txt + rosbags (bag writing) + matplotlib (plots).
# Runs after colcon; setuptools<80 keeps colcon-core working for later rebuilds (torch pulls a newer one).
# psutil comes from Debian (no pip RECORD), so pip must replace it rather than uninstall it.
# numpy<2 keeps the apt numpy ABI that cv_bridge and other ROS Python bindings are built against.
RUN pip install --no-cache-dir --break-system-packages --ignore-installed psutil \
    && pip install --no-cache-dir --break-system-packages \
        "numpy<2" "setuptools<80" networkx numba open3d Pillow scikit_learn scipy sortedcontainers \
        torch torchvision distinctipy numpy-quaternion rosbags matplotlib

# CosPlace ResNet-18/64-d weights (cslam's default frontend.nn_checkpoint is models/resnet18_64.pth,
# resolved relative to the cslam share dir; upstream models/download.sh is only a TODO).
# Symlink it there so the default works; an absolute path in the config also works.
RUN mkdir -p /opt/cslam_models \
    && curl -fsSL -o /opt/cslam_models/resnet18_64.pth \
        https://github.com/gmberton/CosPlace/releases/download/v1.0/resnet18_64_cosplace.pth \
    && share=$(. install/setup.bash && ros2 pkg prefix cslam)/share/cslam \
    && mkdir -p "$share/models" && ln -sf /opt/cslam_models/resnet18_64.pth "$share/models/resnet18_64.pth"

# cosplace_utils/network.py calls resnet18(pretrained=True): pre-fill the torch hub cache (offline runs).
RUN python3 -c "import torchvision; torchvision.models.resnet18(weights='IMAGENET1K_V1')"

# Source ROS + overlay for interactive shells and one-shot commands alike.
RUN echo "source /opt/cslam_ws/install/setup.bash" >> /root/.bashrc \
    && sed -i 's|/opt/ros/\$ROS_DISTRO/setup.bash"|/opt/cslam_ws/install/setup.bash"|' /ros_entrypoint.sh
WORKDIR /root/ws
