"""Bridge between ``multi_rover_setup`` run folders and MISTLab Swarm-SLAM (``cslam``).

Isaac-free and ROS-free: bags are written with the pure-Python ``rosbags``
package, and cslam's logs are read back for evaluation. See ``README.md``.
"""

from .run_data import CameraStream, load_run

__all__ = ("CameraStream", "load_run")
