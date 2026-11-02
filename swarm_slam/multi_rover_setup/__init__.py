"""Multi-rover scene setup for swarm SLAM on top of Space Robotics Bench.

Importing this package does not start Isaac Sim; :class:`MultiRoverSetup`
launches it when entered (``with MultiRoverSetup(cfg) as scene: ...``).
"""

from .config import DEFAULT_PRESET, MultiRoverSetupCfg, parse_cli
from .setup import MultiRoverSetup

__all__ = ("DEFAULT_PRESET", "MultiRoverSetup", "MultiRoverSetupCfg", "parse_cli")
