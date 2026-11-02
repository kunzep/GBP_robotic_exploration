"""Configuration of a multi-rover scene (Isaac-free).

The rover list is defined only in YAML config files (``presets/default.yaml``
unless ``--config`` points elsewhere); every other section can additionally be
overridden from the command line.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
SWARM_SLAM_DIR = PACKAGE_DIR.parent
PRESETS_DIR = PACKAGE_DIR / "presets"
DEFAULT_PRESET = PRESETS_DIR / "default.yaml"
DEFAULT_OUTPUT_ROOT = SWARM_SLAM_DIR / "out" / "multi_rover_setup"

TERRAIN_KINDS = (
    "moon_surface",
    "mars_surface",
    "apollo17",
    "apollo15",
    "jezero",
    "jezero_v2",
    "flat",
)
LAYOUT_KINDS = ("ring", "line", "grid", "random", "explicit")
HEADINGS = ("chain", "tangent", "center", "random")
MOTION_KINDS = ("follow", "orbit", "wander", "constant", "idle")
OVERSEER_MODES = ("track", "fixed", "orbit", "off")
DEPTH_FORMATS = ("png16", "npy", "both", "none")
QUALITIES = ("quality", "balanced", "performance")
SKYDOMES = ("high_res", "low_res", "none")
LODS = ("low", "medium", "high")
AUGMENTS = ("none", "simforge", "database")


@dataclass
class RoverSpec:
    type: str
    """Registered SRB robot name (``srb ls robot``), e.g. ``leo_rover`` or ``name+payload``."""


@dataclass
class TerrainCfg:
    kind: str = "moon_surface"
    # Procedural (moon_surface / mars_surface)
    size_m: float = 48.0
    texture_px: int = 4096
    density: float | None = None
    flat_area_m: float | None = None
    # Real DEM sites (apollo17 / apollo15 / jezero / jezero_v2)
    lod: str = "high"
    patch_size_m: float | None = None
    augment: str = "none"


@dataclass
class LayoutCfg:
    kind: str = "ring"
    spacing_m: float = 4.0
    heading: str = "chain"
    positions: list[list[float]] | None = None
    """Explicit ``[x, y]`` or ``[x, y, yaw_deg]`` per rover (``kind: explicit``)."""
    spawn_clearance_m: float = 0.3


@dataclass
class MotionCfg:
    kind: str = "follow"
    speed_mps: float = 0.25
    cmd: list[float] = field(default_factory=lambda: [0.2, 0.0])
    """``[v m/s, w rad/s]`` applied to every rover for ``kind: constant``."""


@dataclass
class CamerasCfg:
    resolution: list[int] = field(default_factory=lambda: [640, 480])
    hfov_deg: float | None = 90.0
    """Horizontal field of view of every rover camera; ``None`` keeps each asset's own lens."""
    depth_format: str = "png16"


@dataclass
class OverseerCfg:
    mode: str = "track"
    resolution: list[int] = field(default_factory=lambda: [1920, 1080])
    distance_m: float | None = None
    """Horizontal eye distance from the rover centroid; ``None`` scales with the group."""
    height_m: float | None = None
    """Eye height above the centroid; ``None`` scales with the group."""
    azimuth_deg: float = -120.0
    focal_length: float = 18.0
    orbit_deg_per_s: float = 6.0


@dataclass
class SimCfg:
    steps: int = 500
    settle_steps: int = 60
    warmup_steps: int = 8
    seed: int = 0
    headless: bool = True
    quality: str = "quality"
    skydome: str = "high_res"
    sun_rpy: list[float] = field(default_factory=lambda: [45.0, 30.0, 0.0])
    kit_args: str = ""


@dataclass
class OutputCfg:
    root: str = str(DEFAULT_OUTPUT_ROOT)
    name: str | None = None
    """Replaces the descriptive part of the run folder name (after the timestamp)."""
    record_every: int = 5
    video: bool = False
    snapshot_only: bool = False


_SECTIONS: dict[str, type] = {
    "terrain": TerrainCfg,
    "layout": LayoutCfg,
    "motion": MotionCfg,
    "cameras": CamerasCfg,
    "overseer": OverseerCfg,
    "sim": SimCfg,
    "output": OutputCfg,
}
# Keys written into ``scene.yaml`` for provenance; ignored when loading it back.
_PROVENANCE_KEYS = ("run",)


@dataclass
class MultiRoverSetupCfg:
    rovers: list[RoverSpec] = field(default_factory=list)
    terrain: TerrainCfg = field(default_factory=TerrainCfg)
    layout: LayoutCfg = field(default_factory=LayoutCfg)
    motion: MotionCfg = field(default_factory=MotionCfg)
    cameras: CamerasCfg = field(default_factory=CamerasCfg)
    overseer: OverseerCfg = field(default_factory=OverseerCfg)
    sim: SimCfg = field(default_factory=SimCfg)
    output: OutputCfg = field(default_factory=OutputCfg)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiRoverSetupCfg:
        data = dict(data or {})
        for key in _PROVENANCE_KEYS:
            data.pop(key, None)
        unknown = set(data) - {"rovers", *_SECTIONS}
        if unknown:
            raise ValueError(f"Unknown config sections: {sorted(unknown)}")
        rovers = []
        for i, entry in enumerate(data.get("rovers") or []):
            if isinstance(entry, str):
                entry = {"type": entry}
            if not isinstance(entry, dict) or "type" not in entry:
                raise ValueError(f"rovers[{i}] must be a robot name or {{type: <name>}}")
            rovers.append(RoverSpec(type=str(entry["type"])))
        sections = {
            name: _section_from_dict(section_cls, data.get(name) or {}, name)
            for name, section_cls in _SECTIONS.items()
        }
        return cls(rovers=rovers, **sections)

    @classmethod
    def from_yaml(cls, path: str | Path) -> MultiRoverSetupCfg:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def validate(self) -> None:
        if not self.rovers:
            raise ValueError("No rovers configured; add a `rovers:` list to the config file")
        _check_choice("terrain.kind", self.terrain.kind, TERRAIN_KINDS)
        _check_choice("terrain.lod", self.terrain.lod, LODS)
        _check_choice("terrain.augment", self.terrain.augment, AUGMENTS)
        _check_choice("layout.kind", self.layout.kind, LAYOUT_KINDS)
        _check_choice("layout.heading", self.layout.heading, HEADINGS)
        _check_choice("motion.kind", self.motion.kind, MOTION_KINDS)
        _check_choice("overseer.mode", self.overseer.mode, OVERSEER_MODES)
        _check_choice("cameras.depth_format", self.cameras.depth_format, DEPTH_FORMATS)
        _check_choice("sim.quality", self.sim.quality, QUALITIES)
        _check_choice("sim.skydome", self.sim.skydome, SKYDOMES)
        for name, res in (
            ("cameras.resolution", self.cameras.resolution),
            ("overseer.resolution", self.overseer.resolution),
        ):
            if len(res) != 2 or min(res) <= 0:
                raise ValueError(f"{name} must be [width, height] > 0, got {res}")
        if self.cameras.hfov_deg is not None and not 1.0 < self.cameras.hfov_deg < 179.0:
            raise ValueError(f"cameras.hfov_deg must be in (1, 179), got {self.cameras.hfov_deg}")
        if self.terrain.size_m <= 0 or self.terrain.texture_px <= 0:
            raise ValueError("terrain.size_m and terrain.texture_px must be > 0")
        if self.layout.spacing_m <= 0:
            raise ValueError("layout.spacing_m must be > 0")
        if self.layout.kind == "explicit":
            positions = self.layout.positions or []
            if len(positions) != len(self.rovers):
                raise ValueError(
                    f"layout.positions needs one entry per rover "
                    f"({len(self.rovers)}), got {len(positions)}"
                )
            for pos in positions:
                if len(pos) not in (2, 3):
                    raise ValueError(f"layout.positions entries are [x, y(, yaw_deg)], got {pos}")
        if len(self.motion.cmd) != 2:
            raise ValueError(f"motion.cmd must be [v, w], got {self.motion.cmd}")
        if self.output.record_every <= 0:
            raise ValueError("output.record_every must be >= 1")
        if self.sim.steps < 0 or self.sim.settle_steps < 0 or self.sim.warmup_steps < 1:
            raise ValueError("sim.steps/settle_steps must be >= 0 and sim.warmup_steps >= 1")


def _check_choice(name: str, value: str, choices: Sequence[str]) -> None:
    if value not in choices:
        raise ValueError(f"{name}={value!r} is not one of {list(choices)}")


def _section_from_dict(section_cls: type, data: dict[str, Any], name: str) -> Any:
    if not isinstance(data, dict):
        raise ValueError(f"Config section `{name}` must be a mapping")
    known = {f.name for f in dataclasses.fields(section_cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown keys in `{name}`: {sorted(unknown)} (known: {sorted(known)})")
    return section_cls(**data)


#########
## CLI ##
#########


def parse_resolution(text: str) -> list[int]:
    try:
        width, height = (int(v) for v in text.lower().replace(",", "x").split("x"))
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"Expected WIDTHxHEIGHT, got {text!r}") from err
    return [width, height]


def _floats(n: int):
    def parse(text: str) -> list[float]:
        values = [float(v) for v in text.split(",")]
        if len(values) != n:
            raise argparse.ArgumentTypeError(f"Expected {n} comma-separated numbers, got {text!r}")
        return values

    return parse


# (flag, config path, argparse kwargs); `None` defaults keep the config-file value.
_CLI_OPTIONS: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("--terrain", "terrain.kind", {"choices": TERRAIN_KINDS}),
    ("--terrain-size", "terrain.size_m", {"type": float, "help": "procedural extent [m]"}),
    ("--terrain-texture", "terrain.texture_px", {"type": int, "help": "procedural albedo/normal px"}),
    ("--terrain-density", "terrain.density", {"type": float}),
    ("--terrain-flat-area", "terrain.flat_area_m", {"type": float}),
    ("--terrain-lod", "terrain.lod", {"choices": LODS, "help": "real DEM mesh/texture LOD"}),
    ("--terrain-patch", "terrain.patch_size_m", {"type": float, "help": "real DEM patch [m]"}),
    ("--terrain-augment", "terrain.augment", {"choices": AUGMENTS}),
    ("--layout", "layout.kind", {"choices": LAYOUT_KINDS}),
    ("--spacing", "layout.spacing_m", {"type": float}),
    ("--heading", "layout.heading", {"choices": HEADINGS}),
    ("--motion", "motion.kind", {"choices": MOTION_KINDS}),
    ("--speed", "motion.speed_mps", {"type": float}),
    ("--cmd", "motion.cmd", {"type": _floats(2), "help": "v,w for --motion constant"}),
    ("--res", "cameras.resolution", {"type": parse_resolution, "help": "rover cameras WxH"}),
    ("--hfov", "cameras.hfov_deg", {"type": float, "help": "rover camera horizontal FOV [deg]"}),
    ("--depth", "cameras.depth_format", {"choices": DEPTH_FORMATS}),
    ("--overseer", "overseer.mode", {"choices": OVERSEER_MODES}),
    ("--overseer-res", "overseer.resolution", {"type": parse_resolution}),
    ("--overseer-distance", "overseer.distance_m", {"type": float}),
    ("--overseer-height", "overseer.height_m", {"type": float}),
    ("--overseer-azimuth", "overseer.azimuth_deg", {"type": float}),
    ("--steps", "sim.steps", {"type": int}),
    ("--settle-steps", "sim.settle_steps", {"type": int}),
    ("--seed", "sim.seed", {"type": int}),
    ("--quality", "sim.quality", {"choices": QUALITIES}),
    ("--skydome", "sim.skydome", {"choices": SKYDOMES}),
    ("--sun", "sim.sun_rpy", {"type": _floats(3), "help": "roll,pitch,yaw [deg]"}),
    ("--kit-args", "sim.kit_args", {"type": str}),
    ("--out", "output.root", {"type": str, "help": "output root folder"}),
    ("--name", "output.name", {"type": str, "help": "run folder tag"}),
    ("--record-every", "output.record_every", {"type": int}),
)
_CLI_SWITCHES: tuple[tuple[str, str, bool, str], ...] = (
    ("--gui", "sim.headless", False, "open the Isaac Sim window"),
    ("--video", "output.video", True, "encode mp4 per camera at the end"),
    ("--snapshot-only", "output.snapshot_only", True, "spawn, settle, save one frame, exit"),
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multiple rovers on terrain with per-rover and overseer cameras.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PRESET,
        help=f"YAML scene config incl. the rover list (default: {DEFAULT_PRESET.name})",
    )
    for flag, path, kwargs in _CLI_OPTIONS:
        parser.add_argument(flag, dest=path, default=None, **kwargs)
    for flag, path, value, help_text in _CLI_SWITCHES:
        parser.add_argument(flag, dest=path, action="store_const", const=value, help=help_text)
    return parser


def apply_overrides(cfg: MultiRoverSetupCfg, overrides: dict[str, Any]) -> MultiRoverSetupCfg:
    """Set ``section.key`` values (``None`` values are skipped)."""
    for path, value in overrides.items():
        if value is None:
            continue
        section_name, key = path.split(".", 1)
        section = getattr(cfg, section_name)
        if not hasattr(section, key):
            raise ValueError(f"Unknown config key `{path}`")
        setattr(section, key, value)
    return cfg


def parse_cli(argv: Sequence[str] | None = None) -> MultiRoverSetupCfg:
    args = vars(build_arg_parser().parse_args(argv))
    cfg = MultiRoverSetupCfg.from_yaml(args.pop("config"))
    apply_overrides(cfg, args)
    cfg.validate()
    return cfg
