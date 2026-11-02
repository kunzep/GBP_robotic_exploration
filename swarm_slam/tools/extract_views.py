#!/usr/bin/env python3
"""Extract camera views (RGB + depth) from SRB collect HDF5 logs.

Every `obs/image_*` key is exported, so a 3rd-person camera shows up
automatically once it is recorded (e.g. `image_3rd` / `image_scene`).

Usage (from swarm_slam/):
    python3 tools/extract_views.py                 # latest log, frame 50
    python3 tools/extract_views.py -f 0 50 99      # several frames
    python3 tools/extract_views.py -f all -d demo_0
    python3 tools/extract_views.py --hdf5 ../logs/.../x.hdf5
"""

import argparse
import glob
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent.parent  # swarm_slam/
LOGS = HERE.parent / "logs"


def latest_hdf5(task: str | None = None) -> Path:
    root = LOGS / task if task else LOGS
    paths = glob.glob(str(root / "**/*.hdf5"), recursive=True)
    if not paths:
        raise SystemExit(
            f"No HDF5 files under {root} — record one with `srb agent collect` first"
        )
    return Path(max(paths, key=lambda p: Path(p).stat().st_mtime))


def depth_to_png(d: np.ndarray) -> Image.Image:
    d = d.astype(np.float32)
    finite = np.isfinite(d)
    if not finite.any():
        return Image.fromarray(np.zeros(d.shape, np.uint8))
    lo, hi = d[finite].min(), d[finite].max()
    d = np.where(finite, (d - lo) / max(hi - lo, 1e-6), 0.0)
    return Image.fromarray((d * 255).astype(np.uint8))


def save_view(img: np.ndarray, out: Path, stem: str) -> None:
    rgb = img[..., :3]
    if rgb.dtype != np.uint8:  # float images in [0,1]
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(rgb).save(out / f"{stem}_rgb.png")
    if img.shape[-1] > 3:
        depth_to_png(img[..., 3]).save(out / f"{stem}_depth.png")
        np.save(out / f"{stem}_depth.npy", img[..., 3])  # raw metric depth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", type=Path, default=None)
    ap.add_argument("-t", "--task", default=None, help="logs/<task> folder to search")
    ap.add_argument("-d", "--demo", default=None, help="default: all demos")
    ap.add_argument("-f", "--frames", nargs="+", default=["50"])
    ap.add_argument("-o", "--out", type=Path, default=HERE / "out" / "views")
    args = ap.parse_args()

    path = args.hdf5 or latest_hdf5(args.task)
    if not path.is_file():
        raise SystemExit(f"Not an HDF5 file: '{path}' (did `srb agent collect` run?)")
    print(f"Reading {path}")
    with h5py.File(path, "r") as f:
        demos = [args.demo] if args.demo else sorted(f["data"].keys())
        for demo in demos:
            obs = f[f"data/{demo}/obs"]
            cams = [k for k in obs.keys() if k.startswith("image")]
            if not cams:
                print(f"{demo}: no image observations")
                continue
            n = obs[cams[0]].shape[0]
            frames = range(n) if args.frames == ["all"] else [int(i) for i in args.frames]
            out = args.out / path.stem / demo
            out.mkdir(parents=True, exist_ok=True)
            for cam in cams:
                for i in frames:
                    if not -n <= i < n:
                        print(f"skip {cam}[{i}] (len {n})")
                        continue
                    save_view(obs[cam][i], out, f"{cam}_{i % n:04d}")
            print(f"{demo}: {cams} -> {out}")


if __name__ == "__main__":
    main()
