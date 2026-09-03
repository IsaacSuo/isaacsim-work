"""Export temporally stabilized primary XYZ PLY from native Diffuse frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("native_directory", type=Path)
parser.add_argument("output_directory", type=Path)
parser.add_argument("--frames", nargs="+", type=int, required=True)
parser.add_argument(
    "--previous-particle-directory",
    type=Path,
    help="Audited prior sequence used only when the requested first frame needs frame-1.",
)
args = parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ply(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        count = None
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"Truncated PLY header: {path}")
            if line.startswith(b"element vertex "):
                count = int(line.split()[-1])
            if line.strip() == b"end_header":
                break
        payload = stream.read()
    if count is None or len(payload) != count * 12:
        raise ValueError(f"Unexpected XYZ PLY layout: {path}")
    return np.frombuffer(payload, dtype="<f4").reshape(count, 3).copy()


def native_positions(frame: int) -> np.ndarray:
    path = args.native_directory / f"native_diffuse_{frame:04d}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as cache:
            return np.asarray(cache["primary_position_inv_mass"][:, :3], dtype=np.float32)
    if args.previous_particle_directory is not None:
        fallback = args.previous_particle_directory / f"particles_{frame:04d}.ply"
        if fallback.is_file():
            return read_ply(fallback)
    raise FileNotFoundError(path)


def write_ply(path: Path, positions: np.ndarray) -> None:
    xyz = np.ascontiguousarray(positions, dtype="<f4")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment stabilized same-index PhysX primary particles\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        xyz.tofile(stream)


frames = sorted(args.frames)
if len(frames) != len(set(frames)) or any(b != a + 1 for a, b in zip(frames, frames[1:])):
    raise ValueError("frames must be unique and contiguous")
if args.output_directory.exists():
    raise FileExistsError(f"Refusing to overwrite output: {args.output_directory}")
args.output_directory.mkdir(parents=True)
records = []
for frame in frames:
    previous = native_positions(frame - 1) if frame > frames[0] or args.previous_particle_directory else native_positions(frame)
    current = native_positions(frame)
    following = native_positions(frame + 1) if frame < frames[-1] else current
    if previous.shape != current.shape or following.shape != current.shape:
        raise RuntimeError(f"Stable particle order/count contract failed at frame {frame}")
    stabilized = 0.25 * previous + 0.50 * current + 0.25 * following
    output_path = args.output_directory / f"particles_{frame:04d}.ply"
    write_ply(output_path, stabilized)
    records.append(
        {
            "frame": frame,
            "file": output_path.name,
            "particles": len(stabilized),
            "sha256": sha256_file(output_path),
            "maximum_displacement_from_raw_m": float(
                np.linalg.norm(stabilized.astype(np.float64) - current, axis=1).max()
            ),
        }
    )
    print(f"[native-primary-ply] frame={frame:04d} particles={len(stabilized)}", flush=True)

manifest = {
    "schema": 1,
    "product": "native_primary_stabilized_ply",
    "state": {"complete": True, "frames": len(records)},
    "source": str(args.native_directory.resolve()),
    "previous_frame_fallback": (
        str(args.previous_particle_directory.resolve())
        if args.previous_particle_directory is not None
        else None
    ),
    "filter": {"previous": 0.25, "current": 0.50, "following": 0.25},
    "stable_identity": "native primary array index",
    "frames": records,
}
(args.output_directory / "manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
