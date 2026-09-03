"""Compare native primary positions with an existing binary XYZ PLY sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("native_directory", type=Path)
parser.add_argument("particle_directory", type=Path)
parser.add_argument("report", type=Path)
parser.add_argument("--frames", nargs="+", type=int, required=True)
args = parser.parse_args()


def read_xyz_ply(path: Path) -> np.ndarray:
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
        raise ValueError(f"Unexpected binary XYZ PLY layout: {path}")
    return np.frombuffer(payload, dtype="<f4").reshape(count, 3).copy()


records = []
for frame in args.frames:
    native_path = args.native_directory / f"native_diffuse_{frame:04d}.npz"
    particle_path = args.particle_directory / f"particles_{frame:04d}.ply"
    with np.load(native_path, allow_pickle=False) as cache:
        native = np.asarray(cache["primary_position_inv_mass"][:, :3], dtype=np.float32)
    particles = read_xyz_ply(particle_path)
    if native.shape != particles.shape:
        raise RuntimeError(
            f"Frame {frame} shape mismatch: native={native.shape}, PLY={particles.shape}"
        )
    error = np.linalg.norm(native.astype(np.float64) - particles.astype(np.float64), axis=1)
    records.append(
        {
            "frame": frame,
            "particles": len(native),
            "maximum_error_m": float(error.max()),
            "p99_error_m": float(np.percentile(error, 99.0)),
            "mean_error_m": float(error.mean()),
            "exact_float32_match": bool(np.array_equal(native, particles)),
        }
    )

report = {
    "schema": 1,
    "product": "native_diffuse_primary_alignment_audit",
    "native_directory": str(args.native_directory.resolve()),
    "particle_directory": str(args.particle_directory.resolve()),
    "frames": records,
    "exact_sequence_match": all(record["exact_float32_match"] for record in records),
    "maximum_error_m": max(record["maximum_error_m"] for record in records),
}
args.report.parent.mkdir(parents=True, exist_ok=True)
args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
