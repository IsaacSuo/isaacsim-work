#!/usr/bin/env python3
"""Convert audited PhysX primary-particle samples to Partio BGEO v5.

This converter deliberately has no finite-difference velocity path.  Every
output velocity is copied bit-for-bit from the ``velocities`` array exported
by PhysX/UsdGeom.PointInstancer after ``fetch_results()``.

The BGEO writer implements the small, documented subset consumed by the
Partio version bundled with SPlisHSPlasH: ``position`` plus one vector
attribute named ``velocity``.  Files are written atomically and then fully
read back by default so a malformed or reordered payload cannot silently enter
FoamGenerator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import BinaryIO

import numpy as np


SCHEMA = "physx-native-primary-partio/v1"
BGEO_MAGIC = b"Bgeo"
BGEO_VERSION_MARKER = b"V"
BGEO_VERSION = 5
VELOCITY_ATTRIBUTE = b"velocity"
HOUDINI_VECTOR_TYPE = 5
FLOATS_PER_PARTICLE = 7  # position.xyz, homogeneous w, velocity.xyz
BYTES_PER_PARTICLE = FLOATS_PER_PARTICLE * 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert PhysX-native PointInstancer positions and velocities from "
            "NPZ to Partio BGEO without estimating, smoothing, or transforming velocity."
        )
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-sample", type=int, default=None)
    parser.add_argument("--end-sample", type=int, default=None)
    parser.add_argument(
        "--chunk-particles",
        type=int,
        default=65536,
        help="Particles processed per bounded-memory write/read block.",
    )
    parser.add_argument(
        "--no-readback-verify",
        action="store_true",
        help="Skip full payload readback. Not recommended for production conversion.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only outputs already recorded and hashed in the output manifest.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def write_houdini_string(stream: BinaryIO, value: bytes) -> None:
    if len(value) > 0xFFFF:
        raise ValueError("BGEO attribute name is too long")
    stream.write(struct.pack(">H", len(value)))
    stream.write(value)


def write_bgeo(
    path: Path,
    positions: np.ndarray,
    velocities: np.ndarray,
    chunk_particles: int,
) -> None:
    count = int(positions.shape[0])
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(BGEO_MAGIC)
            stream.write(BGEO_VERSION_MARKER)
            # version, points, prims, point groups, prim groups, point attrs,
            # vertex attrs, primitive attrs, fixed/detail attrs
            stream.write(struct.pack(">9i", BGEO_VERSION, count, 0, 0, 0, 1, 0, 0, 0))
            write_houdini_string(stream, VELOCITY_ATTRIBUTE)
            stream.write(struct.pack(">Hi", 3, HOUDINI_VECTOR_TYPE))
            stream.write(struct.pack(">3i", 0, 0, 0))

            for begin in range(0, count, chunk_particles):
                end = min(begin + chunk_particles, count)
                block = np.empty((end - begin, FLOATS_PER_PARTICLE), dtype=">f4")
                block[:, 0:3] = positions[begin:end]
                block[:, 3] = 1.0
                block[:, 4:7] = velocities[begin:end]
                stream.write(memoryview(block).cast("B"))

            # No primitives or fixed attributes; this is Partio's BGEO trailer.
            stream.write(b"\x00\xff")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_and_validate_header(stream: BinaryIO, expected_count: int) -> None:
    magic = stream.read(4)
    marker = stream.read(1)
    header_raw = stream.read(9 * 4)
    if magic != BGEO_MAGIC or marker != BGEO_VERSION_MARKER or len(header_raw) != 36:
        raise ValueError("Invalid BGEO magic or truncated header")
    values = struct.unpack(">9i", header_raw)
    expected = (BGEO_VERSION, expected_count, 0, 0, 0, 1, 0, 0, 0)
    if values != expected:
        raise ValueError(f"Unexpected BGEO header: {values!r}, expected {expected!r}")

    name_length_raw = stream.read(2)
    if len(name_length_raw) != 2:
        raise ValueError("Truncated BGEO attribute name length")
    name_length = struct.unpack(">H", name_length_raw)[0]
    name = stream.read(name_length)
    definition = stream.read(6)
    defaults = stream.read(12)
    if name != VELOCITY_ATTRIBUTE or len(definition) != 6 or len(defaults) != 12:
        raise ValueError("Missing exact BGEO velocity vector attribute")
    size, attribute_type = struct.unpack(">Hi", definition)
    if size != 3 or attribute_type != HOUDINI_VECTOR_TYPE or defaults != b"\x00" * 12:
        raise ValueError("Malformed BGEO velocity vector definition")


def verify_bgeo_payload(
    path: Path,
    positions: np.ndarray,
    velocities: np.ndarray,
    chunk_particles: int,
) -> None:
    count = int(positions.shape[0])
    expected_size = 4 + 1 + 9 * 4 + 2 + len(VELOCITY_ATTRIBUTE) + 2 + 4 + 12
    expected_size += count * BYTES_PER_PARTICLE + 2
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"BGEO size mismatch: {actual_size} != {expected_size}")

    with path.open("rb") as stream:
        read_and_validate_header(stream, count)
        for begin in range(0, count, chunk_particles):
            end = min(begin + chunk_particles, count)
            particle_count = end - begin
            raw = stream.read(particle_count * BYTES_PER_PARTICLE)
            if len(raw) != particle_count * BYTES_PER_PARTICLE:
                raise ValueError("Truncated BGEO particle payload")
            block = np.frombuffer(raw, dtype=">f4").reshape(particle_count, FLOATS_PER_PARTICLE)
            # Comparison is exact: BGEO stores the same IEEE-754 float32 bits,
            # changing only byte order on disk.
            if not np.array_equal(block[:, 0:3], positions[begin:end]):
                raise ValueError(f"Position readback mismatch at particles {begin}:{end}")
            if not np.all(block[:, 3] == np.float32(1.0)):
                raise ValueError(f"Homogeneous coordinate mismatch at particles {begin}:{end}")
            if not np.array_equal(block[:, 4:7], velocities[begin:end]):
                raise ValueError(f"Native velocity readback mismatch at particles {begin}:{end}")
        if stream.read(2) != b"\x00\xff" or stream.read(1) != b"":
            raise ValueError("Invalid BGEO trailer or trailing bytes")


def validate_source_arrays(
    source_path: Path,
    expected_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(source_path, allow_pickle=False) as sample:
        if "positions" not in sample.files:
            raise ValueError(f"{source_path}: missing required positions array")
        if "velocities" not in sample.files:
            raise ValueError(
                f"{source_path}: missing PhysX-native velocities; finite-difference fallback is forbidden"
            )
        positions = np.ascontiguousarray(sample["positions"])
        velocities = np.ascontiguousarray(sample["velocities"])

    expected_shape = (expected_count, 3)
    for name, array in (("positions", positions), ("velocities", velocities)):
        if array.dtype != np.dtype("float32"):
            raise ValueError(f"{source_path}: {name} must be float32, got {array.dtype}")
        if array.shape != expected_shape:
            raise ValueError(f"{source_path}: {name} shape {array.shape} != {expected_shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{source_path}: {name} contains NaN or infinity")
    return positions, velocities


def validate_manifest(source_manifest_path: Path, manifest: dict) -> tuple[list[dict], float]:
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Source manifest has no samples")
    particle_count = manifest.get("particle_count")
    if not isinstance(particle_count, int) or particle_count <= 0:
        raise ValueError("Source manifest particle_count is invalid")

    indices = [sample.get("sample_index") for sample in samples]
    if indices != list(range(len(samples))):
        raise ValueError("Source sample indices are not contiguous from zero")
    times = np.asarray([sample.get("simulation_time") for sample in samples], dtype=np.float64)
    if not np.isfinite(times).all() or len(times) < 2:
        raise ValueError("At least two finite simulation times are required")
    deltas = np.diff(times)
    timestep = float(deltas[0])
    tolerance = max(1.0e-12, abs(timestep) * 1.0e-9)
    if timestep <= 0.0 or not np.all(np.abs(deltas - timestep) <= tolerance):
        raise ValueError(
            "FoamGenerator requires one fixed timestep; source manifest is not uniformly sampled"
        )

    steps = [sample.get("physics_step") for sample in samples]
    if not all(isinstance(step, int) for step in steps):
        raise ValueError("Source physics_step values are missing or invalid")
    step_deltas = np.diff(np.asarray(steps, dtype=np.int64))
    if not np.all(step_deltas == step_deltas[0]) or int(step_deltas[0]) <= 0:
        raise ValueError("Source physics steps are not uniformly sampled")

    for sample in samples:
        source_path = source_manifest_path.parent / str(sample.get("file", ""))
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source sample: {source_path}")
    return samples, timestep


def load_resume_manifest(path: Path, expected_source_sha256: str) -> dict[int, dict]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        existing = json.load(stream)
    if existing.get("schema") != SCHEMA:
        raise ValueError(f"Cannot resume incompatible manifest schema in {path}")
    if existing.get("source_manifest_sha256") != expected_source_sha256:
        raise ValueError("Cannot resume: source manifest hash changed")
    return {int(item["sample_index"]): item for item in existing.get("samples", [])}


def main() -> int:
    args = parse_args()
    source_manifest_path = args.source_manifest.resolve()
    output_directory = args.output.resolve()
    if args.chunk_particles <= 0:
        raise ValueError("chunk-particles must be positive")
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    with source_manifest_path.open("r", encoding="utf-8") as stream:
        source_manifest = json.load(stream)

    samples, timestep = validate_manifest(source_manifest_path, source_manifest)
    first = 0 if args.start_sample is None else args.start_sample
    last = len(samples) - 1 if args.end_sample is None else args.end_sample
    if first < 0 or last < first or last >= len(samples):
        raise ValueError(f"Invalid sample range {first}..{last} for {len(samples)} samples")

    output_directory.mkdir(parents=True, exist_ok=True)
    output_manifest_path = output_directory / "manifest.json"
    source_manifest_sha256 = sha256_file(source_manifest_path)
    prior = load_resume_manifest(output_manifest_path, source_manifest_sha256) if args.resume else {}
    if output_manifest_path.exists() and not args.resume:
        raise FileExistsError(
            f"Output manifest already exists: {output_manifest_path}; use a new directory or --resume"
        )

    selected = samples[first : last + 1]
    result_samples: list[dict] = []
    manifest = {
        "schema": SCHEMA,
        "complete": False,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": source_manifest_sha256,
        "source_coordinates": source_manifest.get("coordinates"),
        "output_coordinates": source_manifest.get("coordinates"),
        "particle_count": int(source_manifest["particle_count"]),
        "sample_range": [first, last],
        "sample_count": len(selected),
        "timestep_s": timestep,
        "sampling_fps": 1.0 / timestep,
        "attributes": {
            "position": {"dtype": "float32", "shape": [int(source_manifest["particle_count"]), 3]},
            "velocity": {"dtype": "float32", "shape": [int(source_manifest["particle_count"]), 3]},
        },
        "velocity_authority": (
            "PhysX/UsdGeom.PointInstancer velocities captured after fetch_results(); "
            "copied bit-for-bit; no finite differences"
        ),
        "finite_difference_velocity_fallback": False,
        "position_filtering": "none",
        "velocity_filtering": "none",
        "coordinate_transform": "none",
        "readback_verification": not args.no_readback_verify,
        "samples": result_samples,
    }

    for ordinal, sample in enumerate(selected, start=1):
        sample_index = int(sample["sample_index"])
        source_path = source_manifest_path.parent / sample["file"]
        output_name = f"fluid_{sample_index:06d}.bgeo"
        output_path = output_directory / output_name
        old = prior.get(sample_index)
        if args.resume and old is not None and output_path.is_file():
            if output_path.stat().st_size == old.get("bytes") and sha256_file(output_path) == old.get("sha256"):
                result_samples.append(old)
                print(f"[{ordinal}/{len(selected)}] verified existing {output_name}", flush=True)
                continue
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite unverified output: {output_path}")

        positions, velocities = validate_source_arrays(
            source_path, int(source_manifest["particle_count"])
        )
        write_bgeo(output_path, positions, velocities, args.chunk_particles)
        if not args.no_readback_verify:
            verify_bgeo_payload(output_path, positions, velocities, args.chunk_particles)
        output_record = {
            "sample_index": sample_index,
            "simulation_time": float(sample["simulation_time"]),
            "physics_step": int(sample["physics_step"]),
            "source_file": sample["file"],
            "file": output_name,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "readback_verified": not args.no_readback_verify,
        }
        result_samples.append(output_record)
        write_json_atomic(output_manifest_path, manifest)
        print(f"[{ordinal}/{len(selected)}] wrote and verified {output_name}", flush=True)

    manifest["complete"] = len(result_samples) == len(selected)
    write_json_atomic(output_manifest_path, manifest)
    if not manifest["complete"]:
        raise RuntimeError("Conversion ended without all selected samples")
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output_directory),
                "samples": len(result_samples),
                "particle_count": manifest["particle_count"],
                "timestep_s": timestep,
                "finite_difference_velocity_fallback": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; completed files and manifest remain resumable.", file=sys.stderr)
        raise SystemExit(130)
