#!/usr/bin/env python3
"""Audit FoamGenerator birth handoff events against native PhysX primary data."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import struct
from typing import BinaryIO

import numpy as np


BIRTH_PATTERN = re.compile(r"^birth_(\d{6})\.bgeo$")
LIFECYCLE_KINDS = ("foam", "spray", "bubbles")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--primary-pattern", type=str, required=True)
    parser.add_argument("--lifecycle-directory", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--particle-radius", type=float, required=True)
    parser.add_argument("--timestep", type=float, required=True)
    parser.add_argument("--lifetime-min", type=float, required=True)
    parser.add_argument("--lifetime-max", type=float, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def open_maybe_gzip(path: Path) -> BinaryIO:
    with path.open("rb") as stream:
        signature = stream.read(2)
    return gzip.open(path, "rb") if signature == b"\x1f\x8b" else path.open("rb")


def read_exact(stream: BinaryIO, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ValueError(f"Unexpected end of BGEO while reading {size} bytes")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_bgeo(path: Path) -> tuple[dict[str, dict[str, int]], dict[str, np.ndarray]]:
    with open_maybe_gzip(path) as stream:
        if read_exact(stream, 4) != b"Bgeo" or read_exact(stream, 1) != b"V":
            raise ValueError(f"{path}: invalid BGEO v5 signature")
        header = struct.unpack(">9i", read_exact(stream, 36))
        version, count = header[0], header[1]
        if version != 5 or count < 0 or any(value != 0 for value in header[2:5] + header[6:]):
            raise ValueError(f"{path}: unsupported BGEO structure")

        attributes: dict[str, dict[str, int]] = {
            "position": {"offset": 0, "count": 3, "type": 5}
        }
        particle_words = 4
        for _ in range(header[5]):
            name_length = struct.unpack(">H", read_exact(stream, 2))[0]
            name = read_exact(stream, name_length).decode("utf-8")
            attribute_count, attribute_type = struct.unpack(">Hi", read_exact(stream, 6))
            if attribute_type not in (0, 1, 5):
                raise ValueError(f"{path}: unsupported attribute type {attribute_type} for {name}")
            read_exact(stream, attribute_count * 4)
            attributes[name] = {
                "offset": particle_words,
                "count": attribute_count,
                "type": attribute_type,
            }
            particle_words += attribute_count

        raw = read_exact(stream, count * particle_words * 4)
        if read_exact(stream, 2) != b"\x00\xff" or stream.read(1) != b"":
            raise ValueError(f"{path}: invalid BGEO trailer or trailing payload")

    floats = np.frombuffer(raw, dtype=">f4").reshape(count, particle_words)
    integers = np.frombuffer(raw, dtype=">i4").reshape(count, particle_words)
    if count and not np.all(floats[:, 3] == np.float32(1.0)):
        raise ValueError(f"{path}: homogeneous position coordinate is not one")

    arrays: dict[str, np.ndarray] = {}
    for name, attribute in attributes.items():
        offset = attribute["offset"]
        width = attribute["count"]
        source = integers if attribute["type"] == 1 else floats
        value = source[:, offset : offset + width].copy()
        arrays[name] = value[:, 0] if width == 1 else value
    return attributes, arrays


def require_attribute(
    path: Path,
    attributes: dict[str, dict[str, int]],
    name: str,
    attribute_type: int,
    width: int,
) -> None:
    value = attributes.get(name)
    if value is None or (value["type"], value["count"]) != (attribute_type, width):
        raise ValueError(f"{path}: {name} must be type/count {attribute_type}/{width}")


def resolve_frame_pattern(pattern: str, frame: int) -> Path:
    marker = "######"
    if pattern.count(marker) != 1:
        raise ValueError("Primary pattern must contain exactly one ###### marker")
    return Path(pattern.replace(marker, f"{frame:06d}"))


def lifecycle_ids(directory: Path, frame: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    for kind in LIFECYCLE_KINDS:
        path = directory / f"secondary_{frame:06d}_{kind}.bgeo"
        if not path.exists():
            continue
        attributes, arrays = read_bgeo(path)
        require_attribute(path, attributes, "id", 1, 1)
        parts.append(arrays["id"].astype(np.int64, copy=False))
    if not parts:
        return np.empty(0, dtype=np.int64)
    result = np.concatenate(parts)
    if len(np.unique(result)) != len(result):
        raise ValueError(f"Lifecycle frame {frame}: duplicate stable ids across split files")
    return result


def atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if (
        args.end_frame < args.start_frame
        or args.particle_radius <= 0.0
        or args.timestep <= 0.0
        or args.lifetime_min <= 0.0
        or args.lifetime_max < args.lifetime_min
    ):
        raise ValueError("Invalid audit parameters")

    directory = args.directory.resolve()
    lifecycle_directory = args.lifecycle_directory.resolve()
    indexed: dict[int, Path] = {}
    for path in directory.glob("birth_*.bgeo"):
        match = BIRTH_PATTERN.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected birth-event filename: {path.name}")
        frame = int(match.group(1))
        if frame in indexed:
            raise ValueError(f"Duplicate birth-event frame: {frame}")
        indexed[frame] = path

    required_birth_attributes = {
        "position": (5, 3),
        "velocity": (5, 3),
        "id": (1, 1),
        "remaining_lifetime": (0, 1),
        "birth_frame": (1, 1),
        "particle_type": (1, 1),
        "source_particle_index": (1, 1),
    }
    velocity_tolerance = max(2.0e-6, args.particle_radius**2 * 2.0e-3)
    position_tolerance = max(3.0e-6, args.particle_radius**2 * 3.0e-3)
    lifetime_tolerance = 2.0e-6
    maximum_radial_velocity = args.particle_radius**2

    all_ids: list[np.ndarray] = []
    frame_records: list[dict] = []
    total_particles = 0
    maximum_velocity_delta = 0.0
    maximum_velocity_axial_error = 0.0
    maximum_position_orthogonal_error = 0.0
    maximum_position_axial_fraction = 0.0

    for frame in range(args.start_frame, args.end_frame + 1):
        path = indexed.get(frame)
        if path is None:
            frame_records.append({"frame": frame, "particle_count": 0, "birth_file": None})
            continue

        attributes, birth = read_bgeo(path)
        for name, (attribute_type, width) in required_birth_attributes.items():
            require_attribute(path, attributes, name, attribute_type, width)

        ids = birth["id"].astype(np.int64, copy=False)
        source_indices = birth["source_particle_index"].astype(np.int64, copy=False)
        birth_frames = birth["birth_frame"].astype(np.int64, copy=False)
        particle_types = birth["particle_type"].astype(np.int64, copy=False)
        remaining = birth["remaining_lifetime"].astype(np.float64, copy=False)
        positions = birth["position"].astype(np.float64, copy=False)
        velocities = birth["velocity"].astype(np.float64, copy=False)

        count = len(ids)
        if count == 0:
            raise ValueError(f"{path}: empty birth files should not be emitted")
        if len(np.unique(ids)) != count or np.any(ids < 0):
            raise ValueError(f"{path}: birth ids are negative or duplicated")
        if np.any(birth_frames != frame):
            raise ValueError(f"{path}: birth_frame does not match filename")
        if np.any(particle_types != 3):
            raise ValueError(f"{path}: birth particle_type must be unclassified (3)")
        if np.any(remaining < args.lifetime_min - lifetime_tolerance) or np.any(
            remaining > args.lifetime_max + lifetime_tolerance
        ):
            raise ValueError(f"{path}: initial lifetime is outside the configured range")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise ValueError(f"{path}: birth position or velocity is non-finite")

        primary_path = resolve_frame_pattern(args.primary_pattern, frame).resolve()
        primary_attributes, primary = read_bgeo(primary_path)
        require_attribute(primary_path, primary_attributes, "position", 5, 3)
        require_attribute(primary_path, primary_attributes, "velocity", 5, 3)
        primary_count = len(primary["position"])
        if np.any(source_indices < 0) or np.any(source_indices >= primary_count):
            raise ValueError(f"{path}: source_particle_index is outside the primary array")

        source_positions = primary["position"][source_indices].astype(np.float64)
        source_velocities = primary["velocity"][source_indices].astype(np.float64)
        source_speeds = np.linalg.norm(source_velocities, axis=1)
        if np.any(source_speeds <= 0.0) or not np.all(np.isfinite(source_speeds)):
            raise ValueError(f"{path}: a source primary velocity is zero or non-finite")
        directions = source_velocities / source_speeds[:, None]

        velocity_delta = velocities - source_velocities
        velocity_delta_norm = np.linalg.norm(velocity_delta, axis=1)
        velocity_axial_error = np.abs(np.einsum("ij,ij->i", velocity_delta, directions))
        if np.any(velocity_delta_norm > maximum_radial_velocity + velocity_tolerance):
            raise ValueError(f"{path}: birth velocity is not the native primary velocity plus cylinder jitter")
        if np.any(velocity_axial_error > velocity_tolerance):
            raise ValueError(f"{path}: velocity jitter is not orthogonal to the primary velocity")

        position_delta = positions - source_positions
        position_without_radial = position_delta - velocity_delta
        axial_offset = np.einsum("ij,ij->i", position_without_radial, directions)
        orthogonal_offset = position_without_radial - axial_offset[:, None] * directions
        orthogonal_error = np.linalg.norm(orthogonal_offset, axis=1)
        axial_limit = 0.5 * args.timestep * source_speeds
        if np.any(orthogonal_error > position_tolerance):
            raise ValueError(f"{path}: birth position does not use the same cylinder radial sample")
        if np.any(np.abs(axial_offset) > axial_limit + position_tolerance):
            raise ValueError(f"{path}: birth position exceeds the FoamGenerator cylinder height")

        next_ids = lifecycle_ids(lifecycle_directory, frame + 1)
        missing_next_frame = ids[~np.isin(ids, next_ids, assume_unique=False)]
        if len(missing_next_frame):
            raise ValueError(
                f"{path}: {len(missing_next_frame)} birth ids are absent from lifecycle frame {frame + 1}"
            )

        all_ids.append(ids)
        total_particles += count
        maximum_velocity_delta = max(maximum_velocity_delta, float(velocity_delta_norm.max()))
        maximum_velocity_axial_error = max(
            maximum_velocity_axial_error, float(velocity_axial_error.max())
        )
        maximum_position_orthogonal_error = max(
            maximum_position_orthogonal_error, float(orthogonal_error.max())
        )
        nonzero_axial_limits = axial_limit > 0.0
        maximum_position_axial_fraction = max(
            maximum_position_axial_fraction,
            float(np.max(np.abs(axial_offset[nonzero_axial_limits]) / axial_limit[nonzero_axial_limits])),
        )
        frame_records.append(
            {
                "frame": frame,
                "birth_file": path.name,
                "birth_file_sha256": sha256_file(path),
                "primary_file": str(primary_path),
                "primary_particle_count": primary_count,
                "particle_count": count,
                "id_minimum": int(ids.min()),
                "id_maximum": int(ids.max()),
                "source_particle_index_minimum": int(source_indices.min()),
                "source_particle_index_maximum": int(source_indices.max()),
                "next_lifecycle_frame": frame + 1,
                "next_lifecycle_membership_verified": True,
            }
        )

    combined_ids = np.concatenate(all_ids) if all_ids else np.empty(0, dtype=np.int64)
    if len(combined_ids) == 0:
        raise ValueError("No non-empty birth event files were audited")
    if len(np.unique(combined_ids)) != len(combined_ids):
        raise ValueError("A stable id occurs in more than one birth frame")
    expected_ids = np.arange(int(combined_ids.min()), int(combined_ids.max()) + 1, dtype=np.int64)
    if not np.array_equal(np.sort(combined_ids), expected_ids):
        raise ValueError("Birth stable ids are not globally contiguous across the audited range")

    report = {
        "schema": "foamgenerator-physx-native-birth-audit/v1",
        "valid": True,
        "directory": str(directory),
        "lifecycle_directory": str(lifecycle_directory),
        "primary_pattern": args.primary_pattern,
        "frame_range": [args.start_frame, args.end_frame],
        "particle_radius_m": args.particle_radius,
        "timestep_s": args.timestep,
        "configured_lifetime_s": [args.lifetime_min, args.lifetime_max],
        "birth_particles_audited": total_particles,
        "stable_id_range": [int(combined_ids.min()), int(combined_ids.max())],
        "maximum_velocity_delta_m_per_s": maximum_velocity_delta,
        "allowed_velocity_delta_m_per_s": maximum_radial_velocity + velocity_tolerance,
        "maximum_velocity_axial_error_m_per_s": maximum_velocity_axial_error,
        "maximum_position_orthogonal_error_m": maximum_position_orthogonal_error,
        "maximum_position_axial_fraction": maximum_position_axial_fraction,
        "native_primary_velocity_attribute_used": True,
        "finite_difference_velocity_used": False,
        "source_particle_index_verified": True,
        "next_frame_lifecycle_membership_verified": True,
        "frames": frame_records,
    }
    atomic_json(args.output_report.resolve(), report)
    print(
        json.dumps(
            {
                "valid": True,
                "birth_particles_audited": total_particles,
                "stable_id_range": report["stable_id_range"],
                "maximum_velocity_delta_m_per_s": maximum_velocity_delta,
                "maximum_velocity_axial_error_m_per_s": maximum_velocity_axial_error,
                "maximum_position_orthogonal_error_m": maximum_position_orthogonal_error,
                "next_frame_lifecycle_membership_verified": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
