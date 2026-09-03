#!/usr/bin/env python3
"""Audit stable FoamGenerator lifecycle metadata across split BGEO frames."""

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


FILE_PATTERN = re.compile(r"^secondary_(\d{6})_(foam|spray|bubbles)\.bgeo$")
KINDS = ("foam", "spray", "bubbles")
TYPE_BY_KIND = {"foam": 0, "spray": 1, "bubbles": 2}
REQUIRED_ATTRIBUTES = {
    "position": (5, 3),
    "velocity": (5, 3),
    "id": (1, 1),
    "remaining_lifetime": (0, 1),
    "birth_frame": (1, 1),
    "particle_type": (1, 1),
    "source_particle_index": (1, 1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--timestep", type=float, required=True)
    parser.add_argument(
        "--lifetime-decay",
        choices=("upstream", "chronological"),
        default="upstream",
    )
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


def read_lifecycle_file(path: Path, expected_type: int) -> tuple[dict, dict[str, np.ndarray]]:
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

        for name, (attribute_type, attribute_count) in REQUIRED_ATTRIBUTES.items():
            attribute = attributes.get(name)
            if attribute is None or (attribute["type"], attribute["count"]) != (
                attribute_type,
                attribute_count,
            ):
                raise ValueError(
                    f"{path}: {name} must be type/count {attribute_type}/{attribute_count}"
                )

        raw = read_exact(stream, count * particle_words * 4)
        if read_exact(stream, 2) != b"\x00\xff" or stream.read(1) != b"":
            raise ValueError(f"{path}: invalid BGEO trailer or trailing payload")

    float_words = np.frombuffer(raw, dtype=">f4").reshape(count, particle_words)
    int_words = np.frombuffer(raw, dtype=">i4").reshape(count, particle_words)
    position_offset = attributes["position"]["offset"]
    velocity_offset = attributes["velocity"]["offset"]
    ids = int_words[:, attributes["id"]["offset"]].astype(np.int64)
    remaining = float_words[:, attributes["remaining_lifetime"]["offset"]].astype(
        np.float32
    )
    birth_frames = int_words[:, attributes["birth_frame"]["offset"]].astype(np.int64)
    particle_types = int_words[:, attributes["particle_type"]["offset"]].astype(np.int8)
    source_particle_indices = int_words[
        :, attributes["source_particle_index"]["offset"]
    ].astype(np.int64)
    positions = float_words[:, position_offset : position_offset + 3].astype(np.float32)
    velocities = float_words[:, velocity_offset : velocity_offset + 3].astype(np.float32)

    if not np.all(float_words[:, 3] == np.float32(1.0)):
        raise ValueError(f"{path}: homogeneous position coordinate is not one")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise ValueError(f"{path}: non-finite position or velocity")
    if not np.all(np.isfinite(remaining)):
        raise ValueError(f"{path}: non-finite remaining lifetime")
    if np.any(ids < 0) or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{path}: stable ids are negative or duplicated")
    if np.any(source_particle_indices < 0):
        raise ValueError(f"{path}: source particle indices are negative")
    if np.any(particle_types != expected_type):
        raise ValueError(f"{path}: particle_type does not match split output kind")

    arrays = {
        "id": ids,
        "remaining_lifetime": remaining,
        "birth_frame": birth_frames,
        "particle_type": particle_types,
        "source_particle_index": source_particle_indices,
        "position": positions,
        "velocity": velocities,
    }
    record = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "particle_count": int(count),
        "attributes": sorted(attributes),
        "id_minimum": int(ids.min()) if count else None,
        "id_maximum": int(ids.max()) if count else None,
        "remaining_lifetime_minimum": float(remaining.min()) if count else None,
        "remaining_lifetime_maximum": float(remaining.max()) if count else None,
        "source_particle_index_minimum": int(source_particle_indices.min()) if count else None,
        "source_particle_index_maximum": int(source_particle_indices.max()) if count else None,
    }
    return record, arrays


def empty_frame() -> dict[str, np.ndarray]:
    return {
        "id": np.empty(0, dtype=np.int64),
        "remaining_lifetime": np.empty(0, dtype=np.float32),
        "birth_frame": np.empty(0, dtype=np.int64),
        "particle_type": np.empty(0, dtype=np.int8),
        "source_particle_index": np.empty(0, dtype=np.int64),
        "position": np.empty((0, 3), dtype=np.float32),
        "velocity": np.empty((0, 3), dtype=np.float32),
    }


def combine(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        return empty_frame()
    combined = {
        key: np.concatenate([part[key] for part in parts]) for key in parts[0]
    }
    order = np.argsort(combined["id"], kind="stable")
    return {key: value[order] for key, value in combined.items()}


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
    if args.end_frame < args.start_frame or not np.isfinite(args.timestep) or args.timestep <= 0:
        raise ValueError("Invalid frame range or timestep")
    directory = args.directory.resolve()
    indexed: dict[tuple[int, str], Path] = {}
    for path in directory.glob("secondary_*.bgeo"):
        match = FILE_PATTERN.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected secondary filename: {path.name}")
        indexed[(int(match.group(1)), match.group(2))] = path

    previous = empty_frame()
    maximum_seen_id = -1
    first_nonempty_frame: int | None = None
    frame_records: list[dict] = []
    file_records: list[dict] = []
    transition_counts = np.zeros((3, 3), dtype=np.int64)
    total_births = 0
    total_deaths = 0
    maximum_count = 0
    tolerance = max(2.0e-6, args.timestep * 2.0e-5)

    for frame in range(args.start_frame, args.end_frame + 1):
        parts: list[dict[str, np.ndarray]] = []
        counts: dict[str, int] = {}
        for kind in KINDS:
            path = indexed.get((frame, kind))
            if path is None:
                counts[kind] = 0
                continue
            record, arrays = read_lifecycle_file(path, TYPE_BY_KIND[kind])
            record.update({"frame": frame, "kind": kind})
            file_records.append(record)
            parts.append(arrays)
            counts[kind] = int(len(arrays["id"]))
        current = combine(parts)
        if len(np.unique(current["id"])) != len(current["id"]):
            raise ValueError(f"Frame {frame}: stable id appears in more than one split kind")
        if np.any(current["birth_frame"] > frame):
            raise ValueError(f"Frame {frame}: particle has a future birth_frame")
        if len(current["id"]) and first_nonempty_frame is None:
            first_nonempty_frame = frame

        previous_ids = previous["id"]
        current_ids = current["id"]
        common_ids, previous_indices, current_indices = np.intersect1d(
            previous_ids, current_ids, assume_unique=True, return_indices=True
        )
        new_mask = ~np.isin(current_ids, common_ids, assume_unique=True)
        dead_mask = ~np.isin(previous_ids, common_ids, assume_unique=True)
        new_ids = current_ids[new_mask]
        dead_ids = previous_ids[dead_mask]

        if len(new_ids):
            if np.any(new_ids <= maximum_seen_id):
                raise ValueError(f"Frame {frame}: a stable id was reused")
            expected = np.arange(maximum_seen_id + 1, int(new_ids.max()) + 1, dtype=np.int64)
            if not np.array_equal(new_ids, expected):
                raise ValueError(f"Frame {frame}: newly assigned stable ids are not contiguous")
            if first_nonempty_frame != frame and np.any(current["birth_frame"][new_mask] != frame - 1):
                raise ValueError(f"Frame {frame}: new particles do not identify the preceding generation frame")
            maximum_seen_id = int(new_ids.max())
        if len(dead_ids) and np.any(
            previous["remaining_lifetime"][dead_mask] > tolerance
        ):
            raise ValueError(f"Frame {frame}: a positive-lifetime particle disappeared")

        if len(common_ids):
            if not np.array_equal(
                previous["birth_frame"][previous_indices],
                current["birth_frame"][current_indices],
            ):
                raise ValueError(f"Frame {frame}: birth_frame changed for a stable id")
            if not np.array_equal(
                previous["source_particle_index"][previous_indices],
                current["source_particle_index"][current_indices],
            ):
                raise ValueError(
                    f"Frame {frame}: source_particle_index changed for a stable id"
                )
            previous_lifetime = previous["remaining_lifetime"][previous_indices].astype(np.float64)
            current_lifetime = current["remaining_lifetime"][current_indices].astype(np.float64)
            current_type = current["particle_type"][current_indices]
            if args.lifetime_decay == "chronological":
                expected_decrement = np.full(len(current_type), args.timestep)
            else:
                expected_decrement = np.where(
                    current_type == TYPE_BY_KIND["foam"], args.timestep, 0.0
                )
            if np.any(np.abs((previous_lifetime - current_lifetime) - expected_decrement) > tolerance):
                raise ValueError(f"Frame {frame}: remaining_lifetime does not follow FoamGenerator semantics")
            previous_type = previous["particle_type"][previous_indices]
            np.add.at(transition_counts, (previous_type, current_type), 1)

        births = int(len(new_ids))
        deaths = int(len(dead_ids))
        total_births += births
        total_deaths += deaths
        maximum_count = max(maximum_count, len(current_ids))
        frame_records.append(
            {
                "frame": frame,
                "counts": counts,
                "total": int(len(current_ids)),
                "births": births,
                "deaths": deaths,
                "persistent": int(len(common_ids)),
            }
        )
        previous = current

    if not file_records or first_nonempty_frame is None:
        raise ValueError(
            "Lifecycle audit found no non-empty BGEO files; a missing output directory "
            "must not be reported as a valid empty simulation"
        )

    report = {
        "schema": "foamgenerator-lifecycle-partio-audit/v2",
        "valid": True,
        "directory": str(directory),
        "frame_range": [args.start_frame, args.end_frame],
        "timestep_s": args.timestep,
        "lifetime_decay": args.lifetime_decay,
        "required_attributes": sorted(REQUIRED_ATTRIBUTES),
        "finite_difference_velocity_used": False,
        "first_nonempty_frame": first_nonempty_frame,
        "maximum_particles": maximum_count,
        "maximum_stable_id": maximum_seen_id,
        "total_first_observations": total_births,
        "total_deaths": total_deaths,
        "transition_matrix": transition_counts.tolist(),
        "transition_axis": list(KINDS),
        "files_audited": len(file_records),
        "frames": frame_records,
        "files": file_records,
    }
    atomic_json(args.output_report.resolve(), report)
    print(
        json.dumps(
            {
                "valid": True,
                "files_audited": len(file_records),
                "first_nonempty_frame": first_nonempty_frame,
                "maximum_particles": maximum_count,
                "maximum_stable_id": maximum_seen_id,
                "total_deaths": total_deaths,
                "transition_matrix": transition_counts.tolist(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
