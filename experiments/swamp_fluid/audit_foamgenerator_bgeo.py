#!/usr/bin/env python3
"""Audit split FoamGenerator BGEO files, including generated velocities."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
from typing import BinaryIO

import numpy as np


FILE_PATTERN = re.compile(r"^secondary_(\d{6})_(foam|spray|bubbles)\.bgeo$")
KINDS = ("foam", "spray", "bubbles")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--chunk-particles", type=int, default=65536)
    return parser.parse_args()


def open_maybe_gzip(path: Path) -> BinaryIO:
    with path.open("rb") as stream:
        signature = stream.read(2)
    if signature == b"\x1f\x8b":
        return gzip.open(path, "rb")
    return path.open("rb")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_exact(stream: BinaryIO, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ValueError(f"Unexpected end of BGEO while reading {size} bytes")
    return value


def audit_file(path: Path, chunk_particles: int) -> dict:
    with open_maybe_gzip(path) as stream:
        if read_exact(stream, 4) != b"Bgeo" or read_exact(stream, 1) != b"V":
            raise ValueError(f"{path}: invalid BGEO v5 signature")
        header = struct.unpack(">9i", read_exact(stream, 36))
        (
            version,
            count,
            primitive_count,
            point_group_count,
            primitive_group_count,
            point_attribute_count,
            vertex_attribute_count,
            primitive_attribute_count,
            fixed_attribute_count,
        ) = header
        if version != 5 or count < 0:
            raise ValueError(f"{path}: invalid BGEO version/count {version}/{count}")
        if any(
            value != 0
            for value in (
                primitive_count,
                point_group_count,
                primitive_group_count,
                vertex_attribute_count,
                primitive_attribute_count,
                fixed_attribute_count,
            )
        ):
            raise ValueError(f"{path}: unsupported non-particle BGEO content")

        attributes: dict[str, dict] = {
            "position": {"offset": 0, "count": 3, "type": 5}
        }
        particle_words = 4
        for _ in range(point_attribute_count):
            name_length = struct.unpack(">H", read_exact(stream, 2))[0]
            name = read_exact(stream, name_length).decode("utf-8")
            attribute_count, attribute_type = struct.unpack(">Hi", read_exact(stream, 6))
            if attribute_type not in (0, 1, 5):
                raise ValueError(f"{path}: unsupported attribute type {attribute_type} for {name}")
            read_exact(stream, attribute_count * 4)  # Houdini default values
            attributes[name] = {
                "offset": particle_words,
                "count": attribute_count,
                "type": attribute_type,
            }
            particle_words += attribute_count

        velocity = attributes.get("velocity")
        particle_id = attributes.get("id")
        if velocity is None or velocity["count"] != 3 or velocity["type"] != 5:
            raise ValueError(f"{path}: exact float3/vector velocity attribute is required")
        if particle_id is None or particle_id["count"] != 1 or particle_id["type"] != 1:
            raise ValueError(f"{path}: exact int id attribute is required")

        position_min = np.full(3, np.inf, dtype=np.float64)
        position_max = np.full(3, -np.inf, dtype=np.float64)
        speed_min = math.inf
        speed_max = 0.0
        next_id = 0
        velocity_offset = int(velocity["offset"])
        id_offset = int(particle_id["offset"])
        for begin in range(0, count, chunk_particles):
            end = min(begin + chunk_particles, count)
            rows = end - begin
            raw = read_exact(stream, rows * particle_words * 4)
            floats = np.frombuffer(raw, dtype=">f4").reshape(rows, particle_words)
            positions = floats[:, 0:3]
            velocities = floats[:, velocity_offset : velocity_offset + 3]
            if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
                raise ValueError(f"{path}: non-finite position or velocity")
            if not np.all(floats[:, 3] == np.float32(1.0)):
                raise ValueError(f"{path}: homogeneous coordinate is not one")
            words = np.frombuffer(raw, dtype=">i4").reshape(rows, particle_words)
            ids = words[:, id_offset]
            expected_ids = np.arange(next_id, next_id + rows, dtype=np.int32)
            if not np.array_equal(ids, expected_ids):
                raise ValueError(f"{path}: Partio ids are not contiguous")
            next_id += rows
            if rows:
                position_min = np.minimum(position_min, positions.min(axis=0))
                position_max = np.maximum(position_max, positions.max(axis=0))
                speed = np.linalg.norm(velocities.astype(np.float64), axis=1)
                speed_min = min(speed_min, float(speed.min()))
                speed_max = max(speed_max, float(speed.max()))
        if read_exact(stream, 2) != b"\x00\xff" or stream.read(1) != b"":
            raise ValueError(f"{path}: invalid BGEO trailer or trailing payload")

    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "particle_count": count,
        "attributes": sorted(attributes),
        "position_minimum": position_min.tolist() if count else None,
        "position_maximum": position_max.tolist() if count else None,
        "speed_minimum": speed_min if count else None,
        "speed_maximum": speed_max if count else None,
        "finite": True,
    }


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.end_frame < args.start_frame or args.chunk_particles <= 0:
        raise ValueError("Invalid frame range or chunk size")
    directory = args.directory.resolve()
    files: dict[tuple[int, str], Path] = {}
    unexpected: list[str] = []
    for path in directory.glob("secondary_*.bgeo"):
        match = FILE_PATTERN.match(path.name)
        if match is None:
            unexpected.append(path.name)
            continue
        key = (int(match.group(1)), match.group(2))
        if key in files:
            raise ValueError(f"Duplicate secondary output for {key}")
        files[key] = path
    if unexpected:
        raise ValueError(f"Unexpected secondary filenames: {unexpected}")

    frame_records: list[dict] = []
    file_records: list[dict] = []
    peak_total = 0
    peak_frame = args.start_frame
    peak_by_kind = {kind: 0 for kind in KINDS}
    for frame in range(args.start_frame, args.end_frame + 1):
        counts: dict[str, int] = {}
        for kind in KINDS:
            path = files.get((frame, kind))
            if path is None:
                counts[kind] = 0
                continue
            record = audit_file(path, args.chunk_particles)
            record.update({"frame": frame, "kind": kind})
            file_records.append(record)
            counts[kind] = int(record["particle_count"])
            peak_by_kind[kind] = max(peak_by_kind[kind], counts[kind])
        total = sum(counts.values())
        if total > peak_total:
            peak_total = total
            peak_frame = frame
        frame_records.append({"frame": frame, "counts": counts, "total": total})

    report = {
        "schema": "foamgenerator-partio-audit/v1",
        "valid": True,
        "directory": str(directory),
        "frame_range": [args.start_frame, args.end_frame],
        "attributes_required": ["position", "velocity", "id"],
        "finite_difference_velocity_used": False,
        "files_audited": len(file_records),
        "peak_total": peak_total,
        "peak_frame": peak_frame,
        "peak_by_kind": peak_by_kind,
        "frames": frame_records,
        "files": file_records,
    }
    output = args.output_report.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite report: {output}")
    write_json_atomic(output, report)
    print(json.dumps({key: report[key] for key in ("valid", "files_audited", "peak_total", "peak_frame", "peak_by_kind")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
