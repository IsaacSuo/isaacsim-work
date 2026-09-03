#!/usr/bin/env python3
"""Build unified id/position/velocity BGEO frames from split lifecycle output."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import struct
from typing import BinaryIO

import numpy as np


KINDS = ("foam", "spray", "bubbles")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
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


def read_state(path: Path) -> dict[str, np.ndarray]:
    with open_maybe_gzip(path) as stream:
        if read_exact(stream, 4) != b"Bgeo" or read_exact(stream, 1) != b"V":
            raise ValueError(f"{path}: invalid BGEO v5 signature")
        header = struct.unpack(">9i", read_exact(stream, 36))
        version, count = header[0], header[1]
        if version != 5 or count < 0 or any(value != 0 for value in header[2:5] + header[6:]):
            raise ValueError(f"{path}: unsupported BGEO structure")
        attributes: dict[str, tuple[int, int, int]] = {"position": (0, 3, 5)}
        particle_words = 4
        for _ in range(header[5]):
            name_length = struct.unpack(">H", read_exact(stream, 2))[0]
            name = read_exact(stream, name_length).decode("utf-8")
            width, attribute_type = struct.unpack(">Hi", read_exact(stream, 6))
            read_exact(stream, width * 4)
            attributes[name] = (particle_words, width, attribute_type)
            particle_words += width
        raw = read_exact(stream, count * particle_words * 4)
        if read_exact(stream, 2) != b"\x00\xff" or stream.read(1) != b"":
            raise ValueError(f"{path}: invalid BGEO trailer")

    for name, expected in {
        "position": (3, 5),
        "velocity": (3, 5),
        "id": (1, 1),
    }.items():
        actual = attributes.get(name)
        if actual is None or actual[1:] != expected:
            raise ValueError(f"{path}: invalid or missing {name} attribute")
    floats = np.frombuffer(raw, dtype=">f4").reshape(count, particle_words)
    integers = np.frombuffer(raw, dtype=">i4").reshape(count, particle_words)
    position_offset = attributes["position"][0]
    velocity_offset = attributes["velocity"][0]
    id_offset = attributes["id"][0]
    return {
        "position": floats[:, position_offset : position_offset + 3].astype(np.float32),
        "velocity": floats[:, velocity_offset : velocity_offset + 3].astype(np.float32),
        "id": integers[:, id_offset].astype(np.int32),
    }


def attribute_definition(name: str, width: int, attribute_type: int) -> bytes:
    encoded = name.encode("utf-8")
    return (
        struct.pack(">H", len(encoded))
        + encoded
        + struct.pack(">Hi", width, attribute_type)
        + bytes(width * 4)
    )


def write_state(path: Path, position: np.ndarray, velocity: np.ndarray, ids: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite fixture: {path}")
    count = len(ids)
    if position.shape != (count, 3) or velocity.shape != (count, 3):
        raise ValueError("Fixture arrays are not aligned")
    if len(np.unique(ids)) != count or np.any(ids < 0):
        raise ValueError("Fixture stable ids are negative or duplicated")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
        raise ValueError("Fixture contains non-finite state")

    words = np.empty((count, 8), dtype=">u4")
    float_words = words.view(">f4")
    int_words = words.view(">i4")
    float_words[:, 0:3] = position
    float_words[:, 3] = np.float32(1.0)
    float_words[:, 4:7] = velocity
    int_words[:, 7] = ids
    payload = (
        b"BgeoV"
        + struct.pack(">9i", 5, count, 0, 0, 0, 2, 0, 0, 0)
        + attribute_definition("velocity", 3, 5)
        + attribute_definition("id", 1, 1)
        + words.tobytes()
        + b"\x00\xff"
    )
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {path}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.end_frame < args.start_frame:
        raise ValueError("Invalid frame range")
    input_directory = args.input_directory.resolve()
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for frame in range(args.start_frame, args.end_frame + 1):
        parts: list[dict[str, np.ndarray]] = []
        source_files: list[str] = []
        for kind in KINDS:
            path = input_directory / f"secondary_{frame:06d}_{kind}.bgeo"
            if not path.exists():
                continue
            parts.append(read_state(path))
            source_files.append(path.name)
        if not parts:
            raise FileNotFoundError(f"Frame {frame}: no split lifecycle state found")
        combined = {
            key: np.concatenate([part[key] for part in parts]) for key in parts[0]
        }
        order = np.argsort(combined["id"], kind="stable")
        combined = {key: value[order] for key, value in combined.items()}
        expected = np.arange(len(combined["id"]), dtype=np.int32)
        if not np.array_equal(combined["id"], expected):
            raise ValueError(f"Frame {frame}: fixture gate expects contiguous ids from zero")
        output_path = output_directory / f"external_{frame:06d}.bgeo"
        write_state(
            output_path,
            combined["position"],
            combined["velocity"],
            combined["id"],
        )
        records.append(
            {
                "frame": frame,
                "particle_count": len(combined["id"]),
                "source_files": source_files,
                "output_file": output_path.name,
            }
        )

    manifest = {
        "schema": "foamgenerator-external-motion-fixture/v1",
        "valid": True,
        "purpose": "Identity fixture for the external PhysX motion handoff contract",
        "input_directory": str(input_directory),
        "output_directory": str(output_directory),
        "frame_range": [args.start_frame, args.end_frame],
        "frames": records,
    }
    atomic_json(args.output_manifest.resolve(), manifest)
    print(json.dumps({"valid": True, "frames": len(records), "maximum_particles": max(r["particle_count"] for r in records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
