"""Small strict BGEO-v5 contract reader/writer for the whitewater bridge."""

from __future__ import annotations

import gzip
import os
from pathlib import Path
import struct
from typing import BinaryIO

import numpy as np


PARTIO_FLOAT = 0
PARTIO_INT = 1
PARTIO_VECTOR = 5


def _open_maybe_gzip(path: Path) -> BinaryIO:
    with path.open("rb") as stream:
        signature = stream.read(2)
    return gzip.open(path, "rb") if signature == b"\x1f\x8b" else path.open("rb")


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ValueError(f"Unexpected end of BGEO while reading {size} bytes")
    return value


def read_bgeo(path: Path) -> tuple[dict[str, tuple[int, int]], dict[str, np.ndarray]]:
    path = path.resolve()
    with _open_maybe_gzip(path) as stream:
        if _read_exact(stream, 4) != b"Bgeo" or _read_exact(stream, 1) != b"V":
            raise ValueError(f"{path}: invalid BGEO-v5 signature")
        header = struct.unpack(">9i", _read_exact(stream, 36))
        version, particle_count = header[0], header[1]
        if version != 5 or particle_count < 0 or any(
            value != 0 for value in header[2:5] + header[6:]
        ):
            raise ValueError(f"{path}: unsupported non-particle BGEO structure")
        attributes: dict[str, tuple[int, int, int]] = {
            "position": (0, 3, PARTIO_VECTOR)
        }
        particle_words = 4
        for _ in range(header[5]):
            name_length = struct.unpack(">H", _read_exact(stream, 2))[0]
            name = _read_exact(stream, name_length).decode("utf-8")
            width, attribute_type = struct.unpack(">Hi", _read_exact(stream, 6))
            if attribute_type not in (PARTIO_FLOAT, PARTIO_INT, PARTIO_VECTOR):
                raise ValueError(f"{path}: unsupported attribute type for {name}")
            _read_exact(stream, width * 4)
            attributes[name] = (particle_words, width, attribute_type)
            particle_words += width
        raw = _read_exact(stream, particle_count * particle_words * 4)
        if _read_exact(stream, 2) != b"\x00\xff" or stream.read(1) != b"":
            raise ValueError(f"{path}: invalid BGEO trailer")

    floats = np.frombuffer(raw, dtype=">f4").reshape(particle_count, particle_words)
    integers = np.frombuffer(raw, dtype=">i4").reshape(particle_count, particle_words)
    if particle_count and not np.all(floats[:, 3] == np.float32(1.0)):
        raise ValueError(f"{path}: homogeneous position coordinate is not one")
    arrays: dict[str, np.ndarray] = {}
    schema: dict[str, tuple[int, int]] = {}
    for name, (offset, width, attribute_type) in attributes.items():
        source = integers if attribute_type == PARTIO_INT else floats
        value = source[:, offset : offset + width].copy()
        arrays[name] = value[:, 0] if width == 1 else value
        schema[name] = (attribute_type, width)
    return schema, arrays


def require_schema(
    path: Path,
    schema: dict[str, tuple[int, int]],
    required: dict[str, tuple[int, int]],
) -> None:
    for name, expected in required.items():
        if schema.get(name) != expected:
            raise ValueError(f"{path}: {name} must be type/count {expected[0]}/{expected[1]}")


def read_birth_events(path: Path) -> dict[str, np.ndarray]:
    schema, arrays = read_bgeo(path)
    require_schema(
        path,
        schema,
        {
            "position": (PARTIO_VECTOR, 3),
            "velocity": (PARTIO_VECTOR, 3),
            "id": (PARTIO_INT, 1),
            "remaining_lifetime": (PARTIO_FLOAT, 1),
            "birth_frame": (PARTIO_INT, 1),
            "particle_type": (PARTIO_INT, 1),
            "source_particle_index": (PARTIO_INT, 1),
        },
    )
    count = len(arrays["id"])
    if (
        len(np.unique(arrays["id"])) != count
        or np.any(arrays["id"] < 0)
        or np.any(arrays["birth_frame"] < 0)
        or np.any(arrays["particle_type"] != 3)
        or np.any(arrays["source_particle_index"] < 0)
        or np.any(arrays["remaining_lifetime"] <= 0.0)
        or not np.all(np.isfinite(arrays["position"]))
        or not np.all(np.isfinite(arrays["velocity"]))
        or not np.all(np.isfinite(arrays["remaining_lifetime"]))
    ):
        raise ValueError(f"{path}: invalid birth-event values")
    return arrays


def _attribute_definition(name: str, width: int, attribute_type: int) -> bytes:
    encoded = name.encode("utf-8")
    return (
        struct.pack(">H", len(encoded))
        + encoded
        + struct.pack(">Hi", width, attribute_type)
        + bytes(width * 4)
    )


def _write_rows(
    path: Path,
    attributes: list[tuple[str, int, int]],
    float_values: dict[str, np.ndarray],
    int_values: dict[str, np.ndarray],
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite BGEO: {path}")
    position = np.asarray(float_values["position"], dtype=np.float32)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("position must have shape (N, 3)")
    particle_count = len(position)
    particle_words = 4 + sum(width for _, width, _ in attributes)
    words = np.empty((particle_count, particle_words), dtype=">u4")
    float_words = words.view(">f4")
    int_words = words.view(">i4")
    float_words[:, 0:3] = position
    float_words[:, 3] = np.float32(1.0)
    offset = 4
    for name, width, attribute_type in attributes:
        values = int_values[name] if attribute_type == PARTIO_INT else float_values[name]
        values = np.asarray(values)
        expected_shape = (particle_count,) if width == 1 else (particle_count, width)
        if values.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {values.shape}")
        target = int_words if attribute_type == PARTIO_INT else float_words
        if width == 1:
            target[:, offset] = values
        else:
            target[:, offset : offset + width] = values
        offset += width
    payload = (
        b"BgeoV"
        + struct.pack(">9i", 5, particle_count, 0, 0, 0, len(attributes), 0, 0, 0)
        + b"".join(_attribute_definition(*attribute) for attribute in attributes)
        + words.tobytes()
        + b"\x00\xff"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_motion_state(
    path: Path, positions: np.ndarray, velocities: np.ndarray, stable_ids: np.ndarray
) -> None:
    positions = np.asarray(positions, dtype=np.float32)
    velocities = np.asarray(velocities, dtype=np.float32)
    stable_ids = np.asarray(stable_ids, dtype=np.int32)
    if positions.shape != velocities.shape or len(positions) != len(stable_ids):
        raise ValueError("Motion-state arrays are not aligned")
    if len(np.unique(stable_ids)) != len(stable_ids) or np.any(stable_ids < 0):
        raise ValueError("Motion-state stable ids are negative or duplicated")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise ValueError("Motion state is non-finite")
    _write_rows(
        path,
        [("velocity", 3, PARTIO_VECTOR), ("id", 1, PARTIO_INT)],
        {"position": positions, "velocity": velocities},
        {"id": stable_ids},
    )


def write_birth_events(
    path: Path,
    positions: np.ndarray,
    velocities: np.ndarray,
    stable_ids: np.ndarray,
    remaining_lifetimes: np.ndarray,
    birth_frame: int,
    source_particle_indices: np.ndarray,
) -> None:
    positions = np.asarray(positions, dtype=np.float32)
    velocities = np.asarray(velocities, dtype=np.float32)
    stable_ids = np.asarray(stable_ids, dtype=np.int32)
    remaining_lifetimes = np.asarray(remaining_lifetimes, dtype=np.float32)
    source_particle_indices = np.asarray(source_particle_indices, dtype=np.int32)
    count = len(stable_ids)
    if (
        positions.shape != (count, 3)
        or velocities.shape != (count, 3)
        or remaining_lifetimes.shape != (count,)
        or source_particle_indices.shape != (count,)
    ):
        raise ValueError("Birth-event arrays are not aligned")
    if birth_frame < 0 or np.any(remaining_lifetimes <= 0.0):
        raise ValueError("Birth frame/lifetime is invalid")
    _write_rows(
        path,
        [
            ("velocity", 3, PARTIO_VECTOR),
            ("id", 1, PARTIO_INT),
            ("remaining_lifetime", 1, PARTIO_FLOAT),
            ("birth_frame", 1, PARTIO_INT),
            ("particle_type", 1, PARTIO_INT),
            ("source_particle_index", 1, PARTIO_INT),
        ],
        {
            "position": positions,
            "velocity": velocities,
            "remaining_lifetime": remaining_lifetimes,
        },
        {
            "id": stable_ids,
            "birth_frame": np.full(count, birth_frame, dtype=np.int32),
            "particle_type": np.full(count, 3, dtype=np.int32),
            "source_particle_index": source_particle_indices,
        },
    )
