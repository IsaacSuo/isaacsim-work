"""Versioned surface-cache and manifest helpers for the liquid video pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CACHE_SCHEMA = "realistic-liquid-surface-cache"
CACHE_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def config_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def immutable_job_config(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_uuid": str(job["job_uuid"]),
        "take_id": str(job["take_id"]),
        "video": job["video"],
        "physics": job["physics"],
        "render": job["render"],
    }


def simulation_provenance_payload(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_uuid": str(job["job_uuid"]),
        "take_id": str(job["take_id"]),
        "video": job["video"],
        "physics": job["physics"],
        "physics_script_sha256": job["script_sha256"]["physx_realistic_liquid.py"],
    }


def validate_job(job: dict[str, Any]) -> dict[str, Any]:
    if int(job.get("schema_version", -1)) != 2:
        raise ValueError(f"Unsupported job schema: {job.get('schema_version')}")
    expected_config_hash = config_hash(immutable_job_config(job))
    if str(job.get("config_hash")) != expected_config_hash:
        raise ValueError(
            "job.json immutable configuration hash mismatch; create a new job "
            "instead of editing video/physics/render settings"
        )
    expected_simulation_hash = config_hash(simulation_provenance_payload(job))
    if str(job.get("simulation_provenance_hash")) != expected_simulation_hash:
        raise ValueError("job.json simulation provenance hash mismatch")
    video = job["video"]
    physics = job["physics"]
    output_frames = expected_output_frames(
        float(video["duration_seconds"]),
        int(video["output_fps"]),
    )
    if int(video["output_frames"]) != output_frames:
        raise ValueError("video.output_frames does not equal duration * output_fps")
    capture_stride = require_integer_capture_stride(
        int(physics["physics_fps"]),
        int(video["output_fps"]),
    )
    if int(physics["capture_stride"]) != capture_stride:
        raise ValueError("physics.capture_stride is inconsistent with frame rates")
    timeline = job["timeline"]
    expected_last = (output_frames - 1) / int(video["output_fps"])
    expected_duration = output_frames / int(video["output_fps"])
    if not math.isclose(
        float(timeline["last_sample_seconds"]),
        expected_last,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("timeline.last_sample_seconds is inconsistent")
    if not math.isclose(
        float(timeline["container_duration_seconds"]),
        expected_duration,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("timeline.container_duration_seconds is inconsistent")
    if not timeline.get("uniform"):
        raise ValueError("Only a uniform output timeline is supported")
    return job


def render_provenance_hash(
    job: dict[str, Any],
    *,
    template_sha256: str,
    renderer_sha256: str,
    isaac_version: str | None,
) -> str:
    validate_job(job)
    return config_hash(
        {
            "job_uuid": job["job_uuid"],
            "take_id": job["take_id"],
            "config_hash": job["config_hash"],
            "simulation_provenance_hash": job["simulation_provenance_hash"],
            "render": job["render"],
            "template_sha256": template_sha256,
            "renderer_sha256": renderer_sha256,
            "isaac_version": isaac_version,
            "cached_mesh_prim": job["paths"]["cached_mesh_prim"],
            "water_material_prim": job["paths"]["water_material_prim"],
            "camera_prim": job["paths"]["camera_prim"],
        }
    )


def file_sha256(path: os.PathLike[str] | str, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: os.PathLike[str] | str, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def append_jsonl(path: os.PathLike[str] | str, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(value) + "\n"
    with open(target, "a", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def atomic_write_jsonl(
    path: os.PathLike[str] | str,
    rows: Iterable[dict[str, Any]],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(canonical_json(row))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def read_jsonl(path: os.PathLike[str] | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Manifest row at {path}:{line_number} is not an object")
            rows.append(row)
    return rows


def _as_points(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


def _as_indices(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.int32)
    if array.ndim != 1:
        raise ValueError(f"face_vertex_indices must be one-dimensional, got {array.shape}")
    return np.ascontiguousarray(array)


def _as_counts(value: Any | None, index_count: int) -> tuple[np.ndarray, bool]:
    if value is None:
        if index_count % 3:
            raise ValueError("Triangle-only indices must be divisible by three")
        return np.empty(0, dtype=np.int32), True
    array = np.asarray(value, dtype=np.int32)
    if array.ndim != 1 or np.any(array <= 0):
        raise ValueError("face_vertex_counts must be a positive one-dimensional array")
    if int(array.sum(dtype=np.int64)) != index_count:
        raise ValueError("face counts do not sum to the index count")
    triangles_only = bool(np.all(array == 3))
    return np.ascontiguousarray(array), triangles_only


def validate_surface_arrays(
    points: Any,
    face_vertex_indices: Any,
    face_vertex_counts: Any | None = None,
    normals: Any | None = None,
) -> dict[str, Any]:
    points_array = _as_points("points", points)
    indices_array = _as_indices(face_vertex_indices)
    counts_array, triangles_only = _as_counts(
        face_vertex_counts,
        len(indices_array),
    )
    if len(points_array) == 0:
        raise ValueError("Surface cache cannot contain an empty mesh")
    if len(indices_array) == 0:
        raise ValueError("Surface cache cannot contain empty topology")
    minimum_index = int(indices_array.min(initial=0))
    maximum_index = int(indices_array.max(initial=-1))
    if minimum_index < 0 or maximum_index >= len(points_array):
        raise ValueError(
            f"Topology index range [{minimum_index}, {maximum_index}] is invalid for "
            f"{len(points_array)} points"
        )
    normals_array = np.empty((0, 3), dtype=np.float32)
    if normals is not None:
        normals_array = _as_points("normals", normals)
        if len(normals_array) not in (len(points_array), len(counts_array)):
            raise ValueError(
                "Normals must be per-point or per-face; "
                f"got {len(normals_array)} for {len(points_array)} points and "
                f"{len(counts_array)} faces"
            )
    bounds_min = points_array.min(axis=0)
    bounds_max = points_array.max(axis=0)
    if not np.all(bounds_min <= bounds_max):
        raise ValueError("Invalid surface bounds")
    return {
        "points": points_array,
        "face_vertex_indices": indices_array,
        "face_vertex_counts": counts_array,
        "normals": normals_array,
        "triangles_only": triangles_only,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "vertex_count": int(len(points_array)),
        "face_count": int(
            len(indices_array) // 3 if triangles_only and not len(counts_array) else len(counts_array)
        ),
    }


def write_surface_cache(
    path: os.PathLike[str] | str,
    *,
    points: Any,
    face_vertex_indices: Any,
    metadata: dict[str, Any],
    face_vertex_counts: Any | None = None,
    normals: Any | None = None,
    compressed: bool = False,
) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = validate_surface_arrays(
        points,
        face_vertex_indices,
        face_vertex_counts,
        normals,
    )
    required_metadata = (
        "output_index",
        "sim_step",
        "sim_time_seconds",
        "take_id",
        "config_hash",
        "simulation_provenance_hash",
    )
    missing = [name for name in required_metadata if name not in metadata]
    if missing:
        raise ValueError(f"Surface metadata is missing: {', '.join(missing)}")
    output_index = int(metadata["output_index"])
    if output_index < 0:
        raise ValueError("output_index must be non-negative")
    cache_metadata = {
        **metadata,
        "schema": CACHE_SCHEMA,
        "schema_version": CACHE_SCHEMA_VERSION,
        "vertex_count": arrays["vertex_count"],
        "face_count": arrays["face_count"],
        "triangles_only": arrays["triangles_only"],
        "bounds_min": arrays["bounds_min"].tolist(),
        "bounds_max": arrays["bounds_max"].tolist(),
        "normals_count": int(len(arrays["normals"])),
    }
    payload = {
        "points": arrays["points"],
        "face_vertex_indices": arrays["face_vertex_indices"],
        "face_vertex_counts": arrays["face_vertex_counts"],
        "normals": arrays["normals"],
        "metadata_json": np.frombuffer(
            canonical_json(cache_metadata).encode("utf-8"),
            dtype=np.uint8,
        ),
    }
    fd, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    try:
        with open(temporary_name, "wb") as stream:
            if compressed:
                np.savez_compressed(stream, **payload)
            else:
                np.savez(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        loaded = load_surface_cache(temporary_name)
        if loaded["metadata"]["output_index"] != output_index:
            raise RuntimeError("Surface cache round-trip changed output_index")
        digest = file_sha256(temporary_name)
        byte_count = os.path.getsize(temporary_name)
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "output_index": output_index,
        "sim_step": int(metadata["sim_step"]),
        "sim_time_seconds": float(metadata["sim_time_seconds"]),
        "take_id": str(metadata["take_id"]),
        "config_hash": str(metadata["config_hash"]),
        "simulation_provenance_hash": str(
            metadata.get("simulation_provenance_hash", "")
        ),
        "cache_file": target.name,
        "cache_bytes": byte_count,
        "cache_sha256": digest,
        "vertex_count": arrays["vertex_count"],
        "face_count": arrays["face_count"],
        "bounds_min": arrays["bounds_min"].tolist(),
        "bounds_max": arrays["bounds_max"].tolist(),
        "status": "cached",
    }


def load_surface_cache(
    path: os.PathLike[str] | str,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    source = Path(path)
    if expected_sha256 is not None:
        actual_hash = file_sha256(source)
        if actual_hash != expected_sha256:
            raise ValueError(
                f"Surface cache hash mismatch for {source}: {actual_hash} != {expected_sha256}"
            )
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "points",
            "face_vertex_indices",
            "face_vertex_counts",
            "normals",
            "metadata_json",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"Surface cache {source} is missing arrays: {', '.join(missing)}")
        metadata_bytes = np.asarray(archive["metadata_json"], dtype=np.uint8).tobytes()
        try:
            metadata = json.loads(metadata_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Surface cache {source} has invalid metadata") from exc
        arrays = validate_surface_arrays(
            archive["points"],
            archive["face_vertex_indices"],
            archive["face_vertex_counts"] if len(archive["face_vertex_counts"]) else None,
            archive["normals"] if len(archive["normals"]) else None,
        )
        result = {
            "metadata": metadata,
            **arrays,
        }
    if metadata.get("schema") != CACHE_SCHEMA:
        raise ValueError(f"Unknown cache schema in {source}: {metadata.get('schema')}")
    if int(metadata.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported cache schema version in {source}: {metadata.get('schema_version')}"
        )
    for name in ("vertex_count", "face_count"):
        if int(metadata.get(name, -1)) != int(result[name]):
            raise ValueError(f"Surface cache metadata mismatch for {name} in {source}")
    return result


def validate_contiguous_manifest(
    rows: Iterable[dict[str, Any]],
    *,
    expected_frames: int,
    expected_take_id: str | None = None,
    expected_config_hash: str | None = None,
    expected_simulation_provenance_hash: str | None = None,
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: int(row["output_index"]))
    if len(ordered) != expected_frames:
        raise ValueError(f"Expected {expected_frames} manifest rows, found {len(ordered)}")
    indices = [int(row["output_index"]) for row in ordered]
    expected = list(range(expected_frames))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        duplicates = sorted(index for index in set(indices) if indices.count(index) > 1)
        raise ValueError(f"Non-contiguous manifest; missing={missing[:20]}, duplicates={duplicates[:20]}")
    take_ids = {str(row.get("take_id")) for row in ordered}
    hashes = {str(row.get("config_hash")) for row in ordered}
    simulation_hashes = {
        str(row.get("simulation_provenance_hash", "")) for row in ordered
    }
    if len(take_ids) != 1 or (expected_take_id is not None and take_ids != {expected_take_id}):
        raise ValueError(f"Manifest mixes take IDs: {sorted(take_ids)}")
    if len(hashes) != 1 or (
        expected_config_hash is not None and hashes != {expected_config_hash}
    ):
        raise ValueError(f"Manifest mixes config hashes: {sorted(hashes)}")
    if len(simulation_hashes) != 1 or (
        expected_simulation_provenance_hash is not None
        and simulation_hashes != {expected_simulation_provenance_hash}
    ):
        raise ValueError(
            "Manifest mixes simulation provenance hashes: "
            f"{sorted(simulation_hashes)}"
        )
    return ordered


def png_dimensions(path: os.PathLike[str] | str) -> tuple[int, int]:
    source = Path(path)
    idat_chunks: list[bytes] = []
    saw_ihdr = False
    saw_iend = False
    width = height = 0
    with open(source, "rb") as stream:
        if stream.read(8) != PNG_SIGNATURE:
            raise ValueError(f"Not a PNG file: {source}")
        while not saw_iend:
            length_bytes = stream.read(4)
            if len(length_bytes) != 4:
                raise ValueError(f"Truncated PNG chunk length: {source}")
            length = struct.unpack(">I", length_bytes)[0]
            chunk_type = stream.read(4)
            if len(chunk_type) != 4:
                raise ValueError(f"Truncated PNG chunk type: {source}")
            chunk_data = stream.read(length)
            crc_bytes = stream.read(4)
            if len(chunk_data) != length or len(crc_bytes) != 4:
                raise ValueError(f"Truncated PNG chunk data: {source}")
            expected_crc = struct.unpack(">I", crc_bytes)[0]
            actual_crc = zlib.crc32(chunk_type)
            actual_crc = zlib.crc32(chunk_data, actual_crc) & 0xFFFFFFFF
            if actual_crc != expected_crc:
                raise ValueError(
                    f"PNG CRC mismatch for {source} chunk {chunk_type!r}"
                )
            if chunk_type == b"IHDR":
                if saw_ihdr or length != 13:
                    raise ValueError(f"Invalid PNG IHDR in {source}")
                width, height = struct.unpack(">II", chunk_data[:8])
                saw_ihdr = True
            elif chunk_type == b"IDAT":
                if not saw_ihdr:
                    raise ValueError(f"PNG IDAT precedes IHDR in {source}")
                idat_chunks.append(chunk_data)
            elif chunk_type == b"IEND":
                if length != 0:
                    raise ValueError(f"Invalid PNG IEND in {source}")
                saw_iend = True
        if stream.read(1):
            raise ValueError(f"Unexpected data after PNG IEND: {source}")
    if not saw_ihdr or not saw_iend or width <= 0 or height <= 0:
        raise ValueError(f"Incomplete PNG structure: {source}")
    if not idat_chunks:
        raise ValueError(f"PNG contains no image data: {source}")
    try:
        decompressed = zlib.decompress(b"".join(idat_chunks))
    except zlib.error as exc:
        raise ValueError(f"PNG image data cannot be decompressed: {source}") from exc
    if not decompressed:
        raise ValueError(f"PNG image data is empty: {source}")
    return width, height


def validate_png(
    path: os.PathLike[str] | str,
    *,
    width: int,
    height: int,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.stat().st_size <= 0:
        raise ValueError(f"Missing or empty PNG: {source}")
    actual_dimensions = png_dimensions(source)
    if actual_dimensions != (width, height):
        raise ValueError(
            f"PNG size mismatch for {source}: {actual_dimensions} != {(width, height)}"
        )
    digest = file_sha256(source)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"PNG hash mismatch for {source}: {digest} != {expected_sha256}")
    return {
        "png_file": source.name,
        "png_bytes": source.stat().st_size,
        "png_sha256": digest,
        "width": width,
        "height": height,
    }


def require_integer_capture_stride(physics_fps: int, output_fps: int) -> int:
    if physics_fps <= 0 or output_fps <= 0:
        raise ValueError("Frame rates must be positive")
    if physics_fps % output_fps:
        raise ValueError(
            f"Physics FPS {physics_fps} is not evenly divisible by output FPS {output_fps}"
        )
    return physics_fps // output_fps


def expected_output_frames(duration_seconds: float, output_fps: int) -> int:
    frames = duration_seconds * output_fps
    rounded = round(frames)
    if not math.isclose(frames, rounded, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(
            f"duration_seconds * output_fps must be an integer, got {frames}"
        )
    if rounded <= 0:
        raise ValueError("Video must contain at least one frame")
    return int(rounded)


class SurfaceCacheTakeWriter:
    """Write one non-resumable continuous simulation take in timeline order."""

    def __init__(
        self,
        take_dir: os.PathLike[str] | str,
        *,
        take_id: str,
        config_hash_value: str,
        simulation_provenance_hash_value: str,
        expected_frames: int,
        physics_fps: int = 60,
        output_fps: int = 30,
        compressed: bool = False,
    ) -> None:
        self.take_dir = Path(take_dir)
        self.cache_dir = self.take_dir / "cache"
        self.manifest_path = self.take_dir / "simulation_manifest.jsonl"
        self.completion_path = self.take_dir / "simulation_complete.json"
        self.template_path = self.take_dir / "render_template.usda"
        self.take_id = str(take_id)
        self.config_hash = str(config_hash_value)
        self.simulation_provenance_hash = str(
            simulation_provenance_hash_value
        )
        self.expected_frames = int(expected_frames)
        self.physics_fps = int(physics_fps)
        self.output_fps = int(output_fps)
        self.capture_stride = require_integer_capture_stride(
            self.physics_fps,
            self.output_fps,
        )
        self.compressed = bool(compressed)
        self.next_output_index = 0
        self.rows: list[dict[str, Any]] = []
        if self.expected_frames <= 0:
            raise ValueError("expected_frames must be positive")
        if self.take_dir.exists():
            existing = []
            for entry in self.take_dir.iterdir():
                if entry == self.cache_dir and entry.is_dir() and not any(entry.iterdir()):
                    continue
                existing.append(entry)
            if existing:
                raise FileExistsError(
                    "A physical take cannot resume or mix with existing output: "
                    f"{self.take_dir}; existing={[entry.name for entry in existing]}"
                )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def write_frame(
        self,
        *,
        points: Any,
        face_vertex_indices: Any,
        face_vertex_counts: Any | None = None,
        normals: Any | None = None,
        sim_step: int,
        sim_time_seconds: float,
        frame_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        output_index = self.next_output_index
        if output_index >= self.expected_frames:
            raise RuntimeError("Surface cache take received too many frames")
        expected_sim_step = output_index * self.capture_stride
        expected_sim_time = output_index / self.output_fps
        if int(sim_step) != expected_sim_step:
            raise ValueError(
                f"Frame {output_index} was captured at physical step {sim_step}; "
                f"expected {expected_sim_step}"
            )
        if not math.isclose(
            float(sim_time_seconds),
            expected_sim_time,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                f"Frame {output_index} was captured at simulation time "
                f"{sim_time_seconds}; expected {expected_sim_time}"
            )
        metadata = {
            **(frame_metadata or {}),
            "output_index": output_index,
            "sim_step": int(sim_step),
            "sim_time_seconds": float(sim_time_seconds),
            "take_id": self.take_id,
            "config_hash": self.config_hash,
            "simulation_provenance_hash": self.simulation_provenance_hash,
        }
        cache_path = self.cache_dir / f"surface_{output_index:06d}.npz"
        row = write_surface_cache(
            cache_path,
            points=points,
            face_vertex_indices=face_vertex_indices,
            face_vertex_counts=face_vertex_counts,
            normals=normals,
            metadata=metadata,
            compressed=self.compressed,
        )
        self.rows.append(row)
        atomic_write_jsonl(self.manifest_path, self.rows)
        self.next_output_index += 1
        return row

    def finalize(self, *, completion_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.next_output_index != self.expected_frames:
            raise RuntimeError(
                f"Cannot complete take: {self.next_output_index}/{self.expected_frames} frames"
            )
        if not self.template_path.is_file() or self.template_path.stat().st_size <= 0:
            raise RuntimeError(f"Missing non-empty render template: {self.template_path}")
        ordered = validate_contiguous_manifest(
            read_jsonl(self.manifest_path),
            expected_frames=self.expected_frames,
            expected_take_id=self.take_id,
            expected_config_hash=self.config_hash,
            expected_simulation_provenance_hash=self.simulation_provenance_hash,
        )
        total_bytes = sum(int(row["cache_bytes"]) for row in ordered)
        template_sha256 = file_sha256(self.template_path)
        completion = {
            **(completion_metadata or {}),
            "schema_version": 1,
            "valid": True,
            "take_id": self.take_id,
            "config_hash": self.config_hash,
            "simulation_provenance_hash": self.simulation_provenance_hash,
            "frame_count": self.expected_frames,
            "physics_fps": self.physics_fps,
            "output_fps": self.output_fps,
            "capture_stride": self.capture_stride,
            "first_sim_time_seconds": 0.0,
            "last_sim_time_seconds": (self.expected_frames - 1) / self.output_fps,
            "cache_bytes": total_bytes,
            "render_template": self.template_path.name,
            "render_template_sha256": template_sha256,
        }
        atomic_write_json(self.completion_path, completion)
        return completion
