"""Immutable fixed-topology mesh video jobs and atomic frame caches.

This module is intentionally Isaac-Sim independent.  A simulation adapter writes
one static topology plus per-frame points/transforms; a renderer can then replay
the take without running PhysX again.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from liquid_video_cache import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    config_hash,
    file_sha256,
    read_jsonl,
)


JOB_TYPE = "fixed_topology_mesh_video"
SCHEMA_VERSION = 1

REQUIRED_PATHS = (
    "take_dir",
    "cache_dir",
    "topology",
    "simulation_manifest",
    "simulation_complete",
    "render_template",
    "render_dir",
    "frames_dir",
    "render_segments_dir",
    "render_manifest",
    "render_complete",
    "video_dir",
    "reports_dir",
    "logs_dir",
)


def immutable_job_config(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_type": job["job_type"],
        "adapter": job["adapter"],
        "take_id": job["take_id"],
        "video": job["video"],
        "simulation": job["simulation"],
        "render": job["render"],
        "paths": job["paths"],
    }


def simulation_provenance_payload(job: dict[str, Any]) -> dict[str, Any]:
    hashes = job["script_sha256"]
    simulation_script = str(job["simulation"]["script"])
    return {
        "job_type": job["job_type"],
        "adapter": job["adapter"],
        "take_id": job["take_id"],
        "video_timeline": {
            "duration_seconds": job["video"]["duration_seconds"],
            "output_fps": job["video"]["output_fps"],
            "output_frames": job["video"]["output_frames"],
        },
        "simulation": job["simulation"],
        "simulation_script_sha256": hashes[simulation_script],
        "simulation_source_sha256": hashes["soft_body_bounce_hero.py"],
        "cache_contract_sha256": hashes["fixed_topology_video.py"],
    }


def render_provenance_hash(
    job: dict[str, Any],
    *,
    template_sha256: str,
    topology_sha256: str,
    renderer_sha256: str,
    isaac_version: str | None,
) -> str:
    return config_hash(
        {
            "job_type": job["job_type"],
            "take_id": job["take_id"],
            "config_hash": job["config_hash"],
            "simulation_provenance_hash": job["simulation_provenance_hash"],
            "render": job["render"],
            "template_sha256": template_sha256,
            "topology_sha256": topology_sha256,
            "renderer_sha256": renderer_sha256,
            "isaac_version": isaac_version,
        }
    )


def validate_job(job: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(job, dict):
        raise ValueError("job must be an object")
    if int(job.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported fixed-topology job schema: {job.get('schema_version')}")
    if job.get("job_type") != JOB_TYPE:
        raise ValueError(f"Unsupported job_type: {job.get('job_type')!r}")
    if job.get("adapter") != "soft_body_bounce":
        raise ValueError(f"Unsupported adapter: {job.get('adapter')!r}")
    for name in ("job_id", "take_id", "config_hash", "simulation_provenance_hash"):
        if not isinstance(job.get(name), str) or not job[name]:
            raise ValueError(f"job.{name} must be a non-empty string")
    paths = job.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("job.paths must be an object")
    for name in REQUIRED_PATHS:
        value = paths.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"job.paths.{name} must be a non-empty string")
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"job.paths.{name} escapes the job root")
    video = job.get("video", {})
    simulation = job.get("simulation", {})
    render = job.get("render", {})
    for name in ("output_fps", "output_frames", "width", "height"):
        if int(video.get(name, 0)) <= 0:
            raise ValueError(f"job.video.{name} must be positive")
    if float(video.get("duration_seconds", 0.0)) <= 0:
        raise ValueError("job.video.duration_seconds must be positive")
    physics_fps = int(simulation.get("physics_fps", 0))
    output_fps = int(video["output_fps"])
    if physics_fps <= 0 or physics_fps % output_fps:
        raise ValueError("simulation.physics_fps must be an integer multiple of output_fps")
    if int(simulation.get("capture_stride", 0)) != physics_fps // output_fps:
        raise ValueError("simulation.capture_stride does not match the frame rates")
    if simulation.get("cache_schema") != "fixed_topology_points_v1":
        raise ValueError("Unsupported simulation.cache_schema")
    for name in ("script", "dynamic_root_prim", "visual_mesh_prim", "camera_prim"):
        if not isinstance(simulation.get(name), str) or not simulation[name]:
            raise ValueError(f"job.simulation.{name} must be a non-empty string")
    if render.get("renderer") != "PathTracing":
        raise ValueError("Only PathTracing is supported for cached rendering")
    if int(render.get("path_spp", 0)) <= 0 or int(render.get("segment_frames", 0)) <= 0:
        raise ValueError("render.path_spp and render.segment_frames must be positive")
    hashes = job.get("script_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("job.script_sha256 must be an object")
    for name in (
        str(simulation["script"]),
        "fixed_topology_video.py",
        "fixed_topology_video_pipeline.py",
        "render_fixed_topology_video.py",
        "encode_fixed_topology_video.py",
        "run_fixed_topology_video_stage.ps1",
        "run_fixed_topology_video.bat",
        "soft_body_bounce_hero.py",
    ):
        if not isinstance(hashes.get(name), str) or not hashes[name]:
            raise ValueError(f"Missing script hash for {name}")
    if config_hash(immutable_job_config(job)) != job["config_hash"]:
        raise ValueError("job.config_hash does not match immutable job content")
    expected_simulation_hash = config_hash(simulation_provenance_payload(job))
    if expected_simulation_hash != job["simulation_provenance_hash"]:
        raise ValueError("job.simulation_provenance_hash does not match the job")
    return job


def load_job(job_dir: os.PathLike[str] | str) -> dict[str, Any]:
    root = Path(job_dir).resolve()
    with open(root / "job.json", "r", encoding="utf-8-sig") as stream:
        return validate_job(json.load(stream))


def job_path(job_dir: os.PathLike[str] | str, job: dict[str, Any], name: str) -> Path:
    root = Path(job_dir).resolve()
    result = (root / Path(job["paths"][name])).resolve()
    result.relative_to(root)
    return result


def _points(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
        raise ValueError(f"points must have shape (N, 3), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("points contain non-finite values")
    return np.ascontiguousarray(array)


def _translate(value: Any | None) -> np.ndarray:
    if value is None:
        return np.zeros(3, dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("translate must contain three finite values")
    return np.ascontiguousarray(array)


def _atomic_npz(path: Path, *, compressed: bool, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        writer = np.savez_compressed if compressed else np.savez
        with open(temporary_name, "wb") as stream:
            writer(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_topology(
    path: os.PathLike[str] | str,
    *,
    face_vertex_counts: Any,
    face_vertex_indices: Any,
    vertex_count: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    counts = np.asarray(face_vertex_counts, dtype=np.int32)
    indices = np.asarray(face_vertex_indices, dtype=np.int32)
    if counts.ndim != 1 or len(counts) == 0 or np.any(counts <= 0):
        raise ValueError("face_vertex_counts must be a non-empty positive vector")
    if indices.ndim != 1 or int(counts.sum(dtype=np.int64)) != len(indices):
        raise ValueError("face topology counts and indices disagree")
    if vertex_count <= 0 or len(indices) == 0 or int(indices.min()) < 0 or int(indices.max()) >= vertex_count:
        raise ValueError("topology indices are outside the fixed vertex range")
    target = Path(path)
    payload = {
        **metadata,
        "schema_version": SCHEMA_VERSION,
        "cache_schema": "fixed_topology_points_v1",
        "vertex_count": int(vertex_count),
        "face_count": int(len(counts)),
    }
    _atomic_npz(
        target,
        compressed=True,
        face_vertex_counts=np.ascontiguousarray(counts),
        face_vertex_indices=np.ascontiguousarray(indices),
        metadata_json=np.asarray(canonical_json(payload)),
    )
    return {
        "topology_file": target.name,
        "topology_bytes": target.stat().st_size,
        "topology_sha256": file_sha256(target),
        "vertex_count": int(vertex_count),
        "face_count": int(len(counts)),
    }


def load_topology(
    path: os.PathLike[str] | str, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    target = Path(path)
    if expected_sha256 is not None and file_sha256(target) != expected_sha256:
        raise ValueError(f"Topology hash mismatch: {target}")
    with np.load(target, allow_pickle=False) as data:
        counts = np.asarray(data["face_vertex_counts"], dtype=np.int32)
        indices = np.asarray(data["face_vertex_indices"], dtype=np.int32)
        metadata = json.loads(str(data["metadata_json"].item()))
    if int(metadata["vertex_count"]) <= 0:
        raise ValueError("Topology vertex count is invalid")
    if int(counts.sum(dtype=np.int64)) != len(indices):
        raise ValueError("Topology counts and indices disagree")
    return {"face_vertex_counts": counts, "face_vertex_indices": indices, "metadata": metadata}


def write_frame(
    path: os.PathLike[str] | str,
    *,
    points: Any,
    translate: Any | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    points_array = _points(points)
    translate_array = _translate(translate)
    payload = {
        **metadata,
        "schema_version": SCHEMA_VERSION,
        "cache_schema": "fixed_topology_points_v1",
        "vertex_count": int(len(points_array)),
    }
    target = Path(path)
    _atomic_npz(
        target,
        compressed=True,
        points=points_array,
        translate=translate_array,
        bounds_min=points_array.min(axis=0),
        bounds_max=points_array.max(axis=0),
        metadata_json=np.asarray(canonical_json(payload)),
    )
    return {
        "cache_file": target.name,
        "cache_bytes": target.stat().st_size,
        "cache_sha256": file_sha256(target),
        "vertex_count": int(len(points_array)),
    }


def load_frame(
    path: os.PathLike[str] | str,
    *,
    expected_sha256: str | None = None,
    expected_vertex_count: int | None = None,
) -> dict[str, Any]:
    target = Path(path)
    if expected_sha256 is not None and file_sha256(target) != expected_sha256:
        raise ValueError(f"Frame hash mismatch: {target}")
    with np.load(target, allow_pickle=False) as data:
        points = _points(data["points"])
        translate = _translate(data["translate"])
        bounds_min = np.asarray(data["bounds_min"], dtype=np.float32)
        bounds_max = np.asarray(data["bounds_max"], dtype=np.float32)
        metadata = json.loads(str(data["metadata_json"].item()))
    if expected_vertex_count is not None and len(points) != expected_vertex_count:
        raise ValueError(f"Frame vertex count changed: {len(points)} != {expected_vertex_count}")
    return {
        "points": points,
        "translate": translate,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "metadata": metadata,
    }


def validate_manifest(
    rows: list[dict[str, Any]], *, job: dict[str, Any]
) -> list[dict[str, Any]]:
    expected_frames = int(job["video"]["output_frames"])
    if len(rows) != expected_frames:
        raise ValueError(f"Manifest has {len(rows)} frames, expected {expected_frames}")
    prior_step = -1
    prior_time = -1.0
    for index, row in enumerate(rows):
        if int(row.get("output_index", -1)) != index:
            raise ValueError(f"Manifest is not contiguous at frame {index}")
        if str(row.get("take_id")) != job["take_id"]:
            raise ValueError(f"Frame {index} belongs to another take")
        if str(row.get("config_hash")) != job["config_hash"]:
            raise ValueError(f"Frame {index} belongs to another config")
        if str(row.get("simulation_provenance_hash")) != job["simulation_provenance_hash"]:
            raise ValueError(f"Frame {index} belongs to another provenance")
        step = int(row["sim_step"])
        time_value = float(row["sim_time_seconds"])
        if step <= prior_step or time_value <= prior_time:
            if index != 0:
                raise ValueError("Simulation time is not strictly increasing")
        prior_step = step
        prior_time = time_value
    return rows


class FixedTopologyTakeWriter:
    def __init__(self, job_dir: os.PathLike[str] | str, job: dict[str, Any]):
        self.job_dir = Path(job_dir).resolve()
        self.job = validate_job(job)
        self.take_dir = job_path(self.job_dir, job, "take_dir")
        self.cache_dir = job_path(self.job_dir, job, "cache_dir")
        self.topology_path = job_path(self.job_dir, job, "topology")
        self.manifest_path = job_path(self.job_dir, job, "simulation_manifest")
        self.completion_path = job_path(self.job_dir, job, "simulation_complete")
        self.expected_frames = int(job["video"]["output_frames"])
        if self.completion_path.exists() or self.manifest_path.exists():
            raise FileExistsError("A fixed-topology take is non-resumable; create a new job")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []
        self.topology_record: dict[str, Any] | None = None

    def write_topology(self, counts: Any, indices: Any, vertex_count: int) -> dict[str, Any]:
        if self.topology_record is not None:
            raise RuntimeError("Topology was already written")
        self.topology_record = write_topology(
            self.topology_path,
            face_vertex_counts=counts,
            face_vertex_indices=indices,
            vertex_count=vertex_count,
            metadata={
                "take_id": self.job["take_id"],
                "config_hash": self.job["config_hash"],
                "simulation_provenance_hash": self.job["simulation_provenance_hash"],
                "visual_mesh_prim": self.job["simulation"]["visual_mesh_prim"],
            },
        )
        return self.topology_record

    def write_frame(
        self,
        *,
        points: Any,
        translate: Any | None,
        sim_step: int,
        sim_time_seconds: float,
        frame_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.topology_record is None:
            raise RuntimeError("Write the static topology before frame data")
        output_index = len(self.rows)
        if output_index >= self.expected_frames:
            raise RuntimeError("Attempted to write more frames than the job declares")
        if output_index == 0:
            if sim_step != 0 or abs(sim_time_seconds) > 1e-9:
                raise ValueError("Frame zero must be the initial state at step/time zero")
        else:
            stride = int(self.job["simulation"]["capture_stride"])
            expected_step = output_index * stride
            expected_time = output_index / int(self.job["video"]["output_fps"])
            if sim_step != expected_step or abs(sim_time_seconds - expected_time) > 1e-8:
                raise ValueError(
                    f"Off-cadence frame {output_index}: step={sim_step}, time={sim_time_seconds}"
                )
        frame_name = f"frame_{output_index:06d}.npz"
        record = write_frame(
            self.cache_dir / frame_name,
            points=points,
            translate=translate,
            metadata={
                "output_index": output_index,
                "sim_step": sim_step,
                "sim_time_seconds": sim_time_seconds,
                "take_id": self.job["take_id"],
                "config_hash": self.job["config_hash"],
                "simulation_provenance_hash": self.job["simulation_provenance_hash"],
                "frame_metadata": frame_metadata or {},
            },
        )
        if record["vertex_count"] != int(self.topology_record["vertex_count"]):
            raise ValueError("Fixed-topology frame changed vertex count")
        row = {
            "schema_version": SCHEMA_VERSION,
            "output_index": output_index,
            "sim_step": sim_step,
            "sim_time_seconds": sim_time_seconds,
            "take_id": self.job["take_id"],
            "config_hash": self.job["config_hash"],
            "simulation_provenance_hash": self.job["simulation_provenance_hash"],
            **record,
        }
        self.rows.append(row)
        atomic_write_jsonl(self.manifest_path, self.rows)
        return row

    def finalize(
        self,
        *,
        render_template: os.PathLike[str] | str,
        isaac_version: str | None,
        validation: dict[str, Any],
    ) -> dict[str, Any]:
        if self.topology_record is None or len(self.rows) != self.expected_frames:
            raise RuntimeError(
                f"Cannot finalize incomplete take: {len(self.rows)}/{self.expected_frames} frames"
            )
        validate_manifest(read_jsonl(self.manifest_path), job=self.job)
        template = Path(render_template)
        if not template.is_file() or template.stat().st_size <= 0:
            raise RuntimeError("Render template is missing or empty")
        completion = {
            "schema_version": SCHEMA_VERSION,
            "valid": True,
            "job_type": JOB_TYPE,
            "take_id": self.job["take_id"],
            "config_hash": self.job["config_hash"],
            "simulation_provenance_hash": self.job["simulation_provenance_hash"],
            "frame_count": len(self.rows),
            "topology_sha256": self.topology_record["topology_sha256"],
            "topology_bytes": self.topology_record["topology_bytes"],
            "render_template_sha256": file_sha256(template),
            "manifest_sha256": file_sha256(self.manifest_path),
            "isaac_version": isaac_version,
            "validation": validation,
        }
        atomic_write_json(self.completion_path, completion)
        return completion
