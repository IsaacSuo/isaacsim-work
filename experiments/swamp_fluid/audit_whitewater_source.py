"""Audit the raw high-frequency PhysX source cache for whitewater simulation.

This audit intentionally validates evidence that the cache came from native,
temporally coherent PhysX state.  Passing file-shape checks alone is not enough:
native particle and rigid velocities are compared against observed displacement
over every cached interval and every particle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source_directory", type=Path)
parser.add_argument(
    "--run-report",
    type=Path,
    help="Parent PhysX run_complete.json; defaults to the source directory parent.",
)
parser.add_argument(
    "--maximum-particle-speed",
    type=float,
    default=6.05,
    help="Hard bound slightly above the authored PhysX maxVelocity.",
)
parser.add_argument(
    "--maximum-velocity-p99-error",
    type=float,
    default=0.35,
    help="Maximum worst-interval P99 native-vs-displacement error in m/s.",
)
parser.add_argument(
    "--maximum-active-relative-p99-error",
    type=float,
    default=1.0,
    help="Maximum relative P99 error for particles above the active-speed floor.",
)
parser.add_argument(
    "--minimum-active-particle-speed",
    type=float,
    default=0.05,
    help=(
        "Minimum reference speed in m/s for the relative-error gate. "
        "The absolute-error gate still evaluates every particle."
    ),
)
parser.add_argument(
    "--maximum-sphere-velocity-error",
    type=float,
    default=0.35,
    help="Maximum sphere midpoint-velocity displacement error in m/s.",
)
parser.add_argument(
    "--allow-missing-run-report",
    action="store_true",
    help="Permit structural QA before a parent run report exists.",
)
args = parser.parse_args()
if args.minimum_active_particle_speed <= 0.0:
    parser.error("--minimum-active-particle-speed must be positive")


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def require_scalar(array, dtype, name):
    if array.shape != ():
        raise ValueError(f"{name} must be scalar, got {array.shape}")
    if array.dtype != np.dtype(dtype):
        raise ValueError(f"{name} must have dtype {np.dtype(dtype)}, got {array.dtype}")
    return array.item()


def require_array(array, shape, dtype, name):
    if array.shape != tuple(shape):
        raise ValueError(f"{name} shape {array.shape} != {tuple(shape)}")
    if array.dtype != np.dtype(dtype):
        raise ValueError(f"{name} dtype {array.dtype} != {np.dtype(dtype)}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


source_directory = args.source_directory.resolve()
manifest_path = source_directory / "manifest.json"
if not manifest_path.is_file():
    raise FileNotFoundError(manifest_path)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest_schema = int(manifest.get("schema", -1))
if manifest_schema not in {1, 2}:
    raise ValueError(f"Unsupported manifest schema: {manifest.get('schema')!r}")

particle_count = int(manifest["particle_count"])
maximum_particle_count = int(manifest.get("maximum_particle_count", particle_count))
particle_id_contract = manifest.get("particle_ids", {})
append_only_births = (
    manifest_schema == 2
    and particle_id_contract.get("storage") == "per_sample"
    and particle_id_contract.get("dtype") == "int64"
    and particle_id_contract.get("births") == "append_only"
)
sampling = manifest["sampling"]
physics_fps = int(sampling["physics_fps"])
expected_steps = [int(value) for value in sampling["expected_physics_steps"]]
sample_rows = manifest["samples"]
expected_sample_count = int(manifest["state"]["expected_samples"])
run_report_path = (
    args.run_report.resolve()
    if args.run_report is not None
    else source_directory.parent / "run_complete.json"
)

required_keys = {
    "schema",
    "sample_index",
    "physics_step",
    "simulation_time",
    "positions",
    "velocities",
    "sphere_transform",
    "sphere_linear_velocity",
    "sphere_angular_velocity",
}
if append_only_births or "particle_ids" in manifest.get("arrays", {}):
    required_keys.add("particle_ids")
criteria = {
    "manifest_complete": manifest["state"].get("complete") is True,
    "manifest_sample_count": (
        len(sample_rows) == expected_sample_count == len(expected_steps)
    ),
    "manifest_indices_contiguous": all(
        int(row.get("sample_index", -1)) == index
        for index, row in enumerate(sample_rows)
    ),
    "manifest_steps_exact": [
        int(row.get("physics_step", -1)) for row in sample_rows
    ]
    == expected_steps,
    "manifest_particle_count_contract": (
        particle_count > 0
        and maximum_particle_count >= particle_count
        and (manifest_schema == 1 or append_only_births)
    ),
}

actual_source_files = sorted(source_directory.glob("source_*.npz"))
expected_names = [f"source_{index:06d}.npz" for index in range(len(sample_rows))]
criteria["source_files_exact"] = (
    [path.name for path in actual_source_files] == expected_names
)

sample_summaries = []
velocity_intervals = []
sphere_velocity_intervals = []
errors = []
maximum_particle_speed = 0.0
maximum_transform_orthogonality_error = 0.0
maximum_transform_determinant_error = 0.0
minimum_active_particle_count = math.inf
maximum_active_particle_count = 0
total_births_between_samples = 0
previous = None

for expected_index, row in enumerate(sample_rows):
    sample_path = source_directory / row["file"]
    try:
        if row["file"] != f"source_{expected_index:06d}.npz":
            raise ValueError(f"non-canonical filename {row['file']!r}")
        if not sample_path.is_file():
            raise FileNotFoundError(sample_path)
        if sample_path.stat().st_size != int(row["bytes"]):
            raise ValueError("file byte count does not match manifest")
        actual_sha256 = sha256_file(sample_path)
        if actual_sha256 != row["sha256"]:
            raise ValueError("file SHA-256 does not match manifest")

        with np.load(sample_path, allow_pickle=False) as cache:
            if set(cache.files) != required_keys:
                raise ValueError(
                    f"keys {sorted(cache.files)} != {sorted(required_keys)}"
                )
            schema = require_scalar(cache["schema"], "<i4", "schema")
            sample_index = require_scalar(
                cache["sample_index"], "<i4", "sample_index"
            )
            physics_step = require_scalar(
                cache["physics_step"], "<i8", "physics_step"
            )
            simulation_time = require_scalar(
                cache["simulation_time"], "<f8", "simulation_time"
            )
            active_particle_count = int(cache["positions"].shape[0])
            if manifest_schema == 1 and active_particle_count != particle_count:
                raise ValueError(
                    f"schema-1 active particle count {active_particle_count} != {particle_count}"
                )
            if not particle_count <= active_particle_count <= maximum_particle_count:
                raise ValueError(
                    "active particle count is outside manifest bounds: "
                    f"{active_particle_count} not in [{particle_count}, {maximum_particle_count}]"
                )
            positions = require_array(
                cache["positions"], (active_particle_count, 3), "<f4", "positions"
            )
            velocities = require_array(
                cache["velocities"],
                (active_particle_count, 3),
                "<f4",
                "velocities",
            )
            particle_ids = (
                require_array(
                    cache["particle_ids"],
                    (active_particle_count,),
                    "<i8",
                    "particle_ids",
                )
                if "particle_ids" in required_keys
                else np.arange(active_particle_count, dtype="<i8")
            )
            sphere_transform = require_array(
                cache["sphere_transform"], (4, 4), "<f8", "sphere_transform"
            )
            sphere_linear_velocity = require_array(
                cache["sphere_linear_velocity"],
                (3,),
                "<f4",
                "sphere_linear_velocity",
            )
            require_array(
                cache["sphere_angular_velocity"],
                (3,),
                "<f4",
                "sphere_angular_velocity",
            )

        # The per-frame binary layout remains schema 1. Manifest schema 2 adds
        # append-only particle identities and variable active counts around it.
        if schema != 1:
            raise ValueError(f"sample schema {schema} != 1")
        if sample_index != expected_index or sample_index != int(row["sample_index"]):
            raise ValueError("sample index does not match manifest/order")
        if physics_step != expected_steps[expected_index]:
            raise ValueError("physics step does not match planned schedule")
        if physics_step != int(row["physics_step"]):
            raise ValueError("physics step does not match manifest row")
        expected_time = physics_step / physics_fps
        if not math.isclose(simulation_time, expected_time, abs_tol=1.0e-12):
            raise ValueError("simulation time does not equal physics_step / physics_fps")
        if not math.isclose(
            simulation_time, float(row["simulation_time"]), abs_tol=1.0e-12
        ):
            raise ValueError("simulation time does not match manifest row")

        rotation = sphere_transform[:3, :3]
        orthogonality_error = float(
            np.max(np.abs(rotation @ rotation.T - np.eye(3)))
        )
        determinant_error = abs(float(np.linalg.det(rotation)) - 1.0)
        maximum_transform_orthogonality_error = max(
            maximum_transform_orthogonality_error, orthogonality_error
        )
        maximum_transform_determinant_error = max(
            maximum_transform_determinant_error, determinant_error
        )

        particle_speeds = np.linalg.norm(velocities, axis=1)
        sample_maximum_speed = float(particle_speeds.max(initial=0.0))
        maximum_particle_speed = max(maximum_particle_speed, sample_maximum_speed)
        minimum_active_particle_count = min(
            minimum_active_particle_count, active_particle_count
        )
        maximum_active_particle_count = max(
            maximum_active_particle_count, active_particle_count
        )

        if append_only_births:
            if not np.array_equal(
                particle_ids, np.arange(active_particle_count, dtype="<i8")
            ):
                raise ValueError(
                    "schema-2 particle IDs are not the canonical append-only sequence"
                )

        if previous is not None:
            dt = simulation_time - previous["simulation_time"]
            if dt <= 0.0:
                raise ValueError("sample times are not strictly increasing")
            previous_count = len(previous["particle_ids"])
            if active_particle_count < previous_count:
                raise ValueError("append-only source lost active particles")
            if not np.array_equal(
                particle_ids[:previous_count], previous["particle_ids"]
            ):
                raise ValueError("persistent particle ID prefix changed between samples")
            births = active_particle_count - previous_count
            total_births_between_samples += births
            persistent_positions = positions[:previous_count]
            persistent_velocities = velocities[:previous_count]
            displacement_velocity = (
                persistent_positions - previous["positions"]
            ) / dt
            native_midpoint_velocity = 0.5 * (
                persistent_velocities + previous["velocities"]
            )
            velocity_error = np.linalg.norm(
                displacement_velocity - native_midpoint_velocity, axis=1
            )
            reference_speed = np.maximum(
                np.linalg.norm(displacement_velocity, axis=1),
                np.linalg.norm(native_midpoint_velocity, axis=1),
            )
            active = reference_speed >= args.minimum_active_particle_speed
            absolute_p99 = float(np.quantile(velocity_error, 0.99))
            active_relative_p99 = (
                float(np.quantile(velocity_error[active] / reference_speed[active], 0.99))
                if np.any(active)
                else 0.0
            )
            velocity_intervals.append(
                {
                    "from_sample": expected_index - 1,
                    "to_sample": expected_index,
                    "dt": dt,
                    "absolute_error_p99": absolute_p99,
                    "absolute_error_maximum": float(velocity_error.max(initial=0.0)),
                    "active_particles": int(np.count_nonzero(active)),
                    "persistent_particles": previous_count,
                    "births": births,
                    "active_relative_error_p99": active_relative_p99,
                }
            )

            sphere_position = sphere_transform[3, :3]
            previous_sphere_position = previous["sphere_transform"][3, :3]
            sphere_displacement_velocity = (
                sphere_position - previous_sphere_position
            ) / dt
            sphere_native_midpoint = 0.5 * (
                sphere_linear_velocity + previous["sphere_linear_velocity"]
            )
            sphere_error = float(
                np.linalg.norm(sphere_displacement_velocity - sphere_native_midpoint)
            )
            sphere_velocity_intervals.append(
                {
                    "from_sample": expected_index - 1,
                    "to_sample": expected_index,
                    "error": sphere_error,
                }
            )

        sample_summaries.append(
            {
                "sample_index": expected_index,
                "physics_step": physics_step,
                "simulation_time": simulation_time,
                "maximum_particle_speed": sample_maximum_speed,
                "active_particle_count": active_particle_count,
                "births_since_previous": (
                    0
                    if previous is None
                    else active_particle_count - len(previous["particle_ids"])
                ),
                "position_minimum": positions.min(axis=0).astype(float).tolist(),
                "position_maximum": positions.max(axis=0).astype(float).tolist(),
            }
        )
        previous = {
            "simulation_time": simulation_time,
            "positions": positions,
            "velocities": velocities,
            "sphere_transform": sphere_transform,
            "sphere_linear_velocity": sphere_linear_velocity,
            "particle_ids": particle_ids,
        }
    except Exception as exc:
        errors.append(
            {
                "sample_index": expected_index,
                "file": str(sample_path),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        previous = None

worst_velocity_p99 = max(
    (row["absolute_error_p99"] for row in velocity_intervals), default=math.inf
)
worst_active_relative_p99 = max(
    (row["active_relative_error_p99"] for row in velocity_intervals),
    default=math.inf,
)
worst_sphere_velocity_error = max(
    (row["error"] for row in sphere_velocity_intervals), default=math.inf
)
criteria.update(
    {
        "all_samples_readable": not errors and len(sample_summaries) == len(sample_rows),
        "append_only_particle_identity": (
            manifest_schema == 1
            or (
                append_only_births
                and not errors
                and maximum_active_particle_count <= maximum_particle_count
            )
        ),
        "transform_is_rigid": (
            maximum_transform_orthogonality_error <= 1.0e-6
            and maximum_transform_determinant_error <= 1.0e-6
        ),
        "particle_speed_within_authored_limit": (
            maximum_particle_speed <= args.maximum_particle_speed
        ),
        "native_particle_velocity_matches_displacement": (
            len(velocity_intervals) == max(0, len(sample_rows) - 1)
            and worst_velocity_p99 <= args.maximum_velocity_p99_error
            and worst_active_relative_p99 <= args.maximum_active_relative_p99_error
        ),
        "native_sphere_velocity_matches_displacement": (
            len(sphere_velocity_intervals) == max(0, len(sample_rows) - 1)
            and worst_sphere_velocity_error <= args.maximum_sphere_velocity_error
        ),
    }
)

run_report = None
if run_report_path.is_file():
    run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
    source_entry = run_report.get("whitewater_source_cache")
    criteria["parent_run_valid"] = run_report.get("valid") is True
    criteria["parent_run_references_manifest"] = bool(
        source_entry
        and Path(source_entry["manifest"]).resolve() == manifest_path.resolve()
        and source_entry["manifest_sha256"] == sha256_file(manifest_path)
    )
else:
    criteria["parent_run_valid"] = args.allow_missing_run_report
    criteria["parent_run_references_manifest"] = args.allow_missing_run_report

report = {
    "schema": 1,
    "valid": all(criteria.values()),
    "audited_utc": utc_now_iso(),
    "source_directory": str(source_directory),
    "manifest": str(manifest_path),
    "manifest_sha256": sha256_file(manifest_path),
    "run_report": str(run_report_path) if run_report_path.is_file() else None,
    "particle_count": particle_count,
    "maximum_particle_count": maximum_particle_count,
    "sample_count": len(sample_rows),
    "physics_fps": physics_fps,
    "criteria": criteria,
    "thresholds": {
        "maximum_particle_speed": args.maximum_particle_speed,
        "maximum_velocity_p99_error": args.maximum_velocity_p99_error,
        "maximum_active_relative_p99_error": args.maximum_active_relative_p99_error,
        "minimum_active_particle_speed": args.minimum_active_particle_speed,
        "maximum_sphere_velocity_error": args.maximum_sphere_velocity_error,
    },
    "metrics": {
        "maximum_particle_speed": maximum_particle_speed,
        "worst_velocity_absolute_error_p99": worst_velocity_p99,
        "worst_active_velocity_relative_error_p99": worst_active_relative_p99,
        "worst_sphere_velocity_error": worst_sphere_velocity_error,
        "maximum_transform_orthogonality_error": (
            maximum_transform_orthogonality_error
        ),
        "maximum_transform_determinant_error": maximum_transform_determinant_error,
        "minimum_active_particle_count": (
            int(minimum_active_particle_count)
            if math.isfinite(minimum_active_particle_count)
            else 0
        ),
        "maximum_active_particle_count": maximum_active_particle_count,
        "total_births_between_samples": total_births_between_samples,
    },
    "errors": errors,
    "samples": sample_summaries,
    "velocity_intervals": velocity_intervals,
    "sphere_velocity_intervals": sphere_velocity_intervals,
}
audit_path = source_directory / "audit_report.json"
atomic_write_json(audit_path, report)
print(json.dumps({key: value for key, value in report.items() if key not in {
    "samples", "velocity_intervals", "sphere_velocity_intervals"
}}, indent=2))
if not report["valid"]:
    raise SystemExit(1)
