"""Independently audit full-particle whitewater feature caches."""

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
parser.add_argument("feature_directory", type=Path)
parser.add_argument("--output", type=Path)
parser.add_argument("--active-surface-threshold", type=float, default=0.1)
parser.add_argument("--normal-length-tolerance", type=float, default=2.0e-4)
parser.add_argument("--formula-absolute-tolerance", type=float, default=5.0e-5)
parser.add_argument("--formula-relative-tolerance", type=float, default=2.0e-5)
parser.add_argument("--minimum-impact-potential-gain", type=float, default=5.0)
parser.add_argument("--maximum-static-active-fraction", type=float, default=0.20)
parser.add_argument(
    "--minimum-top-curvature-quality-median", type=float, default=0.50
)
args = parser.parse_args()


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


def quantiles(values):
    values = np.asarray(values)
    if not values.size:
        return None
    return {
        str(q): float(np.quantile(values, q))
        for q in (0.0, 0.5, 0.9, 0.99, 1.0)
    }


def maximum_error(actual, expected):
    return float(np.max(np.abs(actual - expected))) if actual.size else 0.0


def sample_terrain_nearest(positions, x_values, z_values, terrain_y):
    spacing_x = float(np.median(np.diff(x_values)))
    spacing_z = float(np.median(np.diff(z_values)))
    ix = np.rint((positions[:, 0] - x_values[0]) / spacing_x).astype(np.int64)
    iz = np.rint((positions[:, 2] - z_values[0]) / spacing_z).astype(np.int64)
    valid = (
        (ix >= 0)
        & (ix < len(x_values))
        & (iz >= 0)
        & (iz < len(z_values))
    )
    sampled = np.full(len(positions), np.nan, dtype=np.float32)
    sampled[valid] = terrain_y[ix[valid], iz[valid]]
    valid &= np.isfinite(sampled)
    return sampled, valid


feature_directory = args.feature_directory.resolve()
manifest_path = feature_directory / "manifest.json"
if not manifest_path.is_file():
    raise FileNotFoundError(manifest_path)
output_path = (
    (feature_directory / "audit_report.json")
    if args.output is None
    else args.output.resolve()
)
if output_path.exists():
    raise FileExistsError(f"Refusing to overwrite audit report: {output_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
errors = []


def require(condition, message):
    if not condition:
        errors.append(message)
    return bool(condition)


require(manifest.get("schema") == 1, "Unsupported feature manifest schema")
state = manifest.get("state", {})
require(state.get("complete") is True, "Feature manifest is incomplete")
require(
    state.get("completed_samples") == state.get("expected_samples"),
    "Feature sample counts do not match",
)
sample_rows = manifest.get("samples", [])
sample_indices = manifest.get("sample_indices", [])
require(len(sample_rows) == len(sample_indices), "Manifest sample row count mismatch")
require(
    [row.get("source_sample_index") for row in sample_rows] == sample_indices,
    "Feature samples are not in the declared order",
)
require(bool(sample_indices) and sample_indices[0] == 0, "Sample 0 is required")

for producer_key in ("script", "kernel_module"):
    producer_path = Path(manifest["producer"][producer_key])
    hash_key = producer_key + "_sha256"
    require(producer_path.is_file(), f"Missing producer file: {producer_path}")
    if producer_path.is_file():
        require(
            sha256_file(producer_path) == manifest["producer"][hash_key],
            f"Producer hash mismatch: {producer_path}",
        )

source_manifest_path = Path(manifest["source"]["manifest"])
source_audit_path = Path(manifest["source"]["audit"])
run_report_path = Path(manifest["source"]["run_report"])
terrain_path = Path(manifest["terrain"]["heightfield"])
referenced_files = (
    (source_manifest_path, manifest["source"]["manifest_sha256"]),
    (source_audit_path, manifest["source"]["audit_sha256"]),
    (run_report_path, manifest["source"]["run_report_sha256"]),
    (terrain_path, manifest["terrain"]["heightfield_sha256"]),
)
for referenced_path, expected_hash in referenced_files:
    require(referenced_path.is_file(), f"Missing referenced file: {referenced_path}")
    if referenced_path.is_file():
        require(
            sha256_file(referenced_path) == expected_hash,
            f"Referenced file hash mismatch: {referenced_path}",
        )

source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
require(source_audit.get("valid") is True, "Source audit is invalid")
require(run_report.get("valid") is True, "Parent PhysX report is invalid")
require(source_manifest.get("state", {}).get("complete") is True, "Source is incomplete")
require(
    source_audit.get("manifest_sha256") == sha256_file(source_manifest_path),
    "Source audit does not reference the current source manifest",
)

with np.load(terrain_path, allow_pickle=False) as terrain_cache:
    terrain_x = np.asarray(terrain_cache["x_values"])
    terrain_z = np.asarray(terrain_cache["z_values"])
    terrain_y = np.asarray(terrain_cache["terrain_y"], dtype=np.float32)
require(
    terrain_y.shape == (len(terrain_x), len(terrain_z)),
    "Terrain coordinate and height shapes do not match",
)

particle_count = int(source_manifest["particle_count"])
spacing = float(manifest["configuration"]["particle_spacing"])
support_radius = float(manifest["configuration"]["support_radius"])
solid_start = float(
    manifest["configuration"]["solid_clearance_start_support_radii"]
)
sphere_radius = float(manifest["configuration"]["sphere_radius"])
expected_array_keys = set(manifest["arrays"])
required_scalar_keys = {
    "schema",
    "source_sample_index",
    "physics_step",
    "simulation_time",
}
source_rows = source_manifest["samples"]
sample_metrics = []

for row in sample_rows:
    sample_index = int(row["source_sample_index"])
    feature_path = feature_directory / row["file"]
    require(feature_path.is_file(), f"Missing feature file: {feature_path}")
    if not feature_path.is_file():
        continue
    require(feature_path.stat().st_size == row["bytes"], f"Byte mismatch: {feature_path}")
    require(sha256_file(feature_path) == row["sha256"], f"Hash mismatch: {feature_path}")

    source_row = source_rows[sample_index]
    source_path = source_manifest_path.parent / source_row["file"]
    require(source_path.is_file(), f"Missing source sample: {source_path}")
    if not source_path.is_file():
        continue
    require(sha256_file(source_path) == source_row["sha256"], f"Source hash mismatch: {source_path}")

    with np.load(feature_path, allow_pickle=False) as feature_cache:
        keys = set(feature_cache.files)
        require(
            keys == expected_array_keys | required_scalar_keys,
            f"Unexpected feature keys in sample {sample_index}: {sorted(keys)}",
        )
        arrays = {key: np.asarray(feature_cache[key]) for key in expected_array_keys}
        feature_schema = int(feature_cache["schema"])
        feature_index = int(feature_cache["source_sample_index"])
        feature_step = int(feature_cache["physics_step"])
        feature_time = float(feature_cache["simulation_time"])
    require(feature_schema == 1, f"Feature schema mismatch at sample {sample_index}")

    for key, specification in manifest["arrays"].items():
        array = arrays[key]
        require(array.shape == tuple(specification["shape"]), f"Shape mismatch: {sample_index}/{key}")
        require(array.dtype == np.dtype(specification["dtype"]), f"Dtype mismatch: {sample_index}/{key}")
        require(np.isfinite(array).all(), f"Non-finite values: {sample_index}/{key}")

    with np.load(source_path, allow_pickle=False) as source_cache:
        positions = np.asarray(source_cache["positions"], dtype=np.float32)
        velocities = np.asarray(source_cache["velocities"], dtype=np.float32)
        source_index = int(source_cache["sample_index"])
        source_step = int(source_cache["physics_step"])
        source_time = float(source_cache["simulation_time"])
        sphere_transform = np.asarray(source_cache["sphere_transform"], dtype=np.float64)
    require(positions.shape == (particle_count, 3), f"Source position shape mismatch: {sample_index}")
    require(velocities.shape == (particle_count, 3), f"Source velocity shape mismatch: {sample_index}")
    require(feature_index == source_index == sample_index, f"Index mismatch at sample {sample_index}")
    require(feature_step == source_step, f"Physics-step mismatch at sample {sample_index}")
    require(math.isclose(feature_time, source_time, abs_tol=1.0e-12), f"Time mismatch at sample {sample_index}")

    confidence = arrays["surface_confidence"]
    gradient_quality = arrays["gradient_quality"]
    curvature_quality = arrays["curvature_quality"]
    require(np.logical_and(confidence >= 0.0, confidence <= 1.0).all(), f"Surface confidence out of range: {sample_index}")
    require(np.logical_and(gradient_quality >= 0.0, gradient_quality <= 1.00001).all(), f"Gradient quality out of range: {sample_index}")
    require(np.logical_and(curvature_quality >= 0.0, curvature_quality <= 1.00001).all(), f"Curvature quality out of range: {sample_index}")
    require((arrays["neighbor_count"] >= 1).all(), f"Zero-neighbor particle: {sample_index}")
    require((arrays["number_density"] > 0.0).all(), f"Non-positive number density: {sample_index}")
    require((arrays["kinetic_energy"] >= 0.0).all(), f"Negative kinetic energy: {sample_index}")
    require((arrays["trapped_air_potential"] >= 0.0).all(), f"Negative trapped-air potential: {sample_index}")
    require((arrays["wave_crest_potential"] >= 0.0).all(), f"Negative crest potential: {sample_index}")

    active = confidence >= args.active_surface_threshold
    require(np.any(active), f"No active surface particles: {sample_index}")
    normal_lengths = np.linalg.norm(arrays["surface_normal"], axis=1)
    normal_error = float(np.max(np.abs(normal_lengths[active] - 1.0)))
    require(normal_error <= args.normal_length_tolerance, f"Surface normal length error {normal_error}: {sample_index}")

    sampled_terrain, terrain_valid = sample_terrain_nearest(
        positions, terrain_x, terrain_z, terrain_y
    )
    require(terrain_valid.all(), f"Terrain does not cover all particles: {sample_index}")
    sphere_center = sphere_transform[3, :3].astype(np.float32)
    expected_clearance = np.minimum(
        positions[:, 1] - sampled_terrain,
        np.linalg.norm(positions - sphere_center, axis=1) - sphere_radius,
    ).astype(np.float32)
    clearance_error = maximum_error(arrays["solid_clearance"], expected_clearance)
    require(clearance_error <= 2.0e-6, f"Solid-clearance alignment error {clearance_error}: {sample_index}")
    low_clearance = expected_clearance <= solid_start * support_radius
    low_clearance_max = float(np.max(confidence[low_clearance])) if np.any(low_clearance) else 0.0
    require(low_clearance_max <= 1.0e-6, f"Solid-boundary surface leakage {low_clearance_max}: {sample_index}")

    speed = np.linalg.norm(velocities, axis=1)
    expected_kinetic = (0.5 * speed * speed).astype(np.float32)
    expected_normal_velocity = np.sum(
        velocities * arrays["surface_normal"], axis=1
    ).astype(np.float32)
    convergence = np.maximum(-arrays["velocity_divergence"], 0.0)
    expected_trapped = (
        confidence
        * gradient_quality
        * convergence
        * arrays["velocity_dispersion"]
    ).astype(np.float32)
    curvature_number = np.minimum(
        np.abs(arrays["surface_curvature"]) * spacing, 2.0
    )
    expected_crest = (
        confidence
        * curvature_quality
        * np.maximum(expected_normal_velocity, 0.0)
        * curvature_number
    ).astype(np.float32)
    comparisons = (
        ("kinetic_energy", arrays["kinetic_energy"], expected_kinetic),
        ("normal_velocity", arrays["normal_velocity"], expected_normal_velocity),
        ("trapped_air_potential", arrays["trapped_air_potential"], expected_trapped),
        ("wave_crest_potential", arrays["wave_crest_potential"], expected_crest),
    )
    formula_errors = {}
    for name, actual, expected in comparisons:
        formula_errors[name] = maximum_error(actual, expected)
        require(
            np.allclose(
                actual,
                expected,
                atol=args.formula_absolute_tolerance,
                rtol=args.formula_relative_tolerance,
            ),
            f"Per-index formula/alignment mismatch: {sample_index}/{name}",
        )

    active_curvature = np.abs(arrays["surface_curvature"][active])
    curvature_cutoff = float(np.quantile(active_curvature, 0.99))
    top_curvature = active & (np.abs(arrays["surface_curvature"]) >= curvature_cutoff)
    top_curvature_quality_median = float(np.median(curvature_quality[top_curvature]))
    require(
        top_curvature_quality_median >= args.minimum_top_curvature_quality_median,
        f"High-curvature neighborhood quality is too low: {sample_index}",
    )

    active_fraction = float(np.mean(active))
    sample_metrics.append(
        {
            "source_sample_index": sample_index,
            "simulation_time": source_time,
            "active_surface_particles": int(np.count_nonzero(active)),
            "active_surface_fraction": active_fraction,
            "active_normal_length_quantiles": quantiles(normal_lengths[active]),
            "low_clearance_particles": int(np.count_nonzero(low_clearance)),
            "low_clearance_maximum_surface_confidence": low_clearance_max,
            "solid_clearance_maximum_error": clearance_error,
            "top_curvature_quality_median": top_curvature_quality_median,
            "formula_maximum_absolute_errors": formula_errors,
            "trapped_air_potential_sum": float(arrays["trapped_air_potential"].sum(dtype=np.float64)),
            "wave_crest_potential_sum": float(arrays["wave_crest_potential"].sum(dtype=np.float64)),
            "active_number_density_quantiles": quantiles(arrays["number_density"][active]),
            "inactive_number_density_quantiles": quantiles(arrays["number_density"][~active]),
        }
    )

if sample_metrics:
    baseline = sample_metrics[0]
    require(
        baseline["active_surface_fraction"] <= args.maximum_static_active_fraction,
        "Static active-surface fraction is implausibly high",
    )
    peak_trapped = max(row["trapped_air_potential_sum"] for row in sample_metrics[1:])
    peak_crest = max(row["wave_crest_potential_sum"] for row in sample_metrics[1:])
    trapped_gain = peak_trapped / max(baseline["trapped_air_potential_sum"], 1.0e-12)
    crest_gain = peak_crest / max(baseline["wave_crest_potential_sum"], 1.0e-12)
    require(trapped_gain >= args.minimum_impact_potential_gain, "Impact trapped-air gain is too small")
    require(crest_gain >= args.minimum_impact_potential_gain, "Impact wave-crest gain is too small")
else:
    trapped_gain = None
    crest_gain = None

report = {
    "schema": 1,
    "valid": not errors,
    "created_utc": utc_now_iso(),
    "feature_directory": str(feature_directory),
    "manifest": str(manifest_path),
    "manifest_sha256": sha256_file(manifest_path),
    "configuration": {
        "active_surface_threshold": args.active_surface_threshold,
        "normal_length_tolerance": args.normal_length_tolerance,
        "formula_absolute_tolerance": args.formula_absolute_tolerance,
        "formula_relative_tolerance": args.formula_relative_tolerance,
        "minimum_impact_potential_gain": args.minimum_impact_potential_gain,
        "maximum_static_active_fraction": args.maximum_static_active_fraction,
        "minimum_top_curvature_quality_median": args.minimum_top_curvature_quality_median,
    },
    "impact_gains": {
        "trapped_air_potential_sum": trapped_gain,
        "wave_crest_potential_sum": crest_gain,
    },
    "samples": sample_metrics,
    "errors": errors,
}
output_path.parent.mkdir(parents=True, exist_ok=True)
atomic_write_json(output_path, report)
print(json.dumps(report, indent=2))
if errors:
    raise SystemExit(1)
