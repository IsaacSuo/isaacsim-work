"""Independent audit for a foam/surface-bubble optical-depth atlas cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


EXPECTED_MODEL = {
    "foam_birth_film_thickness": 0.00030,
    "foam_drained_film_thickness": 0.00008,
    "foam_drainage_time": 0.45,
    "foam_target_peak_tau": 0.80,
    "bubble_target_peak_tau": 0.65,
    "maximum_foam_anisotropy": 2.5,
    "foam_anisotropy_speed": 0.25,
    "foam_anisotropy_age": 0.30,
    "gaussian_cutoff_sigma": 3.0,
    "foam_raft_sigma_birth": 0.006,
    "foam_raft_sigma_mature": 0.012,
    "foam_raft_coalescence_time": 0.35,
}
EXPECTED_DT = 1.0 / 120.0
EXPECTED_SAMPLE_STRIDE = 4
MAXIMUM_INTEGRAL_RELATIVE_ERROR = 0.03
MINIMUM_SHORELINE_RETENTION = 0.75
MAXIMUM_STABLE_PAIR_TURNOVER = 1.25
MINIMUM_STABLE_PAIR_JACCARD = 0.35


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("atlas_directory", type=Path)
    return parser.parse_args()


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def relative_error(actual, expected):
    if expected == 0.0:
        return abs(actual)
    return abs(actual - expected) / expected


args = parse_args()
atlas_directory = args.atlas_directory.resolve()
manifest_path = atlas_directory / "manifest.json"
manifest = load_json(manifest_path)
errors = []

if manifest.get("schema") != 1:
    errors.append("Unsupported manifest schema")
if manifest.get("state", {}).get("complete") is not True:
    errors.append("Atlas manifest is incomplete")
if manifest.get("model") != EXPECTED_MODEL:
    errors.append("Atlas model parameters differ from the audited physical model")

configuration = manifest["configuration"]
sample_indices = manifest["sample_indices"]
expected_indices = list(
    range(
        configuration["start_sample"],
        configuration["end_sample"] + 1,
        configuration["sample_stride"],
    )
)
if configuration["sample_stride"] != EXPECTED_SAMPLE_STRIDE:
    errors.append("Atlas is not sampled at 30 FPS from the 120 Hz source")
if sample_indices != expected_indices:
    errors.append("Sample mapping is not a complete fixed-stride sequence")
if len(manifest["samples"]) != len(sample_indices):
    errors.append("Manifest sample count differs from the requested sequence")

source = manifest["source"]
state_directory = Path(source["state_directory"])
state_manifest_path = state_directory / "manifest.json"
state_audit_path = state_directory / "audit_report.json"
source_directory = Path(source["source_directory"])
source_manifest_path = source_directory / "manifest.json"
source_audit_path = source_directory / "audit_report.json"
linked_paths = {
    "state_manifest_sha256": state_manifest_path,
    "state_audit_sha256": state_audit_path,
    "source_manifest_sha256": source_manifest_path,
    "source_audit_sha256": source_audit_path,
    "terrain_sha256": Path(source["terrain_path"]),
}
for key, path in linked_paths.items():
    if not path.is_file() or sha256_file(path) != source[key]:
        errors.append(f"Upstream hash mismatch: {key}")
state_manifest = load_json(state_manifest_path)
state_audit = load_json(state_audit_path)
source_audit = load_json(source_audit_path)
if state_audit.get("valid") is not True or source_audit.get("valid") is not True:
    errors.append("An upstream cache no longer has a valid audit")
if state_audit.get("manifest_sha256") != sha256_file(state_manifest_path):
    errors.append("State audit does not reference the current state manifest")
if source_audit.get("manifest_sha256") != sha256_file(source_manifest_path):
    errors.append("Source audit does not reference the current source manifest")

producer = manifest["producer"]
for path_key, hash_key in (
    ("script", "script_sha256"),
    ("density_module", "density_module_sha256"),
    ("model_module", "model_module_sha256"),
):
    path = Path(producer[path_key])
    if not path.is_file() or sha256_file(path) != producer[hash_key]:
        errors.append(f"Producer changed after cache creation: {path_key}")

coordinates_path = atlas_directory / manifest["coordinates"]["file"]
if not coordinates_path.is_file():
    errors.append("Atlas coordinate file is missing")
elif sha256_file(coordinates_path) != manifest["coordinates"]["sha256"]:
    errors.append("Atlas coordinate hash mismatch")
with np.load(coordinates_path, allow_pickle=False) as coordinates:
    x_values = np.asarray(coordinates["x_values"])
    z_values = np.asarray(coordinates["z_values"])
    terrain_y = np.asarray(coordinates["terrain_y"])
shape = tuple(configuration["atlas_shape"])
if x_values.dtype != np.float64 or z_values.dtype != np.float64:
    errors.append("Atlas coordinates must retain float64 precision")
if terrain_y.dtype != np.float32 or terrain_y.shape != shape:
    errors.append("Atlas terrain field has the wrong dtype or shape")
if len(x_values) != shape[0] or len(z_values) != shape[1]:
    errors.append("Atlas shape does not match coordinate arrays")
if not (
    np.allclose(np.diff(x_values), configuration["atlas_spacing"], atol=1.0e-12)
    and np.allclose(np.diff(z_values), configuration["atlas_spacing"], atol=1.0e-12)
):
    errors.append("Atlas coordinates are not a regular 4 mm grid")

state_by_index = {
    int(item["source_sample_index"]): item for item in state_manifest["samples"]
}
sample_by_index = {
    int(item["source_sample_index"]): item for item in manifest["samples"]
}
cell_area = float(configuration["atlas_spacing"]) ** 2
maximum_foam_integral_error = 0.0
maximum_bubble_integral_error = 0.0
minimum_foam_retention = 1.0
minimum_bubble_retention = 1.0
minimum_orientation_eigenvalue = math.inf
maximum_orientation_trace_error = 0.0
maximum_dry_foam_tau = 0.0
maximum_dry_bubble_tau = 0.0
maximum_coverage_formula_error = 0.0
coverage_turnovers = []
stable_pair_turnovers = []
stable_pair_jaccards = []
previous_coverage = None
previous_foam_ids = None
pre_contact_nonzero_cells = 0
total_bytes = coordinates_path.stat().st_size
array_names = (
    "foam_tau",
    "foam_coverage",
    "foam_mean_age",
    "foam_orientation_xx",
    "foam_orientation_xz",
    "foam_orientation_zz",
    "bubble_tau",
    "bubble_coverage",
    "bubble_mean_radius",
    "liquid_top_y",
)

for expected_frame, source_sample_index in enumerate(sample_indices):
    if source_sample_index not in sample_by_index:
        errors.append(f"Missing manifest entry for sample {source_sample_index}")
        continue
    item = sample_by_index[source_sample_index]
    frame_path = atlas_directory / item["file"]
    if item["render_frame"] != expected_frame:
        errors.append(f"Render-frame mapping mismatch at sample {source_sample_index}")
    if not frame_path.is_file():
        errors.append(f"Missing frame: {frame_path.name}")
        continue
    total_bytes += frame_path.stat().st_size
    if frame_path.stat().st_size != item["bytes"]:
        errors.append(f"Byte count mismatch: {frame_path.name}")
    if sha256_file(frame_path) != item["sha256"]:
        errors.append(f"Hash mismatch: {frame_path.name}")
        continue

    state_item = state_by_index[source_sample_index]
    state_path = state_directory / state_item["file"]
    if sha256_file(state_path) != state_item["sha256"]:
        errors.append(f"State hash mismatch at sample {source_sample_index}")
        continue
    with np.load(state_path, allow_pickle=False) as state_cache:
        state = np.asarray(state_cache["state"], dtype=np.uint8)
        state_age = np.asarray(state_cache["state_age"], dtype=np.float64)
        liquid_volume = np.asarray(state_cache["liquid_volume"], dtype=np.float64)
        radius = np.asarray(state_cache["radius"], dtype=np.float64)
        weight = np.asarray(
            state_cache["representative_weight"], dtype=np.float64
        )
        marker_ids = np.asarray(state_cache["id"], dtype=np.uint64)
        state_time = float(state_cache["simulation_time"])
    foam = state == 2
    bubbles = state == 4
    thickness = EXPECTED_MODEL["foam_drained_film_thickness"] + (
        EXPECTED_MODEL["foam_birth_film_thickness"]
        - EXPECTED_MODEL["foam_drained_film_thickness"]
    ) * np.exp(-np.maximum(state_age[foam], 0.0) / EXPECTED_MODEL["foam_drainage_time"])
    expected_foam_area = float(
        np.sum(liquid_volume[foam] / thickness, dtype=np.float64)
    )
    expected_bubble_area = float(
        np.sum(math.pi * radius[bubbles] ** 2 * weight[bubbles], dtype=np.float64)
    )
    if relative_error(item["expected_foam_area"], expected_foam_area) > 1.0e-8:
        errors.append(f"Foam area model mismatch at sample {source_sample_index}")
    if relative_error(item["expected_bubble_area"], expected_bubble_area) > 1.0e-8:
        errors.append(f"Bubble area model mismatch at sample {source_sample_index}")
    foam_error = relative_error(item["raw_foam_tau_integral"], expected_foam_area)
    bubble_error = relative_error(item["raw_bubble_tau_integral"], expected_bubble_area)
    maximum_foam_integral_error = max(maximum_foam_integral_error, foam_error)
    maximum_bubble_integral_error = max(maximum_bubble_integral_error, bubble_error)

    with np.load(frame_path, allow_pickle=False) as frame:
        if int(frame["schema"]) != 1:
            errors.append(f"Frame schema mismatch: {frame_path.name}")
        if int(frame["render_frame"]) != expected_frame:
            errors.append(f"Frame scalar index mismatch: {frame_path.name}")
        if int(frame["source_sample_index"]) != source_sample_index:
            errors.append(f"Source sample scalar mismatch: {frame_path.name}")
        if abs(float(frame["simulation_time"]) - state_time) > 1.0e-12:
            errors.append(f"Simulation time mismatch: {frame_path.name}")
        arrays = {name: np.asarray(frame[name]) for name in array_names}
        wet_mask = np.asarray(frame["wet_mask"])
    for name, array in arrays.items():
        if array.shape != shape or array.dtype != np.float32:
            errors.append(f"{frame_path.name}:{name} has wrong shape/dtype")
        if name != "liquid_top_y" and not np.isfinite(array).all():
            errors.append(f"{frame_path.name}:{name} contains non-finite values")
    if wet_mask.shape != shape or wet_mask.dtype != np.uint8:
        errors.append(f"{frame_path.name}:wet_mask has wrong shape/dtype")
    if not np.all((wet_mask == 0) | (wet_mask == 1)):
        errors.append(f"{frame_path.name}:wet_mask is not binary")

    foam_tau = arrays["foam_tau"]
    bubble_tau = arrays["bubble_tau"]
    foam_coverage = arrays["foam_coverage"]
    bubble_coverage = arrays["bubble_coverage"]
    if np.min(foam_tau) < 0.0 or np.min(bubble_tau) < 0.0:
        errors.append(f"Negative optical depth: {frame_path.name}")
    if (
        np.min(foam_coverage) < 0.0
        or np.max(foam_coverage) >= 1.0
        or np.min(bubble_coverage) < 0.0
        or np.max(bubble_coverage) >= 1.0
    ):
        errors.append(f"Coverage outside [0,1): {frame_path.name}")
    formula_error = max(
        float(np.max(np.abs(foam_coverage - (-np.expm1(-foam_tau))))),
        float(np.max(np.abs(bubble_coverage - (-np.expm1(-bubble_tau))))),
    )
    maximum_coverage_formula_error = max(maximum_coverage_formula_error, formula_error)
    dry = wet_mask == 0
    maximum_dry_foam_tau = max(maximum_dry_foam_tau, float(np.max(foam_tau[dry], initial=0.0)))
    maximum_dry_bubble_tau = max(maximum_dry_bubble_tau, float(np.max(bubble_tau[dry], initial=0.0)))
    wet = ~dry
    invalid_wet = wet & (
        ~np.isfinite(arrays["liquid_top_y"])
        | ~np.isfinite(terrain_y)
        | (
            arrays["liquid_top_y"]
            < terrain_y + configuration["minimum_liquid_depth"] - 1.0e-6
        )
    )
    if np.any(invalid_wet):
        errors.append(f"Wet mask violates terrain clearance: {frame_path.name}")

    actual_foam_integral = float(np.sum(foam_tau, dtype=np.float64) * cell_area)
    actual_bubble_integral = float(np.sum(bubble_tau, dtype=np.float64) * cell_area)
    if relative_error(actual_foam_integral, item["clipped_foam_tau_integral"]) > 1.0e-7:
        errors.append(f"Clipped foam integral mismatch: {frame_path.name}")
    if relative_error(actual_bubble_integral, item["clipped_bubble_tau_integral"]) > 1.0e-7:
        errors.append(f"Clipped bubble integral mismatch: {frame_path.name}")
    if expected_foam_area > 0.0:
        retention = actual_foam_integral / item["raw_foam_tau_integral"]
        minimum_foam_retention = min(minimum_foam_retention, retention)
    if expected_bubble_area > 0.0:
        retention = actual_bubble_integral / item["raw_bubble_tau_integral"]
        minimum_bubble_retention = min(minimum_bubble_retention, retention)

    active = foam_tau > 1.0e-6
    if np.any(active):
        xx = arrays["foam_orientation_xx"][active]
        xz = arrays["foam_orientation_xz"][active]
        zz = arrays["foam_orientation_zz"][active]
        trace = xx + zz
        discriminant = np.sqrt(np.maximum((xx - zz) ** 2 + 4.0 * xz**2, 0.0))
        minimum_eigenvalue = 0.5 * (trace - discriminant)
        minimum_orientation_eigenvalue = min(
            minimum_orientation_eigenvalue, float(np.min(minimum_eigenvalue))
        )
        maximum_orientation_trace_error = max(
            maximum_orientation_trace_error, float(np.max(np.abs(trace - 1.0)))
        )
        ages = arrays["foam_mean_age"][active]
        if np.min(ages) < -1.0e-6 or np.max(ages) > np.max(state_age[foam], initial=0.0) + 1.0e-5:
            errors.append(f"Foam mean age outside marker bounds: {frame_path.name}")
    bubble_active = bubble_tau > 1.0e-6
    if np.any(bubble_active):
        means = arrays["bubble_mean_radius"][bubble_active]
        if np.min(means) < np.min(radius[bubbles], initial=0.0) - 1.0e-6 or np.max(means) > np.max(radius[bubbles], initial=0.0) + 1.0e-6:
            errors.append(f"Bubble mean radius outside marker bounds: {frame_path.name}")

    combined_coverage = 1.0 - (1.0 - foam_coverage) * (1.0 - bubble_coverage)
    if source_sample_index < int(state_audit["impact_sample"]):
        pre_contact_nonzero_cells += int(np.count_nonzero(combined_coverage))
    foam_ids = marker_ids[foam]
    if previous_coverage is not None:
        union_weight = float(np.sum(np.maximum(previous_coverage, combined_coverage), dtype=np.float64))
        turnover = (
            float(np.sum(np.abs(combined_coverage - previous_coverage), dtype=np.float64)) / union_weight
            if union_weight > 0.0
            else 0.0
        )
        coverage_turnovers.append(turnover)
        persistent = len(np.intersect1d(previous_foam_ids, foam_ids, assume_unique=True))
        persistence = persistent / max(len(previous_foam_ids), len(foam_ids), 1)
        if persistence >= 0.90 and union_weight > 0.0:
            stable_pair_turnovers.append(turnover)
            previous_active = previous_coverage > 0.05
            current_active = combined_coverage > 0.05
            union = np.count_nonzero(previous_active | current_active)
            intersection = np.count_nonzero(previous_active & current_active)
            stable_pair_jaccards.append(intersection / union if union else 1.0)
    previous_coverage = combined_coverage
    previous_foam_ids = foam_ids

criteria = {
    "structural_checks_pass": len(errors) == 0,
    "foam_tau_integral_is_conservative": maximum_foam_integral_error < MAXIMUM_INTEGRAL_RELATIVE_ERROR,
    "bubble_tau_integral_is_conservative": maximum_bubble_integral_error < MAXIMUM_INTEGRAL_RELATIVE_ERROR,
    "foam_shoreline_retention_is_sufficient": minimum_foam_retention >= MINIMUM_SHORELINE_RETENTION,
    "bubble_shoreline_retention_is_sufficient": minimum_bubble_retention >= MINIMUM_SHORELINE_RETENTION,
    "dry_cells_have_zero_optical_depth": maximum_dry_foam_tau == 0.0 and maximum_dry_bubble_tau == 0.0,
    "coverage_matches_optical_depth": maximum_coverage_formula_error <= 1.0e-7,
    "orientation_tensor_is_positive_semidefinite": minimum_orientation_eigenvalue >= -1.0e-5,
    "orientation_tensor_has_unit_trace": maximum_orientation_trace_error <= 1.0e-5,
    "contact_gates_emission": pre_contact_nonzero_cells == 0,
    "stable_pairs_do_not_flicker": (
        not stable_pair_turnovers
        or (
            max(stable_pair_turnovers) <= MAXIMUM_STABLE_PAIR_TURNOVER
            and min(stable_pair_jaccards) >= MINIMUM_STABLE_PAIR_JACCARD
        )
    ),
}
criteria = {name: bool(value) for name, value in criteria.items()}
metrics = {
    "samples": len(sample_indices),
    "atlas_shape": list(shape),
    "total_bytes": total_bytes,
    "maximum_foam_integral_relative_error": maximum_foam_integral_error,
    "maximum_bubble_integral_relative_error": maximum_bubble_integral_error,
    "minimum_foam_shoreline_retention": minimum_foam_retention,
    "minimum_bubble_shoreline_retention": minimum_bubble_retention,
    "maximum_dry_foam_tau": maximum_dry_foam_tau,
    "maximum_dry_bubble_tau": maximum_dry_bubble_tau,
    "maximum_coverage_formula_error": maximum_coverage_formula_error,
    "minimum_orientation_eigenvalue": minimum_orientation_eigenvalue,
    "maximum_orientation_trace_error": maximum_orientation_trace_error,
    "pre_contact_nonzero_cells": pre_contact_nonzero_cells,
    "coverage_turnover_maximum": max(coverage_turnovers, default=0.0),
    "stable_pair_count": len(stable_pair_turnovers),
    "stable_pair_turnover_maximum": max(stable_pair_turnovers, default=0.0),
    "stable_pair_jaccard_minimum": min(stable_pair_jaccards, default=1.0),
}
report = {
    "schema": 1,
    "created_utc": utc_now_iso(),
    "valid": all(criteria.values()),
    "atlas_directory": str(atlas_directory),
    "manifest_sha256": sha256_file(manifest_path),
    "criteria": criteria,
    "metrics": metrics,
    "errors": errors,
    "thresholds": {
        "maximum_integral_relative_error": MAXIMUM_INTEGRAL_RELATIVE_ERROR,
        "minimum_shoreline_retention": MINIMUM_SHORELINE_RETENTION,
        "maximum_stable_pair_turnover": MAXIMUM_STABLE_PAIR_TURNOVER,
        "minimum_stable_pair_jaccard": MINIMUM_STABLE_PAIR_JACCARD,
    },
}
atomic_json(atlas_directory / "audit_report.json", report)
print(json.dumps(report, indent=2))
if not report["valid"]:
    raise SystemExit(1)
