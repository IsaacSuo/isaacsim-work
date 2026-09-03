"""Independently audit unified whitewater marker sequences and phase ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.state_machine import MARKER_DTYPE, WhitewaterState, sphere_volume


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("state_directory", type=Path)
parser.add_argument("--output", type=Path)
parser.add_argument("--maximum-frame-displacement", type=float, default=0.08)
parser.add_argument("--maximum-emission-relative-error", type=float, default=0.05)
args = parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sample_height_nearest(points, x_values, z_values, values):
    dx = float(np.median(np.diff(x_values)))
    dz = float(np.median(np.diff(z_values)))
    ix = np.rint((points[:, 0] - x_values[0]) / dx).astype(np.int64)
    iz = np.rint((points[:, 2] - z_values[0]) / dz).astype(np.int64)
    valid = (ix >= 0) & (ix < len(x_values)) & (iz >= 0) & (iz < len(z_values))
    sampled = np.full(len(points), np.nan, dtype=np.float32)
    sampled[valid] = values[ix[valid], iz[valid]]
    valid &= np.isfinite(sampled)
    return sampled, valid


state_directory = args.state_directory.resolve()
manifest_path = state_directory / "manifest.json"
if not manifest_path.is_file():
    raise FileNotFoundError(manifest_path)
output_path = state_directory / "audit_report.json" if args.output is None else args.output.resolve()
if output_path.exists():
    raise FileExistsError(f"Refusing to overwrite audit: {output_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
errors = []


def require(condition, message):
    if not condition:
        errors.append(message)
    return bool(condition)


require(manifest.get("schema") == 1, "Unsupported state manifest schema")
require(manifest.get("state", {}).get("complete") is True, "State sequence is incomplete")
rows = manifest.get("samples", [])
require(
    len(rows) == manifest["state"].get("expected_samples") == manifest["state"].get("completed_samples"),
    "State sample counts do not match",
)
require(
    [row.get("source_sample_index") for row in rows] == list(range(len(rows))),
    "State sample indices are not consecutive",
)
for key in ("script", "state_module"):
    path = Path(manifest["producer"][key])
    require(path.is_file(), f"Missing producer: {path}")
    if path.is_file():
        require(
            sha256_file(path) == manifest["producer"][key + "_sha256"],
            f"Producer hash mismatch: {path}",
        )

source_directory = Path(manifest["source"]["directory"])
source_manifest_path = source_directory / "manifest.json"
source_audit_path = source_directory / "audit_report.json"
require(sha256_file(source_manifest_path) == manifest["source"]["manifest_sha256"], "Source manifest hash mismatch")
require(sha256_file(source_audit_path) == manifest["source"]["audit_sha256"], "Source audit hash mismatch")
source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
require(source_audit.get("valid") is True, "Source audit is invalid")

terrain_path = Path(manifest["terrain"]["path"])
require(sha256_file(terrain_path) == manifest["terrain"]["sha256"], "Terrain hash mismatch")
with np.load(terrain_path, allow_pickle=False) as terrain_cache:
    terrain_x = np.asarray(terrain_cache["x_values"], dtype=np.float64)
    terrain_z = np.asarray(terrain_cache["z_values"], dtype=np.float64)
    terrain_y = np.asarray(terrain_cache["terrain_y"], dtype=np.float32)

expected_keys = set(MARKER_DTYPE.names) | {
    "schema",
    "source_sample_index",
    "simulation_time",
}
expected_dtypes = {name: MARKER_DTYPE.fields[name][0] for name in MARKER_DTYPE.names}
allowed_transitions = {
    int(WhitewaterState.SPRAY): {int(WhitewaterState.SPRAY), int(WhitewaterState.SURFACE_FOAM)},
    int(WhitewaterState.SURFACE_FOAM): {int(WhitewaterState.SURFACE_FOAM)},
    int(WhitewaterState.ENTRAINED_BUBBLE): {
        int(WhitewaterState.ENTRAINED_BUBBLE),
        int(WhitewaterState.SURFACE_BUBBLE),
    },
    int(WhitewaterState.SURFACE_BUBBLE): {int(WhitewaterState.SURFACE_BUBBLE)},
}
previous = None
seen_ids = set()
disappeared_ids = set()
maximum_displacement = 0.0
minimum_sphere_clearance = float("inf")
minimum_terrain_clearance = float("inf")
maximum_shape_volume_error = 0.0
transition_counts = {}
sample_metrics = []

for row in rows:
    sample_index = int(row["source_sample_index"])
    state_path = state_directory / row["file"]
    require(state_path.is_file(), f"Missing state file: {state_path}")
    if not state_path.is_file():
        continue
    require(state_path.stat().st_size == row["bytes"], f"Byte mismatch: {state_path}")
    require(sha256_file(state_path) == row["sha256"], f"Hash mismatch: {state_path}")
    with np.load(state_path, allow_pickle=False) as cache:
        require(set(cache.files) == expected_keys, f"Unexpected keys: {state_path}")
        arrays = {name: np.asarray(cache[name]) for name in MARKER_DTYPE.names}
        cache_schema = int(cache["schema"])
        cache_index = int(cache["source_sample_index"])
        cache_time = float(cache["simulation_time"])
    count = len(arrays["id"])
    require(cache_schema == 1 and cache_index == sample_index, f"Scalar metadata mismatch: {sample_index}")
    require(abs(cache_time - float(row["simulation_time"])) <= 1.0e-12, f"Time mismatch: {sample_index}")
    for name, array in arrays.items():
        expected_shape = (count,) + expected_dtypes[name].shape
        require(array.shape == expected_shape, f"Shape mismatch: {sample_index}/{name}")
        require(array.dtype == expected_dtypes[name].base, f"Dtype mismatch: {sample_index}/{name}")
        if array.dtype.kind == "f":
            require(np.isfinite(array).all(), f"Non-finite values: {sample_index}/{name}")
    if count:
        require(np.all(np.diff(arrays["id"]) > 0), f"IDs are not strictly increasing: {sample_index}")
        require(np.all(arrays["id"] > 0), f"Zero marker ID: {sample_index}")
        require(np.all(arrays["radius"] > 0.0), f"Non-positive radius: {sample_index}")
        require(np.all(arrays["representative_weight"] > 0.0), f"Non-positive weight: {sample_index}")
        require(np.all(arrays["birth_sample"] <= sample_index), f"Future birth sample: {sample_index}")
        require(np.all(arrays["state_age"] >= 0.0), f"Negative state age: {sample_index}")
        valid_states = np.isin(
            arrays["state"],
            [
                np.uint8(WhitewaterState.SPRAY),
                np.uint8(WhitewaterState.SURFACE_FOAM),
                np.uint8(WhitewaterState.ENTRAINED_BUBBLE),
                np.uint8(WhitewaterState.SURFACE_BUBBLE),
            ],
        )
        require(valid_states.all(), f"Invalid state value: {sample_index}")
        liquid = np.isin(
            arrays["state"],
            [np.uint8(WhitewaterState.SPRAY), np.uint8(WhitewaterState.SURFACE_FOAM)],
        )
        gas = ~liquid
        expected_volume = sphere_volume(arrays["radius"]) * arrays["representative_weight"]
        require(np.allclose(arrays["liquid_volume"][liquid], expected_volume[liquid], rtol=3.0e-7, atol=0.0), f"Liquid marker volume mismatch: {sample_index}")
        require(np.all(arrays["gas_volume"][liquid] == 0.0), f"Liquid marker carries gas: {sample_index}")
        require(np.allclose(arrays["gas_volume"][gas], expected_volume[gas], rtol=3.0e-7, atol=0.0), f"Gas marker volume mismatch: {sample_index}")
        require(np.all(arrays["liquid_volume"][gas] == 0.0), f"Gas marker carries liquid: {sample_index}")
        shape_error = float(np.max(np.abs(np.prod(arrays["shape"][gas], axis=1) - 1.0))) if np.any(gas) else 0.0
        maximum_shape_volume_error = max(maximum_shape_volume_error, shape_error)

    source_path = source_directory / source_manifest["samples"][sample_index]["file"]
    with np.load(source_path, allow_pickle=False) as source_cache:
        sphere_transform = np.asarray(source_cache["sphere_transform"], dtype=np.float64)
    if count:
        sphere_center = sphere_transform[3, :3]
        sphere_radius = 0.08
        sphere_clearance = (
            np.linalg.norm(arrays["position"] - sphere_center, axis=1)
            - sphere_radius
            - arrays["radius"]
        )
        minimum_sphere_clearance = min(minimum_sphere_clearance, float(sphere_clearance.min()))
        require(np.all(sphere_clearance >= -2.0e-6), f"Impactor overlap: {sample_index}")
        terrain_height, terrain_valid = sample_height_nearest(
            arrays["position"], terrain_x, terrain_z, terrain_y
        )
        if np.any(terrain_valid):
            terrain_clearance = arrays["position"][terrain_valid, 1] - terrain_height[terrain_valid] + arrays["radius"][terrain_valid]
            minimum_terrain_clearance = min(minimum_terrain_clearance, float(terrain_clearance.min()))
            require(np.all(terrain_clearance >= -2.0e-6), f"Terrain penetration: {sample_index}")

    current_ids = set(arrays["id"].tolist())
    require(not (current_ids & disappeared_ids), f"A dead ID reappeared: {sample_index}")
    if previous is not None and count and len(previous["id"]):
        shared, old_index, new_index = np.intersect1d(
            previous["id"], arrays["id"], assume_unique=True, return_indices=True
        )
        disappeared_ids.update(set(previous["id"].tolist()) - current_ids)
        if len(shared):
            displacement = np.linalg.norm(
                arrays["position"][new_index] - previous["position"][old_index], axis=1
            )
            maximum_displacement = max(maximum_displacement, float(displacement.max()))
            require(np.all(displacement <= args.maximum_frame_displacement), f"Excessive displacement: {sample_index}")
            require(np.array_equal(arrays["gas_volume"][new_index], previous["gas_volume"][old_index]), f"Shared-ID gas volume changed: {sample_index}")
            require(np.array_equal(arrays["liquid_volume"][new_index], previous["liquid_volume"][old_index]), f"Shared-ID liquid volume changed: {sample_index}")
            for old_state, new_state in zip(previous["state"][old_index], arrays["state"][new_index]):
                edge = (int(old_state), int(new_state))
                transition_counts[str(edge)] = transition_counts.get(str(edge), 0) + 1
                require(int(new_state) in allowed_transitions[int(old_state)], f"Forbidden transition {edge}: {sample_index}")
    elif previous is not None:
        disappeared_ids.update(set(previous["id"].tolist()) - current_ids)
    seen_ids.update(current_ids)
    previous = arrays
    require(count == int(row["alive_markers"]), f"Alive count mismatch: {sample_index}")
    sample_metrics.append({"sample": sample_index, "count": count, "counts": row["counts"]})

impact_sample = manifest["state"].get("impact_sample")
require(impact_sample is not None, "No impact sample was recorded")
if impact_sample is not None:
    require(all(row["alive_markers"] == 0 for row in rows[:impact_sample]), "Pre-contact emission occurred")
    require(rows[impact_sample]["alive_markers"] > 0, "No markers emitted at contact")

ledger = manifest["ledger"]
final_liquid = float(previous["liquid_volume"].sum(dtype=np.float64)) if previous is not None else 0.0
final_gas = float(previous["gas_volume"].sum(dtype=np.float64)) if previous is not None else 0.0
liquid_balance_error = abs(
    ledger["emitted_spray_volume"]
    - ledger["returned_liquid_volume"]
    - final_liquid
)
gas_balance_error = abs(
    ledger["emitted_air_volume"] - ledger["released_air_volume"] - final_gas
)
require(liquid_balance_error <= 2.0e-10, f"Liquid ledger imbalance: {liquid_balance_error}")
require(gas_balance_error <= 2.0e-10, f"Gas ledger imbalance: {gas_balance_error}")
spray_emission_error = abs(ledger["emitted_spray_volume"] - ledger["requested_spray_volume"]) / max(ledger["requested_spray_volume"], 1.0e-30)
air_emission_error = abs(ledger["emitted_air_volume"] - ledger["requested_air_volume"]) / max(ledger["requested_air_volume"], 1.0e-30)
require(spray_emission_error <= args.maximum_emission_relative_error, "Spray emission discretization error is too large")
require(air_emission_error <= args.maximum_emission_relative_error, "Air emission discretization error is too large")

report = {
    "schema": 1,
    "valid": not errors,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "state_directory": str(state_directory),
    "manifest_sha256": sha256_file(manifest_path),
    "samples": len(rows),
    "unique_marker_ids": len(seen_ids),
    "impact_sample": impact_sample,
    "maximum_frame_displacement": maximum_displacement,
    "minimum_sphere_clearance": minimum_sphere_clearance,
    "minimum_terrain_clearance": minimum_terrain_clearance,
    "maximum_shape_volume_error": maximum_shape_volume_error,
    "transition_counts": transition_counts,
    "ledger": {
        **ledger,
        "final_active_liquid_volume": final_liquid,
        "final_active_gas_volume": final_gas,
        "liquid_balance_error": liquid_balance_error,
        "gas_balance_error": gas_balance_error,
        "spray_emission_relative_error": spray_emission_error,
        "air_emission_relative_error": air_emission_error,
    },
    "errors": errors,
}
atomic_write_json(output_path, report)
print(json.dumps(report, indent=2))
if errors:
    raise SystemExit(1)
