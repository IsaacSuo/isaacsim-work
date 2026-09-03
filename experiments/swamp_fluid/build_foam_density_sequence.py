"""Build a conservative 30 FPS foam and surface-bubble optical-depth atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.foam_density_warp import FoamDensityComputer
from whitewater.foam_model import (
    BUBBLE_TARGET_PEAK_TAU,
    FOAM_ANISOTROPY_AGE,
    FOAM_ANISOTROPY_SPEED,
    FOAM_BIRTH_FILM_THICKNESS,
    FOAM_DRAINAGE_TIME,
    FOAM_DRAINED_FILM_THICKNESS,
    FOAM_TARGET_PEAK_TAU,
    FOAM_RAFT_SIGMA_BIRTH,
    FOAM_RAFT_SIGMA_MATURE,
    FOAM_RAFT_COALESCENCE_TIME,
    GAUSSIAN_CUTOFF_SIGMA,
    MAXIMUM_FOAM_ANISOTROPY,
    surface_marker_kernels,
)


DEFAULT_STATE = Path(
    r"Y:\isaacsim_work\output\swamp_fluid_preview"
    r"\whitewater_v3_state_impact_1p5s_full"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-directory", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--start-sample", type=int, default=0)
    parser.add_argument("--end-sample", type=int, default=180)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--atlas-spacing", type=float, default=0.004)
    parser.add_argument("--atlas-margin", type=float, default=0.15)
    parser.add_argument("--wet-support-radius", type=float, default=0.006)
    parser.add_argument("--minimum-liquid-depth", type=float, default=0.002)
    parser.add_argument("--maximum-support", type=float, default=0.06)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
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
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def regular_axis(minimum, maximum, spacing, margin):
    first = math.floor((float(minimum) - margin) / spacing) * spacing
    last = math.ceil((float(maximum) + margin) / spacing) * spacing
    count = int(round((last - first) / spacing)) + 1
    return first + np.arange(count, dtype=np.float64) * spacing


def sample_height_nearest(x, z, terrain_x, terrain_z, terrain_y):
    dx = float(np.median(np.diff(terrain_x)))
    dz = float(np.median(np.diff(terrain_z)))
    ix = np.rint((x[:, None] - terrain_x[0]) / dx).astype(np.int64)
    iz = np.rint((z[None, :] - terrain_z[0]) / dz).astype(np.int64)
    valid_x = (ix >= 0) & (ix < len(terrain_x))
    valid_z = (iz >= 0) & (iz < len(terrain_z))
    clipped_x = np.clip(ix, 0, len(terrain_x) - 1)
    clipped_z = np.clip(iz, 0, len(terrain_z) - 1)
    sampled = terrain_y[clipped_x, clipped_z]
    sampled = np.asarray(sampled, dtype=np.float32)
    sampled[~(valid_x & valid_z)] = np.nan
    return sampled


def liquid_top_and_wet_mask(
    positions,
    x_values,
    z_values,
    terrain_y,
    support_radius,
    minimum_depth,
):
    spacing = float(np.median(np.diff(x_values)))
    ix = np.rint((positions[:, 0] - x_values[0]) / spacing).astype(np.int64)
    iz = np.rint((positions[:, 2] - z_values[0]) / spacing).astype(np.int64)
    inside = (
        (ix >= 0)
        & (ix < len(x_values))
        & (iz >= 0)
        & (iz < len(z_values))
    )
    top = np.full((len(x_values), len(z_values)), -np.inf, dtype=np.float32)
    np.maximum.at(top, (ix[inside], iz[inside]), positions[inside, 1])
    cells = int(math.ceil(support_radius / spacing))
    expanded = top.copy()
    for ox in range(-cells, cells + 1):
        for oz in range(-cells, cells + 1):
            if math.hypot(ox * spacing, oz * spacing) > support_radius + 1.0e-12:
                continue
            sx = slice(max(0, -ox), min(len(x_values), len(x_values) - ox))
            sz = slice(max(0, -oz), min(len(z_values), len(z_values) - oz))
            tx = slice(max(0, ox), min(len(x_values), len(x_values) + ox))
            tz = slice(max(0, oz), min(len(z_values), len(z_values) + oz))
            expanded[tx, tz] = np.maximum(expanded[tx, tz], top[sx, sz])
    expanded[~np.isfinite(expanded)] = np.nan
    wet = (
        np.isfinite(expanded)
        & np.isfinite(terrain_y)
        & (expanded >= terrain_y + minimum_depth)
    )
    return expanded, wet


def safe_weighted_mean(weighted, tau):
    output = np.zeros_like(tau, dtype=np.float32)
    np.divide(weighted, tau, out=output, where=tau > 1.0e-12)
    return output


args = parse_args()
if min(
    args.sample_stride,
    args.atlas_spacing,
    args.atlas_margin,
    args.wet_support_radius,
    args.minimum_liquid_depth,
    args.maximum_support,
) <= 0:
    raise ValueError("All sampling and spatial parameters must be positive")
if args.start_sample < 0 or args.end_sample < args.start_sample:
    raise ValueError("Invalid sample range")
if args.start_sample % args.sample_stride or args.end_sample % args.sample_stride:
    raise ValueError("Start and end samples must align to sample-stride")

state_directory = args.state_directory.resolve()
output_directory = args.output_directory.resolve()
state_manifest_path = state_directory / "manifest.json"
state_audit_path = state_directory / "audit_report.json"
state_manifest = load_json(state_manifest_path)
state_audit = load_json(state_audit_path)
if state_audit.get("valid") is not True:
    raise ValueError("State cache has not passed its independent audit")
if state_audit.get("manifest_sha256") != sha256_file(state_manifest_path):
    raise ValueError("State audit does not reference the current manifest")
if state_manifest.get("state", {}).get("complete") is not True:
    raise ValueError("State cache is incomplete")

source_directory = Path(state_manifest["source"]["directory"])
source_manifest_path = source_directory / "manifest.json"
source_audit_path = source_directory / "audit_report.json"
source_manifest = load_json(source_manifest_path)
source_audit = load_json(source_audit_path)
if source_audit.get("valid") is not True:
    raise ValueError("Source cache has not passed its independent audit")
if source_audit.get("manifest_sha256") != sha256_file(source_manifest_path):
    raise ValueError("Source audit does not reference the current manifest")
if args.end_sample >= len(state_manifest["samples"]):
    raise ValueError("Requested range exceeds the state cache")

sample_indices = list(
    range(args.start_sample, args.end_sample + 1, args.sample_stride)
)
state_by_index = {
    int(item["source_sample_index"]): item for item in state_manifest["samples"]
}
source_by_index = {
    int(item["sample_index"]): item for item in source_manifest["samples"]
}
if not all(index in state_by_index and index in source_by_index for index in sample_indices):
    raise ValueError("State/source caches do not cover every requested sample")

first_source_path = source_directory / source_by_index[0]["file"]
with np.load(first_source_path, allow_pickle=False) as first_source:
    first_positions = np.asarray(first_source["positions"], dtype=np.float32)
x_values = regular_axis(
    first_positions[:, 0].min(),
    first_positions[:, 0].max(),
    args.atlas_spacing,
    args.atlas_margin,
)
z_values = regular_axis(
    first_positions[:, 2].min(),
    first_positions[:, 2].max(),
    args.atlas_spacing,
    args.atlas_margin,
)

terrain_path = Path(state_manifest["terrain"]["path"])
if sha256_file(terrain_path) != state_manifest["terrain"]["sha256"]:
    raise ValueError("Terrain heightfield hash differs from the state manifest")
with np.load(terrain_path, allow_pickle=False) as terrain:
    terrain_x = np.asarray(terrain["x_values"], dtype=np.float64)
    terrain_z = np.asarray(terrain["z_values"], dtype=np.float64)
    full_terrain_y = np.asarray(terrain["terrain_y"], dtype=np.float32)
atlas_terrain_y = sample_height_nearest(
    x_values, z_values, terrain_x, terrain_z, full_terrain_y
)

configuration = {
    "start_sample": args.start_sample,
    "end_sample": args.end_sample,
    "sample_stride": args.sample_stride,
    "atlas_spacing": args.atlas_spacing,
    "atlas_margin": args.atlas_margin,
    "atlas_shape": [len(x_values), len(z_values)],
    "wet_support_radius": args.wet_support_radius,
    "minimum_liquid_depth": args.minimum_liquid_depth,
    "maximum_support": args.maximum_support,
    "minimum_kernel_sigma": args.atlas_spacing,
}
model = {
    "foam_birth_film_thickness": FOAM_BIRTH_FILM_THICKNESS,
    "foam_drained_film_thickness": FOAM_DRAINED_FILM_THICKNESS,
    "foam_drainage_time": FOAM_DRAINAGE_TIME,
    "foam_target_peak_tau": FOAM_TARGET_PEAK_TAU,
    "bubble_target_peak_tau": BUBBLE_TARGET_PEAK_TAU,
    "maximum_foam_anisotropy": MAXIMUM_FOAM_ANISOTROPY,
    "foam_anisotropy_speed": FOAM_ANISOTROPY_SPEED,
    "foam_anisotropy_age": FOAM_ANISOTROPY_AGE,
    "gaussian_cutoff_sigma": GAUSSIAN_CUTOFF_SIGMA,
    "foam_raft_sigma_birth": FOAM_RAFT_SIGMA_BIRTH,
    "foam_raft_sigma_mature": FOAM_RAFT_SIGMA_MATURE,
    "foam_raft_coalescence_time": FOAM_RAFT_COALESCENCE_TIME,
}
manifest_path = output_directory / "manifest.json"
coordinates_path = output_directory / "atlas_coordinates.npz"
source_links = {
    "state_directory": str(state_directory),
    "state_manifest_sha256": sha256_file(state_manifest_path),
    "state_audit_sha256": sha256_file(state_audit_path),
    "source_directory": str(source_directory),
    "source_manifest_sha256": sha256_file(source_manifest_path),
    "source_audit_sha256": sha256_file(source_audit_path),
    "terrain_path": str(terrain_path),
    "terrain_sha256": sha256_file(terrain_path),
}

if output_directory.exists() and any(output_directory.iterdir()):
    if not args.resume:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_directory}")
    manifest = load_json(manifest_path)
    if manifest["configuration"] != configuration or manifest["source"] != source_links:
        raise ValueError("Resume configuration/source does not match the existing cache")
    if sha256_file(coordinates_path) != manifest["coordinates"]["sha256"]:
        raise ValueError("Existing atlas coordinates failed hash verification")
else:
    output_directory.mkdir(parents=True, exist_ok=True)
    atomic_npz(
        coordinates_path,
        schema=np.int32(1),
        x_values=x_values,
        z_values=z_values,
        terrain_y=atlas_terrain_y,
    )
    manifest = {
        "schema": 1,
        "created_utc": utc_now_iso(),
        "producer": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "density_module": str(
                (Path(__file__).parent / "whitewater" / "foam_density_warp.py").resolve()
            ),
            "density_module_sha256": sha256_file(
                Path(__file__).parent / "whitewater" / "foam_density_warp.py"
            ),
            "model_module": str(
                (Path(__file__).parent / "whitewater" / "foam_model.py").resolve()
            ),
            "model_module_sha256": sha256_file(
                Path(__file__).parent / "whitewater" / "foam_model.py"
            ),
        },
        "configuration": configuration,
        "model": model,
        "source": source_links,
        "coordinates": {
            "file": coordinates_path.name,
            "sha256": sha256_file(coordinates_path),
            "bytes": coordinates_path.stat().st_size,
        },
        "sample_indices": sample_indices,
        "samples": [],
        "state": {"complete": False, "completed_samples": 0},
    }
    atomic_json(manifest_path, manifest)

existing = {int(item["source_sample_index"]): item for item in manifest["samples"]}
computer = FoamDensityComputer(
    x_values,
    z_values,
    maximum_markers=max(item["alive_markers"] for item in state_manifest["samples"]),
    maximum_support=args.maximum_support,
    device=args.device,
    grid_dimension=128,
)
cell_area = args.atlas_spacing**2

for frame_number, source_sample_index in enumerate(sample_indices):
    if source_sample_index in existing:
        item = existing[source_sample_index]
        cached_path = output_directory / item["file"]
        if cached_path.is_file() and sha256_file(cached_path) == item["sha256"]:
            print(f"Verified existing {cached_path.name}")
            continue
        raise ValueError(f"Existing frame failed verification: {cached_path}")

    state_item = state_by_index[source_sample_index]
    source_item = source_by_index[source_sample_index]
    state_path = state_directory / state_item["file"]
    source_path = source_directory / source_item["file"]
    if sha256_file(state_path) != state_item["sha256"]:
        raise ValueError(f"State frame hash mismatch: {state_path}")
    if sha256_file(source_path) != source_item["sha256"]:
        raise ValueError(f"Source frame hash mismatch: {source_path}")
    with np.load(state_path, allow_pickle=False) as state_cache:
        snapshot = {name: np.asarray(state_cache[name]) for name in state_cache.files}
    with np.load(source_path, allow_pickle=False) as source_cache:
        liquid_positions = np.asarray(source_cache["positions"], dtype=np.float32)

    kernels = surface_marker_kernels(snapshot, args.atlas_spacing)
    if len(kernels["positions"]):
        required_support = 3.0 * float(np.max(kernels["sigma_major"]))
        if required_support > args.maximum_support * (1.0 + 1.0e-6):
            raise ValueError(
                f"Sample {source_sample_index} needs support {required_support:.6f} m"
            )
    raw = computer.compute(
        kernels["positions"],
        kernels["directions"],
        kernels["sigma_major"],
        kernels["sigma_minor"],
        kernels["peak_optical_depth"],
        kernels["age"],
        kernels["radius"],
        kernels["kind"],
    )
    liquid_top_y, wet_mask = liquid_top_and_wet_mask(
        liquid_positions,
        x_values,
        z_values,
        atlas_terrain_y,
        args.wet_support_radius,
        args.minimum_liquid_depth,
    )
    wet = wet_mask.astype(np.float32)
    foam_tau = raw["foam_tau"] * wet
    bubble_tau = raw["bubble_tau"] * wet
    foam_coverage = -np.expm1(-foam_tau).astype(np.float32)
    bubble_coverage = -np.expm1(-bubble_tau).astype(np.float32)
    foam_mean_age = safe_weighted_mean(raw["foam_age_weighted"] * wet, foam_tau)
    foam_orientation_xx = safe_weighted_mean(raw["foam_orientation_xx"] * wet, foam_tau)
    foam_orientation_xz = safe_weighted_mean(raw["foam_orientation_xz"] * wet, foam_tau)
    foam_orientation_zz = safe_weighted_mean(raw["foam_orientation_zz"] * wet, foam_tau)
    bubble_mean_radius = safe_weighted_mean(
        raw["bubble_radius_weighted"] * wet, bubble_tau
    )

    output_path = output_directory / f"foam_atlas_{frame_number:06d}.npz"
    atomic_npz(
        output_path,
        schema=np.int32(1),
        render_frame=np.int32(frame_number),
        source_sample_index=np.int32(source_sample_index),
        simulation_time=np.float64(snapshot["simulation_time"]),
        foam_tau=foam_tau.astype(np.float32),
        foam_coverage=foam_coverage,
        foam_mean_age=foam_mean_age,
        foam_orientation_xx=foam_orientation_xx,
        foam_orientation_xz=foam_orientation_xz,
        foam_orientation_zz=foam_orientation_zz,
        bubble_tau=bubble_tau.astype(np.float32),
        bubble_coverage=bubble_coverage,
        bubble_mean_radius=bubble_mean_radius,
        wet_mask=wet_mask.astype(np.uint8),
        liquid_top_y=liquid_top_y.astype(np.float32),
    )
    raw_foam_integral = float(np.sum(raw["foam_tau"], dtype=np.float64) * cell_area)
    raw_bubble_integral = float(np.sum(raw["bubble_tau"], dtype=np.float64) * cell_area)
    clipped_foam_integral = float(np.sum(foam_tau, dtype=np.float64) * cell_area)
    clipped_bubble_integral = float(np.sum(bubble_tau, dtype=np.float64) * cell_area)
    item = {
        "render_frame": frame_number,
        "source_sample_index": source_sample_index,
        "simulation_time": float(snapshot["simulation_time"]),
        "file": output_path.name,
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "foam_markers": int(np.count_nonzero(kernels["kind"] == 0)),
        "bubble_markers": int(np.count_nonzero(kernels["kind"] == 1)),
        "expected_foam_area": kernels["foam_area"],
        "expected_bubble_area": kernels["bubble_area"],
        "raw_foam_tau_integral": raw_foam_integral,
        "raw_bubble_tau_integral": raw_bubble_integral,
        "clipped_foam_tau_integral": clipped_foam_integral,
        "clipped_bubble_tau_integral": clipped_bubble_integral,
        "foam_shoreline_retention": (
            clipped_foam_integral / raw_foam_integral if raw_foam_integral else 1.0
        ),
        "bubble_shoreline_retention": (
            clipped_bubble_integral / raw_bubble_integral if raw_bubble_integral else 1.0
        ),
        "wet_cells": int(np.count_nonzero(wet_mask)),
        "maximum_foam_tau": float(np.max(foam_tau, initial=0.0)),
        "maximum_bubble_tau": float(np.max(bubble_tau, initial=0.0)),
    }
    manifest["samples"].append(item)
    manifest["samples"].sort(key=lambda value: value["source_sample_index"])
    manifest["state"]["completed_samples"] = len(manifest["samples"])
    atomic_json(manifest_path, manifest)
    print(
        f"Built {output_path.name}: sample={source_sample_index} "
        f"foam={item['foam_markers']} bubbles={item['bubble_markers']} "
        f"retention=({item['foam_shoreline_retention']:.3f}, "
        f"{item['bubble_shoreline_retention']:.3f})"
    )

manifest["state"] = {
    "complete": len(manifest["samples"]) == len(sample_indices),
    "completed_samples": len(manifest["samples"]),
    "expected_samples": len(sample_indices),
    "completed_utc": utc_now_iso(),
}
atomic_json(manifest_path, manifest)
print(json.dumps(manifest["state"], indent=2))
