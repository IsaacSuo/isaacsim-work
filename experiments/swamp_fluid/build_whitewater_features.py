"""Build full-particle whitewater feature fields from an audited PhysX cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.features_warp import WarpFeatureComputer, wp


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source_directory", type=Path)
parser.add_argument("output_directory", type=Path)
parser.add_argument("--terrain-heightfield", type=Path, required=True)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--grid-dimension", type=int, default=128)
parser.add_argument("--support-radius-multiplier", type=float, default=2.5)
parser.add_argument("--start-sample", type=int, default=0)
parser.add_argument("--end-sample", type=int)
parser.add_argument("--sample-stride", type=int, default=1)
parser.add_argument("--density-deficiency-start", type=float, default=0.92)
parser.add_argument("--density-deficiency-width", type=float, default=0.35)
parser.add_argument("--centroid-offset-start", type=float, default=0.02)
parser.add_argument("--centroid-offset-width", type=float, default=0.18)
parser.add_argument("--solid-clearance-start", type=float, default=1.25)
parser.add_argument("--solid-clearance-width", type=float, default=0.75)
args = parser.parse_args()

if args.grid_dimension < 16:
    raise ValueError("grid-dimension must be at least 16")
if not 1.5 <= args.support_radius_multiplier <= 4.0:
    raise ValueError("support-radius-multiplier must stay within 1.5..4.0")
if args.sample_stride < 1:
    raise ValueError("sample-stride must be at least 1")
if args.start_sample < 0:
    raise ValueError("start-sample must be non-negative")
if args.density_deficiency_width <= 0.0:
    raise ValueError("density-deficiency-width must be positive")
if args.centroid_offset_width <= 0.0:
    raise ValueError("centroid-offset-width must be positive")
if args.solid_clearance_width <= 0.0:
    raise ValueError("solid-clearance-width must be positive")


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


def atomic_write_npz(path, **arrays):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def smoothstep01(value):
    clamped = np.clip(value, 0.0, 1.0)
    return clamped * clamped * (3.0 - 2.0 * clamped)


def quantiles(values):
    if not len(values):
        return None
    return {
        str(q): float(np.quantile(values, q))
        for q in (0.0, 0.5, 0.9, 0.99, 1.0)
    }


def load_source_sample(sample_path, particle_count):
    with np.load(sample_path, allow_pickle=False) as cache:
        positions = np.ascontiguousarray(cache["positions"], dtype=np.float32)
        velocities = np.ascontiguousarray(cache["velocities"], dtype=np.float32)
        sphere_transform = np.ascontiguousarray(
            cache["sphere_transform"], dtype=np.float64
        )
        sample_index = int(cache["sample_index"])
        physics_step = int(cache["physics_step"])
        simulation_time = float(cache["simulation_time"])
    expected_shape = (particle_count, 3)
    if positions.shape != expected_shape or velocities.shape != expected_shape:
        raise ValueError(
            f"Source shape mismatch in {sample_path}: "
            f"{positions.shape}, {velocities.shape}, expected {expected_shape}"
        )
    return {
        "sample_index": sample_index,
        "physics_step": physics_step,
        "simulation_time": simulation_time,
        "positions": positions,
        "velocities": velocities,
        "sphere_transform": sphere_transform,
    }


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


source_directory = args.source_directory.resolve()
output_directory = args.output_directory.resolve()
terrain_path = args.terrain_heightfield.resolve()
source_manifest_path = source_directory / "manifest.json"
source_audit_path = source_directory / "audit_report.json"
run_report_path = source_directory.parent / "run_complete.json"
for required in (
    source_manifest_path,
    source_audit_path,
    run_report_path,
    terrain_path,
):
    if not required.is_file():
        raise FileNotFoundError(required)
source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
if source_manifest.get("schema") != 1:
    raise ValueError("Unsupported whitewater source schema")
if source_manifest["state"].get("complete") is not True:
    raise ValueError("Whitewater source manifest is incomplete")
if source_audit.get("valid") is not True:
    raise ValueError("Whitewater source audit is not valid")
if run_report.get("valid") is not True:
    raise ValueError("Parent PhysX run is not valid")
if source_audit["manifest_sha256"] != sha256_file(source_manifest_path):
    raise ValueError("Source audit does not reference the current manifest")

if output_directory.exists() and any(output_directory.iterdir()):
    raise FileExistsError(
        f"Refusing to overwrite non-empty feature directory: {output_directory}"
    )
output_directory.mkdir(parents=True, exist_ok=True)
feature_manifest_path = output_directory / "manifest.json"

particle_count = int(source_manifest["particle_count"])
source_rows = source_manifest["samples"]
last_sample = len(source_rows) - 1
end_sample = last_sample if args.end_sample is None else args.end_sample
if not 0 <= args.start_sample <= end_sample <= last_sample:
    raise ValueError(
        f"Requested sample range {args.start_sample}..{end_sample} "
        f"is outside 0..{last_sample}"
    )
sample_indices = list(
    range(args.start_sample, end_sample + 1, args.sample_stride)
)
if not sample_indices or sample_indices[0] != 0:
    raise ValueError(
        "Feature builds must include source sample 0 to calibrate bulk density"
    )

particle_spacing = float(run_report["particle_spacing"])
support_radius = args.support_radius_multiplier * particle_spacing
sphere_radius = float(run_report["impactor"]["radius"])
water_level = float(run_report["water_level"]["simulated"])

with np.load(terrain_path, allow_pickle=False) as terrain_cache:
    terrain_x = np.asarray(terrain_cache["x_values"], dtype=np.float64)
    terrain_z = np.asarray(terrain_cache["z_values"], dtype=np.float64)
    terrain_y = np.asarray(terrain_cache["terrain_y"], dtype=np.float32)
if terrain_y.shape != (len(terrain_x), len(terrain_z)):
    raise ValueError("Terrain heightfield shape does not match coordinate arrays")

feature_manifest = {
    "schema": 1,
    "producer": {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "kernel_module": str(
            (Path(__file__).parent / "whitewater" / "features_warp.py").resolve()
        ),
        "kernel_module_sha256": sha256_file(
            Path(__file__).parent / "whitewater" / "features_warp.py"
        ),
        "warp_version": wp.__version__,
        "device": str(wp.get_device(args.device)),
    },
    "created_utc": utc_now_iso(),
    "source": {
        "directory": str(source_directory),
        "manifest": str(source_manifest_path),
        "manifest_sha256": sha256_file(source_manifest_path),
        "audit": str(source_audit_path),
        "audit_sha256": sha256_file(source_audit_path),
        "run_report": str(run_report_path),
        "run_report_sha256": sha256_file(run_report_path),
    },
    "terrain": {
        "heightfield": str(terrain_path),
        "heightfield_sha256": sha256_file(terrain_path),
        "sampling": "nearest_cell_matching_PhysX_containment_audit",
    },
    "configuration": {
        "particle_spacing": particle_spacing,
        "support_radius": support_radius,
        "support_radius_multiplier": args.support_radius_multiplier,
        "grid_dimension": args.grid_dimension,
        "density_deficiency_start": args.density_deficiency_start,
        "density_deficiency_width": args.density_deficiency_width,
        "centroid_offset_start": args.centroid_offset_start,
        "centroid_offset_width": args.centroid_offset_width,
        "solid_clearance_start_support_radii": args.solid_clearance_start,
        "solid_clearance_width_support_radii": args.solid_clearance_width,
        "sphere_radius": sphere_radius,
        "water_level": water_level,
        "potential_normalization": None,
    },
    "potential_definitions": {
        "kinetic_energy": "0.5 * dot(velocity, velocity)",
        "trapped_air_potential": (
            "surface_confidence * gradient_quality * "
            "max(-velocity_divergence, 0) * velocity_dispersion"
        ),
        "wave_crest_potential": (
            "surface_confidence * curvature_quality * "
            "max(normal_velocity, 0) * min(abs(curvature) * spacing, 2)"
        ),
    },
    "sample_indices": sample_indices,
    "density_reference": None,
    "arrays": {
        "neighbor_count": {"dtype": "int16", "shape": [particle_count]},
        "number_density": {"dtype": "float32", "shape": [particle_count]},
        "centroid_offset": {"dtype": "float32", "shape": [particle_count, 3]},
        "surface_normal": {"dtype": "float32", "shape": [particle_count, 3]},
        "surface_confidence": {"dtype": "float32", "shape": [particle_count]},
        "solid_clearance": {"dtype": "float32", "shape": [particle_count]},
        "gradient_quality": {"dtype": "float32", "shape": [particle_count]},
        "velocity_divergence": {"dtype": "float32", "shape": [particle_count]},
        "vorticity": {"dtype": "float32", "shape": [particle_count, 3]},
        "strain_rate": {"dtype": "float32", "shape": [particle_count]},
        "velocity_dispersion": {"dtype": "float32", "shape": [particle_count]},
        "surface_curvature": {"dtype": "float32", "shape": [particle_count]},
        "curvature_quality": {"dtype": "float32", "shape": [particle_count]},
        "normal_velocity": {"dtype": "float32", "shape": [particle_count]},
        "kinetic_energy": {"dtype": "float32", "shape": [particle_count]},
        "trapped_air_potential": {"dtype": "float32", "shape": [particle_count]},
        "wave_crest_potential": {"dtype": "float32", "shape": [particle_count]},
    },
    "state": {
        "complete": False,
        "completed_samples": 0,
        "expected_samples": len(sample_indices),
        "updated_utc": utc_now_iso(),
    },
    "samples": [],
}
atomic_write_json(feature_manifest_path, feature_manifest)

computer = WarpFeatureComputer(
    particle_count=particle_count,
    radius=support_radius,
    device=args.device,
    grid_dimension=args.grid_dimension,
)
density_reference = None

for item_index, source_index in enumerate(sample_indices):
    started = time.perf_counter()
    source_row = source_rows[source_index]
    sample_path = source_directory / source_row["file"]
    source = load_source_sample(sample_path, particle_count)
    if source["sample_index"] != source_index:
        raise ValueError("Source sample index does not match manifest")

    local = computer.compute_local(source["positions"], source["velocities"])
    kernel_sum = np.asarray(local["kernel_sum"], dtype=np.float32)
    if density_reference is None:
        density_reference = float(np.quantile(kernel_sum, 0.875))
        if not math.isfinite(density_reference) or density_reference <= 0.0:
            raise RuntimeError("Could not calibrate a positive bulk density reference")
        feature_manifest["density_reference"] = density_reference

    number_density = kernel_sum / density_reference
    centroid_magnitude = np.linalg.norm(local["centroid_offset"], axis=1)
    density_score = np.clip(
        (
            args.density_deficiency_start
            - number_density
        )
        / args.density_deficiency_width,
        0.0,
        1.0,
    )
    centroid_score = np.clip(
        (
            centroid_magnitude / support_radius
            - args.centroid_offset_start
        )
        / args.centroid_offset_width,
        0.0,
        1.0,
    )
    raw_surface_confidence = np.maximum(density_score, centroid_score)

    sampled_terrain, terrain_valid = sample_terrain_nearest(
        source["positions"], terrain_x, terrain_z, terrain_y
    )
    invalid_terrain_particles = int(np.count_nonzero(~terrain_valid))
    if invalid_terrain_particles:
        raise RuntimeError(
            f"{invalid_terrain_particles} particles lack terrain height coverage"
        )
    terrain_clearance = source["positions"][:, 1] - sampled_terrain
    sphere_center = source["sphere_transform"][3, :3].astype(np.float32)
    sphere_clearance = (
        np.linalg.norm(source["positions"] - sphere_center, axis=1)
        - sphere_radius
    )
    solid_clearance = np.minimum(terrain_clearance, sphere_clearance)
    solid_gate = smoothstep01(
        (
            solid_clearance / support_radius
            - args.solid_clearance_start
        )
        / args.solid_clearance_width
    )
    normal_length = np.linalg.norm(local["surface_normal"], axis=1)
    surface_confidence = (
        raw_surface_confidence
        * solid_gate
        * (normal_length >= 0.5).astype(np.float32)
    ).astype(np.float32)

    curvature_result = computer.compute_curvature(
        local["surface_normal"], surface_confidence
    )
    surface_curvature = np.asarray(
        curvature_result["curvature"], dtype=np.float32
    )
    velocities = source["velocities"]
    speed = np.linalg.norm(velocities, axis=1)
    normal_velocity = np.sum(
        velocities * local["surface_normal"], axis=1
    ).astype(np.float32)
    kinetic_energy = (0.5 * speed * speed).astype(np.float32)
    convergence = np.maximum(-local["velocity_divergence"], 0.0)
    trapped_air_potential = (
        surface_confidence
        * local["gradient_quality"]
        * convergence
        * local["velocity_dispersion"]
    ).astype(np.float32)
    curvature_number = np.minimum(
        np.abs(surface_curvature) * particle_spacing, 2.0
    )
    wave_crest_potential = (
        surface_confidence
        * curvature_result["curvature_quality"]
        * np.maximum(normal_velocity, 0.0)
        * curvature_number
    ).astype(np.float32)

    if int(local["neighbor_count"].max()) > np.iinfo(np.int16).max:
        raise RuntimeError("Neighbor count exceeds int16 output representation")
    output_path = output_directory / f"features_{source_index:06d}.npz"
    atomic_write_npz(
        output_path,
        schema=np.asarray(1, dtype="<i4"),
        source_sample_index=np.asarray(source_index, dtype="<i4"),
        physics_step=np.asarray(source["physics_step"], dtype="<i8"),
        simulation_time=np.asarray(source["simulation_time"], dtype="<f8"),
        neighbor_count=np.asarray(local["neighbor_count"], dtype="<i2"),
        number_density=np.asarray(number_density, dtype="<f4"),
        centroid_offset=np.asarray(local["centroid_offset"], dtype="<f4"),
        surface_normal=np.asarray(local["surface_normal"], dtype="<f4"),
        surface_confidence=np.asarray(surface_confidence, dtype="<f4"),
        solid_clearance=np.asarray(solid_clearance, dtype="<f4"),
        gradient_quality=np.asarray(local["gradient_quality"], dtype="<f4"),
        velocity_divergence=np.asarray(
            local["velocity_divergence"], dtype="<f4"
        ),
        vorticity=np.asarray(local["vorticity"], dtype="<f4"),
        strain_rate=np.asarray(local["strain_rate"], dtype="<f4"),
        velocity_dispersion=np.asarray(
            local["velocity_dispersion"], dtype="<f4"
        ),
        surface_curvature=np.asarray(surface_curvature, dtype="<f4"),
        curvature_quality=np.asarray(
            curvature_result["curvature_quality"], dtype="<f4"
        ),
        normal_velocity=np.asarray(normal_velocity, dtype="<f4"),
        kinetic_energy=np.asarray(kinetic_energy, dtype="<f4"),
        trapped_air_potential=np.asarray(
            trapped_air_potential, dtype="<f4"
        ),
        wave_crest_potential=np.asarray(
            wave_crest_potential, dtype="<f4"
        ),
    )
    active_surface = surface_confidence >= 0.1
    elapsed = time.perf_counter() - started
    summary = {
        "source_sample_index": source_index,
        "physics_step": source["physics_step"],
        "simulation_time": source["simulation_time"],
        "file": output_path.name,
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "elapsed_seconds": elapsed,
        "neighbor_count_quantiles": quantiles(local["neighbor_count"]),
        "number_density_quantiles": quantiles(number_density),
        "surface_particles_0p1": int(np.count_nonzero(active_surface)),
        "surface_particles_0p5": int(
            np.count_nonzero(surface_confidence >= 0.5)
        ),
        "surface_confidence_quantiles": quantiles(surface_confidence),
        "solid_clearance_minimum": float(solid_clearance.min()),
        "kinetic_energy_quantiles": quantiles(kinetic_energy),
        "trapped_air_active_quantiles": quantiles(
            trapped_air_potential[active_surface]
        ),
        "wave_crest_active_quantiles": quantiles(
            wave_crest_potential[active_surface]
        ),
    }
    feature_manifest["samples"].append(summary)
    feature_manifest["state"].update(
        {
            "completed_samples": item_index + 1,
            "updated_utc": utc_now_iso(),
        }
    )
    atomic_write_json(feature_manifest_path, feature_manifest)
    print(
        "[whitewater-features] "
        f"sample={source_index:04d} item={item_index + 1}/{len(sample_indices)} "
        f"surface={summary['surface_particles_0p1']} "
        f"elapsed={elapsed:.3f}s",
        flush=True,
    )

feature_manifest["state"].update(
    {
        "complete": True,
        "completed_samples": len(sample_indices),
        "completed_utc": utc_now_iso(),
        "updated_utc": utc_now_iso(),
    }
)
atomic_write_json(feature_manifest_path, feature_manifest)
print(
    json.dumps(
        {
            "valid": True,
            "output_directory": str(output_directory),
            "samples": len(sample_indices),
            "density_reference": density_reference,
        },
        indent=2,
    )
)
