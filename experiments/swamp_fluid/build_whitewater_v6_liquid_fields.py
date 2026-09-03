"""Build audited v6 three-dimensional liquid fields from a PhysX source cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.liquid_fields import (
    GridSpec,
    aligned_grid_spec,
    material_acceleration,
    reconstruct_liquid_fields,
    terrain_collision_sdf,
    union_collision_sdf,
)
from whitewater.collision_fields import build_collider_fields, resolve_motion_matrix
from whitewater.domain_partition import (
    load_domain_partition_contract,
    load_particle_classification,
)
from whitewater.scene_contract import SceneContract, legacy_sphere_impact_contract
from whitewater.terrain_fields import open_terrain_collision_sdf


SCHEMA = 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--terrain-heightfield",
        type=Path,
        help=(
            "Legacy schema-1 heightfield adapter. Schema-2 scene contracts carry "
            "their audited open terrain asset directly and must omit this option."
        ),
    )
    parser.add_argument("--run-report", type=Path)
    parser.add_argument(
        "--scene-contract",
        type=Path,
        help=(
            "Versioned scene-independent collider contract. If omitted, the "
            "historical run-report sphere is exposed through a strict legacy adapter."
        ),
    )
    parser.add_argument("--start-sample", type=int, default=0)
    parser.add_argument("--end-sample", type=int)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--voxel-size", type=float, default=0.016)
    parser.add_argument("--padding", type=float, default=0.048)
    parser.add_argument("--iso-fraction", type=float, default=0.30)
    parser.add_argument("--velocity-smoothing-sigma", type=float, default=0.85)
    parser.add_argument("--narrow-band-cells", type=float, default=4.0)
    parser.add_argument("--maximum-grid-cells", type=int, default=5_000_000)
    parser.add_argument(
        "--fixed-grid-manifest",
        type=Path,
        help=(
            "Reuse the exact node-centred grid from an existing liquid-field "
            "manifest. This is intended for audited sequence extensions where "
            "rare escaped particles must not expand the reconstruction domain."
        ),
    )
    parser.add_argument(
        "--domain-manifest",
        type=Path,
        help=(
            "Audited water-body partition. Its selected body domain replaces "
            "the global particle AABB and its per-frame classification routes "
            "detached particles to the secondary phase."
        ),
    )
    parser.add_argument(
        "--domain-body-id",
        help=(
            "Stable water body to reconstruct. Optional only when the domain "
            "partition contains exactly one body."
        ),
    )
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


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
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


def source_snapshot(
    path,
    minimum_particles,
    maximum_particles,
    transform_fields,
    metres_per_unit,
    particle_indices=None,
    append_only_ids=False,
):
    with np.load(path, allow_pickle=False) as cache:
        all_positions = np.ascontiguousarray(
            np.asarray(cache["positions"], dtype=np.float64) * metres_per_unit,
            dtype=np.float32,
        )
        all_velocities = np.ascontiguousarray(
            np.asarray(cache["velocities"], dtype=np.float64) * metres_per_unit,
            dtype=np.float32,
        )
        active_particles = int(len(all_positions))
        if (
            all_positions.ndim != 2
            or all_positions.shape[1:] != (3,)
            or all_velocities.shape != all_positions.shape
            or not minimum_particles <= active_particles <= maximum_particles
        ):
            raise ValueError(f"Unexpected particle arrays in {path}")
        particle_ids = (
            np.ascontiguousarray(cache["particle_ids"], dtype=np.int64)
            if "particle_ids" in cache
            else np.arange(active_particles, dtype=np.int64)
        )
        if particle_ids.shape != (active_particles,):
            raise ValueError(f"Unexpected particle IDs in {path}")
        if append_only_ids and not np.array_equal(
            particle_ids, np.arange(active_particles, dtype=np.int64)
        ):
            raise ValueError(f"Append-only particle IDs are not canonical in {path}")
        if particle_indices is None:
            selected_indices = np.arange(active_particles, dtype=np.int64)
        else:
            selected_indices = np.ascontiguousarray(particle_indices, dtype=np.int64)
            if (
                selected_indices.ndim != 1
                or np.any(selected_indices < 0)
                or np.any(selected_indices >= active_particles)
                or (len(selected_indices) > 1 and np.any(np.diff(selected_indices) <= 0))
            ):
                raise ValueError("Particle selection must be sorted unique source indices")
        positions = np.ascontiguousarray(all_positions[selected_indices])
        velocities = np.ascontiguousarray(all_velocities[selected_indices])
        snapshot = {
            "schema": int(cache["schema"]),
            "sample_index": int(cache["sample_index"]),
            "physics_step": int(cache["physics_step"]),
            "simulation_time": float(cache["simulation_time"]),
            "positions": positions,
            "velocities": velocities,
            "particle_ids": np.ascontiguousarray(particle_ids[selected_indices]),
            "source_particle_count": active_particles,
            "selected_particle_indices": selected_indices,
        }
        for field in transform_fields:
            if field not in cache:
                raise ValueError(f"Source frame {path} is missing transform field {field!r}")
            snapshot[field] = np.asarray(cache[field], dtype=np.float64)
    if snapshot["schema"] != 1:
        raise ValueError(f"Unsupported source frame schema in {path}")
    if velocities.shape != positions.shape or not len(positions):
        raise ValueError(f"Particle selection is empty or inconsistent in {path}")
    for field in transform_fields:
        if snapshot[field].shape != (4, 4) or not np.isfinite(snapshot[field]).all():
            raise ValueError(f"Unexpected transform {field!r} in {path}")
    return snapshot


args = parse_args()
if args.start_sample < 0 or args.sample_stride < 1:
    raise ValueError("Invalid sample range or stride")
if min(
    args.voxel_size,
    args.padding,
    args.velocity_smoothing_sigma,
    args.narrow_band_cells,
) <= 0.0:
    raise ValueError("All spatial and smoothing parameters must be positive")
if args.maximum_grid_cells < 27:
    raise ValueError("maximum-grid-cells is too small")
if args.fixed_grid_manifest is not None and args.domain_manifest is not None:
    raise ValueError("--fixed-grid-manifest and --domain-manifest are mutually exclusive")
if args.domain_body_id is not None and args.domain_manifest is None:
    raise ValueError("--domain-body-id requires --domain-manifest")

source_directory = args.source_directory.resolve()
output_directory = args.output_directory.resolve()
terrain_path = (
    args.terrain_heightfield.resolve() if args.terrain_heightfield is not None else None
)
source_manifest_path = source_directory / "manifest.json"
source_audit_path = source_directory / "audit_report.json"
run_report_path = (
    args.run_report.resolve()
    if args.run_report is not None
    else source_directory.parent / "run_complete.json"
)
scene_contract_path = (
    args.scene_contract.resolve() if args.scene_contract is not None else None
)
required_paths = [
    source_manifest_path,
    source_audit_path,
    run_report_path,
]
if terrain_path is not None:
    required_paths.append(terrain_path)
for path in required_paths:
    if not path.is_file():
        raise FileNotFoundError(path)
if scene_contract_path is not None and not scene_contract_path.is_file():
    raise FileNotFoundError(scene_contract_path)

source_manifest = load_json(source_manifest_path)
source_audit = load_json(source_audit_path)
run_report = load_json(run_report_path)
source_manifest_schema = int(source_manifest.get("schema", -1))
if source_manifest_schema not in {1, 2} or source_manifest.get("state", {}).get("complete") is not True:
    raise ValueError("Source manifest is unsupported or incomplete")
if source_audit.get("valid") is not True:
    raise ValueError("Source cache has not passed its audit")
if source_audit.get("manifest_sha256") != sha256_file(source_manifest_path):
    raise ValueError("Source audit does not reference the current manifest")
if run_report.get("valid") is not True:
    raise ValueError("Parent PhysX run is not valid")
scene_contract = (
    SceneContract.load(scene_contract_path)
    if scene_contract_path is not None
    else legacy_sphere_impact_contract(run_report)
)
if scene_contract.terrain is None and terrain_path is None:
    raise ValueError(
        "Schema-1/legacy scenes require --terrain-heightfield; schema 2 embeds terrain"
    )
if scene_contract.terrain is not None and terrain_path is not None:
    raise ValueError(
        "Schema-2 terrain and --terrain-heightfield are mutually exclusive"
    )
metres_per_unit = float(scene_contract.metres_per_unit)
transform_fields = tuple(
    sorted(
        {
            collider.motion.source_field
            for collider in scene_contract.colliders
            if collider.motion.source_field is not None
        }
    )
)
if output_directory.exists() and any(output_directory.iterdir()):
    raise FileExistsError(f"Refusing to overwrite non-empty output: {output_directory}")
output_directory.mkdir(parents=True, exist_ok=True)

source_rows = {int(row["sample_index"]): row for row in source_manifest["samples"]}
audit_rows = {
    int(row["sample_index"]): row for row in source_audit.get("samples", [])
}
last_sample = max(source_rows)
end_sample = args.start_sample if args.end_sample is None else args.end_sample
if not 0 <= args.start_sample <= end_sample <= last_sample:
    raise ValueError(f"Requested range is outside source samples 0..{last_sample}")
sample_indices = list(range(args.start_sample, end_sample + 1, args.sample_stride))
if any(index not in source_rows or index not in audit_rows for index in sample_indices):
    raise ValueError("Source manifest/audit do not cover every requested sample")

particle_count = int(source_manifest["particle_count"])
maximum_particle_count = int(
    source_manifest.get("maximum_particle_count", particle_count)
)
append_only_ids = source_manifest_schema == 2
if maximum_particle_count < particle_count:
    raise ValueError("Source maximum particle count is below its initial count")
fixed_grid_manifest_path = None
fixed_grid_manifest_sha256 = None
domain_contract = None
if args.fixed_grid_manifest is not None:
    fixed_grid_manifest_path = args.fixed_grid_manifest.resolve()
    fixed_grid_manifest = load_json(fixed_grid_manifest_path)
    if fixed_grid_manifest.get("product") != "whitewater_v6_liquid_fields":
        raise ValueError("Fixed-grid manifest has the wrong product")
    fixed_grid = fixed_grid_manifest["grid"]
    spec = GridSpec(
        tuple(fixed_grid["origin"]),
        float(fixed_grid["spacing"]),
        tuple(fixed_grid["shape"]),
    )
    if not np.isclose(spec.spacing, args.voxel_size, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            "--voxel-size must exactly match the fixed-grid manifest spacing"
        )
    fixed_grid_manifest_sha256 = sha256_file(fixed_grid_manifest_path)
elif args.domain_manifest is not None:
    domain_contract = load_domain_partition_contract(
        args.domain_manifest,
        source_manifest_path=source_manifest_path,
        source_manifest=source_manifest,
        requested_spacing=args.voxel_size,
        body_id=args.domain_body_id,
    )
    missing_domain_samples = sorted(
        set(sample_indices) - set(domain_contract["sample_rows"])
    )
    if missing_domain_samples:
        raise ValueError(
            "Domain partition does not classify requested samples: "
            + ", ".join(map(str, missing_domain_samples))
        )
    domain = domain_contract["body"]["domain"]
    spec = GridSpec(
        tuple(domain["origin"]),
        float(domain["spacing"]),
        tuple(domain["shape"]),
    )
else:
    minimum = np.min(
        [
            np.asarray(audit_rows[index]["position_minimum"])
            for index in sample_indices
        ],
        axis=0,
    ) * metres_per_unit
    maximum = np.max(
        [
            np.asarray(audit_rows[index]["position_maximum"])
            for index in sample_indices
        ],
        axis=0,
    ) * metres_per_unit
    spec = aligned_grid_spec(minimum, maximum, args.voxel_size, args.padding)
if spec.cell_count > args.maximum_grid_cells:
    raise ValueError(
        f"Grid has {spec.cell_count:,} cells, exceeding limit "
        f"{args.maximum_grid_cells:,}"
    )

particle_spacing = metres_per_unit * float(run_report["particle_spacing"])
# Each CIC particle represents one particle-spacing cube.  This is fixed for
# the entire sequence and does not fluctuate with per-frame compression.
bulk_weight_reference = (args.voxel_size / particle_spacing) ** 3

if scene_contract.terrain is not None:
    terrain_sdf, terrain_valid, open_terrain_metadata = open_terrain_collision_sdf(
        spec,
        scene_contract.terrain,
        length_scale=metres_per_unit,
    )
    terrain_manifest = {
        "representation": scene_contract.terrain.representation,
        "path": scene_contract.terrain.parameters["path"],
        "sha256": scene_contract.terrain.parameters["sha256"],
        "selection_path": scene_contract.terrain.parameters["selection_path"],
        "selection_sha256": scene_contract.terrain.parameters["selection_sha256"],
        "normal_convention": scene_contract.terrain.parameters["normal_convention"],
        "distance_method": open_terrain_metadata["distance_method"],
        "adapter_metadata": open_terrain_metadata,
        "sign_convention": "negative_on_solid_side",
    }
else:
    with np.load(terrain_path, allow_pickle=False) as terrain:
        terrain_x = metres_per_unit * np.asarray(terrain["x_values"], dtype=np.float64)
        terrain_z = metres_per_unit * np.asarray(terrain["z_values"], dtype=np.float64)
        terrain_y = metres_per_unit * np.asarray(terrain["terrain_y"], dtype=np.float32)
    terrain_sdf, terrain_valid = terrain_collision_sdf(
        spec, terrain_x, terrain_z, terrain_y
    )
    terrain_manifest = {
        "representation": "regular_heightfield_legacy_adapter",
        "path": str(terrain_path),
        "sha256": sha256_file(terrain_path),
        "distance_method": "vertical_height_difference",
        "sign_convention": "negative_inside_solid",
    }
if not np.all(terrain_valid):
    missing = int(np.size(terrain_valid) - np.count_nonzero(terrain_valid))
    raise ValueError(
        f"Fixed grid extends beyond the audited terrain heightfield in {missing} nodes"
    )
static_collider_fields = build_collider_fields(
    spec, scene_contract, snapshot=None, include_motion={"static"}
)

coordinates_path = output_directory / "grid_and_static_fields.npz"
grid_x, grid_y, grid_z = spec.axes(dtype=np.float32)
static_arrays = {
    "schema": np.int32(SCHEMA),
    "x": grid_x,
    "y": grid_y,
    "z": grid_z,
    "terrain_collision_sdf": terrain_sdf,
    "terrain_valid": terrain_valid,
}
if "collider_collision_sdf" in static_collider_fields:
    static_arrays["static_collider_collision_sdf"] = static_collider_fields[
        "collider_collision_sdf"
    ]
if "churn_source_sdf" in static_collider_fields:
    static_arrays["static_churn_source_sdf"] = static_collider_fields[
        "churn_source_sdf"
    ]
atomic_npz(coordinates_path, **static_arrays)

configuration = {
    "start_sample": args.start_sample,
    "end_sample": end_sample,
    "sample_stride": args.sample_stride,
    "voxel_size": args.voxel_size,
    "padding": args.padding,
    "iso_fraction": args.iso_fraction,
    "velocity_smoothing_sigma": args.velocity_smoothing_sigma,
    "narrow_band_cells": args.narrow_band_cells,
    "fixed_grid_manifest": (
        str(fixed_grid_manifest_path) if fixed_grid_manifest_path else None
    ),
    "fixed_grid_manifest_sha256": fixed_grid_manifest_sha256,
    "domain_manifest": (
        str(domain_contract["manifest_path"]) if domain_contract else None
    ),
    "domain_manifest_sha256": (
        domain_contract["manifest_sha256"] if domain_contract else None
    ),
    "domain_body_id": (
        domain_contract["body"]["body_id"] if domain_contract else None
    ),
    "particle_spacing": particle_spacing,
    "metres_per_unit": metres_per_unit,
    "bulk_weight_reference": bulk_weight_reference,
    "bulk_weight_definition": "(voxel_size / particle_spacing) ** 3",
    "whitewater_sample_rate_hz": 1.0
    / float(source_rows[1]["simulation_time"] - source_rows[0]["simulation_time"]),
}
manifest = {
    "schema": SCHEMA,
    "product": "whitewater_v6_liquid_fields",
    "created_utc": utc_now_iso(),
    "producer": {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "field_module": str(
            (Path(__file__).parent / "whitewater" / "liquid_fields.py").resolve()
        ),
        "field_module_sha256": sha256_file(
            Path(__file__).parent / "whitewater" / "liquid_fields.py"
        ),
        "terrain_module": str(
            (Path(__file__).parent / "whitewater" / "terrain_fields.py").resolve()
        ),
        "terrain_module_sha256": sha256_file(
            Path(__file__).parent / "whitewater" / "terrain_fields.py"
        ),
        "implementation": "numpy_scipy_cpu_reference",
    },
    "source": {
        "directory": str(source_directory),
        "manifest": str(source_manifest_path),
        "manifest_sha256": sha256_file(source_manifest_path),
        "audit": str(source_audit_path),
        "audit_sha256": sha256_file(source_audit_path),
        "run_report": str(run_report_path),
        "run_report_sha256": sha256_file(run_report_path),
        "particle_count": particle_count,
        "maximum_particle_count": maximum_particle_count,
        "selected_stable_body_particle_count": (
            int(len(domain_contract["body_particle_indices"]))
            if domain_contract
            else particle_count
        ),
    },
    "domain_partition": (
        {
            "manifest": str(domain_contract["manifest_path"]),
            "manifest_sha256": domain_contract["manifest_sha256"],
            "membership": str(domain_contract["membership_path"]),
            "membership_sha256": domain_contract["membership_sha256"],
            "body_id": domain_contract["body"]["body_id"],
            "body_index": domain_contract["body_index"],
            "stable_body_particles": int(
                len(domain_contract["body_particle_indices"])
            ),
            "particle_id_contract": source_manifest["particle_ids"],
            "liquid_core_rule": (
                "selected stable body AND particle_state=core_liquid AND "
                "outside_body_domain=0"
            ),
            "secondary_handoff_rule": (
                "selected stable body AND (particle_state!=core_liquid OR "
                "outside_body_domain!=0)"
            ),
        }
        if domain_contract
        else None
    ),
    "scene_contract": {
        "mode": "file" if scene_contract_path is not None else "legacy_adapter",
        "path": str(scene_contract_path) if scene_contract_path is not None else None,
        "sha256": sha256_file(scene_contract_path) if scene_contract_path is not None else None,
        "resolved": scene_contract.metadata(),
        "transform_fields": list(transform_fields),
        "static_collider_fields": static_collider_fields["metadata"],
    },
    "terrain": terrain_manifest,
    "grid": spec.metadata(),
    "configuration": configuration,
    "static_fields": {
        "file": coordinates_path.name,
        "sha256": sha256_file(coordinates_path),
        "bytes": coordinates_path.stat().st_size,
        "arrays": sorted(static_arrays),
    },
    "field_contract": {
        "liquid_phi_sign": "negative_inside_liquid",
        "collision_phi_sign": "negative_inside_solid",
        "collision_sdf": "union of terrain and every collider with the solid role",
        "dynamic_collision_sdf": "optional union of source-driven dynamic solids",
        "churn_source_sdf": "optional union of colliders explicitly tagged churn_source",
        "dynamic_churn_source_sdf": "optional dynamic-only churn union retained for composition audit",
        "normal_direction": "outward_from_liquid",
        "curvature": "divergence_of_outward_normal",
        "acceleration": "material_Du_Dt",
        "axis_order": "xyz",
        "vector_component_order": "xyz",
    },
    "samples": [],
    "state": {
        "complete": False,
        "completed_samples": 0,
        "expected_samples": len(sample_indices),
    },
}
manifest_path = output_directory / "manifest.json"
atomic_json(manifest_path, manifest)


def reconstruct(snapshot):
    return reconstruct_liquid_fields(
        snapshot["positions"],
        snapshot["velocities"],
        spec,
        bulk_weight_reference=bulk_weight_reference,
        iso_fraction=args.iso_fraction,
        velocity_smoothing_sigma=args.velocity_smoothing_sigma,
        narrow_band_cells=args.narrow_band_cells,
    )


previous_snapshot = None
previous_fields = None
if sample_indices[0] > 0 and (
    domain_contract is None
    or sample_indices[0] - 1 in domain_contract["sample_rows"]
):
    predecessor_index = sample_indices[0] - 1
    predecessor_row = source_rows[predecessor_index]
    predecessor_path = source_directory / predecessor_row["file"]
    if sha256_file(predecessor_path) != predecessor_row["sha256"]:
        raise ValueError(f"Warm-up source hash mismatch: {predecessor_path}")
    predecessor_selection = None
    if domain_contract is not None:
        predecessor_classification = load_particle_classification(
            domain_contract, predecessor_index, particle_count
        )
        predecessor_selection = predecessor_classification["selected_core_indices"]
    previous_snapshot = source_snapshot(
        predecessor_path,
        particle_count,
        maximum_particle_count,
        transform_fields,
        metres_per_unit,
        particle_indices=predecessor_selection,
        append_only_ids=append_only_ids,
    )
    previous_fields = reconstruct(previous_snapshot)

for output_frame, source_index in enumerate(sample_indices):
    started = time.perf_counter()
    source_row = source_rows[source_index]
    source_path = source_directory / source_row["file"]
    if sha256_file(source_path) != source_row["sha256"]:
        raise ValueError(f"Source hash mismatch: {source_path}")
    classification = None
    particle_selection = None
    if domain_contract is not None:
        classification = load_particle_classification(
            domain_contract, source_index, particle_count
        )
        particle_selection = classification["selected_core_indices"]
    snapshot = source_snapshot(
        source_path,
        particle_count,
        maximum_particle_count,
        transform_fields,
        metres_per_unit,
        particle_indices=particle_selection,
        append_only_ids=append_only_ids,
    )
    if snapshot["sample_index"] != source_index:
        raise ValueError(f"Source index mismatch in {source_path}")
    fields = reconstruct(snapshot)

    if previous_fields is None:
        acceleration = np.zeros_like(fields["velocity"], dtype=np.float32)
        acceleration_valid = np.zeros(spec.shape, dtype=np.uint8)
    else:
        dt = snapshot["simulation_time"] - previous_snapshot["simulation_time"]
        if dt <= 0.0:
            raise ValueError("Source simulation times are not strictly increasing")
        acceleration_valid = (
            (fields["velocity_valid"] != 0)
            & (previous_fields["velocity_valid"] != 0)
        )
        acceleration = material_acceleration(
            fields["velocity"],
            previous_fields["velocity"],
            dt,
            spec.spacing,
            valid_mask=acceleration_valid,
        )

    dynamic_collider_fields = build_collider_fields(
        spec,
        scene_contract,
        snapshot=snapshot,
        include_motion={"source_matrix"},
    )
    collider_components = [
        value
        for value in (
            static_collider_fields.get("collider_collision_sdf"),
            dynamic_collider_fields.get("collider_collision_sdf"),
        )
        if value is not None
    ]
    collider_sdf = (
        union_collision_sdf(*collider_components) if collider_components else None
    )
    churn_components = [
        value
        for value in (
            static_collider_fields.get("churn_source_sdf"),
            dynamic_collider_fields.get("churn_source_sdf"),
        )
        if value is not None
    ]
    churn_source_sdf = (
        union_collision_sdf(*churn_components) if churn_components else None
    )
    collision_sdf = (
        terrain_sdf
        if collider_sdf is None
        else union_collision_sdf(terrain_sdf, collider_sdf)
    )
    output_path = output_directory / f"liquid_fields_{output_frame:06d}.npz"
    output_arrays = {
        "schema": np.int32(SCHEMA),
        "output_frame": np.int32(output_frame),
        "source_sample_index": np.int32(source_index),
        "physics_step": np.int64(snapshot["physics_step"]),
        "simulation_time": np.float64(snapshot["simulation_time"]),
        "particle_weight": fields["particle_weight"],
        "number_density": fields["number_density"],
        "fluid_mask": fields["fluid_mask"],
        "phi": fields["phi"],
        "depth": fields["depth"],
        "velocity": fields["velocity"],
        "velocity_valid": fields["velocity_valid"],
        "normal": fields["normal"],
        "curvature": fields["curvature"],
        "surface_valid": fields["surface_valid"],
        "divergence": fields["divergence"],
        "vorticity": fields["vorticity"],
        "strain_rate": fields["strain_rate"],
        "acceleration": acceleration,
        "acceleration_valid": acceleration_valid,
        "collision_sdf": collision_sdf,
    }
    if collider_sdf is not None:
        output_arrays["collider_collision_sdf"] = collider_sdf
    if "dynamic_collision_sdf" in dynamic_collider_fields:
        output_arrays["dynamic_collision_sdf"] = dynamic_collider_fields[
            "dynamic_collision_sdf"
        ]
    if "churn_source_sdf" in dynamic_collider_fields:
        output_arrays["dynamic_churn_source_sdf"] = dynamic_collider_fields[
            "churn_source_sdf"
        ]
    if churn_source_sdf is not None:
        output_arrays["churn_source_sdf"] = churn_source_sdf
    if scene_contract_path is None:
        # Preserve the exact v6 cache interface while all downstream readers
        # migrate to role-based names. This alias is never emitted for a new
        # scene contract and therefore cannot become a new hidden dependency.
        legacy_collider = scene_contract.colliders[0]
        _, sphere_center = resolve_motion_matrix(
            legacy_collider.motion, snapshot, length_scale=metres_per_unit
        )
        output_arrays["sphere_center"] = np.asarray(sphere_center, dtype=np.float32)
        output_arrays["sphere_collision_sdf"] = dynamic_collider_fields[
            "dynamic_collision_sdf"
        ]
    atomic_npz(output_path, **output_arrays)
    surface = fields["surface_valid"] != 0
    elapsed = time.perf_counter() - started
    item = {
        "output_frame": output_frame,
        "source_sample_index": source_index,
        "physics_step": snapshot["physics_step"],
        "simulation_time": snapshot["simulation_time"],
        "source_file": source_row["file"],
        "source_sha256": source_row["sha256"],
        "source_particle_count": int(snapshot["source_particle_count"]),
        "liquid_core_input_particles": int(len(snapshot["positions"])),
        "domain_classification_file": (
            classification["path"].name if classification else None
        ),
        "domain_classification_sha256": (
            classification["sha256"] if classification else None
        ),
        "detached_secondary_candidates": (
            int(len(classification["selected_detached_indices"]))
            if classification
            else 0
        ),
        "outside_body_domain_candidates": (
            int(len(classification["selected_outside_indices"]))
            if classification
            else 0
        ),
        "secondary_handoff_particles": (
            int(len(classification["selected_secondary_indices"]))
            if classification
            else 0
        ),
        "file": output_path.name,
        "sha256": sha256_file(output_path),
        "bytes": output_path.stat().st_size,
        "elapsed_seconds": elapsed,
        "fluid_nodes": int(np.count_nonzero(fields["fluid_mask"])),
        "surface_nodes": int(np.count_nonzero(surface)),
        "velocity_valid_nodes": int(np.count_nonzero(fields["velocity_valid"])),
        "acceleration_valid_nodes": int(np.count_nonzero(acceleration_valid)),
        "maximum_speed": float(
            np.max(np.linalg.norm(fields["velocity"], axis=-1), initial=0.0)
        ),
        "maximum_acceleration": float(
            np.max(np.linalg.norm(acceleration, axis=-1), initial=0.0)
        ),
        "maximum_absolute_curvature": float(
            np.max(np.abs(fields["curvature"][surface]), initial=0.0)
        ),
        "liquid_phi_range": [float(fields["phi"].min()), float(fields["phi"].max())],
        "collision_phi_range": [
            float(collision_sdf.min()),
            float(collision_sdf.max()),
        ],
    }
    manifest["samples"].append(item)
    manifest["state"]["completed_samples"] = len(manifest["samples"])
    atomic_json(manifest_path, manifest)
    print(
        f"[v6-fields] source={source_index:04d} grid={spec.shape} "
        f"fluid={item['fluid_nodes']} surface={item['surface_nodes']} "
        f"elapsed={elapsed:.2f}s",
        flush=True,
    )
    previous_snapshot = snapshot
    previous_fields = fields

manifest["state"]["complete"] = True
manifest["state"]["completed_utc"] = utc_now_iso()
atomic_json(manifest_path, manifest)
print(json.dumps(manifest["state"], indent=2))
