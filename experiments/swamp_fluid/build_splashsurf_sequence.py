"""Batch-reconstruct and terrain-clip a PhysX particle PLY sequence."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.scene_contract import SceneContract
from whitewater.domain_partition import (
    load_domain_partition_contract,
    load_particle_classification,
)
from whitewater.terrain_fields import (
    open_terrain_world_arrays,
    rasterize_open_terrain_heightfield,
)


MANIFEST_SCHEMA = 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("particle_directory", type=Path)
    parser.add_argument(
        "terrain_source",
        type=Path,
        help="Legacy heightfield NPZ or schema-2 scene-contract JSON.",
    )
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--water-level", type=float, required=True)
    parser.add_argument("--spacing", type=float, default=0.008)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        help="Splashsurf threads per process (default: CPU count divided by workers).",
    )
    parser.add_argument(
        "--particle-radius",
        type=float,
        help="Particle volume radius (default: equal-volume radius for spacing).",
    )
    parser.add_argument(
        "--smoothing-length",
        type=float,
        help="Kernel radius in particle-radius units (default keeps radius at spacing).",
    )
    parser.add_argument(
        "--cube-size",
        type=float,
        help="Voxel size in particle-radius units (default keeps voxel at 0.375 spacing).",
    )
    parser.add_argument("--surface-threshold", type=float, default=0.6)
    parser.add_argument("--mesh-smoothing-iters", type=int, default=10)
    parser.add_argument("--normal-smoothing-iters", type=int, default=10)
    parser.add_argument(
        "--shoreline-mode",
        choices=("auto", "terrain", "reference-ply", "dynamic"),
        default="auto",
    )
    parser.add_argument(
        "--shoreline-particles-ply",
        type=Path,
        help="Fixed particle frame used by reference-ply shoreline mode.",
    )
    parser.add_argument("--minimum-layers", type=int, default=8)
    parser.add_argument("--shoreline-erosion-cells", type=int, default=4)
    parser.add_argument("--impact-x", type=float, default=-0.73)
    parser.add_argument("--impact-z", type=float, default=0.78)
    parser.add_argument("--impact-radius", type=float, default=0.40)
    parser.add_argument(
        "--aabb-margin",
        type=float,
        help="X/Z reconstruction margin beyond heightfield bounds (default: 2 spacing).",
    )
    parser.add_argument(
        "--terrain-raster-bounds",
        type=float,
        nargs=4,
        metavar=("X_MIN", "X_MAX", "Z_MIN", "Z_MAX"),
        help=(
            "Explicit support-raster bounds before AABB margin. Intended for "
            "audited migration; automatic water-body domains are a later stage."
        ),
    )
    parser.add_argument(
        "--terrain-raster-axis-dtype",
        choices=("float32", "float64"),
        default="float64",
        help="Axis storage precision; float32 exists for exact legacy migration.",
    )
    parser.add_argument(
        "--aabb-height",
        type=float,
        default=0.6,
        help="Maximum reconstruction height above the water level.",
    )
    parser.add_argument(
        "--no-particle-aabb",
        action="store_true",
        help="Disable heightfield-derived particle filtering (diagnostics only).",
    )
    parser.add_argument(
        "--domain-manifest",
        type=Path,
        help=(
            "Audited water-body partition. In this mode particle_directory must "
            "be its source NPZ directory; core PLY and secondary handoff files "
            "are derived from the shared per-frame classification ledger."
        ),
    )
    parser.add_argument(
        "--domain-body-id",
        help="Stable water body ID; optional only for a one-body partition.",
    )
    parser.add_argument(
        "--frames",
        nargs="+",
        type=int,
        help="Optional frame numbers to reconstruct, for representative-frame QA.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--terrain-source-mode",
        choices=("auto", "heightfield", "scene-contract"),
        default="auto",
    )
    parser.add_argument(
        "--splashsurf",
        type=Path,
        default=Path(sys.executable).with_name("pysplashsurf.exe"),
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def particle_metadata_digest(paths):
    digest = hashlib.sha256()
    total_bytes = 0
    for path in paths:
        stat = path.stat()
        total_bytes += stat.st_size
        digest.update(
            f"{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8")
        )
    return digest.hexdigest(), total_bytes


def run_checked(command):
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(map(str, command))}\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return completed.stdout.strip()


def write_manifest(path, configuration, state):
    payload = {
        "schema": MANIFEST_SCHEMA,
        "configuration": configuration,
        "state": state,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_npz(path, **arrays):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def binary_position_ply_bytes(positions):
    positions = np.ascontiguousarray(positions, dtype="<f4")
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise ValueError("PLY positions must be a finite (n, 3) array")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment core liquid particles selected by whitewater v6 domain ledger\n"
        f"element vertex {len(positions)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode("ascii")
    return header, positions.tobytes(order="C")


def ensure_binary_position_ply(path, positions, force=False):
    header, payload = binary_position_ply_bytes(positions)
    expected_hash = hashlib.sha256(header + payload).hexdigest()
    if path.is_file() and sha256_file(path) == expected_hash:
        return expected_hash
    if path.exists() and not force:
        raise ValueError(f"Existing core PLY does not match its particle ledger: {path}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(header)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return expected_hash


def arrays_match_npz(path, arrays):
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as cache:
            return set(cache.files) == set(arrays) and all(
                np.array_equal(np.asarray(cache[name]), value)
                for name, value in arrays.items()
            )
    except (OSError, ValueError):
        return False


def prepare_domain_particle_inputs(args):
    domain_manifest_path = args.domain_manifest.resolve()
    raw_domain = json.loads(domain_manifest_path.read_text(encoding="utf-8"))
    source_manifest_path = Path(raw_domain["source"]["manifest"]).resolve()
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_directory = Path(raw_domain["source"]["directory"]).resolve()
    if not np.isclose(
        float(args.spacing),
        float(raw_domain["configuration"]["particle_spacing"]),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("--spacing must match the domain source particle spacing")
    if args.particle_directory.resolve() != source_directory:
        raise ValueError(
            "Domain mode requires particle_directory to equal the partition source directory"
        )
    contract = load_domain_partition_contract(
        domain_manifest_path,
        source_manifest_path=source_manifest_path,
        source_manifest=source_manifest,
        requested_spacing=float(raw_domain["configuration"]["domain_spacing"]),
        body_id=args.domain_body_id,
    )
    available = sorted(contract["sample_rows"])
    if args.frames is None:
        sample_indices = available
    else:
        requested = sorted(set(map(int, args.frames)))
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(
                "Domain partition does not classify requested samples: "
                + ", ".join(map(str, missing))
            )
        sample_indices = requested
    source_rows = {
        int(item["sample_index"]): item for item in source_manifest["samples"]
    }
    core_directory = args.output_directory.resolve() / "core_particles"
    secondary_directory = args.output_directory.resolve() / "secondary_handoff"
    core_directory.mkdir(parents=True, exist_ok=True)
    secondary_directory.mkdir(parents=True, exist_ok=True)
    frame_provenance = []
    core_paths = []
    for sample_index in sample_indices:
        source_row = source_rows[sample_index]
        source_path = source_directory / source_row["file"]
        domain_row = contract["sample_rows"][sample_index]
        if (
            sha256_file(source_path) != source_row["sha256"]
            or domain_row["source_sha256"] != source_row["sha256"]
        ):
            raise ValueError(f"Domain/source frame provenance mismatch: {sample_index}")
        classification = load_particle_classification(
            contract, sample_index, int(source_manifest["particle_count"])
        )
        with np.load(source_path, allow_pickle=False) as cache:
            positions = np.asarray(cache["positions"], dtype=np.float32)
            velocities = np.asarray(cache["velocities"], dtype=np.float32)
            simulation_time = np.float64(cache["simulation_time"])
            physics_step = np.int64(cache["physics_step"])
        core_indices = classification["selected_core_indices"]
        secondary_indices = classification["selected_secondary_indices"]
        core_path = core_directory / f"particles_{sample_index:06d}.ply"
        core_hash = ensure_binary_position_ply(
            core_path, positions[core_indices], force=args.force
        )
        secondary_path = secondary_directory / (
            f"secondary_candidates_{sample_index:06d}.npz"
        )
        state = classification["state"]
        outside_body = classification["outside_body_domain"]
        secondary_arrays = {
            "schema": np.int32(1),
            "source_sample_index": np.int32(sample_index),
            "physics_step": physics_step,
            "simulation_time": simulation_time,
            "source_particle_indices": np.ascontiguousarray(
                secondary_indices, dtype=np.int64
            ),
            "positions": np.ascontiguousarray(positions[secondary_indices]),
            "velocities": np.ascontiguousarray(velocities[secondary_indices]),
            "particle_state": np.ascontiguousarray(state[secondary_indices]),
            "instantaneous_detached": np.ascontiguousarray(
                classification["instantaneous_detached"][secondary_indices]
            ),
            "outside_body_domain": np.ascontiguousarray(
                outside_body[secondary_indices]
            ),
            "new_secondary_handoff": np.ascontiguousarray(
                classification["new_secondary_handoff"][secondary_indices]
            ),
            "pending_return_to_core": np.ascontiguousarray(
                classification["pending_return_to_core"][secondary_indices]
            ),
            "particle_id_contract_sha256": np.asarray(
                source_manifest["particle_ids"]["sha256"]
            ),
        }
        if not arrays_match_npz(secondary_path, secondary_arrays):
            if secondary_path.exists() and not args.force:
                raise ValueError(
                    f"Existing secondary handoff does not match ledger: {secondary_path}"
                )
            atomic_npz(secondary_path, **secondary_arrays)
        frame_provenance.append(
            {
                "source_sample_index": sample_index,
                "source_file": source_row["file"],
                "source_sha256": source_row["sha256"],
                "classification_file": classification["path"].name,
                "classification_sha256": classification["sha256"],
                "core_particles": int(len(core_indices)),
                "core_ply": core_path.name,
                "core_ply_sha256": core_hash,
                "secondary_particles": int(len(secondary_indices)),
                "pending_proxy_particles": int(
                    len(classification["selected_pending_indices"])
                ),
                "secondary_owned_particles": int(
                    len(classification["selected_owned_indices"])
                ),
                "new_secondary_handoff_particles": int(
                    len(classification["selected_new_handoff_indices"])
                ),
                "pending_return_to_core_particles": int(
                    len(classification["selected_pending_return_indices"])
                ),
                "secondary_handoff": secondary_path.name,
                "secondary_handoff_sha256": sha256_file(secondary_path),
            }
        )
        core_paths.append(core_path)
    return contract, core_paths, frame_provenance


def terrain_source_mode(path, requested):
    if requested != "auto":
        return requested
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict) and payload.get("product") == "whitewater_scene_contract":
            return "scene-contract"
    return "heightfield"


def prepare_terrain(args, domain_raster_bounds=None):
    source_path = args.terrain_source.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    mode = terrain_source_mode(source_path, args.terrain_source_mode)
    aabb_margin = args.aabb_margin if args.aabb_margin is not None else 2.0 * args.spacing
    if mode == "heightfield":
        with np.load(source_path, allow_pickle=False) as heightfield:
            x_values = np.asarray(heightfield["x_values"], dtype=np.float64)
            z_values = np.asarray(heightfield["z_values"], dtype=np.float64)
            terrain_y = np.asarray(heightfield["terrain_y"], dtype=np.float64)
        provenance = {
            "representation": "regular_heightfield_legacy_adapter",
            "heightfield": str(source_path),
            "heightfield_sha256": sha256_file(source_path),
            "raster": {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
                "derived": False,
            },
        }
        return source_path, x_values, z_values, terrain_y, provenance, aabb_margin

    contract = SceneContract.load(source_path)
    if contract.schema != 2 or contract.terrain is None:
        raise ValueError("Splashsurf scene-contract terrain requires schema 2")
    vertices, _faces = open_terrain_world_arrays(
        contract.terrain, length_scale=contract.metres_per_unit
    )
    if args.terrain_raster_bounds is None:
        if domain_raster_bounds is None:
            x_min = math.floor(float(vertices[:, 0].min()) / args.spacing) * args.spacing
            x_max = math.ceil(float(vertices[:, 0].max()) / args.spacing) * args.spacing
            z_min = math.floor(float(vertices[:, 2].min()) / args.spacing) * args.spacing
            z_max = math.ceil(float(vertices[:, 2].max()) / args.spacing) * args.spacing
            raster_domain_mode = "selected_terrain_bounds"
        else:
            x_min, x_max, z_min, z_max = map(float, domain_raster_bounds)
            raster_domain_mode = "automatic_water_body_domain"
    else:
        x_min, x_max, z_min, z_max = map(float, args.terrain_raster_bounds)
        if not x_min < x_max or not z_min < z_max:
            raise ValueError("--terrain-raster-bounds must be strictly increasing")
        raster_domain_mode = "explicit_audited_bounds"
    x_steps = int(round((x_max - x_min) / args.spacing))
    z_steps = int(round((z_max - z_min) / args.spacing))
    if (
        not np.isclose(x_min + x_steps * args.spacing, x_max, atol=1.0e-7, rtol=0.0)
        or not np.isclose(z_min + z_steps * args.spacing, z_max, atol=1.0e-7, rtol=0.0)
    ):
        raise ValueError("Terrain raster bounds must align to --spacing")
    # Preserve explicitly audited endpoints exactly.  The tiny float32 legacy
    # endpoint round-off is part of Splashsurf's particle-AABB/grid provenance.
    x_values = np.linspace(x_min, x_max, x_steps + 1, dtype=np.float64)
    z_values = np.linspace(z_min, z_max, z_steps + 1, dtype=np.float64)
    axis_dtype = np.dtype(args.terrain_raster_axis_dtype)
    x_values = x_values.astype(axis_dtype)
    z_values = z_values.astype(axis_dtype)
    terrain_y, raster_metadata = rasterize_open_terrain_heightfield(
        contract.terrain,
        x_values,
        z_values,
        maximum_y=None,
        length_scale=contract.metres_per_unit,
    )
    raster_path = args.output_directory.resolve() / "terrain_raster_from_scene_contract.npz"
    raster_arrays = {
        "schema": np.int32(1),
        "x_values": x_values,
        "z_values": z_values,
        "terrain_y": terrain_y.astype(np.float32),
        "spacing": np.float64(args.spacing),
        "scene_contract_sha256": np.asarray(sha256_file(source_path)),
        "terrain_mesh_sha256": np.asarray(contract.terrain.parameters["sha256"]),
        "terrain_selection_sha256": np.asarray(
            contract.terrain.parameters["selection_sha256"]
        ),
    }
    if raster_path.exists():
        with np.load(raster_path, allow_pickle=False) as existing:
            matches = set(existing.files) == set(raster_arrays) and all(
                np.array_equal(np.asarray(existing[key]), value, equal_nan=True)
                for key, value in raster_arrays.items()
            )
        if not matches:
            raise ValueError(
                f"Existing terrain raster does not match the scene contract: {raster_path}"
            )
    else:
        atomic_npz(raster_path, **raster_arrays)
    provenance = {
        "representation": "open_triangle_mesh_raster_adapter",
        "scene_contract": str(source_path),
        "scene_contract_sha256": sha256_file(source_path),
        "terrain_mesh": contract.terrain.parameters["path"],
        "terrain_mesh_sha256": contract.terrain.parameters["sha256"],
        "terrain_selection": contract.terrain.parameters["selection_path"],
        "terrain_selection_sha256": contract.terrain.parameters["selection_sha256"],
        "normal_convention": contract.terrain.parameters["normal_convention"],
        "raster": {
            "path": str(raster_path),
            "sha256": sha256_file(raster_path),
            "derived": True,
            "metadata": raster_metadata,
            "domain_mode": raster_domain_mode,
            "bounds": [x_min, x_max, z_min, z_max],
            "aabb_margin": aabb_margin,
            "axis_dtype": args.terrain_raster_axis_dtype,
        },
    }
    return raster_path, x_values, z_values, terrain_y, provenance, aabb_margin


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.domain_body_id is not None and args.domain_manifest is None:
        raise ValueError("--domain-body-id requires --domain-manifest")
    if args.domain_manifest is not None and args.no_particle_aabb:
        raise ValueError("Domain mode cannot disable its audited particle AABB")

    clip_script = Path(__file__).with_name("clip_splashsurf_free_surface.py")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    raw_directory = args.output_directory / "raw"
    surface_directory = args.output_directory / "surface"
    raw_directory.mkdir(parents=True, exist_ok=True)
    surface_directory.mkdir(parents=True, exist_ok=True)

    domain_contract = None
    domain_frame_provenance = None
    if args.domain_manifest is not None:
        domain_contract, particle_paths, domain_frame_provenance = (
            prepare_domain_particle_inputs(args)
        )
    else:
        all_particle_paths = sorted(args.particle_directory.glob("particles_*.ply"))
        if not all_particle_paths:
            raise RuntimeError(f"No particle PLY files found in {args.particle_directory}")
        if args.frames is None:
            particle_paths = all_particle_paths
        else:
            requested = {int(frame) for frame in args.frames}
            particle_paths = [
                path
                for path in all_particle_paths
                if int(path.stem.rsplit("_", 1)[-1]) in requested
            ]
            found = {int(path.stem.rsplit("_", 1)[-1]) for path in particle_paths}
            missing = sorted(requested - found)
            if missing:
                raise RuntimeError(f"Requested particle frames not found: {missing}")

    equal_volume_radius = math.pow(3.0 / (4.0 * math.pi), 1.0 / 3.0) * args.spacing
    particle_radius = args.particle_radius or equal_volume_radius
    smoothing_length = args.smoothing_length or (args.spacing / particle_radius)
    cube_size = args.cube_size or (0.375 * args.spacing / particle_radius)
    voxel_size = particle_radius * cube_size
    threads_per_worker = args.threads_per_worker or max(
        1, (os.cpu_count() or 1) // args.workers
    )
    if min(particle_radius, smoothing_length, cube_size) <= 0:
        raise ValueError("Splashsurf radius, smoothing length, and cube size must be positive")
    if threads_per_worker < 1:
        raise ValueError("--threads-per-worker must be at least 1")

    domain_raster_bounds = None
    if domain_contract is not None:
        domain = domain_contract["body"]["domain"]
        domain_raster_bounds = (
            domain["origin"][0],
            domain["maximum"][0],
            domain["origin"][2],
            domain["maximum"][2],
        )
    (
        terrain_raster_path,
        x_values,
        z_values,
        terrain_y,
        terrain_provenance,
        aabb_margin,
    ) = prepare_terrain(args, domain_raster_bounds=domain_raster_bounds)
    finite_terrain = terrain_y[np.isfinite(terrain_y)]
    if len(finite_terrain) == 0:
        raise RuntimeError("Heightfield contains no finite terrain samples")
    if domain_contract is None:
        particle_aabb_min = [
            float(x_values[0] - aabb_margin),
            float(np.min(finite_terrain) - max(0.05, 4.0 * args.spacing)),
            float(z_values[0] - aabb_margin),
        ]
        particle_aabb_max = [
            float(x_values[-1] + aabb_margin),
            float(args.water_level + args.aabb_height),
            float(z_values[-1] + aabb_margin),
        ]
        particle_aabb_source = "terrain_raster_plus_manual_height"
    else:
        domain = domain_contract["body"]["domain"]
        particle_aabb_min = list(map(float, domain["origin"]))
        particle_aabb_max = list(map(float, domain["maximum"]))
        particle_aabb_source = "audited_water_body_domain"

    splashsurf_version = run_checked([str(args.splashsurf), "--version"])
    metadata_digest, total_particle_bytes = particle_metadata_digest(particle_paths)
    configuration = {
        "schema": MANIFEST_SCHEMA,
        "particle_directory": str(args.particle_directory.resolve()),
        "particle_count": len(particle_paths),
        "particle_first": particle_paths[0].name,
        "particle_last": particle_paths[-1].name,
        "particle_metadata_sha256": metadata_digest,
        "particle_total_bytes": total_particle_bytes,
        "terrain": terrain_provenance,
        "builder_script": str(Path(__file__).resolve()),
        "builder_script_sha256": sha256_file(Path(__file__).resolve()),
        "terrain_module": str(
            Path(__file__).with_name("whitewater").joinpath("terrain_fields.py").resolve()
        ),
        "terrain_module_sha256": sha256_file(
            Path(__file__).with_name("whitewater").joinpath("terrain_fields.py")
        ),
        "clip_script_sha256": sha256_file(clip_script),
        "splashsurf_executable": str(args.splashsurf.resolve()),
        "splashsurf_version": splashsurf_version,
        "water_level": args.water_level,
        "spacing": args.spacing,
        "particle_radius": particle_radius,
        "smoothing_length": smoothing_length,
        "smoothing_length_metres": particle_radius * smoothing_length,
        "cube_size": cube_size,
        "voxel_size_metres": voxel_size,
        "surface_threshold": args.surface_threshold,
        "mesh_smoothing_iters": args.mesh_smoothing_iters,
        "normal_smoothing_iters": args.normal_smoothing_iters,
        "workers": args.workers,
        "threads_per_worker": threads_per_worker,
        "shoreline_mode": args.shoreline_mode,
        "shoreline_particles_ply": (
            str(args.shoreline_particles_ply.resolve())
            if args.shoreline_particles_ply
            else None
        ),
        "minimum_layers": args.minimum_layers,
        "shoreline_erosion_cells": args.shoreline_erosion_cells,
        "impact": [args.impact_x, args.impact_z, args.impact_radius],
        "particle_aabb_enabled": not args.no_particle_aabb,
        "particle_aabb_min": particle_aabb_min,
        "particle_aabb_max": particle_aabb_max,
        "particle_aabb_source": particle_aabb_source,
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
                "core_input_directory": str(particle_paths[0].parent),
                "secondary_handoff_directory": str(
                    args.output_directory.resolve() / "secondary_handoff"
                ),
                "frames": domain_frame_provenance,
                "conservation_rule": (
                    "per frame: core PLY source IDs plus secondary handoff "
                    "source IDs equal the selected stable body exactly"
                ),
            }
            if domain_contract is not None
            else None
        ),
        "selected_frames": args.frames,
    }

    manifest_path = args.output_directory / "splashsurf_manifest.json"
    existing_outputs = list(surface_directory.glob("surface_*_clipped.obj"))
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("configuration") != configuration and not args.force:
            raise RuntimeError(
                f"Existing cache manifest does not match this build: {manifest_path}. "
                "Use a new output directory or pass --force to replace every selected frame."
            )
    elif existing_outputs and not args.force:
        raise RuntimeError(
            f"{surface_directory} contains meshes without a cache manifest. "
            "Use a new output directory or pass --force after verifying the target."
        )

    write_manifest(
        manifest_path,
        configuration,
        {"complete": False, "completed_frames": 0, "total_frames": len(particle_paths)},
    )

    def build_frame(particle_path):
        frame_id = particle_path.stem.rsplit("_", 1)[-1]
        raw_path = raw_directory / f"surface_{frame_id}.obj"
        output_path = surface_directory / f"surface_{frame_id}_clipped.obj"
        temporary_path = surface_directory / f"surface_{frame_id}_clipped.tmp.obj"
        if output_path.is_file() and output_path.stat().st_size > 100_000 and not args.force:
            return {
                "frame_id": frame_id,
                "status": "manifest-matched-cache",
                "diagnostics": "manifest-matched-cache",
                "surface_file": output_path.name,
                "surface_sha256": sha256_file(output_path),
                "surface_bytes": output_path.stat().st_size,
            }

        reconstruct_command = [
            str(args.splashsurf),
            "reconstruct",
            str(particle_path),
            "-r",
            str(particle_radius),
            "-l",
            str(smoothing_length),
            "-c",
            str(cube_size),
            "-t",
            str(args.surface_threshold),
            "--mesh-smoothing-weights=on",
            "--mesh-smoothing-iters",
            str(args.mesh_smoothing_iters),
            "--mesh-cleanup=on",
            "--normals=on",
            "--normals-smoothing-iters",
            str(args.normal_smoothing_iters),
            "--check-mesh=off",
            "--num-threads",
            str(threads_per_worker),
        ]
        if not args.no_particle_aabb:
            reconstruct_command.extend(
                [
                    "--particle-aabb-min",
                    *map(str, particle_aabb_min),
                    "--particle-aabb-max",
                    *map(str, particle_aabb_max),
                ]
            )
        reconstruct_command.extend(["-o", str(raw_path)])
        if not (raw_path.is_file() and raw_path.stat().st_size > 100_000):
            run_checked(reconstruct_command)

        clip_command = [
            sys.executable,
            str(clip_script),
            str(raw_path),
            str(terrain_raster_path),
            str(temporary_path),
            "--particles-ply",
            str(particle_path),
            "--water-level",
            str(args.water_level),
            "--spacing",
            str(args.spacing),
            "--shoreline-mode",
            args.shoreline_mode,
            "--minimum-layers",
            str(args.minimum_layers),
            "--shoreline-erosion-cells",
            str(args.shoreline_erosion_cells),
            "--impact-x",
            str(args.impact_x),
            "--impact-z",
            str(args.impact_z),
            "--impact-radius",
            str(args.impact_radius),
            "--voxel-size",
            str(voxel_size),
        ]
        if args.shoreline_particles_ply is not None:
            clip_command.extend(
                ["--shoreline-particles-ply", str(args.shoreline_particles_ply)]
            )
        clip_output = run_checked(clip_command)
        temporary_path.replace(output_path)
        raw_path.unlink(missing_ok=True)
        return {
            "frame_id": frame_id,
            "status": "rebuilt",
            "diagnostics": clip_output,
            "surface_file": output_path.name,
            "surface_sha256": sha256_file(output_path),
            "surface_bytes": output_path.stat().st_size,
        }

    completed_count = 0
    completed_frames = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(build_frame, path): path for path in particle_paths}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            frame_id = result["frame_id"]
            completed_count += 1
            completed_frames.append(result)
            print(
                f"[splashsurf-sequence] {completed_count:03d}/{len(particle_paths):03d} "
                f"frame={frame_id} size={result['surface_bytes']} "
                f"{result['diagnostics']}",
                flush=True,
            )

    write_manifest(
        manifest_path,
        configuration,
        {
            "complete": True,
            "completed_frames": completed_count,
            "total_frames": len(particle_paths),
            "frames": sorted(completed_frames, key=lambda item: item["frame_id"]),
        },
    )
    print(f"SPLASHSURF_SEQUENCE={surface_directory} frames={len(particle_paths)}")


if __name__ == "__main__":
    main()
