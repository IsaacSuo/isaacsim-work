"""Bind PhysX/FoamGenerator foam parcels to Splashsurf and build Plateau cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.mesh_surface_sampler import TriangleSurfaceSampler
from whitewater.plateau_cells import (
    FILM_DTYPE,
    NODE_DTYPE,
    PlateauCellModel,
    build_plateau_cells,
)
from whitewater.state_machine import splitmix64
from whitewater.surface_raft import RAFT_DTYPE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("render_cache_directory", type=Path)
    parser.add_argument("surface_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--frame-list", nargs="+", required=True, type=int)
    parser.add_argument("--maximum-bind-distance", type=float, default=0.016)
    parser.add_argument("--minimum-opacity", type=float, default=0.02)
    parser.add_argument("--radius-seed", type=int, default=20260902)
    parser.add_argument("--micro-fraction", type=float, default=0.70)
    parser.add_argument("--meso-fraction", type=float, default=0.25)
    parser.add_argument("--micro-radius-range", nargs=2, type=float, default=(0.00045, 0.00090))
    parser.add_argument("--meso-radius-range", nargs=2, type=float, default=(0.00090, 0.00180))
    parser.add_argument("--macro-radius-range", nargs=2, type=float, default=(0.00180, 0.00320))
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def deterministic_radii(ids: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    key = splitmix64(np.asarray(ids, dtype=np.uint64) ^ np.uint64(args.radius_seed))
    uniform = ((key >> np.uint64(11)).astype(np.float64) / float(1 << 53))
    secondary = splitmix64(key ^ np.uint64(0xA17F0A6D))
    within = ((secondary >> np.uint64(11)).astype(np.float64) / float(1 << 53))
    micro_end = args.micro_fraction
    meso_end = args.micro_fraction + args.meso_fraction
    radii = np.empty(len(ids), dtype=np.float64)
    classes = np.empty(len(ids), dtype=np.uint8)
    selections = (
        (uniform < micro_end, args.micro_radius_range, 0),
        ((uniform >= micro_end) & (uniform < meso_end), args.meso_radius_range, 1),
        (uniform >= meso_end, args.macro_radius_range, 2),
    )
    for selection, radius_range, radius_class in selections:
        low, high = map(float, radius_range)
        # Log interpolation avoids over-weighting the largest visible cells.
        radii[selection] = low * (high / low) ** within[selection]
        classes[selection] = radius_class
    return radii, classes


def make_raft(cache: dict, sampler: TriangleSurfaceSampler, args: argparse.Namespace):
    opacity = np.asarray(cache["opacity"], dtype=np.float64)
    active = opacity >= args.minimum_opacity
    ids = np.asarray(cache["id"], dtype=np.uint64)[active]
    positions = np.asarray(cache["position"], dtype=np.float64)[active]
    velocities = np.asarray(cache["velocity"], dtype=np.float64)[active]
    shape = np.asarray(cache["shape"], dtype=np.float64)[active]
    opacity = opacity[active]
    birth_frame = np.asarray(cache["birth_frame"], dtype=np.int64)[active]
    source_frame = int(np.asarray(cache["source_frame"]))

    closest, signed, distance, normals, triangle_ids = sampler.closest(positions)
    bound = distance <= args.maximum_bind_distance
    base_radius, radius_class = deterministic_radii(ids, args)
    # Opacity is a render-only birth/death gate.  Cubic scaling makes visual gas
    # volume continuous without claiming that FoamGenerator exported gas volume.
    effective_radius = base_radius * np.cbrt(np.clip(opacity, 0.0, 1.0))

    raft = np.zeros(int(np.count_nonzero(bound)), dtype=RAFT_DTYPE)
    raft["marker_id"] = ids[bound]
    raft["physical_radius"] = effective_radius[bound].astype(np.float32)
    raft["representative_count"] = 1.0
    raft["phase_volume"] = (
        (4.0 / 3.0) * np.pi * effective_radius[bound] ** 3
    ).astype(np.float32)
    raft["shape"] = shape[bound].astype(np.float32)
    raft["raw_anchor_position"] = positions[bound].astype(np.float32)
    raft["raft_position"] = closest[bound].astype(np.float32)
    raft["raft_velocity"] = velocities[bound].astype(np.float32)
    raft["state_age"] = np.maximum(0.0, source_frame - birth_frame[bound]) / 120.0
    raft["random_key"] = splitmix64(ids[bound] ^ np.uint64(args.radius_seed))
    raft["anchor_displacement"] = distance[bound].astype(np.float32)
    raft["support_value"] = np.clip(
        1.0 - distance[bound] / args.maximum_bind_distance, 0.0, 1.0
    ).astype(np.float32)
    order = np.argsort(raft["marker_id"])
    raft = raft[order]
    diagnostics = {
        "input_count": int(len(cache["id"])),
        "opacity_active_count": int(len(ids)),
        "bound_count": int(np.count_nonzero(bound)),
        "unbound_count": int(np.count_nonzero(~bound)),
        "bound_fraction": float(np.mean(bound)) if len(bound) else 1.0,
        "distance_m": {
            "maximum_bound": float(distance[bound].max(initial=0.0)),
            "median": float(np.median(distance)) if len(distance) else 0.0,
            "p90": float(np.quantile(distance, 0.90)) if len(distance) else 0.0,
            "p99": float(np.quantile(distance, 0.99)) if len(distance) else 0.0,
        },
        "radius_class_counts": {
            "micro": int(np.count_nonzero(bound & (radius_class == 0))),
            "meso": int(np.count_nonzero(bound & (radius_class == 1))),
            "macro": int(np.count_nonzero(bound & (radius_class == 2))),
        },
        "signed_distance_positive_fraction": float(np.mean(signed[bound] >= 0.0)) if np.any(bound) else 0.0,
        "gas_volume_m3": float(raft["phase_volume"].sum(dtype=np.float64)),
    }
    unbound = {
        "unbound_id": ids[~bound].astype(np.uint64),
        "unbound_position": positions[~bound].astype(np.float32),
        "unbound_velocity": velocities[~bound].astype(np.float32),
        "unbound_distance": distance[~bound].astype(np.float32),
        "unbound_nearest_position": closest[~bound].astype(np.float32),
        "unbound_nearest_normal": normals[~bound].astype(np.float32),
        "unbound_triangle_id": triangle_ids[~bound].astype(np.int64),
    }
    return raft, diagnostics, unbound


def main() -> None:
    args = parse_args()
    cache_directory = args.render_cache_directory.resolve()
    surface_directory = args.surface_directory.resolve()
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "manifest.json"
    frames = sorted(set(args.frame_list))
    if not frames:
        raise RuntimeError("No frames requested")
    if not 0.0 < args.micro_fraction < 1.0 or not 0.0 <= args.meso_fraction < 1.0:
        raise ValueError("Invalid radius mixture fractions")
    if args.micro_fraction + args.meso_fraction >= 1.0:
        raise ValueError("Micro and meso fractions leave no macro population")
    for radius_range in (args.micro_radius_range, args.meso_radius_range, args.macro_radius_range):
        if not 0.0 < radius_range[0] <= radius_range[1]:
            raise ValueError("Invalid radius range")

    model = PlateauCellModel()
    manifest = {
        "schema": "physx-foamgenerator-plateau/v1",
        "product": "physx_foamgenerator_surface_plateau",
        "complete": False,
        "created_utc": utc_now(),
        "configuration": {
            "frames": frames,
            "maximum_bind_distance_m": args.maximum_bind_distance,
            "minimum_opacity": args.minimum_opacity,
            "radius_seed": args.radius_seed,
            "radius_model": {
                "authority": "render-only calibrated distribution; FoamGenerator exports no bubble radius",
                "representative_count": 1.0,
                "micro_fraction": args.micro_fraction,
                "meso_fraction": args.meso_fraction,
                "macro_fraction": 1.0 - args.micro_fraction - args.meso_fraction,
                "micro_radius_range_m": list(args.micro_radius_range),
                "meso_radius_range_m": list(args.meso_radius_range),
                "macro_radius_range_m": list(args.macro_radius_range),
                "opacity_policy": "effective radius = calibrated radius * opacity^(1/3)",
            },
            "surface_binding": "exact closest triangle on oriented open Splashsurf mesh",
            "plateau_model": model.metadata(),
            "available_film_area": "sum(4*pi*effective_radius^2) of bound cells",
        },
        "inputs": {
            "render_cache_directory": str(cache_directory),
            "render_cache_manifest_sha256": sha256_file(cache_directory / "secondary_manifest.json"),
            "render_cache_audit_sha256": sha256_file(cache_directory / "audit_report.json"),
            "surface_directory": str(surface_directory),
        },
        "producer": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "plateau_module_sha256": sha256_file(Path(__file__).parent / "whitewater" / "plateau_cells.py"),
            "surface_sampler_module_sha256": sha256_file(Path(__file__).parent / "whitewater" / "mesh_surface_sampler.py"),
        },
        "samples": [],
    }
    atomic_json(manifest_path, manifest)
    previous_films = np.empty(0, dtype=FILM_DTYPE)
    previous_nodes = np.empty(0, dtype=NODE_DTYPE)
    previous_ruptured = np.empty(0, dtype=np.uint64)
    previous_frame = None
    try:
        for frame in frames:
            cache_path = cache_directory / "foam" / f"frame_{frame:04d}.npz"
            surface_path = surface_directory / f"surface_{frame:04d}_clipped.obj"
            if not cache_path.is_file() or not surface_path.is_file():
                raise FileNotFoundError(cache_path if not cache_path.is_file() else surface_path)
            with np.load(cache_path, allow_pickle=False) as source:
                cache = {name: np.asarray(source[name]).copy() for name in source.files}
            sampler = TriangleSurfaceSampler(surface_path)
            raft, binding, unbound = make_raft(cache, sampler, args)
            dt = (1.0 / 30.0) if previous_frame is None else (frame - previous_frame) / 30.0
            film_area = float(
                np.sum(4.0 * np.pi * raft["physical_radius"].astype(np.float64) ** 2)
            )
            topology, metrics = build_plateau_cells(
                raft,
                sampler,
                None,
                active_surface_film_area_m2=film_area,
                event_time=frame / 30.0,
                dt=dt,
                previous_films=previous_films,
                previous_nodes=previous_nodes,
                previous_ruptured_pairs=previous_ruptured,
                model=model,
            )
            output_path = output_directory / f"plateau_{frame:04d}.npz"
            atomic_npz(
                output_path,
                schema=np.asarray("physx-foamgenerator-plateau-frame/v1"),
                output_frame=np.int32(frame),
                source_frame=np.int32(int(np.asarray(cache["source_frame"]))),
                raft=raft,
                **unbound,
                **topology,
            )
            manifest["samples"].append(
                {
                    "output_frame": frame,
                    "source_frame": int(np.asarray(cache["source_frame"])),
                    "file": output_path.name,
                    "bytes": output_path.stat().st_size,
                    "sha256": sha256_file(output_path),
                    "source_cache": str(cache_path),
                    "source_cache_sha256": sha256_file(cache_path),
                    "surface": str(surface_path),
                    "surface_sha256": sha256_file(surface_path),
                    "binding": binding,
                    "topology": metrics,
                }
            )
            atomic_json(manifest_path, manifest)
            previous_films = topology["films"].copy()
            previous_nodes = topology["nodes"].copy()
            previous_ruptured = topology["ruptured_pair_ids"].copy()
            previous_frame = frame
            print(
                f"[plateau-adapter] frame={frame:04d} bound={len(raft)} "
                f"cells={len(topology['cells'])} films={len(topology['films'])} "
                f"nodes={len(topology['nodes'])}",
                flush=True,
            )
        manifest["complete"] = True
        manifest["completed_utc"] = utc_now()
        manifest["summary"] = {
            "frames": len(manifest["samples"]),
            "maximum_bound_cells": max(sample["binding"]["bound_count"] for sample in manifest["samples"]),
            "maximum_shared_films": max(sample["topology"]["shared_films"] for sample in manifest["samples"]),
            "maximum_plateau_nodes": max(sample["topology"]["plateau_nodes"] for sample in manifest["samples"]),
            "maximum_unbound_fraction": max(1.0 - sample["binding"]["bound_fraction"] for sample in manifest["samples"]),
            "maximum_gas_volume_residual_m3": max(
                abs(sample["topology"]["cell_gas_volume_m3"] - sample["topology"]["source_gas_volume_m3"])
                for sample in manifest["samples"]
            ),
            "maximum_analytic_geometry_volume_residual_m3": max(
                abs(sample["topology"]["analytic_geometry_volume_m3"] - sample["topology"]["source_gas_volume_m3"])
                for sample in manifest["samples"]
            ),
        }
        atomic_json(manifest_path, manifest)
        print(json.dumps({"valid": True, **manifest["summary"]}, indent=2))
    except Exception as exc:
        obsolete = {
            "schema": 1,
            "obsolete": True,
            "created_utc": utc_now(),
            "reason": str(exc),
            "traceback": traceback.format_exc(),
            "completed_samples": len(manifest["samples"]),
            "manifest": str(manifest_path),
        }
        atomic_json(output_directory / "OBSOLETE.json", obsolete)
        raise


if __name__ == "__main__":
    main()
