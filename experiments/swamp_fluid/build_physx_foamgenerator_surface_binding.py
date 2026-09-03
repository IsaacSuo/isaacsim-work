"""Bind audited FoamGenerator foam parcels to arbitrary Splashsurf triangles."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.mesh_surface_sampler import TriangleSurfaceSampler


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("render_cache_directory", type=Path)
    parser.add_argument("surface_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--frame-list", nargs="+", required=True, type=int)
    parser.add_argument("--maximum-bind-distance", type=float, default=0.016)
    parser.add_argument("--minimum-opacity", type=float, default=0.02)
    args = parser.parse_args()
    cache_directory = args.render_cache_directory.resolve()
    surface_directory = args.surface_directory.resolve()
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    frames = sorted(set(args.frame_list))
    manifest_path = output_directory / "manifest.json"
    manifest = {
        "schema": "physx-foamgenerator-surface-binding/v1",
        "product": "physx_foamgenerator_surface_binding",
        "complete": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "frames": frames,
            "maximum_bind_distance_m": args.maximum_bind_distance,
            "minimum_opacity": args.minimum_opacity,
            "method": "exact closest oriented triangle on open Splashsurf mesh",
            "height_field_used": False,
            "finite_difference_velocity_used": False,
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
            "surface_sampler_sha256": sha256_file(Path(__file__).parent / "whitewater" / "mesh_surface_sampler.py"),
        },
        "samples": [],
    }
    atomic_json(manifest_path, manifest)
    try:
        for frame in frames:
            foam_path = cache_directory / "foam" / f"frame_{frame:04d}.npz"
            surface_path = surface_directory / f"surface_{frame:04d}_clipped.obj"
            with np.load(foam_path, allow_pickle=False) as cache:
                source = {name: np.asarray(cache[name]).copy() for name in cache.files}
            active = np.asarray(source["opacity"], dtype=np.float64) >= args.minimum_opacity
            active_rows = np.flatnonzero(active)
            positions = np.asarray(source["position"], dtype=np.float64)[active]
            sampler = TriangleSurfaceSampler(surface_path)
            closest, signed, distance, normals, triangle_ids = sampler.closest(positions)
            bound = distance <= args.maximum_bind_distance
            bound_rows = active_rows[bound]
            unbound_rows = active_rows[~bound]
            output_path = output_directory / f"binding_{frame:04d}.npz"
            atomic_npz(
                output_path,
                schema=np.asarray("physx-foamgenerator-surface-binding-frame/v1"),
                output_frame=np.int32(frame),
                source_frame=np.int32(int(np.asarray(source["source_frame"]))),
                id=np.asarray(source["id"], dtype=np.uint64)[bound_rows],
                source_row=bound_rows.astype(np.int32),
                source_position=np.asarray(source["position"], dtype=np.float32)[bound_rows],
                position=closest[bound].astype(np.float32),
                normal=normals[bound].astype(np.float32),
                signed_distance=signed[bound].astype(np.float32),
                distance=distance[bound].astype(np.float32),
                triangle_id=triangle_ids[bound].astype(np.int64),
                velocity=np.asarray(source["velocity"], dtype=np.float32)[bound_rows],
                opacity=np.asarray(source["opacity"], dtype=np.float32)[bound_rows],
                remaining_lifetime=np.asarray(source["remaining_lifetime"], dtype=np.float32)[bound_rows],
                birth_frame=np.asarray(source["birth_frame"], dtype=np.int32)[bound_rows],
                shape=np.asarray(source["shape"], dtype=np.float32)[bound_rows],
                unbound_id=np.asarray(source["id"], dtype=np.uint64)[unbound_rows],
                unbound_source_row=unbound_rows.astype(np.int32),
                unbound_distance=distance[~bound].astype(np.float32),
            )
            record = {
                "output_frame": frame,
                "source_frame": int(np.asarray(source["source_frame"])),
                "file": output_path.name,
                "bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
                "source_foam_cache": str(foam_path),
                "source_foam_cache_sha256": sha256_file(foam_path),
                "surface": str(surface_path),
                "surface_sha256": sha256_file(surface_path),
                "input_count": int(len(source["id"])),
                "active_count": int(np.count_nonzero(active)),
                "bound_count": int(np.count_nonzero(bound)),
                "unbound_count": int(np.count_nonzero(~bound)),
                "bound_fraction": float(np.mean(bound)) if len(bound) else 1.0,
                "distance_m": {
                    "median": float(np.median(distance)) if len(distance) else 0.0,
                    "p90": float(np.quantile(distance, 0.90)) if len(distance) else 0.0,
                    "p99": float(np.quantile(distance, 0.99)) if len(distance) else 0.0,
                    "maximum_bound": float(distance[bound].max(initial=0.0)),
                },
            }
            manifest["samples"].append(record)
            atomic_json(manifest_path, manifest)
            print(
                f"[surface-binding] frame={frame:04d} bound={record['bound_count']} "
                f"unbound={record['unbound_count']}",
                flush=True,
            )
        manifest["complete"] = True
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["summary"] = {
            "frames": len(manifest["samples"]),
            "maximum_bound": max(sample["bound_count"] for sample in manifest["samples"]),
            "maximum_unbound_fraction": max(1.0 - sample["bound_fraction"] for sample in manifest["samples"]),
        }
        atomic_json(manifest_path, manifest)
        print(json.dumps({"valid": True, **manifest["summary"]}, indent=2))
    except Exception as exc:
        atomic_json(
            output_directory / "OBSOLETE.json",
            {
                "schema": 1,
                "obsolete": True,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "reason": str(exc),
                "traceback": traceback.format_exc(),
                "completed_samples": len(manifest["samples"]),
            },
        )
        raise


if __name__ == "__main__":
    main()
