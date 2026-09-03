"""Build conservative surface-film patches from audited FoamGenerator parcels."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.foam_render_payload import FoamRenderModel, compose_foam_render_payload
from whitewater.mesh_surface_sampler import TriangleSurfaceSampler
from whitewater.state_machine import splitmix64
from whitewater.surface_foam import FOAM_DTYPE, REPELLENT_DTYPE


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("render_cache_directory", type=Path)
    parser.add_argument(
        "surface_binding_directory",
        type=Path,
        help=(
            "Completed physx_foamgenerator_surface_binding or "
            "physx_foamgenerator_surface_plateau product"
        ),
    )
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--frame-list", nargs="+", required=True, type=int)
    parser.add_argument("--particle-spacing", type=float, default=0.008)
    parser.add_argument("--support-area-scale", type=float, default=1.0)
    parser.add_argument("--initial-coverage", type=float, default=0.72)
    parser.add_argument("--drainage-seconds", type=float, default=2.5)
    parser.add_argument("--maximum-anisotropy", type=float, default=3.0)
    parser.add_argument("--anisotropy-speed", type=float, default=1.0)
    return parser.parse_args()


def load_bound_state(path: Path, product: str) -> dict[str, np.ndarray]:
    """Load the common, physically authoritative subset of either binding product."""
    with np.load(path, allow_pickle=False) as source:
        source_frame = int(np.asarray(source["source_frame"]))
        if product == "physx_foamgenerator_surface_binding":
            birth_frame = np.asarray(source["birth_frame"], dtype=np.int64)
            # Match the float32 state_age stored in the audited Plateau raft.
            age = (
                np.maximum(0.0, source_frame - birth_frame) / 120.0
            ).astype(np.float32)
            return {
                "id": np.asarray(source["id"], dtype=np.uint64).copy(),
                "position": np.asarray(source["position"], dtype=np.float32).copy(),
                "velocity": np.asarray(source["velocity"], dtype=np.float32).copy(),
                "age": age,
                "source_frame": np.asarray(source_frame, dtype=np.int32),
            }
        if product == "physx_foamgenerator_surface_plateau":
            raft = np.asarray(source["raft"]).copy()
            return {
                "id": np.asarray(raft["marker_id"], dtype=np.uint64),
                "position": np.asarray(raft["raft_position"], dtype=np.float32),
                "velocity": np.asarray(raft["raft_velocity"], dtype=np.float32),
                "age": np.asarray(raft["state_age"], dtype=np.float32),
                "source_frame": np.asarray(source_frame, dtype=np.int32),
            }
    raise RuntimeError(f"Unsupported surface-binding product: {product}")


def main() -> None:
    args = parse_args()
    if min(
        args.particle_spacing,
        args.support_area_scale,
        args.drainage_seconds,
        args.maximum_anisotropy,
        args.anisotropy_speed,
    ) <= 0.0 or not 0.0 < args.initial_coverage <= 1.0:
        raise ValueError("Invalid surface-payload parameters")
    cache_directory = args.render_cache_directory.resolve()
    binding_directory = args.surface_binding_directory.resolve()
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    binding_manifest_path = binding_directory / "manifest.json"
    binding_manifest = json.loads(binding_manifest_path.read_text(encoding="utf-8"))
    binding_product = binding_manifest.get("product")
    allowed_products = {
        "physx_foamgenerator_surface_binding",
        "physx_foamgenerator_surface_plateau",
    }
    if binding_product not in allowed_products:
        raise RuntimeError(f"Unsupported surface-binding product: {binding_product!r}")
    if binding_manifest.get("complete") is not True:
        raise RuntimeError("Surface-binding manifest is incomplete")
    binding_samples = {
        int(sample["output_frame"]): sample for sample in binding_manifest["samples"]
    }
    frames = sorted(set(args.frame_list))
    model = FoamRenderModel(
        micro_radius_m=0.0025,
        macro_radius_m=0.0080,
    )
    manifest_path = output_directory / "manifest.json"
    manifest = {
        "schema": "physx-foamgenerator-surface-payload/v2",
        "product": "physx_foamgenerator_surface_render_payload",
        "complete": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "frames": frames,
            "particle_spacing_m": args.particle_spacing,
            "support_area_scale": args.support_area_scale,
            "support_area_per_full_parcel_m2": args.particle_spacing**2 * args.support_area_scale,
            "support_area_authority": (
                "render sampling footprint calibrated from primary-liquid spacing; "
                "separate from Plateau gas volume and physical bubble radius"
            ),
            "initial_coverage": args.initial_coverage,
            "drainage_seconds": args.drainage_seconds,
            "maximum_anisotropy": args.maximum_anisotropy,
            "anisotropy_speed_m_per_s": args.anisotropy_speed,
            "payload_model": model.metadata(),
            "repellents": "none; no unsupported burst-hole events are invented",
        },
        "inputs": {
            "render_cache_directory": str(cache_directory),
            "render_cache_manifest_sha256": sha256_file(cache_directory / "secondary_manifest.json"),
            "surface_binding_product": binding_product,
            "surface_binding_directory": str(binding_directory),
            "surface_binding_manifest_sha256": sha256_file(binding_manifest_path),
        },
        "producer": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "payload_module_sha256": sha256_file(Path(__file__).parent / "whitewater" / "foam_render_payload.py"),
        },
        "samples": [],
    }
    atomic_json(manifest_path, manifest)
    try:
        for frame in frames:
            if frame not in binding_samples:
                raise RuntimeError(f"Surface binding has no frame {frame}")
            binding_sample = binding_samples[frame]
            binding_path = binding_directory / binding_sample["file"]
            foam_path = cache_directory / "foam" / f"frame_{frame:04d}.npz"
            surface_path = Path(binding_sample["surface"])
            bound_state = load_bound_state(binding_path, binding_product)
            with np.load(foam_path, allow_pickle=False) as foam_cache:
                source_ids = np.asarray(foam_cache["id"], dtype=np.uint64)
                source_opacity = np.asarray(foam_cache["opacity"], dtype=np.float64)
                source_remaining = np.asarray(
                    foam_cache["remaining_lifetime"], dtype=np.float64
                )
            source_index = {int(value): index for index, value in enumerate(source_ids)}
            indices = np.asarray([source_index[int(value)] for value in bound_state["id"]])
            opacity = source_opacity[indices]
            remaining = source_remaining[indices]
            support = (
                args.particle_spacing**2 * args.support_area_scale * opacity
            )
            age = bound_state["age"].astype(np.float64)
            coverage = (
                args.initial_coverage
                * np.exp(-age / args.drainage_seconds)
                * np.clip(remaining / max(1.0 / 30.0, args.drainage_seconds), 0.0, 1.0)
            )
            # The lifetime term only fades the last drainage interval.  Most
            # particles have remaining lifetime above drainage_seconds.
            coverage = np.clip(coverage, 0.0, 1.0)
            sampler = TriangleSurfaceSampler(surface_path)
            normals = sampler.sample_normal(bound_state["position"])
            velocity = bound_state["velocity"].astype(np.float64)
            tangent_velocity = velocity - np.sum(velocity * normals, axis=1)[:, None] * normals
            tangent_speed = np.linalg.norm(tangent_velocity, axis=1)
            principal = tangent_velocity.copy()
            valid = tangent_speed > 1.0e-8
            principal[valid] /= tangent_speed[valid, None]
            principal[~valid] = np.cross(normals[~valid], np.array((1.0, 0.0, 0.0)))
            fallback_length = np.linalg.norm(principal, axis=1)
            second_fallback = fallback_length <= 1.0e-8
            principal[second_fallback] = np.cross(
                normals[second_fallback], np.array((0.0, 0.0, 1.0))
            )
            principal /= np.maximum(np.linalg.norm(principal, axis=1), 1.0e-12)[:, None]
            anisotropy = 1.0 + (args.maximum_anisotropy - 1.0) * np.clip(
                tangent_speed / args.anisotropy_speed, 0.0, 1.0
            )
            foam = np.zeros(len(bound_state["id"]), dtype=FOAM_DTYPE)
            foam["id"] = bound_state["id"]
            foam["source_event_id"] = bound_state["id"]
            foam["source_kind"] = 2
            foam["birth_time"] = frame / 30.0 - age
            foam["age"] = age.astype(np.float32)
            foam["position"] = bound_state["position"]
            foam["velocity"] = bound_state["velocity"]
            foam["support_area"] = support.astype(np.float32)
            foam["film_area"] = (support * coverage).astype(np.float32)
            foam["drainage_time"] = args.drainage_seconds
            foam["principal_direction"] = principal.astype(np.float32)
            foam["anisotropy"] = anisotropy.astype(np.float32)
            foam["random_key"] = splitmix64(bound_state["id"] ^ np.uint64(0xF0A6A11))
            patches, rings, metrics = compose_foam_render_payload(
                foam,
                np.empty(0, dtype=REPELLENT_DTYPE),
                sampler,
                None,
                model,
            )
            output_path = output_directory / f"surface_payload_{frame:04d}.npz"
            atomic_npz(
                output_path,
                schema=np.asarray("physx-foamgenerator-surface-payload-frame/v1"),
                output_frame=np.int32(frame),
                source_frame=np.int32(int(binding_sample["source_frame"])),
                source_foam=foam,
                patches=patches,
                rings=rings,
            )
            sample = {
                "output_frame": frame,
                "source_frame": int(binding_sample["source_frame"]),
                "file": output_path.name,
                "bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
                "source_foam_cache": str(foam_path),
                "source_foam_cache_sha256": sha256_file(foam_path),
                "surface_binding_product": binding_product,
                "surface_binding": str(binding_path),
                "surface_binding_sha256": sha256_file(binding_path),
                "surface": str(surface_path),
                "surface_sha256": sha256_file(surface_path),
                "metrics": metrics,
                "support_area_m2": float(support.sum(dtype=np.float64)),
                "mean_coverage": float(np.mean(coverage)) if len(coverage) else 0.0,
                "maximum_anisotropy": float(anisotropy.max(initial=1.0)),
            }
            manifest["samples"].append(sample)
            atomic_json(manifest_path, manifest)
            print(
                f"[surface-payload] frame={frame:04d} patches={len(patches)} "
                f"support={sample['support_area_m2']:.6g}m2 "
                f"film={metrics['source_film_area_m2']:.6g}m2",
                flush=True,
            )
        manifest["complete"] = True
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["summary"] = {
            "frames": len(manifest["samples"]),
            "maximum_patches": max(sample["metrics"]["render_class_counts"]["bubble_cluster"] for sample in manifest["samples"]),
            "maximum_support_area_m2": max(sample["support_area_m2"] for sample in manifest["samples"]),
            "maximum_area_balance_residual_m2": max(abs(sample["metrics"]["area_balance_residual_m2"]) for sample in manifest["samples"]),
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
