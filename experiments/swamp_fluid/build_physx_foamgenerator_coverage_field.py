"""Build a temporally advected render-coverage mask from native PhysX foam.

The product routes every FoamGenerator foam ID explicitly.  Surface-bound
foam drives a scalar optical field on the complete Splashsurf mesh; active
unbound foam remains available as free-particle render data; lifecycle-faded
foam is explicitly recorded as culled.  No gas volume or physical film area
is inferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.foam_coverage_field import (
    COVERAGE_SOURCE_DTYPE,
    FoamCoverageFieldModel,
    coverage_to_optical_depth,
    rasterize_advected_coverage_field,
)
from whitewater.mesh_surface_sampler import TriangleSurfaceSampler


ROUTE_LIFECYCLE_CULLED = np.uint8(0)
ROUTE_SURFACE_COVERAGE = np.uint8(1)
ROUTE_FREE_FOAM_PARTICLE = np.uint8(2)


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
    parser.add_argument("surface_binding_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--frame-list", nargs="+", required=True, type=int)
    parser.add_argument("--kernel-support-radius", type=float, default=0.008)
    parser.add_argument("--initial-coverage", type=float, default=0.72)
    parser.add_argument("--history-decay-seconds", type=float, default=2.5)
    parser.add_argument("--history-weight", type=float, default=0.68)
    parser.add_argument("--history-remap-radius", type=float, default=0.006)
    parser.add_argument("--history-max-projection-distance", type=float, default=0.016)
    parser.add_argument("--route-switch-confirmation-frames", type=int, default=2)
    parser.add_argument("--route-max-hold-distance", type=float, default=0.024)
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    args = parse_args()
    if min(
        args.kernel_support_radius,
        args.history_decay_seconds,
        args.history_remap_radius,
        args.history_max_projection_distance,
        args.route_max_hold_distance,
    ) <= 0.0:
        raise ValueError("Coverage support, history, and route distances must be positive")
    if not 0.0 < args.initial_coverage < 1.0:
        raise ValueError("initial coverage must be within (0, 1)")
    if not 0.0 <= args.history_weight <= 1.0:
        raise ValueError("history weight must be within [0, 1]")
    if args.route_switch_confirmation_frames < 1:
        raise ValueError("route switch confirmation must be at least one frame")

    cache_directory = args.render_cache_directory.resolve()
    binding_directory = args.surface_binding_directory.resolve()
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)

    cache_manifest_path = cache_directory / "secondary_manifest.json"
    binding_manifest_path = binding_directory / "manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    binding_manifest = json.loads(binding_manifest_path.read_text(encoding="utf-8"))
    require(cache_manifest.get("valid") is True, "Render cache manifest is invalid")
    require(
        binding_manifest.get("complete") is True
        and binding_manifest.get("product") == "physx_foamgenerator_surface_binding",
        "A completed native surface-binding product is required",
    )
    binding_samples = {
        int(sample["output_frame"]): sample for sample in binding_manifest["samples"]
    }
    output_fps = int(cache_manifest["configuration"]["output_fps"])
    frames = sorted(set(args.frame_list))
    require(frames, "At least one output frame is required")

    model = FoamCoverageFieldModel(
        kernel_support_radius_m=args.kernel_support_radius,
        history_weight=args.history_weight,
        history_decay_seconds=args.history_decay_seconds,
        history_remap_radius_m=args.history_remap_radius,
        history_max_projection_distance_m=args.history_max_projection_distance,
    )
    manifest_path = output_directory / "manifest.json"
    manifest = {
        "schema": "physx-foamgenerator-water-surface-mask-prototype/v2",
        "product": "physx_foamgenerator_water_surface_mask_prototype",
        "complete": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "semantics": {
            "maturity": "prototype; not a mature flow-following foam system",
            "field_quantity": "dimensionless render optical depth and coverage",
            "physical_film_area_claimed": False,
            "physical_gas_volume_claimed": False,
            "coverage_integral_is_physical_area": False,
            "native_particle_classification_preserved": True,
            "unbound_foam_policy": "active unbound native foam is retained as free-particle render data",
            "lifecycle_cull_policy": "IDs below the audited binding opacity gate remain explicitly routed as culled",
            "geometry_policy": "attributes on complete liquid mesh; no solid connected membrane",
        },
        "render_only_calibration": {
            "kernel_support_radius_m": args.kernel_support_radius,
            "initial_coverage": args.initial_coverage,
            "history_decay_seconds": args.history_decay_seconds,
            "history_weight": args.history_weight,
            "history_remap_radius_m": args.history_remap_radius,
            "history_max_projection_distance_m": args.history_max_projection_distance,
            "route_switch_confirmation_frames": args.route_switch_confirmation_frames,
            "route_max_hold_distance_m": args.route_max_hold_distance,
            "route_switch_policy": (
                "render-route debounce only; native FoamGenerator foam classification "
                "is unchanged and every pending ID remains visible exactly once"
            ),
            "opacity_authority": (
                "existing render-cache lifecycle opacity; remaining_lifetime is preserved "
                "but is not multiplied a second time"
            ),
        },
        "transport": model.metadata(),
        "route_codes": {
            "0": "lifecycle_culled",
            "1": "surface_coverage",
            "2": "free_foam_particle",
        },
        "inputs": {
            "render_cache_directory": str(cache_directory),
            "render_cache_manifest_sha256": sha256_file(cache_manifest_path),
            "surface_binding_directory": str(binding_directory),
            "surface_binding_manifest_sha256": sha256_file(binding_manifest_path),
        },
        "producer": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "coverage_module_sha256": sha256_file(
                Path(__file__).parent / "whitewater" / "foam_coverage_field.py"
            ),
            "surface_sampler_sha256": sha256_file(
                Path(__file__).parent / "whitewater" / "mesh_surface_sampler.py"
            ),
        },
        "frames": frames,
        "samples": [],
    }
    atomic_json(manifest_path, manifest)

    previous_field = None
    previous_frame = None
    route_states = {}
    try:
        for frame in frames:
            require(frame in binding_samples, f"Surface binding has no frame {frame}")
            binding_sample = binding_samples[frame]
            cache_path = cache_directory / "foam" / f"frame_{frame:04d}.npz"
            binding_path = binding_directory / binding_sample["file"]
            surface_path = Path(binding_sample["surface"]).resolve()
            require(sha256_file(cache_path) == binding_sample["source_foam_cache_sha256"], f"Cache hash mismatch for frame {frame}")
            require(sha256_file(binding_path) == binding_sample["sha256"], f"Binding hash mismatch for frame {frame}")
            require(sha256_file(surface_path) == binding_sample["surface_sha256"], f"Surface hash mismatch for frame {frame}")

            with np.load(cache_path, allow_pickle=False) as loaded:
                cache = {name: np.asarray(loaded[name]).copy() for name in loaded.files}
            with np.load(binding_path, allow_pickle=False) as loaded:
                binding = {name: np.asarray(loaded[name]).copy() for name in loaded.files}
            ids = np.asarray(cache["id"], dtype=np.int64)
            require(len(np.unique(ids)) == len(ids), f"Duplicate native foam IDs in frame {frame}")
            require(np.array_equal(binding["id"].astype(np.int64), ids[binding["source_row"]]), f"Bound source-row mismatch in frame {frame}")
            require(np.array_equal(binding["unbound_id"].astype(np.int64), ids[binding["unbound_source_row"]]), f"Unbound source-row mismatch in frame {frame}")
            active_rows = np.sort(
                np.concatenate(
                    (
                        np.asarray(binding["source_row"], dtype=np.int64),
                        np.asarray(binding["unbound_source_row"], dtype=np.int64),
                    )
                )
            )
            require(len(np.unique(active_rows)) == len(active_rows), f"Bound/unbound routes overlap in frame {frame}")
            require(np.all((active_rows >= 0) & (active_rows < len(ids))), f"Invalid active source row in frame {frame}")
            inactive_mask = np.ones(len(ids), dtype=bool)
            inactive_mask[active_rows] = False
            inactive_rows = np.flatnonzero(inactive_mask)

            raw_route = np.full(
                len(ids), ROUTE_LIFECYCLE_CULLED, dtype=np.uint8
            )
            raw_bound_rows = np.asarray(binding["source_row"], dtype=np.int64)
            raw_unbound_rows = np.asarray(
                binding["unbound_source_row"], dtype=np.int64
            )
            raw_route[raw_bound_rows] = ROUTE_SURFACE_COVERAGE
            raw_route[raw_unbound_rows] = ROUTE_FREE_FOAM_PARTICLE
            raw_surface_distance = np.full(len(ids), np.inf, dtype=np.float32)
            raw_surface_distance[raw_bound_rows] = np.asarray(
                binding["distance"], dtype=np.float32
            )
            raw_surface_distance[raw_unbound_rows] = np.asarray(
                binding["unbound_distance"], dtype=np.float32
            )
            route = raw_route.copy()
            active_id_set = set(map(int, ids[active_rows]))
            route_states = {
                marker_id: state
                for marker_id, state in route_states.items()
                if marker_id in active_id_set
            }
            pending_count = 0
            held_count = 0
            forced_distance_switch_count = 0
            for row in active_rows:
                marker_id = int(ids[row])
                desired = int(raw_route[row])
                state = route_states.get(marker_id)
                if state is None:
                    state = {"route": desired, "pending": desired, "count": 0}
                elif desired == state["route"]:
                    state["pending"] = desired
                    state["count"] = 0
                else:
                    force_surface_exit = (
                        state["route"] == int(ROUTE_SURFACE_COVERAGE)
                        and desired == int(ROUTE_FREE_FOAM_PARTICLE)
                        and float(raw_surface_distance[row])
                        > args.route_max_hold_distance
                    )
                    if force_surface_exit:
                        state["route"] = desired
                        state["pending"] = desired
                        state["count"] = 0
                        forced_distance_switch_count += 1
                    elif desired == state["pending"]:
                        state["count"] += 1
                    else:
                        state["pending"] = desired
                        state["count"] = 1
                    if (
                        not force_surface_exit
                        and state["count"] >= args.route_switch_confirmation_frames
                    ):
                        state["route"] = desired
                        state["count"] = 0
                    elif not force_surface_exit:
                        pending_count += 1
                        held_count += 1
                route[row] = np.uint8(state["route"])
                route_states[marker_id] = state

            sampler = TriangleSurfaceSampler(surface_path)
            vertices = np.asarray(sampler.mesh.vertices, dtype=np.float64)
            triangles = np.asarray(sampler.mesh.faces, dtype=np.int64)
            coverage_rows = np.flatnonzero(route == ROUTE_SURFACE_COVERAGE)
            sources = np.zeros(len(coverage_rows), dtype=COVERAGE_SOURCE_DTYPE)
            sources["id"] = ids[coverage_rows].astype(np.uint64)
            sources["source_row"] = coverage_rows.astype(np.int32)
            binding_index_by_source_row = np.full(len(ids), -1, dtype=np.int64)
            binding_index_by_source_row[raw_bound_rows] = np.arange(
                len(raw_bound_rows), dtype=np.int64
            )
            raw_bound_source = raw_route[coverage_rows] == ROUTE_SURFACE_COVERAGE
            raw_binding_indices = binding_index_by_source_row[
                coverage_rows[raw_bound_source]
            ]
            sources["position"][raw_bound_source] = np.asarray(
                binding["position"], dtype=np.float32
            )[raw_binding_indices]
            sources["normal"][raw_bound_source] = np.asarray(
                binding["normal"], dtype=np.float32
            )[raw_binding_indices]
            sources["anchor_face_index"][raw_bound_source] = np.asarray(
                binding["triangle_id"], dtype=np.int64
            )[raw_binding_indices]
            sources["anchor_authority"][raw_bound_source] = 0
            held_surface = ~raw_bound_source
            if np.any(held_surface):
                held_rows = coverage_rows[held_surface]
                require(
                    np.all(
                        raw_surface_distance[held_rows]
                        <= args.route_max_hold_distance + 1.0e-9
                    ),
                    f"Route debounce held foam beyond maximum distance in frame {frame}",
                )
                closest, _signed, _distance, normals, triangle_ids = sampler.closest(
                    np.asarray(cache["position"], dtype=np.float64)[held_rows]
                )
                sources["position"][held_surface] = closest.astype(np.float32)
                sources["normal"][held_surface] = normals.astype(np.float32)
                sources["anchor_face_index"][held_surface] = triangle_ids
                sources["anchor_authority"][held_surface] = 1
            native_velocity = np.asarray(cache["velocity"], dtype=np.float64)[
                coverage_rows
            ]
            source_normal = sources["normal"].astype(np.float64)
            source_normal /= np.maximum(
                np.linalg.norm(source_normal, axis=1), 1.0e-12
            )[:, None]
            transport_velocity = native_velocity - np.sum(
                native_velocity * source_normal, axis=1
            )[:, None] * source_normal
            sources["native_velocity"] = native_velocity.astype(np.float32)
            sources["velocity"] = transport_velocity.astype(np.float32)
            sources["opacity"] = np.asarray(cache["opacity"], dtype=np.float32)[
                coverage_rows
            ]
            sources["remaining_lifetime"] = np.asarray(
                cache["remaining_lifetime"], dtype=np.float32
            )[coverage_rows]
            sources["kernel_support_radius"] = np.float32(args.kernel_support_radius)
            target_coverage = np.clip(
                args.initial_coverage * sources["opacity"].astype(np.float64),
                0.0,
                1.0 - 1.0e-7,
            )
            sources["instantaneous_target_optical_depth"] = coverage_to_optical_depth(
                target_coverage
            ).astype(np.float32)
            sources["target_optical_depth"] = sources[
                "instantaneous_target_optical_depth"
            ]
            dt = (
                1.0 / output_fps
                if previous_frame is None
                else (frame - previous_frame) / output_fps
            )
            history_projection = None
            if previous_field is not None:
                predicted = (
                    previous_field["active_face_centres"].astype(np.float64)
                    + previous_field["active_face_velocity"].astype(np.float64) * dt
                )
                closest, _signed, distance, normals, triangle_ids = sampler.closest(
                    predicted
                )
                history_projection = {
                    "position": closest,
                    "normal": normals,
                    "face_index": triangle_ids,
                    "distance": distance,
                }
            field, metrics = rasterize_advected_coverage_field(
                vertices,
                triangles,
                sources,
                previous=previous_field,
                history_projection=history_projection,
                dt=dt,
                model=model,
            )
            require(
                metrics["source_without_surface_support_count"] == 0,
                f"A bound foam source failed to reach its anchor triangle in frame {frame}",
            )

            free_rows = np.flatnonzero(route == ROUTE_FREE_FOAM_PARTICLE)
            output_path = output_directory / f"coverage_field_{frame:04d}.npz"
            atomic_npz(
                output_path,
                schema=np.asarray("physx-foamgenerator-water-surface-mask-frame/v2"),
                output_frame=np.int32(frame),
                source_frame=np.int32(int(np.asarray(cache["source_frame"]))),
                surface_vertex_count=np.int32(len(vertices)),
                surface_face_count=np.int32(len(triangles)),
                routing_id=ids.astype(np.int32),
                routing_source_row=np.arange(len(ids), dtype=np.int32),
                raw_routing_code=raw_route,
                raw_routing_surface_distance=raw_surface_distance,
                routing_code=route,
                coverage_sources=sources,
                free_foam_id=ids[free_rows].astype(np.int32),
                free_foam_source_row=free_rows.astype(np.int32),
                free_foam_position=np.asarray(cache["position"], dtype=np.float32)[free_rows],
                free_foam_velocity=np.asarray(cache["velocity"], dtype=np.float32)[free_rows],
                free_foam_radius=np.asarray(cache["radius"], dtype=np.float32)[free_rows],
                free_foam_opacity=np.asarray(cache["opacity"], dtype=np.float32)[free_rows],
                free_foam_shape=np.asarray(cache["shape"], dtype=np.float32)[free_rows],
                lifecycle_culled_id=ids[inactive_rows].astype(np.int32),
                **field,
            )
            record = {
                "output_frame": frame,
                "source_frame": int(np.asarray(cache["source_frame"])),
                "file": output_path.name,
                "bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
                "source_foam_cache": str(cache_path),
                "source_foam_cache_sha256": sha256_file(cache_path),
                "surface_binding": str(binding_path),
                "surface_binding_sha256": sha256_file(binding_path),
                "surface": str(surface_path),
                "surface_sha256": sha256_file(surface_path),
                "inventory": {
                    "native_foam": int(len(ids)),
                    "surface_coverage": int(np.count_nonzero(route == ROUTE_SURFACE_COVERAGE)),
                    "free_foam_particle": int(np.count_nonzero(route == ROUTE_FREE_FOAM_PARTICLE)),
                    "lifecycle_culled": int(np.count_nonzero(route == ROUTE_LIFECYCLE_CULLED)),
                    "missing": 0,
                    "duplicated": 0,
                },
                "raw_inventory": {
                    "surface_coverage": int(
                        np.count_nonzero(raw_route == ROUTE_SURFACE_COVERAGE)
                    ),
                    "free_foam_particle": int(
                        np.count_nonzero(raw_route == ROUTE_FREE_FOAM_PARTICLE)
                    ),
                    "lifecycle_culled": int(
                        np.count_nonzero(raw_route == ROUTE_LIFECYCLE_CULLED)
                    ),
                },
                "route_debounce": {
                    "pending_count": int(pending_count),
                    "held_count": int(held_count),
                    "held_surface_anchor_count": int(np.count_nonzero(held_surface)),
                    "forced_distance_switch_count": int(
                        forced_distance_switch_count
                    ),
                    "maximum_held_surface_distance_m": float(
                        raw_surface_distance[coverage_rows[held_surface]].max(
                            initial=0.0
                        )
                    ),
                },
                "field_metrics": metrics,
                "temporal_metrics": {
                    "history_advected": previous_field is not None,
                    "history_projection_distance_m": {
                        "median": float(np.median(history_projection["distance"]))
                        if history_projection is not None
                        and len(history_projection["distance"])
                        else 0.0,
                        "p95": float(
                            np.quantile(history_projection["distance"], 0.95)
                        )
                        if history_projection is not None
                        and len(history_projection["distance"])
                        else 0.0,
                        "maximum": float(
                            history_projection["distance"].max(initial=0.0)
                        )
                        if history_projection is not None
                        else 0.0,
                    },
                },
            }
            manifest["samples"].append(record)
            atomic_json(manifest_path, manifest)
            previous_field = field
            previous_frame = frame
            print(
                f"[coverage-field] frame={frame:04d} bound={record['inventory']['surface_coverage']} "
                f"free={record['inventory']['free_foam_particle']} "
                f"culled={record['inventory']['lifecycle_culled']} "
                f"faces={metrics['active_face_count']}",
                flush=True,
            )

        manifest["complete"] = True
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["summary"] = {
            "frames": len(manifest["samples"]),
            "total_native_foam_states": int(
                sum(sample["inventory"]["native_foam"] for sample in manifest["samples"])
            ),
            "total_surface_coverage_states": int(
                sum(sample["inventory"]["surface_coverage"] for sample in manifest["samples"])
            ),
            "total_free_foam_states": int(
                sum(sample["inventory"]["free_foam_particle"] for sample in manifest["samples"])
            ),
            "total_lifecycle_culled_states": int(
                sum(sample["inventory"]["lifecycle_culled"] for sample in manifest["samples"])
            ),
            "missing_or_duplicated_states": 0,
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
