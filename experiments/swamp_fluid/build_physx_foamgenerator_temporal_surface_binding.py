"""Build temporally coherent surface anchors for native PhysX foam markers.

The product is deliberately render-only.  It preserves the authoritative
PhysX marker ID, position, velocity and lifetime, but consumes the native
velocity to advect a stable visual anchor over a changing Splashsurf mesh.
No position finite differences are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from whitewater.mesh_surface_sampler import TriangleSurfaceSampler


WORLD_UP = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)


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


def normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lengths = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(lengths, 1.0e-12)


def canonical_new_normals(normals: np.ndarray) -> np.ndarray:
    normals = normalized(normals)
    flip = np.einsum("ij,j->i", normals, WORLD_UP) < 0.0
    normals[flip] *= -1.0
    return normals


def orient_to_previous(normals: np.ndarray, previous: np.ndarray) -> np.ndarray:
    normals = normalized(normals)
    flip = np.einsum("ij,ij->i", normals, previous) < 0.0
    normals[flip] *= -1.0
    return normals


def limited_normal_update(
    previous: np.ndarray,
    target: np.ndarray,
    blend: float,
    maximum_step_degrees: float,
) -> np.ndarray:
    """Normalized interpolation with an explicit angular step limit."""

    previous = normalized(previous)
    target = orient_to_previous(target, previous)
    dot = np.clip(np.einsum("ij,ij->i", previous, target), -1.0, 1.0)
    angles = np.arccos(dot)
    requested = angles * float(blend)
    maximum = math.radians(float(maximum_step_degrees))
    step = np.minimum(requested, maximum)
    result = previous.copy()
    nonzero = angles > 1.0e-8
    if np.any(nonzero):
        fraction = step[nonzero] / angles[nonzero]
        sin_angle = np.sin(angles[nonzero])
        near_singular = np.abs(sin_angle) < 1.0e-8
        weights_previous = np.sin((1.0 - fraction) * angles[nonzero]) / np.maximum(
            sin_angle, 1.0e-12
        )
        weights_target = np.sin(fraction * angles[nonzero]) / np.maximum(
            sin_angle, 1.0e-12
        )
        interpolated = (
            weights_previous[:, None] * previous[nonzero]
            + weights_target[:, None] * target[nonzero]
        )
        if np.any(near_singular):
            interpolated[near_singular] = (
                (1.0 - fraction[near_singular, None]) * previous[nonzero][near_singular]
                + fraction[near_singular, None] * target[nonzero][near_singular]
            )
        result[nonzero] = normalized(interpolated)
    return normalized(result)


def stable_unit_float(marker_id: int, salt: int = 0) -> float:
    value = (int(marker_id) + int(salt)) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 31
    return float(value & 0x00FFFFFF) / 16777216.0


def initial_tangent(normal: np.ndarray, marker_id: int) -> np.ndarray:
    normal = normalized(np.asarray(normal, dtype=np.float64).reshape((1, 3)))[0]
    reference = WORLD_UP if abs(float(np.dot(normal, WORLD_UP))) < 0.85 else np.asarray(
        (1.0, 0.0, 0.0), dtype=np.float64
    )
    first = reference - float(np.dot(reference, normal)) * normal
    first /= max(float(np.linalg.norm(first)), 1.0e-12)
    second = np.cross(normal, first)
    angle = 2.0 * math.pi * stable_unit_float(marker_id, 0x9E3779B97F4A7C15)
    return math.cos(angle) * first + math.sin(angle) * second


def transported_tangent(
    previous_tangent: np.ndarray, current_normal: np.ndarray, marker_id: int
) -> np.ndarray:
    tangent = previous_tangent - float(np.dot(previous_tangent, current_normal)) * current_normal
    length = float(np.linalg.norm(tangent))
    if length <= 1.0e-8:
        return initial_tangent(current_normal, marker_id)
    tangent /= length
    if float(np.dot(tangent, previous_tangent)) < 0.0:
        tangent *= -1.0
    return tangent


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"median": 0.0, "p90": 0.0, "p99": 0.0, "maximum": 0.0}
    return {
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def read_source(path: Path, minimum_opacity: float) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        source = {name: np.asarray(cache[name]).copy() for name in cache.files}
    active = np.asarray(source["opacity"], dtype=np.float64) >= minimum_opacity
    source["active_rows"] = np.flatnonzero(active).astype(np.int32)
    return source


def continuous_surface_density(
    positions: np.ndarray,
    weights: np.ndarray,
    support_length: float,
    cutoff_multiplier: float,
    optical_scale: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return a continuous render-only crowding field in O(N log N + pairs)."""

    positions = np.asarray(positions, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    raw = np.zeros(len(positions), dtype=np.float64)
    if len(positions) < 2:
        return raw.copy(), raw, 0
    cutoff = float(support_length) * float(cutoff_multiplier)
    tree = cKDTree(positions)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")
    if len(pairs):
        delta = positions[pairs[:, 0]] - positions[pairs[:, 1]]
        distance_squared = np.einsum("ij,ij->i", delta, delta)
        kernel = np.exp(-distance_squared / (support_length * support_length))
        np.add.at(raw, pairs[:, 0], kernel * weights[pairs[:, 1]])
        np.add.at(raw, pairs[:, 1], kernel * weights[pairs[:, 0]])
    density = 1.0 - np.exp(-raw / float(optical_scale))
    return np.clip(density, 0.0, 1.0), raw, int(len(pairs))


def triangle_curvature_proxy_degrees(
    sampler: TriangleSurfaceSampler,
    triangle_ids: np.ndarray,
    reference_normals: np.ndarray,
) -> np.ndarray:
    """Maximum vertex-normal deviation on each hit triangle."""

    triangle_ids = np.asarray(triangle_ids, dtype=np.int64)
    if not len(triangle_ids):
        return np.empty(0, dtype=np.float64)
    vertex_ids = sampler.mesh.faces[triangle_ids]
    vertex_normals = np.asarray(sampler.mesh.vertex_normals[vertex_ids], dtype=np.float64)
    references = normalized(reference_normals)
    dots = np.einsum("nij,nj->ni", vertex_normals, references)
    dots = np.abs(np.clip(dots, -1.0, 1.0))
    return np.degrees(np.arccos(np.min(dots, axis=1)))


def curvature_radius_scale(
    curvature_degrees: float,
    full_radius_degrees: float,
    minimum_radius_degrees: float,
    minimum_scale: float,
) -> float:
    if curvature_degrees <= full_radius_degrees:
        return 1.0
    if curvature_degrees >= minimum_radius_degrees:
        return float(minimum_scale)
    unit = (curvature_degrees - full_radius_degrees) / (
        minimum_radius_degrees - full_radius_degrees
    )
    unit = unit * unit * (3.0 - 2.0 * unit)
    return float(1.0 + unit * (minimum_scale - 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("render_cache_directory", type=Path)
    parser.add_argument("surface_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--frame-list", nargs="+", required=True, type=int)
    parser.add_argument("--entry-distance", type=float, default=0.012)
    parser.add_argument("--maximum-surface-distance", type=float, default=0.016)
    parser.add_argument("--minimum-opacity", type=float, default=0.02)
    parser.add_argument("--normal-blend", type=float, default=0.55)
    parser.add_argument("--maximum-normal-candidate-degrees", type=float, default=70.0)
    parser.add_argument("--maximum-normal-step-degrees", type=float, default=30.0)
    parser.add_argument("--quiet-speed-maximum", type=float, default=0.25)
    parser.add_argument("--maximum-quiet-correction-change", type=float, default=0.004)
    parser.add_argument("--prediction-score-scale", type=float, default=0.012)
    parser.add_argument("--prediction-score-weight", type=float, default=0.70)
    parser.add_argument("--normal-score-weight", type=float, default=0.35)
    parser.add_argument("--density-support-length", type=float, default=0.006)
    parser.add_argument("--density-cutoff-multiplier", type=float, default=3.0)
    parser.add_argument("--density-optical-scale", type=float, default=7.5)
    parser.add_argument("--density-ema-alpha", type=float, default=0.45)
    parser.add_argument("--density-maximum-change", type=float, default=0.18)
    parser.add_argument("--cluster-radius-scale", type=float, default=2.25)
    parser.add_argument("--cluster-radius-minimum", type=float, default=0.004)
    parser.add_argument("--cluster-radius-maximum", type=float, default=0.007)
    parser.add_argument("--cell-diameter-minimum", type=float, default=0.00075)
    parser.add_argument("--cell-diameter-maximum", type=float, default=0.00125)
    parser.add_argument("--curvature-full-radius-degrees", type=float, default=20.0)
    parser.add_argument("--curvature-minimum-radius-degrees", type=float, default=55.0)
    parser.add_argument("--curvature-minimum-radius-scale", type=float, default=0.55)
    args = parser.parse_args()

    cache_directory = args.render_cache_directory.resolve()
    surface_directory = args.surface_directory.resolve()
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    frames = sorted(set(args.frame_list))
    if not frames:
        raise ValueError("At least one frame is required")
    if any(b != a + 1 for a, b in zip(frames, frames[1:])):
        raise ValueError("Temporal binding requires a consecutive output-frame list")
    if not 0.0 < args.entry_distance < args.maximum_surface_distance:
        raise ValueError("entry-distance must be positive and below maximum-surface-distance")

    secondary_manifest_path = cache_directory / "secondary_manifest.json"
    secondary_manifest = json.loads(secondary_manifest_path.read_text(encoding="utf-8"))
    source_fps = float(secondary_manifest["configuration"]["source_fps"])
    manifest_path = output_directory / "manifest.json"
    script_path = Path(__file__).resolve()
    sampler_path = Path(__file__).parent / "whitewater" / "mesh_surface_sampler.py"
    manifest = {
        "schema": "physx-foamgenerator-temporal-surface-binding/v2",
        "product": "physx_foamgenerator_temporal_surface_binding",
        "complete": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "frames": frames,
            "entry_distance_m": args.entry_distance,
            "maximum_surface_distance_m": args.maximum_surface_distance,
            "minimum_opacity": args.minimum_opacity,
            "source_fps": source_fps,
            "method": "two-pass stable-ID native-velocity predicted anchors on arbitrary Splashsurf triangles",
            "entry_policy": "two consecutive near-surface samples; first sample is a complementary 50/50 crossfade",
            "exit_policy": "surface weight is zero immediately beyond the absolute maximum distance; eligible preceding sample may pre-fade to 50 percent",
            "normal_policy": {
                "persistent_orientation_flips_allowed": False,
                "blend": args.normal_blend,
                "maximum_candidate_degrees": args.maximum_normal_candidate_degrees,
                "maximum_step_degrees": args.maximum_normal_step_degrees,
            },
            "anchor_continuity_gate": {
                "quiet_native_speed_maximum_m_per_s": args.quiet_speed_maximum,
                "maximum_quiet_projection_correction_change_m": args.maximum_quiet_correction_change,
                "failure_policy": "route marker to complementary free wet-film representation",
            },
            "candidate_score": {
                "source_distance_weight": 1.0,
                "prediction_residual_weight": args.prediction_score_weight,
                "normal_angle_weight": args.normal_score_weight,
                "prediction_scale_m": args.prediction_score_scale,
            },
            "continuous_density": {
                "method": "cKDTree Gaussian pair kernel; no integer-neighbor thresholds",
                "support_length_m": args.density_support_length,
                "cutoff_multiplier": args.density_cutoff_multiplier,
                "normalization": "1-exp(-raw_kernel_sum/optical_scale)",
                "optical_scale": args.density_optical_scale,
                "stable_id_ema_alpha": args.density_ema_alpha,
                "maximum_change_per_output_frame": args.density_maximum_change,
                "physical_density_claimed": False,
                "render_only_calibration": True,
            },
            "microbubble_scale": {
                "cluster_radius_policy": "fixed at first active stable-ID sample; independent of instantaneous density",
                "cluster_radius_scale_from_native_render_radius": args.cluster_radius_scale,
                "base_cluster_radius_range_before_curvature_lod_m": [
                    args.cluster_radius_minimum,
                    args.cluster_radius_maximum,
                ],
                "effective_cluster_radius_minimum_m": (
                    args.cluster_radius_minimum * args.curvature_minimum_radius_scale
                ),
                "cell_diameter_range_m": [
                    args.cell_diameter_minimum,
                    args.cell_diameter_maximum,
                ],
                "cell_diameter_policy": "fixed stable-ID hash; independent of instantaneous density",
                "render_only_calibration": True,
                "physical_bubble_radius_claimed": False,
            },
            "surface_curvature_lod": {
                "proxy": "maximum oriented vertex-normal deviation on the raw closest triangle",
                "aggregation": "maximum over every active sample of the stable marker ID in the requested offline sequence",
                "full_radius_below_degrees": args.curvature_full_radius_degrees,
                "minimum_radius_above_degrees": args.curvature_minimum_radius_degrees,
                "minimum_radius_scale": args.curvature_minimum_radius_scale,
                "radius_is_constant_over_sequence": True,
                "render_only_calibration": True,
            },
            "render_only_calibration": True,
            "finite_difference_velocity_used": False,
            "native_velocity_stored": True,
            "native_velocity_consumed": True,
        },
        "inputs": {
            "render_cache_directory": str(cache_directory),
            "render_cache_manifest_sha256": sha256_file(secondary_manifest_path),
            "render_cache_audit_sha256": sha256_file(cache_directory / "audit_report.json"),
            "surface_directory": str(surface_directory),
        },
        "producer": {
            "script": str(script_path),
            "script_sha256": sha256_file(script_path),
            "surface_sampler_sha256": sha256_file(sampler_path),
        },
        "samples": [],
    }
    atomic_json(manifest_path, manifest)

    try:
        # Pass one records authoritative source samples and raw geometric
        # distances.  Route hysteresis can then use one-frame look-ahead
        # without deriving a velocity from positions.
        raw_frames: list[dict] = []
        for frame in frames:
            foam_path = cache_directory / "foam" / f"frame_{frame:04d}.npz"
            surface_path = surface_directory / f"surface_{frame:04d}_clipped.obj"
            source = read_source(foam_path, args.minimum_opacity)
            rows = source["active_rows"]
            positions = np.asarray(source["position"], dtype=np.float64)[rows]
            if len(positions):
                sampler = TriangleSurfaceSampler(surface_path)
                closest, signed, distance, normal, triangle_id = sampler.closest(positions)
                normal = canonical_new_normals(normal)
                curvature = triangle_curvature_proxy_degrees(
                    sampler, triangle_id, normal
                )
            else:
                # A production sequence legitimately contains pre-impact and
                # post-lifetime frames with no active foam markers.  Trimesh's
                # closest-point query is undefined for an empty query batch,
                # and loading a multi-million-face surface in this case is
                # unnecessary.  Preserve the frame with correctly shaped,
                # typed empty arrays so time alignment remains exact.
                closest = np.empty((0, 3), dtype=np.float64)
                signed = np.empty(0, dtype=np.float64)
                distance = np.empty(0, dtype=np.float64)
                normal = np.empty((0, 3), dtype=np.float64)
                triangle_id = np.empty(0, dtype=np.int64)
                curvature = np.empty(0, dtype=np.float64)
            raw_frames.append(
                {
                    "frame": frame,
                    "foam_path": foam_path,
                    "surface_path": surface_path,
                    "source": source,
                    "rows": rows,
                    "id": np.asarray(source["id"], dtype=np.int64)[rows],
                    "position": positions,
                    "velocity": np.asarray(source["velocity"], dtype=np.float64)[rows],
                    "closest": closest,
                    "signed": signed,
                    "distance": distance,
                    "normal": normal,
                    "triangle_id": triangle_id,
                    "curvature_degrees": curvature,
                }
            )
            print(f"[temporal-binding:measure] frame={frame:04d} active={len(rows)}", flush=True)

        lifetime_curvature: dict[int, float] = {}
        for raw in raw_frames:
            for marker_id_value, curvature in zip(raw["id"], raw["curvature_degrees"]):
                marker_id = int(marker_id_value)
                lifetime_curvature[marker_id] = max(
                    lifetime_curvature.get(marker_id, 0.0), float(curvature)
                )

        state: dict[int, dict] = {}
        for frame_offset, raw in enumerate(raw_frames):
            frame = int(raw["frame"])
            source = raw["source"]
            rows = raw["rows"]
            ids = raw["id"]
            positions = raw["position"]
            velocities = raw["velocity"]
            id_to_index = {int(marker_id): index for index, marker_id in enumerate(ids)}
            next_raw = raw_frames[frame_offset + 1] if frame_offset + 1 < len(raw_frames) else None
            next_distance = {}
            if next_raw is not None:
                next_distance = {
                    int(marker_id): float(distance)
                    for marker_id, distance in zip(next_raw["id"], next_raw["distance"])
                }

            source_frame = int(np.asarray(source["source_frame"]))
            sampler = (
                TriangleSurfaceSampler(raw["surface_path"]) if len(ids) else None
            )
            persistent_indices = [
                index for index, marker_id in enumerate(ids) if int(marker_id) in state
            ]
            selected_position = np.asarray(raw["closest"], dtype=np.float64).copy()
            selected_normal = np.asarray(raw["normal"], dtype=np.float64).copy()
            selected_triangle = np.asarray(raw["triangle_id"], dtype=np.int64).copy()
            selected_signed = np.asarray(raw["signed"], dtype=np.float64).copy()
            selected_distance = np.asarray(raw["distance"], dtype=np.float64).copy()
            prediction_residual = np.zeros(len(ids), dtype=np.float64)
            normal_candidate_change = np.zeros(len(ids), dtype=np.float64)
            candidate_source_projection = np.ones(len(ids), dtype=bool)

            if persistent_indices:
                assert sampler is not None
                current_indices = np.asarray(persistent_indices, dtype=np.int64)
                previous_states = [state[int(ids[index])] for index in current_indices]
                previous_anchor = np.asarray([item["anchor"] for item in previous_states])
                previous_normal = np.asarray([item["normal"] for item in previous_states])
                previous_velocity = np.asarray([item["velocity"] for item in previous_states])
                previous_source_frame = np.asarray(
                    [item["source_frame"] for item in previous_states], dtype=np.float64
                )
                dt = np.maximum((source_frame - previous_source_frame) / source_fps, 0.0)
                tangent_velocity = previous_velocity - np.einsum(
                    "ij,ij->i", previous_velocity, previous_normal
                )[:, None] * previous_normal
                predicted = previous_anchor + tangent_velocity * dt[:, None]
                (
                    predicted_projection,
                    predicted_signed,
                    _predicted_surface_distance,
                    predicted_normal,
                    predicted_triangle,
                ) = sampler.closest(predicted)
                predicted_normal = orient_to_previous(predicted_normal, previous_normal)

                source_projection = selected_position[current_indices]
                source_normal = orient_to_previous(
                    selected_normal[current_indices], previous_normal
                )
                source_triangle = selected_triangle[current_indices]
                source_signed = selected_signed[current_indices]
                candidate_positions = (source_projection, predicted_projection)
                candidate_normals = (source_normal, predicted_normal)
                candidate_triangles = (source_triangle, predicted_triangle)
                candidate_signed = (source_signed, predicted_signed)
                scores = []
                source_distances = []
                prediction_distances = []
                normal_angles = []
                correction_changes = []
                previous_correction = previous_anchor - np.asarray(
                    [item["source_position"] for item in previous_states], dtype=np.float64
                )
                quiet = np.maximum(
                    np.linalg.norm(previous_velocity, axis=1),
                    np.linalg.norm(velocities[current_indices], axis=1),
                ) <= args.quiet_speed_maximum
                for candidate_position, candidate_normal in zip(
                    candidate_positions, candidate_normals
                ):
                    source_distance = np.linalg.norm(
                        candidate_position - positions[current_indices], axis=1
                    )
                    prediction_distance = np.linalg.norm(candidate_position - predicted, axis=1)
                    dot = np.clip(
                        np.einsum("ij,ij->i", previous_normal, candidate_normal),
                        -1.0,
                        1.0,
                    )
                    angle = np.degrees(np.arccos(dot))
                    current_correction = candidate_position - positions[current_indices]
                    correction_change = np.linalg.norm(
                        current_correction - previous_correction, axis=1
                    )
                    score = (
                        source_distance / args.maximum_surface_distance
                        + args.prediction_score_weight
                        * prediction_distance
                        / args.prediction_score_scale
                        + args.normal_score_weight
                        * angle
                        / args.maximum_normal_candidate_degrees
                    )
                    invalid = (
                        (source_distance > args.maximum_surface_distance)
                        | (angle > args.maximum_normal_candidate_degrees)
                        | (quiet & (correction_change > args.maximum_quiet_correction_change))
                    )
                    score[invalid] = np.inf
                    scores.append(score)
                    source_distances.append(source_distance)
                    prediction_distances.append(prediction_distance)
                    normal_angles.append(angle)
                    correction_changes.append(correction_change)
                # Temporal coherence is the primary purpose of this product:
                # prefer the native-velocity predicted projection whenever it
                # passes every hard gate.  The current-marker projection is a
                # fallback for topology changes, not the default attractor.
                use_predicted = np.isfinite(scores[1])
                no_valid_candidate = ~np.isfinite(np.minimum(scores[0], scores[1]))
                use_predicted[no_valid_candidate] = False
                local = np.arange(len(current_indices))
                choice = use_predicted.astype(np.int64)
                selected_position[current_indices] = np.stack(candidate_positions)[choice, local]
                geometric_normal = np.stack(candidate_normals)[choice, local]
                selected_normal[current_indices] = limited_normal_update(
                    previous_normal,
                    geometric_normal,
                    args.normal_blend,
                    args.maximum_normal_step_degrees,
                )
                selected_triangle[current_indices] = np.stack(candidate_triangles)[choice, local]
                selected_signed[current_indices] = np.stack(candidate_signed)[choice, local]
                selected_distance[current_indices] = np.stack(source_distances)[choice, local]
                prediction_residual[current_indices] = np.stack(prediction_distances)[choice, local]
                normal_candidate_change[current_indices] = np.stack(normal_angles)[choice, local]
                candidate_source_projection[current_indices] = ~use_predicted
                # A continuity candidate is never allowed to keep a marker on
                # the surface outside the explicit absolute distance gate.
                selected_distance[current_indices[no_valid_candidate]] = np.inf

            surface_weight = np.zeros(len(ids), dtype=np.float64)
            route_state = np.zeros(len(ids), dtype=np.int8)
            tangents = np.zeros((len(ids), 3), dtype=np.float64)
            route_reason = np.zeros(len(ids), dtype=np.int8)
            # route_reason: 0 stable free, 1 entering, 2 stable surface,
            # 3 pre-fade, 4 hard maximum-distance exit, 5 invalid candidate.
            for index, marker_id_value in enumerate(ids):
                marker_id = int(marker_id_value)
                previous = state.get(marker_id)
                raw_near = float(raw["distance"][index]) <= args.entry_distance
                within_maximum = (
                    float(selected_distance[index]) <= args.maximum_surface_distance
                )
                eligible = raw_near and within_maximum
                next_near = next_distance.get(marker_id, math.inf) <= args.entry_distance
                next_within_maximum = (
                    next_distance.get(marker_id, math.inf) <= args.maximum_surface_distance
                )
                if previous is None:
                    # The first cache frame has no prior routing history.  A
                    # clearly near marker is accepted directly; all later
                    # entries require temporal confirmation.
                    is_surface = eligible
                    weight = 1.0 if is_surface else 0.0
                    reason = 2 if is_surface else 0
                elif bool(previous["is_surface"]):
                    if not within_maximum:
                        is_surface = False
                        weight = 0.0
                        reason = 5 if not np.isfinite(selected_distance[index]) else 4
                    else:
                        is_surface = True
                        weight = 0.5 if not next_within_maximum else 1.0
                        reason = 3 if weight < 1.0 else 2
                else:
                    near_streak = int(previous.get("near_streak", 0)) + 1 if eligible else 0
                    confirmed = eligible and near_streak >= 2
                    is_surface = bool(confirmed)
                    if confirmed:
                        weight = 1.0
                        reason = 2
                    elif eligible and next_near:
                        weight = 0.5
                        reason = 1
                    else:
                        weight = 0.0
                        reason = 0
                if weight > 0.0 and within_maximum:
                    if previous is None:
                        tangents[index] = initial_tangent(selected_normal[index], marker_id)
                    else:
                        tangents[index] = transported_tangent(
                            previous["tangent"], selected_normal[index], marker_id
                        )
                else:
                    tangents[index] = (
                        previous["tangent"]
                        if previous is not None
                        else initial_tangent(selected_normal[index], marker_id)
                    )
                surface_weight[index] = weight
                route_state[index] = int(is_surface)
                route_reason[index] = reason

            free_weight = 1.0 - surface_weight
            surface_mask = surface_weight > 0.0
            free_mask = free_weight > 0.0
            surface_density_target = np.zeros(len(ids), dtype=np.float64)
            density_raw = np.zeros(len(ids), dtype=np.float64)
            density_pair_count = 0
            if np.any(surface_mask):
                target, raw_kernel, density_pair_count = continuous_surface_density(
                    selected_position[surface_mask],
                    surface_weight[surface_mask],
                    args.density_support_length,
                    args.density_cutoff_multiplier,
                    args.density_optical_scale,
                )
                surface_density_target[surface_mask] = target
                density_raw[surface_mask] = raw_kernel
            density = surface_density_target.copy()
            cluster_radius = np.empty(len(ids), dtype=np.float64)
            cell_diameter = np.empty(len(ids), dtype=np.float64)
            marker_radius = np.asarray(source["radius"], dtype=np.float64)[rows]
            for index, marker_id_value in enumerate(ids):
                marker_id = int(marker_id_value)
                previous = state.get(marker_id)
                if previous is not None:
                    previous_density = float(previous.get("density", 0.0))
                    requested_delta = args.density_ema_alpha * (
                        surface_density_target[index] - previous_density
                    )
                    density[index] = previous_density + float(
                        np.clip(
                            requested_delta,
                            -args.density_maximum_change,
                            args.density_maximum_change,
                        )
                    )
                    cluster_radius[index] = float(previous["cluster_radius"])
                    cell_diameter[index] = float(previous["cell_diameter"])
                else:
                    curvature_scale = curvature_radius_scale(
                        lifetime_curvature[marker_id],
                        args.curvature_full_radius_degrees,
                        args.curvature_minimum_radius_degrees,
                        args.curvature_minimum_radius_scale,
                    )
                    cluster_radius[index] = float(
                        np.clip(
                            marker_radius[index] * args.cluster_radius_scale,
                            args.cluster_radius_minimum,
                            args.cluster_radius_maximum,
                        )
                    ) * curvature_scale
                    cell_diameter[index] = (
                        args.cell_diameter_minimum
                        + (args.cell_diameter_maximum - args.cell_diameter_minimum)
                        * stable_unit_float(marker_id, 0xD1B54A32D192ED03)
                    )
            # Patch-local UV spans [-1, 1].  A Voronoi scale of R / D therefore
            # yields approximately 2R / D cells across the patch diameter.
            cell_scale = cluster_radius / cell_diameter
            output_path = output_directory / f"binding_{frame:04d}.npz"
            opacity = np.asarray(source["opacity"], dtype=np.float32)[rows]
            remaining_lifetime = np.asarray(
                source["remaining_lifetime"], dtype=np.float32
            )[rows]
            birth_frame = np.asarray(source["birth_frame"], dtype=np.int32)[rows]
            shape = np.asarray(source["shape"], dtype=np.float32)[rows]
            atomic_npz(
                output_path,
                schema=np.asarray("physx-foamgenerator-temporal-surface-binding-frame/v2"),
                output_frame=np.int32(frame),
                source_frame=np.int32(source_frame),
                active_id=ids.astype(np.uint64),
                active_source_row=rows.astype(np.int32),
                surface_weight=surface_weight.astype(np.float32),
                free_weight=free_weight.astype(np.float32),
                route_state=route_state,
                route_reason=route_reason,
                id=ids[surface_mask].astype(np.uint64),
                source_row=rows[surface_mask].astype(np.int32),
                source_position=positions[surface_mask].astype(np.float32),
                position=selected_position[surface_mask].astype(np.float32),
                geometric_normal=orient_to_previous(
                    np.asarray(raw["normal"])[surface_mask], selected_normal[surface_mask]
                ).astype(np.float32),
                normal=selected_normal[surface_mask].astype(np.float32),
                tangent=tangents[surface_mask].astype(np.float32),
                signed_distance=selected_signed[surface_mask].astype(np.float32),
                distance=selected_distance[surface_mask].astype(np.float32),
                raw_closest_distance=np.asarray(raw["distance"])[surface_mask].astype(np.float32),
                prediction_residual=prediction_residual[surface_mask].astype(np.float32),
                normal_candidate_change_degrees=normal_candidate_change[surface_mask].astype(np.float32),
                candidate_source_projection=candidate_source_projection[surface_mask],
                density_raw=density_raw[surface_mask].astype(np.float32),
                density=density[surface_mask].astype(np.float32),
                cluster_radius=cluster_radius[surface_mask].astype(np.float32),
                cell_diameter=cell_diameter[surface_mask].astype(np.float32),
                cell_scale=cell_scale[surface_mask].astype(np.float32),
                lifetime_curvature_degrees=np.asarray(
                    [lifetime_curvature[int(marker_id)] for marker_id in ids[surface_mask]],
                    dtype=np.float32,
                ),
                triangle_id=selected_triangle[surface_mask].astype(np.int64),
                velocity=velocities[surface_mask].astype(np.float32),
                opacity=opacity[surface_mask],
                surface_opacity=(opacity[surface_mask] * surface_weight[surface_mask]).astype(np.float32),
                remaining_lifetime=remaining_lifetime[surface_mask],
                birth_frame=birth_frame[surface_mask],
                shape=shape[surface_mask],
                unbound_id=ids[free_mask].astype(np.uint64),
                unbound_source_row=rows[free_mask].astype(np.int32),
                unbound_distance=np.asarray(raw["distance"])[free_mask].astype(np.float32),
                unbound_weight=free_weight[free_mask].astype(np.float32),
            )

            next_state: dict[int, dict] = {}
            for index, marker_id_value in enumerate(ids):
                marker_id = int(marker_id_value)
                previous = state.get(marker_id)
                raw_near = float(raw["distance"][index]) <= args.entry_distance
                eligible = raw_near and (
                    float(selected_distance[index]) <= args.maximum_surface_distance
                )
                near_streak = (
                    int(previous.get("near_streak", 0)) + 1
                    if previous is not None and eligible
                    else int(eligible)
                )
                next_state[marker_id] = {
                    "anchor": selected_position[index].copy(),
                    "source_position": positions[index].copy(),
                    "normal": selected_normal[index].copy(),
                    "tangent": tangents[index].copy(),
                    "velocity": velocities[index].copy(),
                    "source_frame": source_frame,
                    "is_surface": bool(route_state[index]),
                    "near_streak": near_streak,
                    "density": float(density[index]),
                    "cluster_radius": float(cluster_radius[index]),
                    "cell_diameter": float(cell_diameter[index]),
                }
            state = next_state

            record = {
                "output_frame": frame,
                "source_frame": source_frame,
                "file": output_path.name,
                "bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
                "source_foam_cache": str(raw["foam_path"]),
                "source_foam_cache_sha256": sha256_file(raw["foam_path"]),
                "surface": str(raw["surface_path"]),
                "surface_sha256": sha256_file(raw["surface_path"]),
                "input_count": int(len(source["id"])),
                "active_count": int(len(ids)),
                "surface_count": int(np.count_nonzero(surface_mask)),
                "free_count": int(np.count_nonzero(free_mask)),
                "crossfade_count": int(np.count_nonzero((surface_weight > 0.0) & (surface_weight < 1.0))),
                "hard_surface_state_count": int(np.count_nonzero(route_state)),
                "surface_fraction": float(np.mean(surface_weight)) if len(ids) else 0.0,
                "surface_distance_m": quantiles(selected_distance[surface_mask]),
                "raw_closest_distance_m": quantiles(np.asarray(raw["distance"])),
                "prediction_residual_m": quantiles(prediction_residual[surface_mask]),
                "density": quantiles(density[surface_mask]),
                "density_raw": quantiles(density_raw[surface_mask]),
                "density_neighbor_pairs": density_pair_count,
                "cluster_radius_m": quantiles(cluster_radius[surface_mask]),
                "cell_diameter_m": quantiles(cell_diameter[surface_mask]),
            }
            manifest["samples"].append(record)
            atomic_json(manifest_path, manifest)
            print(
                f"[temporal-binding:build] frame={frame:04d} surface={record['surface_count']} "
                f"free={record['free_count']} crossfade={record['crossfade_count']}",
                flush=True,
            )

        manifest["complete"] = True
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["summary"] = {
            "frames": len(manifest["samples"]),
            "maximum_surface_markers": max(sample["surface_count"] for sample in manifest["samples"]),
            "maximum_free_fraction": max(
                1.0 - sample["surface_fraction"] for sample in manifest["samples"]
            ),
            "native_velocity_consumed": True,
            "finite_difference_velocity_used": False,
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
