"""Audit v2 temporal FoamGenerator surface binding as hard render gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binding_directory", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--quiet-speed-maximum", type=float, default=0.25)
    parser.add_argument("--maximum-quiet-correction-change-p99", type=float, default=0.004)
    parser.add_argument("--maximum-quiet-anchor-prediction-residual-p99", type=float, default=0.004)
    parser.add_argument("--maximum-tangent-plane-change-p99-degrees", type=float, default=45.0)
    args = parser.parse_args()

    directory = args.binding_directory.resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("product") != "physx_foamgenerator_temporal_surface_binding":
        raise RuntimeError("Input is not a v2 temporal surface-binding product")
    configuration = manifest["configuration"]
    maximum_surface_distance = float(configuration["maximum_surface_distance_m"])
    density_configuration = configuration.get("continuous_density", {})
    maximum_density_change = float(
        density_configuration.get("maximum_change_per_output_frame", 1.0)
    )
    minimum_opacity = float(configuration["minimum_opacity"])
    cache_directory = Path(manifest["inputs"]["render_cache_directory"])
    samples = sorted(manifest["samples"], key=lambda item: int(item["output_frame"]))

    structural_errors: list[str] = []
    states: list[dict] = []
    total_active = 0
    maximum_weight_error = 0.0
    maximum_surface_distance_seen = 0.0
    for sample in samples:
        frame = int(sample["output_frame"])
        binding_path = directory / sample["file"]
        foam_path = cache_directory / "foam" / f"frame_{frame:04d}.npz"
        if sha256_file(binding_path) != sample["sha256"]:
            structural_errors.append(f"frame {frame}: binding SHA-256 mismatch")
        if sha256_file(foam_path) != sample["source_foam_cache_sha256"]:
            structural_errors.append(f"frame {frame}: source foam SHA-256 mismatch")
        with np.load(binding_path, allow_pickle=False) as binding, np.load(
            foam_path, allow_pickle=False
        ) as foam:
            if str(np.asarray(binding["schema"])) != "physx-foamgenerator-temporal-surface-binding-frame/v2":
                structural_errors.append(f"frame {frame}: unsupported frame schema")
            active_id = np.asarray(binding["active_id"], dtype=np.uint64).copy()
            active_rows = np.asarray(binding["active_source_row"], dtype=np.int64)
            source_id = np.asarray(foam["id"], dtype=np.uint64)
            expected_rows = np.flatnonzero(
                np.asarray(foam["opacity"], dtype=np.float64) >= minimum_opacity
            )
            expected_id = source_id[expected_rows]
            if len(np.unique(active_id)) != len(active_id):
                structural_errors.append(f"frame {frame}: active IDs are not unique")
            if not np.array_equal(active_rows, expected_rows):
                structural_errors.append(f"frame {frame}: active row inventory mismatch")
            if not np.array_equal(active_id, expected_id):
                structural_errors.append(f"frame {frame}: active ID inventory mismatch")

            surface_weight = np.asarray(binding["surface_weight"], dtype=np.float64)
            free_weight = np.asarray(binding["free_weight"], dtype=np.float64)
            weight_error = np.abs(surface_weight + free_weight - 1.0)
            maximum_weight_error = max(
                maximum_weight_error, float(weight_error.max(initial=0.0))
            )
            if np.any(surface_weight < 0.0) or np.any(surface_weight > 1.0):
                structural_errors.append(f"frame {frame}: surface weights outside [0,1]")
            if np.any(free_weight < 0.0) or np.any(free_weight > 1.0):
                structural_errors.append(f"frame {frame}: free weights outside [0,1]")
            if np.any(weight_error > 1.0e-6):
                structural_errors.append(f"frame {frame}: route weights do not sum to one")

            surface_mask = surface_weight > 0.0
            free_mask = free_weight > 0.0
            bound_id = np.asarray(binding["id"], dtype=np.uint64)
            bound_rows = np.asarray(binding["source_row"], dtype=np.int64)
            free_id = np.asarray(binding["unbound_id"], dtype=np.uint64)
            free_rows = np.asarray(binding["unbound_source_row"], dtype=np.int64)
            if not np.array_equal(bound_id, active_id[surface_mask]):
                structural_errors.append(f"frame {frame}: surface ID inventory mismatch")
            if not np.array_equal(bound_rows, active_rows[surface_mask]):
                structural_errors.append(f"frame {frame}: surface row inventory mismatch")
            if not np.array_equal(free_id, active_id[free_mask]):
                structural_errors.append(f"frame {frame}: free ID inventory mismatch")
            if not np.array_equal(free_rows, active_rows[free_mask]):
                structural_errors.append(f"frame {frame}: free row inventory mismatch")

            velocity = np.asarray(binding["velocity"], dtype=np.float32)
            if not np.array_equal(velocity, np.asarray(foam["velocity"], dtype=np.float32)[bound_rows]):
                structural_errors.append(f"frame {frame}: native velocity is not bit-exact")
            distance = np.asarray(binding["distance"], dtype=np.float64)
            maximum_surface_distance_seen = max(
                maximum_surface_distance_seen, float(distance.max(initial=0.0))
            )
            if np.any(distance > maximum_surface_distance + 1.0e-7):
                structural_errors.append(f"frame {frame}: surface route exceeds absolute distance")
            normals = np.asarray(binding["normal"], dtype=np.float64)
            tangents = np.asarray(binding["tangent"], dtype=np.float64)
            arrays = (
                active_id,
                surface_weight,
                free_weight,
                np.asarray(binding["position"]),
                np.asarray(binding["source_position"]),
                normals,
                tangents,
                velocity,
                distance,
                np.asarray(binding["prediction_residual"]),
                np.asarray(binding["density"]),
                np.asarray(binding["density_raw"]),
                np.asarray(binding["cluster_radius"]),
                np.asarray(binding["cell_diameter"]),
                np.asarray(binding["cell_scale"]),
                np.asarray(binding["lifetime_curvature_degrees"]),
            )
            if not all(np.all(np.isfinite(value)) for value in arrays):
                structural_errors.append(f"frame {frame}: non-finite state")
            if len(normals):
                normal_length_error = np.abs(np.linalg.norm(normals, axis=1) - 1.0)
                tangent_length_error = np.abs(np.linalg.norm(tangents, axis=1) - 1.0)
                orthogonality_error = np.abs(np.einsum("ij,ij->i", normals, tangents))
                if np.any(normal_length_error > 2.0e-5):
                    structural_errors.append(f"frame {frame}: non-unit stabilized normal")
                if np.any(tangent_length_error > 2.0e-5):
                    structural_errors.append(f"frame {frame}: non-unit tangent")
                if np.any(orthogonality_error > 2.0e-5):
                    structural_errors.append(f"frame {frame}: tangent is not on stabilized plane")

            state_index = {int(marker_id): index for index, marker_id in enumerate(active_id)}
            bound_index = {int(marker_id): index for index, marker_id in enumerate(bound_id)}
            states.append(
                {
                    "frame": frame,
                    "source_frame": int(np.asarray(binding["source_frame"])),
                    "id": active_id,
                    "index": state_index,
                    "surface_weight": surface_weight,
                    "free_weight": free_weight,
                    "route_state": np.asarray(binding["route_state"], dtype=np.int8).copy(),
                    "bound_index": bound_index,
                    "position": np.asarray(binding["position"], dtype=np.float64).copy(),
                    "source_position": np.asarray(binding["source_position"], dtype=np.float64).copy(),
                    "normal": normals.copy(),
                    "tangent": tangents.copy(),
                    "velocity": velocity.astype(np.float64, copy=True),
                    "prediction_residual": np.asarray(
                        binding["prediction_residual"], dtype=np.float64
                    ).copy(),
                    "density": np.asarray(binding["density"], dtype=np.float64).copy(),
                    "cluster_radius": np.asarray(
                        binding["cluster_radius"], dtype=np.float64
                    ).copy(),
                    "cell_diameter": np.asarray(
                        binding["cell_diameter"], dtype=np.float64
                    ).copy(),
                    "lifetime_curvature": np.asarray(
                        binding["lifetime_curvature_degrees"], dtype=np.float64
                    ).copy(),
                }
            )
            total_active += len(active_id)

    transitions = []
    quiet_correction_p99 = []
    quiet_prediction_p99 = []
    plane_change_p99 = []
    normal_flip_count = 0
    tangent_flip_count = 0
    maximum_density_delta = 0.0
    maximum_cluster_radius_delta = 0.0
    maximum_cell_diameter_delta = 0.0
    maximum_lifetime_curvature_delta = 0.0
    for previous, current in zip(states, states[1:]):
        persistent = set(previous["index"]) & set(current["index"])
        persistent_surface = set(previous["bound_index"]) & set(current["bound_index"])
        correction_change = []
        prediction_residual = []
        plane_change = []
        tangent_change = []
        speed = []
        for marker_id in persistent_surface:
            i = previous["bound_index"][marker_id]
            j = current["bound_index"][marker_id]
            previous_correction = previous["position"][i] - previous["source_position"][i]
            current_correction = current["position"][j] - current["source_position"][j]
            correction_change.append(float(np.linalg.norm(current_correction - previous_correction)))
            prediction_residual.append(float(current["prediction_residual"][j]))
            dot = float(np.clip(np.dot(previous["normal"][i], current["normal"][j]), -1.0, 1.0))
            tangent_dot = float(
                np.clip(np.dot(previous["tangent"][i], current["tangent"][j]), -1.0, 1.0)
            )
            plane_change.append(float(np.degrees(np.arccos(abs(dot)))))
            tangent_change.append(float(np.degrees(np.arccos(tangent_dot))))
            normal_flip_count += int(dot < 0.0)
            tangent_flip_count += int(tangent_dot < 0.0)
            maximum_density_delta = max(
                maximum_density_delta,
                abs(float(current["density"][j] - previous["density"][i])),
            )
            maximum_cluster_radius_delta = max(
                maximum_cluster_radius_delta,
                abs(
                    float(
                        current["cluster_radius"][j]
                        - previous["cluster_radius"][i]
                    )
                ),
            )
            maximum_cell_diameter_delta = max(
                maximum_cell_diameter_delta,
                abs(
                    float(
                        current["cell_diameter"][j]
                        - previous["cell_diameter"][i]
                    )
                ),
            )
            maximum_lifetime_curvature_delta = max(
                maximum_lifetime_curvature_delta,
                abs(
                    float(
                        current["lifetime_curvature"][j]
                        - previous["lifetime_curvature"][i]
                    )
                ),
            )
            speed.append(
                max(
                    float(np.linalg.norm(previous["velocity"][i])),
                    float(np.linalg.norm(current["velocity"][j])),
                )
            )
        correction_array = np.asarray(correction_change, dtype=np.float64)
        prediction_array = np.asarray(prediction_residual, dtype=np.float64)
        plane_array = np.asarray(plane_change, dtype=np.float64)
        speed_array = np.asarray(speed, dtype=np.float64)
        quiet = speed_array <= args.quiet_speed_maximum
        correction_stats = quantiles(correction_array)
        prediction_stats = quantiles(prediction_array)
        plane_stats = quantiles(plane_array)
        quiet_correction_stats = quantiles(correction_array[quiet])
        quiet_prediction_stats = quantiles(prediction_array[quiet])
        if np.any(quiet):
            quiet_correction_p99.append(quiet_correction_stats["p99"])
            quiet_prediction_p99.append(quiet_prediction_stats["p99"])
        plane_change_p99.append(plane_stats["p99"])
        hard_switches = 0
        weight_change = []
        for marker_id in persistent:
            i = previous["index"][marker_id]
            j = current["index"][marker_id]
            hard_switches += int(previous["route_state"][i] != current["route_state"][j])
            weight_change.append(
                abs(float(current["surface_weight"][j] - previous["surface_weight"][i]))
            )
        transitions.append(
            {
                "from_output_frame": previous["frame"],
                "to_output_frame": current["frame"],
                "persistent_ids": len(persistent),
                "persistent_surface_ids": len(persistent_surface),
                "hard_route_switches": hard_switches,
                "hard_route_switch_fraction": hard_switches / len(persistent) if persistent else 0.0,
                "surface_weight_change": quantiles(np.asarray(weight_change)),
                "anchor_correction_change_m": correction_stats,
                "quiet_anchor_correction_change_m": quiet_correction_stats,
                "prediction_residual_m": prediction_stats,
                "quiet_prediction_residual_m": quiet_prediction_stats,
                "tangent_plane_change_degrees": plane_stats,
                "tangent_orientation_change_degrees": quantiles(np.asarray(tangent_change)),
            }
        )

    chatter_events = []
    for first, middle, last in zip(states, states[1:], states[2:]):
        persistent = set(first["index"]) & set(middle["index"]) & set(last["index"])
        for marker_id in persistent:
            route = (
                int(first["route_state"][first["index"][marker_id]]),
                int(middle["route_state"][middle["index"][marker_id]]),
                int(last["route_state"][last["index"][marker_id]]),
            )
            if route[0] == route[2] and route[0] != route[1]:
                chatter_events.append(
                    {
                        "id": marker_id,
                        "frames": [first["frame"], middle["frame"], last["frame"]],
                        "route": route,
                    }
                )

    gates = {
        "manifest_complete": manifest.get("complete") is True,
        "structural_integrity": not structural_errors,
        "route_weight_sum_maximum_error": maximum_weight_error,
        "route_weight_sum_pass": maximum_weight_error <= 1.0e-6,
        "maximum_surface_distance_seen_m": maximum_surface_distance_seen,
        "maximum_surface_distance_pass": maximum_surface_distance_seen <= maximum_surface_distance + 1.0e-7,
        "normal_direction_flips": normal_flip_count,
        "normal_direction_flip_pass": normal_flip_count == 0,
        "tangent_direction_flips": tangent_flip_count,
        "tangent_direction_flip_pass": tangent_flip_count == 0,
        "one_frame_route_chatter_events": len(chatter_events),
        "one_frame_route_chatter_pass": len(chatter_events) == 0,
        "maximum_quiet_anchor_correction_change_p99_m": max(quiet_correction_p99, default=0.0),
        "quiet_anchor_correction_pass": max(quiet_correction_p99, default=0.0) <= args.maximum_quiet_correction_change_p99,
        "maximum_quiet_prediction_residual_p99_m": max(quiet_prediction_p99, default=0.0),
        "quiet_prediction_residual_advisory_pass": max(quiet_prediction_p99, default=0.0) <= args.maximum_quiet_anchor_prediction_residual_p99,
        "maximum_tangent_plane_change_p99_degrees": max(plane_change_p99, default=0.0),
        "tangent_plane_change_pass": max(plane_change_p99, default=0.0) <= args.maximum_tangent_plane_change_p99_degrees,
        "density_is_continuous_kernel": density_configuration.get("method", "").startswith("cKDTree Gaussian"),
        "maximum_density_change_per_frame": maximum_density_delta,
        "density_change_pass": maximum_density_delta <= maximum_density_change + 1.0e-6,
        "maximum_cluster_radius_change_m": maximum_cluster_radius_delta,
        "cluster_radius_stability_pass": maximum_cluster_radius_delta <= 1.0e-8,
        "maximum_cell_diameter_change_m": maximum_cell_diameter_delta,
        "cell_diameter_stability_pass": maximum_cell_diameter_delta <= 1.0e-8,
        "maximum_lifetime_curvature_change_degrees": maximum_lifetime_curvature_delta,
        "lifetime_curvature_stability_pass": maximum_lifetime_curvature_delta <= 1.0e-6,
        "native_velocity_declared_consumed": configuration.get("native_velocity_consumed") is True,
        "finite_difference_velocity_forbidden_pass": configuration.get("finite_difference_velocity_used") is False,
    }
    pass_keys = [
        key
        for key in gates
        if (key.endswith("_pass") and "advisory" not in key)
        or key in (
            "manifest_complete",
            "structural_integrity",
            "native_velocity_declared_consumed",
            "density_is_continuous_kernel",
        )
    ]
    valid = all(bool(gates[key]) for key in pass_keys)
    report = {
        "schema": "physx-foamgenerator-temporal-surface-binding-audit/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "valid": valid,
        "binding_directory": str(directory),
        "manifest_sha256": sha256_file(manifest_path),
        "frames_audited": len(states),
        "states_audited": total_active,
        "thresholds": {
            "maximum_surface_distance_m": maximum_surface_distance,
            "quiet_native_speed_maximum_m_per_s": args.quiet_speed_maximum,
            "maximum_quiet_anchor_correction_change_p99_m": args.maximum_quiet_correction_change_p99,
            "maximum_quiet_anchor_prediction_residual_p99_m": args.maximum_quiet_anchor_prediction_residual_p99,
            "maximum_tangent_plane_change_p99_degrees": args.maximum_tangent_plane_change_p99_degrees,
            "maximum_density_change_per_output_frame": maximum_density_change,
            "maximum_cluster_radius_change_m": 1.0e-8,
            "maximum_cell_diameter_change_m": 1.0e-8,
            "maximum_lifetime_curvature_change_degrees": 1.0e-6,
            "normal_direction_flips": 0,
            "one_frame_route_chatter_events": 0,
            "route_weight_sum_tolerance": 1.0e-6,
            "role": "render-only acceptance gates",
        },
        "gates": gates,
        "structural_errors": structural_errors,
        "chatter_examples": chatter_events[:32],
        "transitions": transitions,
    }
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(
        json.dumps(
            {
                "valid": valid,
                "frames_audited": len(states),
                "states_audited": total_active,
                "gates": gates,
                "structural_errors": structural_errors,
            },
            indent=2,
        )
    )
    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
