"""Audit structural integrity and temporal consistency of surface-binding caches."""

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
    parser.add_argument("--maximum-switch-fraction", type=float, default=0.08)
    parser.add_argument(
        "--maximum-correction-change-p99", type=float, default=0.008,
        help="Maximum p99 frame-to-frame change of surface projection correction, metres",
    )
    parser.add_argument("--maximum-normal-change-p99-degrees", type=float, default=75.0)
    parser.add_argument("--quiet-speed-maximum", type=float, default=0.25)
    parser.add_argument("--maximum-quiet-correction-change-p99", type=float, default=0.004)
    parser.add_argument(
        "--maximum-quiet-tangent-plane-change-p99-degrees", type=float, default=45.0
    )
    args = parser.parse_args()

    directory = args.binding_directory.resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("product") != "physx_foamgenerator_surface_binding":
        raise RuntimeError("Input is not a lightweight surface-binding product")
    cache_directory = Path(manifest["inputs"]["render_cache_directory"])
    minimum_opacity = float(manifest["configuration"]["minimum_opacity"])
    maximum_bind_distance = float(manifest["configuration"]["maximum_bind_distance_m"])
    samples = sorted(manifest["samples"], key=lambda item: int(item["output_frame"]))

    structural_errors: list[str] = []
    states: list[dict] = []
    audited_states = 0
    for sample in samples:
        frame = int(sample["output_frame"])
        binding_path = directory / sample["file"]
        foam_path = cache_directory / "foam" / f"frame_{frame:04d}.npz"
        if sha256_file(binding_path) != sample["sha256"]:
            structural_errors.append(f"frame {frame}: binding SHA-256 mismatch")
        with np.load(binding_path, allow_pickle=False) as binding, np.load(
            foam_path, allow_pickle=False
        ) as foam:
            bound_id = np.asarray(binding["id"], dtype=np.uint64).copy()
            unbound_id = np.asarray(binding["unbound_id"], dtype=np.uint64).copy()
            bound_rows = np.asarray(binding["source_row"], dtype=np.int64)
            unbound_rows = np.asarray(binding["unbound_source_row"], dtype=np.int64)
            source_id = np.asarray(foam["id"], dtype=np.uint64)
            active_rows = np.flatnonzero(
                np.asarray(foam["opacity"], dtype=np.float64) >= minimum_opacity
            )
            active_id = source_id[active_rows]
            if len(np.unique(np.concatenate((bound_id, unbound_id)))) != len(active_id):
                structural_errors.append(f"frame {frame}: duplicate bound/unbound IDs")
            if not np.array_equal(
                np.sort(np.concatenate((bound_id, unbound_id))), np.sort(active_id)
            ):
                structural_errors.append(f"frame {frame}: active partition mismatch")
            if not np.array_equal(bound_id, source_id[bound_rows]):
                structural_errors.append(f"frame {frame}: bound source-row ID mismatch")
            if not np.array_equal(unbound_id, source_id[unbound_rows]):
                structural_errors.append(f"frame {frame}: unbound source-row ID mismatch")
            velocity = np.asarray(binding["velocity"], dtype=np.float32)
            if not np.array_equal(velocity, np.asarray(foam["velocity"], dtype=np.float32)[bound_rows]):
                structural_errors.append(f"frame {frame}: native velocity is not bit-exact")
            distance = np.asarray(binding["distance"], dtype=np.float64)
            unbound_distance = np.asarray(binding["unbound_distance"], dtype=np.float64)
            if np.any(distance > maximum_bind_distance + 1.0e-7):
                structural_errors.append(f"frame {frame}: bound distance exceeds threshold")
            if np.any(unbound_distance <= maximum_bind_distance):
                structural_errors.append(f"frame {frame}: unbound distance does not exceed threshold")
            arrays = (
                np.asarray(binding["source_position"]),
                np.asarray(binding["position"]),
                np.asarray(binding["normal"]),
                velocity,
                distance,
                unbound_distance,
            )
            if not all(np.all(np.isfinite(array)) for array in arrays):
                structural_errors.append(f"frame {frame}: non-finite state")
            states.append(
                {
                    "frame": frame,
                    "source_frame": int(np.asarray(binding["source_frame"])),
                    "active_id": active_id.copy(),
                    "bound_id": bound_id,
                    "position": np.asarray(binding["position"], dtype=np.float64).copy(),
                    "source_position": np.asarray(
                        binding["source_position"], dtype=np.float64
                    ).copy(),
                    "normal": np.asarray(binding["normal"], dtype=np.float64).copy(),
                    "velocity": velocity.astype(np.float64, copy=True),
                    "triangle_id": np.asarray(binding["triangle_id"], dtype=np.int64).copy(),
                }
            )
            audited_states += len(active_id)

    transitions = []
    switch_fractions = []
    correction_p99 = []
    normal_p99 = []
    quiet_correction_p99 = []
    quiet_plane_p99 = []
    for previous, current in zip(states, states[1:]):
        previous_active = set(map(int, previous["active_id"]))
        current_active = set(map(int, current["active_id"]))
        persistent = previous_active & current_active
        previous_bound_index = {
            int(value): index for index, value in enumerate(previous["bound_id"])
        }
        current_bound_index = {
            int(value): index for index, value in enumerate(current["bound_id"])
        }
        previous_bound = set(previous_bound_index)
        current_bound = set(current_bound_index)
        switches = (previous_bound ^ current_bound) & persistent
        persistent_bound = previous_bound & current_bound
        switch_fraction = len(switches) / len(persistent) if persistent else 0.0

        correction_change = []
        normal_change = []
        tangent_plane_change = []
        normal_flip_count = 0
        projection_displacement = []
        source_displacement = []
        correction_examples = []
        speed = []
        triangle_change = 0
        for marker_id in persistent_bound:
            i = previous_bound_index[marker_id]
            j = current_bound_index[marker_id]
            previous_correction = previous["position"][i] - previous["source_position"][i]
            current_correction = current["position"][j] - current["source_position"][j]
            correction_delta = float(np.linalg.norm(current_correction - previous_correction))
            correction_change.append(correction_delta)
            projection_displacement.append(
                float(np.linalg.norm(current["position"][j] - previous["position"][i]))
            )
            source_displacement.append(
                float(
                    np.linalg.norm(
                        current["source_position"][j] - previous["source_position"][i]
                    )
                )
            )
            dot = float(np.clip(np.dot(previous["normal"][i], current["normal"][j]), -1.0, 1.0))
            normal_change.append(float(np.degrees(np.arccos(dot))))
            tangent_plane_change.append(float(np.degrees(np.arccos(abs(dot)))))
            speed.append(
                max(
                    float(np.linalg.norm(previous["velocity"][i])),
                    float(np.linalg.norm(current["velocity"][j])),
                )
            )
            normal_flip_count += int(dot < 0.0)
            correction_examples.append(
                {
                    "id": marker_id,
                    "correction_change_m": correction_delta,
                    "previous_surface_position": previous["position"][i].tolist(),
                    "current_surface_position": current["position"][j].tolist(),
                    "previous_source_position": previous["source_position"][i].tolist(),
                    "current_source_position": current["source_position"][j].tolist(),
                }
            )
            triangle_change += int(previous["triangle_id"][i] != current["triangle_id"][j])
        correction_stats = quantiles(np.asarray(correction_change))
        normal_stats = quantiles(np.asarray(normal_change))
        tangent_plane_stats = quantiles(np.asarray(tangent_plane_change))
        speed_array = np.asarray(speed, dtype=np.float64)
        quiet = speed_array <= args.quiet_speed_maximum
        quiet_correction_stats = quantiles(np.asarray(correction_change)[quiet])
        quiet_plane_stats = quantiles(np.asarray(tangent_plane_change)[quiet])
        switch_fractions.append(switch_fraction)
        correction_p99.append(correction_stats["p99"])
        normal_p99.append(tangent_plane_stats["p99"])
        if np.any(quiet):
            quiet_correction_p99.append(quiet_correction_stats["p99"])
            quiet_plane_p99.append(quiet_plane_stats["p99"])
        correction_examples.sort(key=lambda item: item["correction_change_m"], reverse=True)
        transitions.append(
            {
                "from_output_frame": previous["frame"],
                "to_output_frame": current["frame"],
                "source_frame_step": current["source_frame"] - previous["source_frame"],
                "persistent_active_ids": len(persistent),
                "born_or_entered_ids": len(current_active - previous_active),
                "died_or_exited_ids": len(previous_active - current_active),
                "bound_unbound_switches": len(switches),
                "bound_unbound_switch_fraction": switch_fraction,
                "persistent_bound_ids": len(persistent_bound),
                "projection_correction_change_m": correction_stats,
                "projection_position_displacement_m": quantiles(
                    np.asarray(projection_displacement)
                ),
                "source_position_displacement_m_diagnostic_only": quantiles(
                    np.asarray(source_displacement)
                ),
                "oriented_normal_change_degrees_diagnostic_only": normal_stats,
                "tangent_plane_change_degrees": tangent_plane_stats,
                "quiet_region": {
                    "native_speed_maximum_m_per_s": args.quiet_speed_maximum,
                    "persistent_bound_ids": int(np.count_nonzero(quiet)),
                    "projection_correction_change_m": quiet_correction_stats,
                    "tangent_plane_change_degrees": quiet_plane_stats,
                },
                "normal_orientation_flip_fraction_diagnostic_only": (
                    normal_flip_count / len(persistent_bound) if persistent_bound else 0.0
                ),
                "largest_projection_correction_changes": correction_examples[:8],
                # Splashsurf remeshing does not preserve triangle IDs, so this is
                # diagnostic only and is deliberately not a quality gate.
                "triangle_id_change_fraction_diagnostic_only": (
                    triangle_change / len(persistent_bound) if persistent_bound else 0.0
                ),
            }
        )

    temporal_gate = {
        "maximum_raw_switch_fraction_diagnostic_only": max(
            switch_fractions, default=0.0
        ),
        "maximum_quiet_projection_correction_change_p99_m": max(
            quiet_correction_p99, default=0.0
        ),
        "maximum_quiet_tangent_plane_change_p99_degrees": max(
            quiet_plane_p99, default=0.0
        ),
        "dynamic_region_diagnostic_only": {
            "maximum_projection_correction_change_p99_m": max(
                correction_p99, default=0.0
            ),
            "maximum_tangent_plane_change_p99_degrees": max(normal_p99, default=0.0),
        },
    }
    chatter_fractions = []
    chatter_events = 0
    for first, middle, last in zip(states, states[1:], states[2:]):
        active = (
            set(map(int, first["active_id"]))
            & set(map(int, middle["active_id"]))
            & set(map(int, last["active_id"]))
        )
        bound = [set(map(int, state["bound_id"])) for state in (first, middle, last)]
        chatter = {
            marker_id
            for marker_id in active
            if ((marker_id in bound[0]) != (marker_id in bound[1]))
            and ((marker_id in bound[0]) == (marker_id in bound[2]))
        }
        chatter_events += len(chatter)
        chatter_fractions.append(len(chatter) / len(active) if active else 0.0)
    temporal_gate["maximum_one_frame_bound_unbound_chatter_fraction"] = max(
        chatter_fractions, default=0.0
    )
    temporal_gate["total_one_frame_bound_unbound_chatter_events"] = chatter_events
    temporal_gate["bound_unbound_chatter_pass"] = (
        temporal_gate["maximum_one_frame_bound_unbound_chatter_fraction"]
        <= args.maximum_switch_fraction
    )
    anchor_advisory = {}
    anchor_advisory["projection_correction_reference_pass"] = (
        temporal_gate["maximum_quiet_projection_correction_change_p99_m"]
        <= args.maximum_quiet_correction_change_p99
    )
    anchor_advisory["normal_change_reference_pass"] = (
        temporal_gate["maximum_quiet_tangent_plane_change_p99_degrees"]
        <= args.maximum_quiet_tangent_plane_change_p99_degrees
    )
    anchor_advisory["downstream_policy"] = (
        "reported but not silently promoted to a binding failure; connected-membrane "
        "native-velocity hysteresis and stable-ID eligibility debounce are audited separately"
    )
    temporal_gate["near_static_anchor_advisory"] = anchor_advisory
    temporal_gate["valid"] = temporal_gate["bound_unbound_chatter_pass"]
    structural_valid = manifest.get("complete") is True and not structural_errors
    report = {
        "schema": "physx-foamgenerator-surface-binding-sequence-audit/v4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "valid": structural_valid and temporal_gate["valid"],
        "structural_valid": structural_valid,
        "temporal_gate": temporal_gate,
        "thresholds": {
            "maximum_one_frame_bound_unbound_chatter_fraction": args.maximum_switch_fraction,
            "quiet_native_speed_maximum_m_per_s": args.quiet_speed_maximum,
            "maximum_quiet_projection_correction_change_p99_m": args.maximum_quiet_correction_change_p99,
            "maximum_quiet_tangent_plane_change_p99_degrees": args.maximum_quiet_tangent_plane_change_p99_degrees,
            "dynamic_diagnostic_reference_only": {
                "maximum_projection_correction_change_p99_m": args.maximum_correction_change_p99,
                "maximum_tangent_plane_change_p99_degrees": args.maximum_normal_change_p99_degrees,
            },
        },
        "binding_directory": str(directory),
        "manifest_sha256": sha256_file(manifest_path),
        "finite_difference_velocity_used": False,
        "native_velocity_bit_exact_required": True,
        "frames_audited": len(states),
        "states_audited": audited_states,
        "structural_errors": structural_errors,
        "transitions": transitions,
    }
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(json.dumps({key: report[key] for key in ("valid", "structural_valid", "temporal_gate", "frames_audited", "states_audited")}, indent=2))


if __name__ == "__main__":
    main()
