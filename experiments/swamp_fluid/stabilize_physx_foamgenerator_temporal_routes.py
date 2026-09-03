"""Remove isolated hard surface-route pulses from temporal foam bindings.

The temporal binder deliberately treats the absolute surface-distance and
continuity gates as hard constraints.  A marker can therefore complete its
two-sample entry crossfade, become a hard surface marker for one frame, and
be forced back to the free route on the following frame.  This tool performs
an offline, render-only three-frame deglitch pass:

    hard route 0, 1, 0  ->  0, 0, 0
    weight     .5, 1, 0 -> .5, .5, 0

It never extends a surface attachment beyond a failed geometry gate.  The
middle sample remains a balanced surface/free crossfade.  Density is then
recomputed sequentially for every stable ID so the correction cannot leave a
one-frame optical-density impulse in neighbouring patches.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from build_physx_foamgenerator_temporal_surface_binding import (
    atomic_json,
    atomic_npz,
    continuous_surface_density,
    quantiles,
    sha256_file,
)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def id_index(values: np.ndarray) -> dict[int, int]:
    return {int(value): index for index, value in enumerate(values)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Suppress isolated 0-1-0 hard surface-route pulses while preserving "
            "the absolute geometry gates and recomputing temporal density."
        )
    )
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()

    input_directory = args.input_directory.resolve()
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)

    input_manifest_path = input_directory / "manifest.json"
    manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("product") != "physx_foamgenerator_temporal_surface_binding":
        raise RuntimeError("Input is not a temporal FoamGenerator surface binding")
    if manifest.get("complete") is not True:
        raise RuntimeError("Input temporal binding is incomplete")

    samples = list(manifest.get("samples", []))
    frames = [int(sample["output_frame"]) for sample in samples]
    if not frames or any(b != a + 1 for a, b in zip(frames, frames[1:])):
        raise RuntimeError("Input temporal binding must contain consecutive frames")

    route_by_frame: list[dict[int, int]] = []
    weight_by_frame: list[dict[int, float]] = []
    for sample in samples:
        frame = int(sample["output_frame"])
        data = load_npz(input_directory / f"binding_{frame:04d}.npz")
        active_id = np.asarray(data["active_id"], dtype=np.uint64)
        route_state = np.asarray(data["route_state"], dtype=np.int8)
        surface_weight = np.asarray(data["surface_weight"], dtype=np.float64)
        route_by_frame.append(
            {int(marker_id): int(route) for marker_id, route in zip(active_id, route_state)}
        )
        weight_by_frame.append(
            {
                int(marker_id): float(weight)
                for marker_id, weight in zip(active_id, surface_weight)
            }
        )

    suppressed_by_frame: dict[int, set[int]] = {}
    suppressed_events: list[dict] = []
    for offset in range(1, len(frames) - 1):
        first = route_by_frame[offset - 1]
        middle = route_by_frame[offset]
        last = route_by_frame[offset + 1]
        persistent = set(first) & set(middle) & set(last)
        for marker_id in persistent:
            route = (first[marker_id], middle[marker_id], last[marker_id])
            if route != (0, 1, 0):
                continue
            middle_weight = weight_by_frame[offset][marker_id]
            if middle_weight <= 0.5:
                continue
            frame = frames[offset]
            suppressed_by_frame.setdefault(frame, set()).add(marker_id)
            suppressed_events.append(
                {
                    "id": marker_id,
                    "frames": [frames[offset - 1], frame, frames[offset + 1]],
                    "route_before": [0, 1, 0],
                    "middle_surface_weight_before": middle_weight,
                    "middle_surface_weight_after": 0.5,
                }
            )

    configuration = manifest["configuration"]
    density_configuration = configuration["continuous_density"]
    density_support_length = float(density_configuration["support_length_m"])
    density_cutoff_multiplier = float(density_configuration["cutoff_multiplier"])
    density_optical_scale = float(density_configuration["optical_scale"])
    density_ema_alpha = float(density_configuration["stable_id_ema_alpha"])
    density_maximum_change = float(
        density_configuration["maximum_change_per_output_frame"]
    )

    output_manifest = json.loads(json.dumps(manifest))
    output_manifest["complete"] = False
    output_manifest["created_utc"] = datetime.now(timezone.utc).isoformat()
    output_manifest.pop("completed_utc", None)
    output_manifest["samples"] = []
    output_manifest["configuration"]["route_deglitch"] = {
        "method": "offline stable-ID three-frame isolated hard-route suppression",
        "hard_route_pattern_before": [0, 1, 0],
        "hard_route_pattern_after": [0, 0, 0],
        "middle_surface_weight_after": 0.5,
        "middle_free_weight_after": 0.5,
        "absolute_surface_distance_gate_relaxed": False,
        "surface_attachment_extended": False,
        "density_recomputed_after_weight_change": True,
        "suppressed_events": len(suppressed_events),
        "render_only_calibration": True,
    }
    output_manifest["inputs"]["unstabilized_temporal_binding_directory"] = str(
        input_directory
    )
    output_manifest["inputs"]["unstabilized_temporal_binding_manifest_sha256"] = (
        sha256_file(input_manifest_path)
    )
    script_path = Path(__file__).resolve()
    output_manifest["producer"]["route_stabilizer_script"] = str(script_path)
    output_manifest["producer"]["route_stabilizer_script_sha256"] = sha256_file(
        script_path
    )
    output_manifest["route_deglitch_events"] = suppressed_events
    output_manifest_path = output_directory / "manifest.json"
    atomic_json(output_manifest_path, output_manifest)

    previous_density: dict[int, float] = {}
    for sample in samples:
        frame = int(sample["output_frame"])
        data = load_npz(input_directory / f"binding_{frame:04d}.npz")
        active_id = np.asarray(data["active_id"], dtype=np.uint64)
        active_source_row = np.asarray(data["active_source_row"], dtype=np.int32)
        surface_weight = np.asarray(data["surface_weight"], dtype=np.float64).copy()
        free_weight = np.asarray(data["free_weight"], dtype=np.float64).copy()
        route_state = np.asarray(data["route_state"], dtype=np.int8).copy()
        route_reason = np.asarray(data["route_reason"], dtype=np.int8).copy()
        active_lookup = id_index(active_id)

        for marker_id in suppressed_by_frame.get(frame, set()):
            index = active_lookup[marker_id]
            route_state[index] = 0
            route_reason[index] = 6  # isolated hard-surface pulse softened
            surface_weight[index] = 0.5
            free_weight[index] = 0.5

        if np.max(np.abs(surface_weight + free_weight - 1.0), initial=0.0) > 1.0e-12:
            raise RuntimeError(f"Route-weight conservation failed at frame {frame:04d}")

        bound_id = np.asarray(data["id"], dtype=np.uint64)
        bound_lookup = id_index(bound_id)
        bound_active_index = np.asarray(
            [active_lookup[int(marker_id)] for marker_id in bound_id], dtype=np.int64
        )
        bound_weight = surface_weight[bound_active_index]
        if np.any(bound_weight <= 0.0):
            raise RuntimeError(
                f"Deglitch unexpectedly removed a bound payload at frame {frame:04d}"
            )

        raw_distance_by_id = {
            int(marker_id): float(distance)
            for marker_id, distance in zip(bound_id, data["raw_closest_distance"])
        }
        raw_distance_by_id.update(
            {
                int(marker_id): float(distance)
                for marker_id, distance in zip(
                    np.asarray(data["unbound_id"], dtype=np.uint64),
                    np.asarray(data["unbound_distance"], dtype=np.float64),
                )
            }
        )
        free_mask = free_weight > 0.0
        rebuilt_unbound_id = active_id[free_mask]
        try:
            rebuilt_unbound_distance = np.asarray(
                [raw_distance_by_id[int(marker_id)] for marker_id in rebuilt_unbound_id],
                dtype=np.float32,
            )
        except KeyError as exc:
            raise RuntimeError(
                f"Missing raw distance for free marker {int(exc.args[0])} "
                f"at frame {frame:04d}"
            ) from exc

        target_all = np.zeros(len(active_id), dtype=np.float64)
        density_raw_all = np.zeros(len(active_id), dtype=np.float64)
        density_pair_count = 0
        if len(bound_id):
            target, density_raw, density_pair_count = continuous_surface_density(
                np.asarray(data["position"], dtype=np.float64),
                bound_weight,
                density_support_length,
                density_cutoff_multiplier,
                density_optical_scale,
            )
            target_all[bound_active_index] = target
            density_raw_all[bound_active_index] = density_raw

        density_all = target_all.copy()
        for index, marker_id_value in enumerate(active_id):
            marker_id = int(marker_id_value)
            if marker_id not in previous_density:
                continue
            requested_delta = density_ema_alpha * (
                target_all[index] - previous_density[marker_id]
            )
            density_all[index] = previous_density[marker_id] + float(
                np.clip(
                    requested_delta,
                    -density_maximum_change,
                    density_maximum_change,
                )
            )
        previous_density = {
            int(marker_id): float(value)
            for marker_id, value in zip(active_id, density_all)
        }

        data["surface_weight"] = surface_weight.astype(np.float32)
        data["free_weight"] = free_weight.astype(np.float32)
        data["route_state"] = route_state
        data["route_reason"] = route_reason
        data["density_raw"] = density_raw_all[bound_active_index].astype(np.float32)
        data["density"] = density_all[bound_active_index].astype(np.float32)
        data["surface_opacity"] = (
            np.asarray(data["opacity"], dtype=np.float32) * bound_weight
        ).astype(np.float32)
        data["unbound_id"] = rebuilt_unbound_id.astype(np.uint64)
        data["unbound_source_row"] = active_source_row[free_mask].astype(np.int32)
        data["unbound_distance"] = rebuilt_unbound_distance
        data["unbound_weight"] = free_weight[free_mask].astype(np.float32)

        output_path = output_directory / f"binding_{frame:04d}.npz"
        atomic_npz(output_path, **data)

        record = json.loads(json.dumps(sample))
        record["file"] = output_path.name
        record["bytes"] = output_path.stat().st_size
        record["sha256"] = sha256_file(output_path)
        record["surface_count"] = int(len(bound_id))
        record["free_count"] = int(np.count_nonzero(free_mask))
        record["crossfade_count"] = int(
            np.count_nonzero((surface_weight > 0.0) & (surface_weight < 1.0))
        )
        record["hard_surface_state_count"] = int(np.count_nonzero(route_state))
        record["surface_fraction"] = (
            float(np.mean(surface_weight)) if len(surface_weight) else 0.0
        )
        record["density"] = quantiles(density_all[bound_active_index])
        record["density_raw"] = quantiles(density_raw_all[bound_active_index])
        record["density_neighbor_pairs"] = int(density_pair_count)
        record["route_deglitch_count"] = len(suppressed_by_frame.get(frame, set()))
        output_manifest["samples"].append(record)
        atomic_json(output_manifest_path, output_manifest)
        print(
            f"[route-deglitch] frame={frame:04d} "
            f"suppressed={record['route_deglitch_count']} "
            f"surface={record['surface_count']} free={record['free_count']}",
            flush=True,
        )

    output_manifest["complete"] = True
    output_manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    output_manifest["summary"] = {
        "frames": len(output_manifest["samples"]),
        "maximum_surface_markers": max(
            sample["surface_count"] for sample in output_manifest["samples"]
        ),
        "maximum_free_fraction": max(
            1.0 - sample["surface_fraction"] for sample in output_manifest["samples"]
        ),
        "one_frame_route_chatter_suppressed": len(suppressed_events),
        "native_velocity_consumed": True,
        "finite_difference_velocity_used": False,
    }
    atomic_json(output_manifest_path, output_manifest)
    print(json.dumps({"valid": True, **output_manifest["summary"]}, indent=2))


if __name__ == "__main__":
    main()
