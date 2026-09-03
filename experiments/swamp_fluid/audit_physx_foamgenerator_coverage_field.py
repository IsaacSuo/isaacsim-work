"""Independent audit of the water-surface foam mask prototype."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.foam_coverage_field import kernel_integral_area
from whitewater.mesh_surface_sampler import TriangleSurfaceSampler


ROUTE_LIFECYCLE_CULLED = 0
ROUTE_SURFACE_COVERAGE = 1
ROUTE_FREE_FOAM_PARTICLE = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def exact(left, right, message):
    require(np.array_equal(np.asarray(left), np.asarray(right)), message)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coverage_field_directory", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = args.coverage_field_directory.resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(
        manifest.get("schema")
        == "physx-foamgenerator-water-surface-mask-prototype/v2",
        "Unexpected mask schema",
    )
    require(
        manifest.get("product")
        == "physx_foamgenerator_water_surface_mask_prototype"
        and manifest.get("complete") is True,
        "Mask prototype is incomplete",
    )
    semantics = manifest["semantics"]
    require(
        semantics.get("maturity")
        == "prototype; not a mature flow-following foam system",
        "Maturity is overstated",
    )
    require(
        semantics.get("physical_film_area_claimed") is False
        and semantics.get("physical_gas_volume_claimed") is False
        and semantics.get("coverage_integral_is_physical_area") is False,
        "Render proxy is mislabeled as a physical quantity",
    )
    transport = manifest["transport"]
    require(
        transport.get("finite_difference_velocity_used") is False,
        "Finite-difference velocity is declared",
    )
    require(
        transport.get("source_normalization", "").startswith(
            "kernel times actual face area"
        ),
        "Actual-area source normalization is not declared",
    )

    producer = manifest["producer"]
    producer_paths = {
        "script_sha256": Path(producer["script"]),
        "coverage_module_sha256": Path(producer["script"]).parent
        / "whitewater"
        / "foam_coverage_field.py",
        "surface_sampler_sha256": Path(producer["script"]).parent
        / "whitewater"
        / "mesh_surface_sampler.py",
    }
    for hash_key, path in producer_paths.items():
        require(
            path.is_file() and sha256_file(path) == producer[hash_key],
            f"Producer hash mismatch: {path}",
        )

    calibration = manifest["render_only_calibration"]
    limits = {
        "maximum_source_measure_residual_m2": 1.0e-12,
        "maximum_history_measure_residual_m2": 1.0e-12,
        "maximum_one_frame_surface_free_surface_chatter_events": 0,
        "route_max_hold_distance_m": float(calibration["route_max_hold_distance_m"]),
        "history_max_projection_distance_m": float(
            calibration["history_max_projection_distance_m"]
        ),
    }
    frame_reports = []
    transitions = []
    route_history = {}
    totals = {"native": 0, "surface": 0, "free": 0, "culled": 0}
    previous = None
    forbidden = {"film_area", "physical_film_area", "gas_volume"}

    for sample in manifest["samples"]:
        frame = int(sample["output_frame"])
        output_path = directory / sample["file"]
        cache_path = Path(sample["source_foam_cache"])
        binding_path = Path(sample["surface_binding"])
        surface_path = Path(sample["surface"])
        for path, expected, label in (
            (output_path, sample["sha256"], "output"),
            (cache_path, sample["source_foam_cache_sha256"], "source foam"),
            (binding_path, sample["surface_binding_sha256"], "binding"),
            (surface_path, sample["surface_sha256"], "surface"),
        ):
            require(sha256_file(path) == expected, f"{label} hash mismatch at {frame}")
        with np.load(output_path, allow_pickle=False) as loaded:
            data = {name: np.asarray(loaded[name]).copy() for name in loaded.files}
        with np.load(cache_path, allow_pickle=False) as loaded:
            cache = {name: np.asarray(loaded[name]).copy() for name in loaded.files}
        with np.load(binding_path, allow_pickle=False) as loaded:
            binding = {name: np.asarray(loaded[name]).copy() for name in loaded.files}
        require(not forbidden.intersection(data), f"Physical field leaked at {frame}")
        require(
            str(np.asarray(data["schema"]))
            == "physx-foamgenerator-water-surface-mask-frame/v2",
            f"Frame schema mismatch at {frame}",
        )

        ids = np.asarray(cache["id"], dtype=np.int64)
        routing_ids = np.asarray(data["routing_id"], dtype=np.int64)
        rows = np.asarray(data["routing_source_row"], dtype=np.int64)
        raw_route = np.asarray(data["raw_routing_code"], dtype=np.uint8)
        route = np.asarray(data["routing_code"], dtype=np.uint8)
        raw_distance = np.asarray(
            data["raw_routing_surface_distance"], dtype=np.float64
        )
        exact(routing_ids, ids, f"Routing IDs are not total at {frame}")
        exact(rows, np.arange(len(ids)), f"Routing rows are not total at {frame}")
        require(len(np.unique(ids)) == len(ids), f"Duplicate native ID at {frame}")
        require(np.all(np.isin(route, (0, 1, 2))), f"Unknown route at {frame}")

        bound_rows = np.asarray(binding["source_row"], dtype=np.int64)
        unbound_rows = np.asarray(binding["unbound_source_row"], dtype=np.int64)
        inactive = np.ones(len(ids), dtype=bool)
        inactive[np.concatenate((bound_rows, unbound_rows))] = False
        exact(
            np.flatnonzero(raw_route == ROUTE_SURFACE_COVERAGE),
            bound_rows,
            f"Raw surface route mismatch at {frame}",
        )
        exact(
            np.flatnonzero(raw_route == ROUTE_FREE_FOAM_PARTICLE),
            unbound_rows,
            f"Raw free route mismatch at {frame}",
        )
        exact(
            raw_distance[bound_rows].astype(np.float32),
            np.asarray(binding["distance"], dtype=np.float32),
            f"Bound distance mismatch at {frame}",
        )
        exact(
            raw_distance[unbound_rows].astype(np.float32),
            np.asarray(binding["unbound_distance"], dtype=np.float32),
            f"Unbound distance mismatch at {frame}",
        )
        exact(
            np.flatnonzero(route == ROUTE_LIFECYCLE_CULLED),
            np.flatnonzero(inactive),
            f"Lifecycle cull mismatch at {frame}",
        )
        held_surface_rows = np.flatnonzero(
            (route == ROUTE_SURFACE_COVERAGE)
            & (raw_route == ROUTE_FREE_FOAM_PARTICLE)
        )
        require(
            np.all(
                raw_distance[held_surface_rows]
                <= limits["route_max_hold_distance_m"] + 1.0e-9
            ),
            f"Debounce held foam too far from surface at {frame}",
        )

        sources = np.asarray(data["coverage_sources"])
        coverage_rows = np.flatnonzero(route == ROUTE_SURFACE_COVERAGE)
        exact(sources["source_row"], coverage_rows, f"Coverage rows mismatch at {frame}")
        exact(sources["id"].astype(np.int64), ids[coverage_rows], f"Coverage IDs mismatch at {frame}")
        exact(sources["native_velocity"], cache["velocity"][coverage_rows], f"Native velocity changed at {frame}")
        normal = sources["normal"].astype(np.float64)
        normal /= np.maximum(np.linalg.norm(normal, axis=1), 1.0e-12)[:, None]
        native = cache["velocity"][coverage_rows].astype(np.float64)
        expected_tangent = native - np.sum(native * normal, axis=1)[:, None] * normal
        require(
            np.max(np.abs(sources["velocity"] - expected_tangent), initial=0.0)
            <= 1.0e-6,
            f"Surface carrier is not native tangent at {frame}",
        )
        expected_coverage = np.clip(
            float(calibration["initial_coverage"])
            * cache["opacity"][coverage_rows].astype(np.float64),
            0.0,
            1.0 - 1.0e-7,
        )
        expected_tau = -np.log1p(-expected_coverage)
        require(
            np.max(
                np.abs(
                    sources["instantaneous_target_optical_depth"].astype(np.float64)
                    - expected_tau
                ),
                initial=0.0,
            )
            <= 1.0e-6,
            f"Coverage calibration mismatch at {frame}",
        )
        require(
            np.max(
                np.abs(
                    sources["target_optical_depth"]
                    - sources["instantaneous_target_optical_depth"]
                ),
                initial=0.0,
            )
            == 0.0,
            f"Current source was temporally restamped at {frame}",
        )

        sampler = TriangleSurfaceSampler(surface_path)
        direct = sources["anchor_authority"] == 0
        binding_lookup = np.full(len(ids), -1, dtype=np.int64)
        binding_lookup[bound_rows] = np.arange(len(bound_rows))
        direct_binding = binding_lookup[coverage_rows[direct]]
        require(np.all(direct_binding >= 0), f"Direct anchor lacks binding at {frame}")
        exact(sources["position"][direct], binding["position"][direct_binding], f"Direct anchor position mismatch at {frame}")
        exact(sources["normal"][direct], binding["normal"][direct_binding], f"Direct anchor normal mismatch at {frame}")
        exact(sources["anchor_face_index"][direct], binding["triangle_id"][direct_binding], f"Direct anchor face mismatch at {frame}")
        held = sources["anchor_authority"] == 1
        if np.any(held):
            held_rows = coverage_rows[held]
            closest, _signed, _distance, held_normal, held_face = sampler.closest(
                cache["position"][held_rows].astype(np.float64)
            )
            require(np.max(np.abs(sources["position"][held] - closest), initial=0.0) <= 1.0e-6, f"Held anchor position mismatch at {frame}")
            require(np.max(np.abs(sources["normal"][held] - held_normal), initial=0.0) <= 1.0e-6, f"Held anchor normal mismatch at {frame}")
            exact(sources["anchor_face_index"][held], held_face, f"Held anchor face mismatch at {frame}")

        free_rows = np.flatnonzero(route == ROUTE_FREE_FOAM_PARTICLE)
        exact(data["free_foam_source_row"], free_rows, f"Free rows mismatch at {frame}")
        exact(data["free_foam_id"].astype(np.int64), ids[free_rows], f"Free IDs mismatch at {frame}")
        for name in ("position", "velocity", "radius", "opacity", "shape"):
            exact(data[f"free_foam_{name}"], cache[name][free_rows], f"Free {name} changed at {frame}")

        face_count = len(sampler.mesh.faces)
        active = np.asarray(data["active_face_indices"], dtype=np.int64)
        area = np.asarray(data["active_face_area"], dtype=np.float64)
        tau = np.asarray(data["active_face_optical_depth"], dtype=np.float64)
        optical_measure = np.asarray(data["active_face_optical_measure"], dtype=np.float64)
        current_measure = np.asarray(data["active_face_current_optical_measure"], dtype=np.float64)
        history_measure = np.asarray(data["active_face_history_optical_measure"], dtype=np.float64)
        coverage = np.asarray(data["active_face_coverage"], dtype=np.float64)
        require(len(np.unique(active)) == len(active) and np.all((active >= 0) & (active < face_count)), f"Invalid active faces at {frame}")
        require(np.max(np.abs(optical_measure - tau * area), initial=0.0) <= 1.0e-12, f"Optical measure mismatch at {frame}")
        require(np.all(current_measure >= 0.0) and np.all(history_measure >= 0.0), f"Negative measure at {frame}")
        require(np.max(np.abs(coverage - (-np.expm1(-tau))), initial=0.0) <= 1.0e-6, f"Coverage/tau mismatch at {frame}")
        current_tau = current_measure / area
        history_tau = history_measure / area
        expected_output_tau = np.zeros(len(active), dtype=np.float64)
        has_current = current_measure > 0.0
        has_history = history_measure > 0.0
        both = has_current & has_history
        expected_output_tau[both] = float(calibration["history_weight"]) * history_tau[both] + (1.0 - float(calibration["history_weight"])) * current_tau[both]
        expected_output_tau[has_current & ~has_history] = current_tau[has_current & ~has_history]
        expected_output_tau[has_history & ~has_current] = float(calibration["history_weight"]) * history_tau[has_history & ~has_current]
        require(np.max(np.abs(tau - expected_output_tau), initial=0.0) <= 2.0e-6, f"Temporal assimilation mismatch at {frame}")

        target_measure = sources["target_optical_depth"].astype(np.float64) * kernel_integral_area(sources["kernel_support_radius"])
        exact_target = np.asarray(data["source_target_optical_measure"], dtype=np.float64)
        allocated_source = np.asarray(data["source_allocated_optical_measure"], dtype=np.float64)
        require(np.max(np.abs(exact_target - target_measure), initial=0.0) <= 1.0e-12, f"Source target measure mismatch at {frame}")
        source_residual = float(np.max(np.abs(exact_target - allocated_source), initial=0.0))
        require(source_residual <= limits["maximum_source_measure_residual_m2"], f"Source measure did not close at {frame}")
        require(abs(float(current_measure.sum()) - float(allocated_source.sum())) <= 1.0e-12, f"Face/source current measure mismatch at {frame}")
        require(np.all(data["source_face_hit_count"] > 0), f"Coverage source missed surface at {frame}")

        history_report = sample["field_metrics"]["history"]
        if previous is None:
            require(float(history_measure.sum()) == 0.0, f"First frame contains history at {frame}")
        else:
            dt = (frame - previous["frame"]) / 30.0
            predicted = previous["centres"] + previous["velocity"] * dt
            _closest, _signed, projected_distance, projected_normal, _face = sampler.closest(predicted)
            alignment = np.abs(np.sum(previous["normal"] * projected_normal, axis=1))
            accepted = (
                projected_distance <= limits["history_max_projection_distance_m"]
            ) & (
                alignment >= float(transport["history_minimum_normal_alignment"])
            )
            decayed_input = previous["tau"] * previous["area"] * np.exp(
                -dt / float(calibration["history_decay_seconds"])
            )
            expected_input = float(decayed_input.sum(dtype=np.float64))
            expected_accepted = float(decayed_input[accepted].sum(dtype=np.float64))
            actual_history = float(history_measure.sum(dtype=np.float64))
            history_residual = abs(expected_accepted - actual_history)
            require(history_residual <= limits["maximum_history_measure_residual_m2"], f"History remap did not close at {frame}")
            require(abs(expected_input - float(history_report["input_optical_measure_m2"])) <= 1.0e-12, f"History input ledger mismatch at {frame}")
            require(int(history_report["rejected_distance_samples"]) == int(np.count_nonzero(projected_distance > limits["history_max_projection_distance_m"])), f"History distance rejection mismatch at {frame}")
            transitions.append(
                {
                    "previous_frame": previous["frame"],
                    "current_frame": frame,
                    "input_history_measure_m2": expected_input,
                    "accepted_history_measure_m2": actual_history,
                    "rejected_history_measure_m2": expected_input - actual_history,
                    "accepted_history_measure_fraction": actual_history / max(expected_input, 1.0e-30),
                    "maximum_projection_distance_m": float(projected_distance.max(initial=0.0)),
                    "distance_rejected_samples": int(np.count_nonzero(~accepted)),
                    "history_measure_residual_m2": history_residual,
                    "valid": True,
                }
            )

        counts = {
            "native": int(len(ids)),
            "surface": int(np.count_nonzero(route == ROUTE_SURFACE_COVERAGE)),
            "free": int(np.count_nonzero(route == ROUTE_FREE_FOAM_PARTICLE)),
            "culled": int(np.count_nonzero(route == ROUTE_LIFECYCLE_CULLED)),
        }
        require(counts["native"] == counts["surface"] + counts["free"] + counts["culled"], f"Inventory does not close at {frame}")
        for key in totals:
            totals[key] += counts[key]
        for marker_id, route_code in zip(ids, route):
            route_history.setdefault(int(marker_id), []).append((frame, int(route_code)))
        frame_reports.append(
            {
                "output_frame": frame,
                "counts": counts,
                "held_surface_count": int(len(held_surface_rows)),
                "maximum_held_surface_distance_m": float(raw_distance[held_surface_rows].max(initial=0.0)),
                "source_measure_residual_m2": source_residual,
                "active_faces": int(len(active)),
                "coverage_integral_m2_render_proxy": float(np.sum(area * coverage)),
            }
        )
        previous = {
            "frame": frame,
            "centres": np.asarray(data["active_face_centres"], dtype=np.float64),
            "velocity": np.asarray(data["active_face_velocity"], dtype=np.float64),
            "normal": np.asarray(data["active_face_normal"], dtype=np.float64),
            "tau": tau,
            "area": area,
        }

    chatter = 0
    for history in route_history.values():
        for first, middle, last in zip(history, history[1:], history[2:]):
            consecutive = first[0] + 1 == middle[0] and middle[0] + 1 == last[0]
            aba = first[1] == last[1] and first[1] != middle[1]
            chatter += int(consecutive and aba and set((first[1], middle[1])) == {1, 2})
    require(
        chatter <= limits["maximum_one_frame_surface_free_surface_chatter_events"],
        "One-frame surface/free chatter exceeds gate",
    )

    report = {
        "schema": "physx-foamgenerator-water-surface-mask-audit/v2",
        "valid": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "maturity": "water-surface foam mask prototype",
        "limits": limits,
        "gates": {
            "native_id_inventory_closed": True,
            "routes_mutually_exclusive": True,
            "route_debounce_distance_bounded": True,
            "one_frame_route_chatter_zero": True,
            "source_measure_area_normalized_and_closed": True,
            "history_measure_surface_advected": True,
            "history_projection_distance_bounded": True,
            "history_measure_area_normalized_and_closed": True,
            "native_velocity_preserved": True,
            "finite_difference_velocity_used": False,
            "physical_film_area_or_gas_volume_claimed": False,
        },
        "one_frame_surface_free_surface_chatter_events": chatter,
        "totals": totals,
        "transitions": transitions,
        "frames": frame_reports,
    }
    output = args.output.resolve() if args.output else directory / "audit_report.json"
    require(not output.exists(), f"Refusing to overwrite audit: {output}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"valid": True, "output": str(output), "totals": totals, "transitions": transitions}, indent=2))


if __name__ == "__main__":
    main()
