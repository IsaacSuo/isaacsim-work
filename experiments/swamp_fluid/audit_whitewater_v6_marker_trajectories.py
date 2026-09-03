"""Independently audit persistent v6 marker trajectories and events."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.liquid_fields import GridSpec
from whitewater.marker_birth import PhaseKind, sample_scalar_trilinear
from whitewater.marker_solver import (
    EVENT_DTYPE,
    SOLVER_DTYPE,
    SUPPORT_LOSS_DTYPE,
    NonterminalTransitionKind,
    TerminalEventKind,
)
from whitewater.render_surface_support import (
    build_render_surface_support,
    load_render_surface_support_recipe,
)
from whitewater.point_collisions import SparsePointCollisionScene
from whitewater.scene_contract import SceneContract
from whitewater.state_machine import WhitewaterState


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory_directory", type=Path)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    manifest_path = args.trajectory_directory / "manifest.json"
    manifest = load_json(manifest_path)
    issues = []
    if manifest.get("product") != "whitewater_v6_marker_trajectories":
        issues.append("Unexpected trajectory product")
    if not manifest.get("complete"):
        issues.append("Trajectory manifest is incomplete")

    inputs = manifest.get("inputs", {})
    liquid_manifest_path = Path(inputs.get("liquid_manifest", ""))
    birth_manifest_path = Path(inputs.get("birth_manifest", ""))
    input_hashes_match = True
    for label, path, expected_hash in (
        ("liquid", liquid_manifest_path, inputs.get("liquid_manifest_sha256")),
        ("birth", birth_manifest_path, inputs.get("birth_manifest_sha256")),
    ):
        if not path.is_file() or sha256_file(path) != expected_hash:
            issues.append(f"{label.capitalize()} input manifest is missing or changed")
            input_hashes_match = False
    liquid_manifest = load_json(liquid_manifest_path)
    birth_manifest = load_json(birth_manifest_path)
    liquid_samples = liquid_manifest["samples"]
    birth_samples = birth_manifest["samples"]
    samples = manifest.get("samples", [])
    if len(samples) != len(liquid_samples) or len(samples) != len(birth_samples):
        issues.append("Trajectory and input sample counts differ")

    grid = manifest["grid"]
    spec = GridSpec(
        tuple(grid["origin"]), float(grid["spacing"]), tuple(grid["shape"])
    )
    solid_clearance = (
        float(manifest["configuration"]["model"]["solid_clearance_cells"])
        * spec.spacing
    )
    maximum_cfl_target = float(
        manifest["configuration"]["model"]["maximum_cfl"]
    )
    secondary_padding = (
        float(
            manifest["configuration"]["model"].get(
                "secondary_domain_padding_cells", 0.0
            )
        )
        * spec.spacing
    )
    secondary_minimum = np.asarray(spec.origin) - secondary_padding
    secondary_maximum = np.asarray(spec.maximum) + secondary_padding
    previous = np.empty(0, dtype=SOLVER_DTYPE)
    terminated_ids = set()
    cumulative_born = {1: 0.0, 2: 0.0}
    cumulative_terminal = {1: 0.0, 2: 0.0}
    maximum_volume_error = 0.0
    maximum_shape_error = 0.0
    minimum_collision_margin = float("inf")
    maximum_cfl = 0.0
    maximum_shared_displacement = 0.0
    total_surface_support_losses = 0
    total_events = 0
    total_markers_seen = 0
    hashes_ok = True
    structural_ok = True
    identity_ok = True
    transition_ok = True
    spatial_ok = True
    conservation_ok = True
    no_domain_escape = True
    maximum_spray_outside_core = 0
    support_recipe = load_render_surface_support_recipe(
        manifest["configuration"]["render_surface_support"]["recipe_manifest"]
    )
    secondary_scene_path_value = inputs.get("secondary_scene_contract")
    secondary_scene = None
    secondary_scene_input_ok = True
    if secondary_scene_path_value:
        secondary_scene_path = Path(secondary_scene_path_value)
        expected_scene_hash = inputs.get("secondary_scene_contract_sha256")
        if (
            not secondary_scene_path.is_file()
            or sha256_file(secondary_scene_path) != expected_scene_hash
        ):
            issues.append("Secondary scene contract is missing or changed")
            secondary_scene_input_ok = False
        else:
            secondary_scene = SparsePointCollisionScene(
                SceneContract.load(secondary_scene_path)
            )
    source_directory = Path(liquid_manifest["source"]["directory"])
    minimum_secondary_collision_margin = float("inf")
    secondary_endpoint_queries = 0
    secondary_endpoint_valid = 0
    secondary_endpoint_boundary_invalid = 0
    recorded_secondary_queries = 0
    recorded_secondary_projections = 0

    liquid_directory = liquid_manifest_path.parent
    birth_directory = birth_manifest_path.parent
    for index, sample in enumerate(samples):
        path = args.trajectory_directory / sample["file"]
        if not path.is_file():
            issues.append(f"Missing trajectory file {path.name}")
            structural_ok = False
            continue
        if sha256_file(path) != sample.get("sha256"):
            issues.append(f"Trajectory hash mismatch for {path.name}")
            hashes_ok = False
        with np.load(path) as cache:
            markers = np.asarray(cache["markers"]).copy()
            events = np.asarray(cache["terminal_events"]).copy()
            support_losses = np.asarray(cache["surface_support_losses"]).copy()
            source_sample = int(cache["source_sample_index"])
            interval_start = float(cache["interval_start"])
            interval_end = float(cache["interval_end"])
        if (
            markers.dtype != SOLVER_DTYPE
            or events.dtype != EVENT_DTYPE
            or support_losses.dtype != SUPPORT_LOSS_DTYPE
        ):
            issues.append(f"{path.name} has an unexpected structured dtype")
            structural_ok = False
            continue
        if source_sample != int(sample["source_sample_index"]):
            issues.append(f"{path.name} source sample disagrees with manifest")
            structural_ok = False
        if len(markers) != int(sample["active_count"]):
            issues.append(f"{path.name} active count disagrees with manifest")
            structural_ok = False
        if len(markers) and np.any(np.diff(markers["id"]) <= 0):
            issues.append(f"{path.name} marker IDs are not sorted and unique")
            identity_ok = False
        if len(events) and np.any(np.diff(events["event_id"]) <= 0):
            issues.append(f"{path.name} event IDs are not sorted and unique")
            identity_ok = False
        numeric_marker_fields = (
            "birth_time",
            "state_age",
            "position",
            "velocity",
            "physical_radius",
            "representative_count",
            "phase_volume",
            "shape",
        )
        if any(not np.isfinite(markers[name]).all() for name in numeric_marker_fields):
            issues.append(f"{path.name} contains non-finite marker data")
            structural_ok = False
        if len(events) and any(
            not np.isfinite(events[name]).all()
            for name in (
                "event_time",
                "position",
                "normal",
                "phase_volume",
                "physical_radius",
                "representative_count",
                "impact_speed",
            )
        ):
            issues.append(f"{path.name} contains non-finite event data")
            structural_ok = False

        birth_path = birth_directory / birth_samples[index]["file"]
        with np.load(birth_path) as birth_cache:
            births = np.asarray(birth_cache["births"])
        for phase in (1, 2):
            cumulative_born[phase] += float(
                births["phase_volume"][births["phase"] == phase].sum(dtype=np.float64)
            )
            cumulative_terminal[phase] += float(
                events["phase_volume"][events["phase"] == phase].sum(dtype=np.float64)
            )
        total_events += len(events)
        total_surface_support_losses += len(support_losses)
        total_markers_seen += len(markers)

        available_ids = np.concatenate((previous["id"], births["id"]))
        if len(np.unique(available_ids)) != len(available_ids):
            issues.append(f"{path.name} rebirths an already-active marker ID")
            identity_ok = False
        event_marker_ids = events["marker_id"]
        if np.setdiff1d(event_marker_ids, available_ids).size:
            issues.append(f"{path.name} terminates an unknown marker ID")
            identity_ok = False
        for marker_id in event_marker_ids.tolist():
            if marker_id in terminated_ids:
                issues.append(f"Marker {marker_id} has more than one terminal event")
                identity_ok = False
            terminated_ids.add(marker_id)
        expected_survivors = np.setdiff1d(
            available_ids, event_marker_ids, assume_unique=False
        )
        if not np.array_equal(np.sort(markers["id"]), np.sort(expected_survivors)):
            issues.append(f"{path.name} active identity balance is inconsistent")
            identity_ok = False

        if len(support_losses):
            valid_loss_kind = support_losses["kind"] == np.uint8(
                NonterminalTransitionKind.SURFACE_SUPPORT_LOSS_REENTRAINMENT
            )
            if np.any(~valid_loss_kind):
                issues.append(f"{path.name} contains an unknown nonterminal transition")
                transition_ok = False
            if np.setdiff1d(support_losses["marker_id"], available_ids).size:
                issues.append(f"{path.name} re-entrains an unknown marker ID")
                identity_ok = False
            if (
                not np.isfinite(support_losses["event_time"]).all()
                or not np.isfinite(support_losses["position"]).all()
                or not np.isfinite(support_losses["phase_volume"]).all()
                or not np.isfinite(support_losses["sampled_support"]).all()
                or np.any(support_losses["phase_volume"] <= 0.0)
            ):
                issues.append(f"{path.name} contains invalid support-loss records")
                structural_ok = False
            if len(support_losses) != int(sample.get("surface_support_loss_count", -1)):
                issues.append(f"{path.name} support-loss count disagrees with manifest")
                structural_ok = False

        if len(previous):
            shared, old_index, new_index = np.intersect1d(
                previous["id"], markers["id"], return_indices=True, assume_unique=True
            )
            if len(shared):
                immutable = (
                    "channel",
                    "phase",
                    "source_node_id",
                    "source_emission_index",
                    "source_sample",
                    "birth_time",
                    "physical_radius",
                    "representative_count",
                    "phase_volume",
                    "random_key",
                )
                for name in immutable:
                    if not np.array_equal(previous[name][old_index], markers[name][new_index]):
                        issues.append(f"{path.name} changed immutable marker field {name}")
                        identity_ok = False
                displacement = np.linalg.norm(
                    markers["position"][new_index] - previous["position"][old_index], axis=1
                )
                maximum_shared_displacement = max(
                    maximum_shared_displacement, float(displacement.max())
                )
                old_state = previous["state"][old_index]
                new_state = markers["state"][new_index]
                legal = (
                    (old_state == new_state)
                    | (
                        (old_state == np.uint8(WhitewaterState.ENTRAINED_BUBBLE))
                        & (new_state == np.uint8(WhitewaterState.SURFACE_BUBBLE))
                    )
                    | (
                        (old_state == np.uint8(WhitewaterState.SURFACE_BUBBLE))
                        & (new_state == np.uint8(WhitewaterState.ENTRAINED_BUBBLE))
                        & np.isin(shared, support_losses["marker_id"])
                    )
                )
                if np.any(~legal):
                    issues.append(f"{path.name} contains an illegal persistent state transition")
                    transition_ok = False

        valid_states = np.isin(
            markers["state"],
            (
                np.uint8(WhitewaterState.SPRAY),
                np.uint8(WhitewaterState.ENTRAINED_BUBBLE),
                np.uint8(WhitewaterState.SURFACE_BUBBLE),
            ),
        )
        if np.any(~valid_states):
            issues.append(f"{path.name} contains a dead or unknown active state")
            transition_ok = False
        phase_state_ok = (
            (
                (markers["state"] == np.uint8(WhitewaterState.SPRAY))
                & (markers["phase"] == np.uint8(PhaseKind.LIQUID))
            )
            | (
                np.isin(
                    markers["state"],
                    (
                        np.uint8(WhitewaterState.ENTRAINED_BUBBLE),
                        np.uint8(WhitewaterState.SURFACE_BUBBLE),
                    ),
                )
                & (markers["phase"] == np.uint8(PhaseKind.GAS))
            )
        )
        if np.any(~phase_state_ok):
            issues.append(f"{path.name} contains a state/phase mismatch")
            transition_ok = False

        if len(markers):
            computed_volume = (
                (4.0 / 3.0)
                * np.pi
                * markers["physical_radius"].astype(np.float64) ** 3
                * markers["representative_count"].astype(np.float64)
            )
            error = np.max(
                np.abs(markers["phase_volume"] - computed_volume)
                / np.maximum(computed_volume, 1.0e-30)
            )
            maximum_volume_error = max(maximum_volume_error, float(error))
            bubbles = markers["phase"] == np.uint8(PhaseKind.GAS)
            if np.any(bubbles):
                shape_error = np.max(
                    np.abs(np.prod(markers["shape"][bubbles], axis=1) - 1.0)
                )
                maximum_shape_error = max(maximum_shape_error, float(shape_error))

        end_liquid_index = min(index + 1, len(liquid_samples) - 1)
        end_liquid_path = liquid_directory / liquid_samples[end_liquid_index]["file"]
        with np.load(end_liquid_path) as liquid_cache:
            end_phi = np.asarray(liquid_cache["phi"])
            end_collision = np.asarray(liquid_cache["collision_sdf"])
            end_fluid_mask = np.asarray(liquid_cache["fluid_mask"])
        if len(support_losses):
            start_liquid_path = liquid_directory / liquid_samples[index]["file"]
            with np.load(start_liquid_path) as liquid_cache:
                start_fluid_mask = np.asarray(liquid_cache["fluid_mask"])
            start_support, _ = build_render_surface_support(
                start_fluid_mask, spec, support_recipe
            )
            end_support, _ = build_render_surface_support(
                end_fluid_mask, spec, support_recipe
            )
            loss_alpha = np.clip(
                (support_losses["event_time"] - interval_start)
                / (interval_end - interval_start),
                0.0,
                1.0,
            )
            independently_sampled_support = (
                (1.0 - loss_alpha)
                * sample_scalar_trilinear(
                    start_support,
                    support_losses["position"],
                    spec,
                    outside=0.0,
                )
                + loss_alpha
                * sample_scalar_trilinear(
                    end_support,
                    support_losses["position"],
                    spec,
                    outside=0.0,
                )
            )
            if (
                np.any(independently_sampled_support >= 0.5)
                or not np.allclose(
                    independently_sampled_support,
                    support_losses["sampled_support"],
                    rtol=0.0,
                    atol=2.0e-6,
                )
            ):
                issues.append(
                    f"{path.name} has a re-entrainment not justified by support loss"
                )
                transition_ok = False
        if len(markers):
            inside_core = np.all(
                (markers["position"] >= np.asarray(spec.origin))
                & (markers["position"] <= np.asarray(spec.maximum)),
                axis=1,
            )
            inside_secondary = np.all(
                (markers["position"] >= secondary_minimum)
                & (markers["position"] <= secondary_maximum),
                axis=1,
            )
            spray_state = markers["state"] == np.uint8(WhitewaterState.SPRAY)
            maximum_spray_outside_core = max(
                maximum_spray_outside_core,
                int(np.count_nonzero(spray_state & ~inside_core)),
            )
            if np.any(~inside_secondary):
                issues.append(f"{path.name} contains a marker outside the secondary domain")
                spatial_ok = False
            if np.any(~inside_core & ~spray_state):
                issues.append(
                    f"{path.name} contains a non-spray marker outside carrier fields"
                )
                spatial_ok = False
            outside_spray = markers[spray_state & ~inside_core]
            if len(outside_spray) and secondary_scene is None:
                issues.append(
                    f"{path.name} has outside spray without a sparse collision scene"
                )
                spatial_ok = False
            elif len(outside_spray):
                endpoint_sample = liquid_samples[end_liquid_index]
                source_path = source_directory / endpoint_sample["source_file"]
                with np.load(source_path, allow_pickle=False) as source_cache:
                    snapshot = {
                        name: np.asarray(source_cache[name]).copy()
                        for name in source_cache.files
                    }
                query = secondary_scene.query(
                    outside_spray["position"],
                    snapshot0=snapshot,
                    snapshot1=None,
                    alpha=0.0,
                )
                valid = np.asarray(query["valid"], dtype=bool)
                secondary_endpoint_queries += len(outside_spray)
                secondary_endpoint_valid += int(np.count_nonzero(valid))
                secondary_endpoint_boundary_invalid += int(
                    np.count_nonzero(query["open_boundary_invalid"])
                )
                if np.any(valid):
                    secondary_margin = (
                        np.asarray(query["distance"])[valid]
                        - outside_spray["physical_radius"][valid].astype(np.float64)
                        - solid_clearance
                    )
                    minimum_secondary_collision_margin = min(
                        minimum_secondary_collision_margin,
                        float(np.min(secondary_margin)),
                    )
                    if np.any(secondary_margin < -2.0e-6):
                        issues.append(
                            f"{path.name} contains secondary-scene penetration"
                        )
                        spatial_ok = False
            core_markers = markers[inside_core]
            collision = sample_scalar_trilinear(
                end_collision, core_markers["position"], spec
            )
            margin = (
                collision
                - core_markers["physical_radius"].astype(np.float64)
                - solid_clearance
            )
            minimum_collision_margin = min(
                minimum_collision_margin, float(np.nanmin(margin))
            )
            if np.any(~np.isfinite(margin)) or np.any(margin < -2.0e-6):
                issues.append(f"{path.name} contains a materially solid-overlapping marker")
                spatial_ok = False
            phi = sample_scalar_trilinear(end_phi, core_markers["position"], spec)
            surface = core_markers["state"] == np.uint8(WhitewaterState.SURFACE_BUBBLE)
            if np.any(surface) and np.max(np.abs(phi[surface])) > 0.004:
                issues.append(f"{path.name} surface bubbles left the interface band")
                spatial_ok = False

        if len(events):
            computed_event_volume = (
                (4.0 / 3.0)
                * np.pi
                * events["physical_radius"].astype(np.float64) ** 3
                * events["representative_count"].astype(np.float64)
            )
            if not np.allclose(
                events["phase_volume"],
                computed_event_volume,
                rtol=5.0e-7,
                atol=0.0,
            ):
                issues.append(f"{path.name} terminal event phase volume is inconsistent")
                conservation_ok = False
            if np.any(events["event_time"] < interval_start - 1.0e-12) or np.any(
                events["event_time"] > interval_end + 1.0e-12
            ):
                issues.append(f"{path.name} contains an out-of-interval terminal event")
                transition_ok = False
            valid_event_kind = np.isin(
                events["kind"],
                tuple(np.uint8(value) for value in TerminalEventKind),
            )
            if np.any(~valid_event_kind):
                issues.append(f"{path.name} contains an unknown terminal event")
                transition_ok = False
            if np.any(events["kind"] == np.uint8(TerminalEventKind.DOMAIN_ESCAPE)):
                no_domain_escape = False

        observed_cfl = float(sample["solver_metrics"]["maximum_cfl"])
        recorded_secondary_queries += int(
            sample["solver_metrics"].get("secondary_collision_query_count", 0)
        )
        recorded_secondary_projections += int(
            sample["solver_metrics"].get("secondary_solid_projection_count", 0)
        )
        maximum_cfl = max(maximum_cfl, observed_cfl)
        if observed_cfl > maximum_cfl_target + 1.0e-5:
            issues.append(f"{path.name} exceeded the configured CFL target")
        previous = markers

    active_volume = {
        phase: float(
            previous["phase_volume"][previous["phase"] == phase].sum(dtype=np.float64)
        )
        for phase in (1, 2)
    }
    conservation_residual = {
        phase: cumulative_born[phase]
        - active_volume[phase]
        - cumulative_terminal[phase]
        for phase in (1, 2)
    }
    for phase in (1, 2):
        tolerance = max(1.0e-12, cumulative_born[phase] * 2.0e-7)
        if abs(conservation_residual[phase]) > tolerance:
            issues.append(
                f"Phase {phase} volume is not conserved: residual {conservation_residual[phase]}"
            )
            conservation_ok = False
    if maximum_volume_error > 5.0e-7 or maximum_shape_error > 3.0e-6:
        issues.append("Marker volume or bubble shape-volume contract failed")
        conservation_ok = False
    if not no_domain_escape:
        issues.append("One or more markers escaped the reconstruction domain")

    criteria = {
        "manifest_complete": bool(manifest.get("complete")),
        "input_manifest_hashes_match": input_hashes_match,
        "secondary_scene_contract_hash_matches": secondary_scene_input_ok,
        "all_sample_hashes_match": hashes_ok,
        "structural_checks_pass": structural_ok,
        "identity_and_terminal_event_balance_pass": identity_ok,
        "state_transitions_are_legal": transition_ok,
        "spatial_checks_pass": spatial_ok,
        "secondary_sparse_collision_was_queried": (
            maximum_spray_outside_core == 0 or recorded_secondary_queries > 0
        ),
        "phase_volume_is_conserved": conservation_ok,
        "no_domain_escape": no_domain_escape,
        "cfl_target_respected": maximum_cfl <= maximum_cfl_target + 1.0e-5,
    }
    report = {
        "schema": 1,
        "product": "whitewater_v6_marker_trajectories_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trajectory_directory": str(args.trajectory_directory.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "valid": not issues and all(criteria.values()),
        "criteria": criteria,
        "metrics": {
            "samples": len(samples),
            "final_active_markers": len(previous),
            "marker_rows_audited": total_markers_seen,
            "terminal_events": total_events,
            "surface_support_loss_transitions": total_surface_support_losses,
            "maximum_cfl": maximum_cfl,
        "maximum_shared_frame_displacement_m": maximum_shared_displacement,
            "maximum_spray_markers_outside_core_grid": maximum_spray_outside_core,
            "secondary_domain_padding_m": secondary_padding,
            "recorded_secondary_collision_queries": recorded_secondary_queries,
            "recorded_secondary_solid_projections": recorded_secondary_projections,
            "secondary_endpoint_queries": secondary_endpoint_queries,
            "secondary_endpoint_valid_scene_queries": secondary_endpoint_valid,
            "secondary_endpoint_open_boundary_invalid": (
                secondary_endpoint_boundary_invalid
            ),
            "minimum_secondary_endpoint_collision_margin_m": (
                minimum_secondary_collision_margin
                if np.isfinite(minimum_secondary_collision_margin)
                else None
            ),
            "minimum_endpoint_collision_margin_m": (
                minimum_collision_margin if np.isfinite(minimum_collision_margin) else None
            ),
            "maximum_phase_volume_relative_error": maximum_volume_error,
            "maximum_bubble_shape_volume_error": maximum_shape_error,
            "born_phase_volume_m3": {
                "liquid": cumulative_born[1],
                "gas": cumulative_born[2],
            },
            "active_phase_volume_m3": {
                "liquid": active_volume[1],
                "gas": active_volume[2],
            },
            "terminal_phase_volume_m3": {
                "liquid": cumulative_terminal[1],
                "gas": cumulative_terminal[2],
            },
            "conservation_residual_m3": {
                "liquid": conservation_residual[1],
                "gas": conservation_residual[2],
            },
        },
        "errors": issues,
    }
    atomic_json(args.trajectory_directory / "audit_report.json", report)
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
