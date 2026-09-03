"""Audit end-to-end ownership, phase volume and film-area lifecycle ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.marker_birth import BirthChannel, PhaseKind
from whitewater.marker_solver import TerminalEventKind
from whitewater.secondary_handoff import EXTERNAL_SOURCE_NAMESPACE
from whitewater.state_machine import WhitewaterState


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain_directory", type=Path)
    parser.add_argument("birth_directory", type=Path)
    parser.add_argument("trajectory_directory", type=Path)
    parser.add_argument("pending_proxy_directory", type=Path)
    parser.add_argument("surface_foam_directory", type=Path)
    parser.add_argument("surface_raft_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--plateau-directory",
        type=Path,
        help="Optional audited Plateau-cell cache whose structural liquid is debited from the carrier",
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(directory, domain=False):
    path = directory / ("domain_manifest.json" if domain else "manifest.json")
    return path, json.loads(path.read_text(encoding="utf-8"))


def summed(records, selection=None):
    if selection is not None:
        records = records[selection]
    return float(records["phase_volume"].sum(dtype=np.float64))


def main():
    args = parse_args()
    directories = {
        "domain": args.domain_directory.resolve(),
        "births": args.birth_directory.resolve(),
        "trajectories": args.trajectory_directory.resolve(),
        "pending_proxy": args.pending_proxy_directory.resolve(),
        "surface_foam": args.surface_foam_directory.resolve(),
        "surface_raft": args.surface_raft_directory.resolve(),
    }
    if args.plateau_directory is not None:
        directories["plateau"] = args.plateau_directory.resolve()
    manifests = {}
    manifest_paths = {}
    for name, directory in directories.items():
        path, manifest = load_manifest(directory, domain=name == "domain")
        manifest_paths[name] = path
        manifests[name] = manifest

    expected_products = {
        "domain": "whitewater_v6_domain_partition",
        "births": "whitewater_v6_marker_births",
        "trajectories": "whitewater_v6_marker_trajectories",
        "pending_proxy": "whitewater_v6_pending_liquid_proxy",
        "surface_foam": "whitewater_v6_surface_foam",
        "surface_raft": "whitewater_v6_surface_raft",
    }
    if "plateau" in directories:
        expected_products["plateau"] = "whitewater_v6_plateau_cells"
    errors = []
    for name, product in expected_products.items():
        if manifests[name].get("product") != product:
            errors.append(f"{name}: unexpected product")
        complete = manifests[name].get("valid") if name == "domain" else manifests[name].get("complete")
        if not complete:
            errors.append(f"{name}: input is not complete/valid")

    sample_lists = {name: manifest["samples"] for name, manifest in manifests.items()}
    source_indices = {
        name: [
            int(sample.get("source_sample_index", sample.get("sample_index")))
            for sample in samples
        ]
        for name, samples in sample_lists.items()
    }
    reference_indices = source_indices["domain"]
    for name, indices in source_indices.items():
        if indices != reference_indices:
            errors.append(f"{name}: source sample alignment mismatch")

    output_directory = args.output_directory.resolve()
    report_path = output_directory / "audit_report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    output_directory.mkdir(parents=True, exist_ok=True)

    particle_spacing = float(manifests["domain"]["configuration"]["particle_spacing"])
    particle_volume = particle_spacing**3
    source_particle_count = int(sample_lists["domain"][0]["classification"]["core_particles"])
    first_classification = directories["domain"] / sample_lists["domain"][0]["classification"]["file"]
    with np.load(first_classification, allow_pickle=False) as cache:
        source_particle_count = len(cache["particle_state"])
    source_liquid_volume = source_particle_count * particle_volume

    id_origin = {}
    terminal_seen = set()
    born = {"external_liquid": 0.0, "field_liquid": 0.0, "gas": 0.0}
    terminal = {
        "external_reentry": 0.0,
        "external_escape": 0.0,
        "field_reentry": 0.0,
        "field_escape": 0.0,
        "gas_burst": 0.0,
        "gas_escape": 0.0,
    }
    film = {"injected": 0.0, "drained": 0.0, "topology_lost": 0.0}
    maximum = {
        "partition_residual_m3": 0.0,
        "external_link_residual_m3": 0.0,
        "field_liquid_residual_m3": 0.0,
        "global_liquid_residual_m3": 0.0,
        "gas_residual_m3": 0.0,
        "film_area_residual_m2": 0.0,
        "raft_gas_residual_m3": 0.0,
        "plateau_gas_residual_m3": 0.0,
        "plateau_film_area_overdraw_m2": 0.0,
        "plateau_structural_liquid_m3": 0.0,
    }
    cumulative_pending_returns = 0
    previous_state = None
    frame_ledgers = []

    for frame_index, source_sample in enumerate(reference_indices):
        domain_sample = sample_lists["domain"][frame_index]
        birth_sample = sample_lists["births"][frame_index]
        trajectory_sample = sample_lists["trajectories"][frame_index]
        proxy_sample = sample_lists["pending_proxy"][frame_index]
        foam_sample = sample_lists["surface_foam"][frame_index]
        raft_sample = sample_lists["surface_raft"][frame_index]

        classification_path = directories["domain"] / domain_sample["classification"]["file"]
        with np.load(classification_path, allow_pickle=False) as cache:
            state = np.asarray(cache["particle_state"], dtype=np.uint8)
            new_handoff = np.asarray(cache["new_secondary_handoff"], dtype=bool)
            pending_return = np.asarray(cache["pending_return_to_core"], dtype=bool)
        counts = {code: int(np.count_nonzero(state == code)) for code in range(4)}
        core_count = counts[0]
        pending_count = counts[1]
        secondary_count = counts[2] + counts[3]
        partition_residual = (core_count + pending_count + secondary_count) * particle_volume - source_liquid_volume
        maximum["partition_residual_m3"] = max(maximum["partition_residual_m3"], abs(partition_residual))

        if previous_state is not None:
            expected_return = (previous_state == 1) & (state == 0)
            expected_handoff = (previous_state == 1) & ((state == 2) | (state == 3))
            if not np.array_equal(pending_return, expected_return):
                errors.append(f"sample {source_sample}: pending return transition mismatch")
            if not np.array_equal(new_handoff, expected_handoff):
                errors.append(f"sample {source_sample}: confirmed handoff transition mismatch")
            if np.any((previous_state >= 2) & (state < 2)):
                errors.append(f"sample {source_sample}: sticky secondary ownership regressed")
        cumulative_pending_returns += int(np.count_nonzero(pending_return))
        previous_state = state.copy()

        birth_path = directories["births"] / birth_sample["file"]
        with np.load(birth_path, allow_pickle=False) as cache:
            births = np.asarray(cache["births"])
        for record in births:
            marker_id = int(record["id"])
            if marker_id in id_origin:
                errors.append(f"sample {source_sample}: duplicate birth ID {marker_id}")
                continue
            external = bool(
                np.uint32(record["source_node_id"]) & EXTERNAL_SOURCE_NAMESPACE
            )
            if external:
                origin = "external_liquid"
            elif int(record["phase"]) == int(PhaseKind.LIQUID):
                origin = "field_liquid"
            else:
                origin = "gas"
            id_origin[marker_id] = origin
            born[origin] += float(record["phase_volume"])

        trajectory_path = directories["trajectories"] / trajectory_sample["file"]
        with np.load(trajectory_path, allow_pickle=False) as cache:
            markers = np.asarray(cache["markers"])
            events = np.asarray(cache["terminal_events"])
        for event in events:
            marker_id = int(event["marker_id"])
            if marker_id in terminal_seen:
                errors.append(f"sample {source_sample}: duplicate terminal marker {marker_id}")
                continue
            terminal_seen.add(marker_id)
            origin = id_origin.get(marker_id)
            if origin is None:
                errors.append(f"sample {source_sample}: terminal event without a birth")
                continue
            kind = int(event["kind"])
            volume = float(event["phase_volume"])
            if origin == "external_liquid":
                target = "external_reentry" if kind == int(TerminalEventKind.SPRAY_REENTRY) else "external_escape"
            elif origin == "field_liquid":
                target = "field_reentry" if kind == int(TerminalEventKind.SPRAY_REENTRY) else "field_escape"
            else:
                target = "gas_burst" if kind == int(TerminalEventKind.BUBBLE_BURST) else "gas_escape"
            terminal[target] += volume

        active = {"external_liquid": 0.0, "field_liquid": 0.0, "gas": 0.0}
        active_ids = set()
        for marker in markers:
            marker_id = int(marker["id"])
            active_ids.add(marker_id)
            origin = id_origin.get(marker_id)
            if origin is None:
                errors.append(f"sample {source_sample}: active marker without a birth")
                continue
            active[origin] += float(marker["phase_volume"])
        if active_ids & terminal_seen:
            errors.append(f"sample {source_sample}: active/terminal identity overlap")

        external_terminal = terminal["external_reentry"] + terminal["external_escape"]
        field_terminal = terminal["field_reentry"] + terminal["field_escape"]
        gas_terminal = terminal["gas_burst"] + terminal["gas_escape"]
        external_residual = born["external_liquid"] - active["external_liquid"] - external_terminal
        field_residual = born["field_liquid"] - active["field_liquid"] - field_terminal
        gas_residual = born["gas"] - active["gas"] - gas_terminal
        maximum["external_link_residual_m3"] = max(maximum["external_link_residual_m3"], abs(external_residual))
        maximum["field_liquid_residual_m3"] = max(maximum["field_liquid_residual_m3"], abs(field_residual))
        maximum["gas_residual_m3"] = max(maximum["gas_residual_m3"], abs(gas_residual))

        secondary_volume = secondary_count * particle_volume
        external_handoff_residual = secondary_volume - born["external_liquid"]
        maximum["external_link_residual_m3"] = max(
            maximum["external_link_residual_m3"], abs(external_handoff_residual)
        )

        proxy_path = directories["pending_proxy"] / proxy_sample["file"]
        with np.load(proxy_path, allow_pickle=False) as cache:
            proxies = np.asarray(cache["proxies"])
        proxy_volume = float(proxies["liquid_volume"].sum(dtype=np.float64))
        if len(proxies) != pending_count or abs(proxy_volume - pending_count * particle_volume) > 1.0e-12:
            errors.append(f"sample {source_sample}: pending proxy ownership mismatch")

        # Field-born liquid is debited from the carrier at birth and credited on
        # reentry. External liquid has already left the PhysX core partition.
        carrier_before_plateau = (
            (core_count + pending_count) * particle_volume
            - born["field_liquid"]
            + terminal["field_reentry"]
            + terminal["external_reentry"]
        )
        plateau_structural_liquid = 0.0
        plateau_gas = 0.0
        plateau_film_area = 0.0
        if "plateau" in directories:
            plateau_sample = sample_lists["plateau"][frame_index]
            plateau_path = directories["plateau"] / plateau_sample["file"]
            with np.load(plateau_path, allow_pickle=False) as cache:
                plateau_cells = np.asarray(cache["cells"])
                plateau_films = np.asarray(cache["films"])
                plateau_borders = np.asarray(cache["borders"])
                plateau_nodes = np.asarray(cache["nodes"])
            plateau_gas = float(
                plateau_cells["gas_volume"].sum(dtype=np.float64)
            )
            plateau_film_area = float(
                plateau_films["area"].sum(dtype=np.float64)
            )
            plateau_structural_liquid = float(
                plateau_films["liquid_volume"].sum(dtype=np.float64)
                + plateau_borders["liquid_volume"].sum(dtype=np.float64)
                + plateau_nodes["liquid_volume"].sum(dtype=np.float64)
            )
        carrier_volume = carrier_before_plateau - plateau_structural_liquid
        lost_liquid = terminal["field_escape"] + terminal["external_escape"]
        global_liquid = (
            carrier_volume
            + plateau_structural_liquid
            + active["field_liquid"]
            + active["external_liquid"]
            + lost_liquid
        )
        global_liquid_residual = global_liquid - source_liquid_volume
        maximum["global_liquid_residual_m3"] = max(
            maximum["global_liquid_residual_m3"], abs(global_liquid_residual)
        )
        if carrier_volume < -1.0e-12:
            errors.append(f"sample {source_sample}: field spray overdraws carrier")

        raft_path = directories["surface_raft"] / raft_sample["file"]
        with np.load(raft_path, allow_pickle=False) as cache:
            raft = np.asarray(cache["raft"])
        surface_markers = markers[
            markers["state"] == np.uint8(WhitewaterState.SURFACE_BUBBLE)
        ]
        if not np.array_equal(np.sort(raft["marker_id"]), np.sort(surface_markers["id"])):
            errors.append(f"sample {source_sample}: raft identity mismatch")
        raft_residual = summed(raft) - summed(surface_markers)
        maximum["raft_gas_residual_m3"] = max(
            maximum["raft_gas_residual_m3"], abs(raft_residual)
        )
        if "plateau" in directories:
            plateau_gas_residual = plateau_gas - summed(raft)
            maximum["plateau_gas_residual_m3"] = max(
                maximum["plateau_gas_residual_m3"],
                abs(plateau_gas_residual),
            )
            maximum["plateau_structural_liquid_m3"] = max(
                maximum["plateau_structural_liquid_m3"],
                plateau_structural_liquid,
            )

        source_metrics = foam_sample["source_metrics"]
        advance_metrics = foam_sample["advance_metrics"]
        film["injected"] += float(source_metrics["injected_film_area_m2"])
        film["drained"] += float(advance_metrics["drained_film_area_m2"])
        film["topology_lost"] += float(advance_metrics["topology_lost_film_area_m2"])
        foam_path = directories["surface_foam"] / foam_sample["file"]
        with np.load(foam_path, allow_pickle=False) as cache:
            foam = np.asarray(cache["foam"])
        active_film = float(foam["film_area"].sum(dtype=np.float64))
        if "plateau" in directories:
            plateau_area_overdraw = max(plateau_film_area - active_film, 0.0)
            maximum["plateau_film_area_overdraw_m2"] = max(
                maximum["plateau_film_area_overdraw_m2"],
                plateau_area_overdraw,
            )
        film_residual = film["injected"] - film["drained"] - film["topology_lost"] - active_film
        maximum["film_area_residual_m2"] = max(
            maximum["film_area_residual_m2"], abs(film_residual)
        )

        frame_ledgers.append(
            {
                "output_frame": frame_index,
                "source_sample_index": source_sample,
                "physx_partition": {
                    "core_owned_particles": core_count,
                    "pending_proxy_particles": pending_count,
                    "secondary_owned_particles": secondary_count,
                    "partition_residual_m3": partition_residual,
                    "new_secondary_handoff_particles": int(np.count_nonzero(new_handoff)),
                    "pending_return_to_core_particles": int(np.count_nonzero(pending_return)),
                },
                "liquid_volume_m3": {
                    "source_total": source_liquid_volume,
                    "carrier_owned_after_subgrid_debit": carrier_volume,
                    "carrier_before_plateau_structural_debit": carrier_before_plateau,
                    "plateau_film_liquid": (
                        float(plateau_films["liquid_volume"].sum(dtype=np.float64))
                        if "plateau" in directories
                        else 0.0
                    ),
                    "plateau_border_liquid": (
                        float(plateau_borders["liquid_volume"].sum(dtype=np.float64))
                        if "plateau" in directories
                        else 0.0
                    ),
                    "plateau_node_liquid": (
                        float(plateau_nodes["liquid_volume"].sum(dtype=np.float64))
                        if "plateau" in directories
                        else 0.0
                    ),
                    "pending_proxy_owned": proxy_volume,
                    "active_external_spray": active["external_liquid"],
                    "active_field_spray": active["field_liquid"],
                    "returned_external_to_carrier_cumulative": terminal["external_reentry"],
                    "returned_field_to_carrier_cumulative": terminal["field_reentry"],
                    "domain_escape_cumulative": lost_liquid,
                    "global_balance_residual": global_liquid_residual,
                },
                "gas_volume_m3": {
                    "born_cumulative": born["gas"],
                    "active": active["gas"],
                    "burst_to_atmosphere_cumulative": terminal["gas_burst"],
                    "domain_escape_cumulative": terminal["gas_escape"],
                    "balance_residual": gas_residual,
                    "surface_raft_subset": summed(raft),
                    "plateau_cell_subset": plateau_gas,
                },
                "film_area_m2": {
                    "injected_cumulative": film["injected"],
                    "drained_cumulative": film["drained"],
                    "topology_lost_cumulative": film["topology_lost"],
                    "active": active_film,
                    "balance_residual": film_residual,
                    "plateau_shared_membrane_subset": plateau_film_area,
                },
            }
        )

    volume_tolerance = 2.0e-10
    area_tolerance = 2.0e-9
    if maximum["partition_residual_m3"] > volume_tolerance:
        errors.append("PhysX ownership partition volume gate exceeded")
    if maximum["external_link_residual_m3"] > volume_tolerance:
        errors.append("external handoff linkage volume gate exceeded")
    if maximum["field_liquid_residual_m3"] > volume_tolerance:
        errors.append("field spray lifecycle volume gate exceeded")
    if maximum["global_liquid_residual_m3"] > volume_tolerance:
        errors.append("global liquid ownership volume gate exceeded")
    if maximum["gas_residual_m3"] > volume_tolerance:
        errors.append("gas lifecycle volume gate exceeded")
    if maximum["raft_gas_residual_m3"] > volume_tolerance:
        errors.append("surface raft gas subset volume gate exceeded")
    if maximum["plateau_gas_residual_m3"] > volume_tolerance:
        errors.append("Plateau child-cell gas subset volume gate exceeded")
    if maximum["plateau_film_area_overdraw_m2"] > area_tolerance:
        errors.append("Plateau shared-film area budget gate exceeded")
    if maximum["film_area_residual_m2"] > area_tolerance:
        errors.append("film area lifecycle gate exceeded")

    report = {
        "schema": 1,
        "product": "whitewater_v6_lifecycle_conservation_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "valid": not errors,
        "interpretation": {
            "physical_liquid_ownership": (
                "core and pending remain carrier-owned; confirmed PhysX handoffs "
                "are linked by identity to external spray markers; field spray is "
                "debited from a sub-grid carrier reservoir at birth and credited "
                "on reentry, never added twice"
            ),
            "gas_volume": "conserved independently through active, burst and escape states",
            "film_area": (
                "a separate interfacial-area ledger; it is deliberately never "
                "summed with liquid or gas volume"
            ),
            "plateau_structural_liquid": (
                "film, border and node liquid volumes are a per-frame ownership "
                "reclassification debited from the carrier and added back as a "
                "separate liquid inventory; they are never confused with gas or area"
            ),
        },
        "inputs": {
            name: {
                "manifest": str(manifest_paths[name]),
                "manifest_sha256": sha256_file(manifest_paths[name]),
            }
            for name in directories
        },
        "criteria": {
            "all_inputs_complete_and_aligned": not any(
                "input" in error or "alignment" in error for error in errors
            ),
            "physx_partition_is_conservative": maximum["partition_residual_m3"] <= volume_tolerance,
            "pending_returns_and_handoffs_are_exact_transitions": not any(
                "transition" in error or "regressed" in error for error in errors
            ),
            "pending_proxy_is_mutually_exclusive_and_conservative": not any(
                "pending proxy" in error for error in errors
            ),
            "external_handoff_identity_and_volume_are_linked": maximum["external_link_residual_m3"] <= volume_tolerance,
            "field_spray_subgrid_debit_is_conservative": maximum["field_liquid_residual_m3"] <= volume_tolerance,
            "global_liquid_ownership_is_conservative": maximum["global_liquid_residual_m3"] <= volume_tolerance,
            "gas_volume_is_conservative": maximum["gas_residual_m3"] <= volume_tolerance,
            "surface_raft_is_an_exact_active_gas_subset": maximum["raft_gas_residual_m3"] <= volume_tolerance,
            "plateau_cells_are_an_exact_surface_raft_gas_subset": maximum["plateau_gas_residual_m3"] <= volume_tolerance,
            "plateau_shared_films_fit_the_active_area_budget": maximum["plateau_film_area_overdraw_m2"] <= area_tolerance,
            "plateau_structural_liquid_is_debited_from_carrier": (
                all(
                    frame["liquid_volume_m3"]["carrier_owned_after_subgrid_debit"]
                    <= frame["liquid_volume_m3"]["carrier_before_plateau_structural_debit"]
                    + volume_tolerance
                    for frame in frame_ledgers
                )
            ),
            "film_area_is_conservative_and_dimensionally_separate": maximum["film_area_residual_m2"] <= area_tolerance,
            "formal_domain_escape_is_zero": (
                terminal["external_escape"] + terminal["field_escape"] + terminal["gas_escape"]
            )
            <= volume_tolerance,
        },
        "metrics": {
            "samples": len(frame_ledgers),
            "source_particles": source_particle_count,
            "source_liquid_volume_m3": source_liquid_volume,
            "cumulative_pending_returns_to_core": cumulative_pending_returns,
            "born_volume_m3": born,
            "terminal_volume_m3": terminal,
            "maximum_absolute_residuals": maximum,
            "tolerances": {
                "volume_m3": volume_tolerance,
                "film_area_m2": area_tolerance,
            },
        },
        "frames": frame_ledgers,
        "errors": errors,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("valid", "criteria", "metrics", "errors")}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
