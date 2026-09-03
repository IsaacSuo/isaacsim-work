"""Independently audit Plateau-cell conservation, topology and render geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.plateau_cells import (
    BORDER_DTYPE,
    CELL_DTYPE,
    EVENT_DTYPE,
    FILM_DTYPE,
    NODE_DTYPE,
)
from whitewater.surface_raft import RAFT_DTYPE


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plateau_directory", type=Path)
    parser.add_argument("surface_raft_directory", type=Path)
    parser.add_argument("surface_foam_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ids(records):
    return set(int(value) for value in records["id"])


def maximum_absolute(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.max(np.abs(values), initial=0.0))


def main():
    args = parse_args()
    plateau_directory = args.plateau_directory.resolve()
    raft_directory = args.surface_raft_directory.resolve()
    foam_directory = args.surface_foam_directory.resolve()
    directories = {
        "plateau": plateau_directory,
        "raft": raft_directory,
        "foam": foam_directory,
    }
    manifests = {
        name: load_json(directory / "manifest.json")
        for name, directory in directories.items()
    }
    expected_products = {
        "plateau": "whitewater_v6_plateau_cells",
        "raft": "whitewater_v6_surface_raft",
        "foam": "whitewater_v6_surface_foam",
    }
    failures = []
    for name, product in expected_products.items():
        if manifests[name].get("product") != product:
            failures.append(f"{name}: unexpected product")
        if not manifests[name].get("complete"):
            failures.append(f"{name}: manifest is incomplete")

    plateau_manifest = manifests["plateau"]
    recorded_inputs = plateau_manifest.get("inputs", {})
    for name, directory in (("raft", raft_directory), ("foam", foam_directory)):
        path = directory / "manifest.json"
        recorded_path = Path(recorded_inputs.get(f"{name}_manifest", "")).resolve()
        recorded_hash = recorded_inputs.get(f"{name}_manifest_sha256")
        if recorded_path != path.resolve():
            failures.append(f"{name}: recorded input path mismatch")
        if recorded_hash != sha256_file(path):
            failures.append(f"{name}: recorded manifest hash mismatch")
    liquid_path_text = recorded_inputs.get("liquid_manifest")
    if not liquid_path_text:
        failures.append("liquid: recorded input path is missing")
    else:
        liquid_path = Path(liquid_path_text)
        if not liquid_path.is_file():
            failures.append("liquid: recorded input manifest is missing")
        elif recorded_inputs.get("liquid_manifest_sha256") != sha256_file(liquid_path):
            failures.append("liquid: recorded manifest hash mismatch")

    samples = plateau_manifest.get("samples", [])
    raft_samples = manifests["raft"].get("samples", [])
    foam_samples = manifests["foam"].get("samples", [])
    if not (len(samples) == len(raft_samples) == len(foam_samples)):
        failures.append("plateau/raft/foam sample count mismatch")

    model = plateau_manifest["configuration"]["model"]
    maximum_displacement = 0.012
    maximum_cell_aspect = float(model["maximum_cell_aspect"])
    initial_thickness = float(model["initial_film_thickness_m"])
    minimum_thickness = float(model["minimum_film_thickness_m"])
    totals = {
        "cells": 0,
        "films": 0,
        "borders": 0,
        "nodes": 0,
        "triangles": 0,
        "gas_volume_m3": 0.0,
        "film_area_m2": 0.0,
        "film_liquid_volume_m3": 0.0,
        "border_liquid_volume_m3": 0.0,
        "node_liquid_volume_m3": 0.0,
    }
    maxima = {
        "parent_gas_residual_m3": 0.0,
        "frame_gas_residual_m3": 0.0,
        "analytic_geometry_residual_m3": 0.0,
        "radius_inflation_m": 0.0,
        "packing_displacement_m": 0.0,
        "packing_displacement_record_residual_m": 0.0,
        "film_budget_fraction": 0.0,
        "film_liquid_formula_residual_m3": 0.0,
        "border_liquid_formula_residual_m3": 0.0,
        "node_liquid_formula_residual_m3": 0.0,
        "cell_height_to_radius": 0.0,
    }
    counts = {
        "duplicate_cell_ids": 0,
        "missing_parent_packets": 0,
        "unexpected_parent_packets": 0,
        "duplicate_film_pairs": 0,
        "invalid_film_owners": 0,
        "invalid_border_owners": 0,
        "invalid_node_incidence": 0,
        "topology_event_mismatches": 0,
        "nonfinite_records": 0,
        "invalid_mesh_indices": 0,
        "invalid_triangle_contract": 0,
        "invalid_triangle_owners": 0,
        "unresolved_constraints": 0,
        "unsupported_ruptures": 0,
    }
    previous_films = set()
    previous_nodes = set()
    frame_ledgers = []

    for frame_index, (sample, raft_sample, foam_sample) in enumerate(
        zip(samples, raft_samples, foam_samples)
    ):
        source = int(sample["source_sample_index"])
        if source != int(raft_sample["source_sample_index"]) or source != int(
            foam_sample["source_sample_index"]
        ):
            failures.append(f"sample {frame_index}: source alignment mismatch")
            continue
        plateau_path = plateau_directory / sample["file"]
        if sha256_file(plateau_path) != sample["sha256"]:
            failures.append(f"sample {source}: cache hash mismatch")
        with np.load(plateau_path, allow_pickle=False) as cache:
            cells = np.asarray(cache["cells"])
            films = np.asarray(cache["films"])
            borders = np.asarray(cache["borders"])
            nodes = np.asarray(cache["nodes"])
            events = np.asarray(cache["events"])
            ruptured = np.asarray(cache["ruptured_pair_ids"])
            vertices = np.asarray(cache["vertices"])
            triangles = np.asarray(cache["triangles"])
            materials = np.asarray(cache["triangle_material"])
            owners = np.asarray(cache["triangle_owner_id"])
        with np.load(raft_directory / raft_sample["file"], allow_pickle=False) as cache:
            raft = np.asarray(cache["raft"])
        with np.load(foam_directory / foam_sample["file"], allow_pickle=False) as cache:
            foam = np.asarray(cache["foam"])

        for records, dtype, name in (
            (cells, CELL_DTYPE, "cell"),
            (films, FILM_DTYPE, "film"),
            (borders, BORDER_DTYPE, "border"),
            (nodes, NODE_DTYPE, "node"),
            (events, EVENT_DTYPE, "event"),
        ):
            if records.dtype != dtype:
                failures.append(f"sample {source}: {name} dtype mismatch")
        if raft.dtype != RAFT_DTYPE:
            failures.append(f"sample {source}: raft dtype mismatch")

        cell_ids = ids(cells)
        if len(cell_ids) != len(cells):
            counts["duplicate_cell_ids"] += len(cells) - len(cell_ids)
        raft_by_id = {int(row["marker_id"]): row for row in raft}
        cell_parents = set(int(value) for value in cells["parent_marker_id"])
        counts["missing_parent_packets"] += len(set(raft_by_id) - cell_parents)
        counts["unexpected_parent_packets"] += len(cell_parents - set(raft_by_id))
        parent_residuals = []
        radius_inflation = []
        for parent_id, parent in raft_by_id.items():
            children = cells[cells["parent_marker_id"] == np.uint64(parent_id)]
            parent_residuals.append(
                float(children["gas_volume"].sum(dtype=np.float64))
                - float(parent["phase_volume"])
            )
            if len(children):
                radius_inflation.append(
                    float(children["physical_radius"].max())
                    - float(parent["physical_radius"])
                )
        parent_residual = maximum_absolute(parent_residuals)
        frame_gas = float(cells["gas_volume"].sum(dtype=np.float64))
        source_gas = float(raft["phase_volume"].sum(dtype=np.float64))
        analytic_gas = float(
            np.sum(
                cells["footprint_area"]
                * (
                    cells["base_height"]
                    + (cells["peak_height"] - cells["base_height"]) / 3.0
                ),
                dtype=np.float64,
            )
        )
        displacement = np.linalg.norm(
            cells["position"].astype(np.float64)
            - cells["packet_anchor_position"].astype(np.float64),
            axis=1,
        )
        displacement_record_residual = maximum_absolute(
            displacement - cells["packing_displacement"]
        )
        height_to_radius = (
            cells["peak_height"]
            / np.maximum(cells["physical_radius"], 1.0e-30)
            if len(cells)
            else np.empty(0)
        )

        film_pairs = [
            (min(int(row["cell_a"]), int(row["cell_b"])),
             max(int(row["cell_a"]), int(row["cell_b"])))
            for row in films
        ]
        counts["duplicate_film_pairs"] += len(film_pairs) - len(set(film_pairs))
        counts["invalid_film_owners"] += sum(
            a not in cell_ids or b not in cell_ids or a >= b for a, b in film_pairs
        )
        available_film_area = float(foam["film_area"].sum(dtype=np.float64))
        film_area = float(films["area"].sum(dtype=np.float64))
        film_formula_residual = maximum_absolute(
            films["liquid_volume"] - films["area"] * films["thickness"]
        )
        if len(films) and (
            np.any(films["thickness"] < minimum_thickness)
            or np.any(films["thickness"] > initial_thickness + 1.0e-12)
        ):
            failures.append(f"sample {source}: film thickness outside physical bounds")

        film_ids = ids(films)
        node_ids = ids(nodes)
        border_formula_residual = maximum_absolute(
            borders["liquid_volume"]
            - np.pi
            * borders["radius"].astype(np.float64) ** 2
            * np.linalg.norm(
                borders["end"].astype(np.float64)
                - borders["start"].astype(np.float64),
                axis=1,
            )
        )
        for border in borders:
            owner = int(border["owner_id"])
            kind = int(border["kind"])
            if (kind == 1 and owner not in film_ids) or (
                kind == 2 and owner not in node_ids
            ) or kind not in (1, 2):
                counts["invalid_border_owners"] += 1
        node_formula_residual = maximum_absolute(
            nodes["liquid_volume"]
            - 4.0 * np.pi * nodes["radius"].astype(np.float64) ** 3 / 3.0
        )
        counts["invalid_node_incidence"] += int(
            np.count_nonzero(
                (nodes["incident_films"] < 3) | (nodes["incident_cells"] < 3)
            )
        )

        event_ids = {
            kind: set(int(value) for value in events["topology_id"][events["kind"] == kind])
            for kind in range(1, 6)
        }
        current_films = film_ids
        current_nodes = node_ids
        if event_ids[1] != current_films - previous_films:
            counts["topology_event_mismatches"] += 1
        removed_films = previous_films - current_films
        if event_ids[2] | event_ids[3] != removed_films or event_ids[2] & event_ids[3]:
            counts["topology_event_mismatches"] += 1
        if event_ids[4] != current_nodes - previous_nodes:
            counts["topology_event_mismatches"] += 1
        if event_ids[5] != previous_nodes - current_nodes:
            counts["topology_event_mismatches"] += 1
        counts["unsupported_ruptures"] += len(event_ids[3]) + len(ruptured)
        previous_films = current_films
        previous_nodes = current_nodes

        numeric_arrays = (
            cells["physical_radius"], cells["gas_volume"], cells["position"],
            cells["footprint_area"], cells["base_height"], cells["peak_height"],
            films["area"], films["thickness"], films["liquid_volume"],
            borders["radius"], borders["liquid_volume"], nodes["radius"],
            nodes["liquid_volume"], vertices,
        )
        if any(not np.isfinite(array).all() for array in numeric_arrays):
            counts["nonfinite_records"] += 1
        if len(triangles) and (
            triangles.min(initial=0) < 0 or triangles.max(initial=-1) >= len(vertices)
        ):
            counts["invalid_mesh_indices"] += 1
        if len(triangles) != len(materials) or len(triangles) != len(owners):
            counts["invalid_triangle_contract"] += 1
        if len(materials) and (materials.min() < 0 or materials.max() > 4):
            counts["invalid_triangle_contract"] += 1
        owner_sets = {0: cell_ids, 1: cell_ids, 2: film_ids, 3: ids(borders), 4: node_ids}
        for material in range(5):
            material_owners = set(int(value) for value in owners[materials == material])
            counts["invalid_triangle_owners"] += len(
                material_owners - owner_sets[material]
            )

        metrics = sample["metrics"]
        counts["unresolved_constraints"] += int(
            metrics.get("unresolved_containments", 0)
            + metrics.get("unresolved_power_slivers", 0)
        )
        maxima["parent_gas_residual_m3"] = max(maxima["parent_gas_residual_m3"], parent_residual)
        maxima["frame_gas_residual_m3"] = max(maxima["frame_gas_residual_m3"], abs(frame_gas - source_gas))
        maxima["analytic_geometry_residual_m3"] = max(maxima["analytic_geometry_residual_m3"], abs(analytic_gas - frame_gas))
        maxima["radius_inflation_m"] = max(maxima["radius_inflation_m"], max(radius_inflation, default=0.0))
        maxima["packing_displacement_m"] = max(maxima["packing_displacement_m"], float(displacement.max(initial=0.0)))
        maxima["packing_displacement_record_residual_m"] = max(maxima["packing_displacement_record_residual_m"], displacement_record_residual)
        maxima["film_budget_fraction"] = max(maxima["film_budget_fraction"], film_area / max(available_film_area, 1.0e-30))
        maxima["film_liquid_formula_residual_m3"] = max(maxima["film_liquid_formula_residual_m3"], film_formula_residual)
        maxima["border_liquid_formula_residual_m3"] = max(maxima["border_liquid_formula_residual_m3"], border_formula_residual)
        maxima["node_liquid_formula_residual_m3"] = max(maxima["node_liquid_formula_residual_m3"], node_formula_residual)
        maxima["cell_height_to_radius"] = max(maxima["cell_height_to_radius"], float(height_to_radius.max(initial=0.0)))
        totals["cells"] += len(cells)
        totals["films"] += len(films)
        totals["borders"] += len(borders)
        totals["nodes"] += len(nodes)
        totals["triangles"] += len(triangles)
        totals["gas_volume_m3"] += frame_gas
        totals["film_area_m2"] += film_area
        totals["film_liquid_volume_m3"] += float(films["liquid_volume"].sum(dtype=np.float64))
        totals["border_liquid_volume_m3"] += float(borders["liquid_volume"].sum(dtype=np.float64))
        totals["node_liquid_volume_m3"] += float(nodes["liquid_volume"].sum(dtype=np.float64))
        frame_ledgers.append({
            "source_sample_index": source,
            "gas_volume_m3": frame_gas,
            "film_area_m2": film_area,
            "film_liquid_volume_m3": float(films["liquid_volume"].sum(dtype=np.float64)),
            "border_liquid_volume_m3": float(borders["liquid_volume"].sum(dtype=np.float64)),
            "node_liquid_volume_m3": float(nodes["liquid_volume"].sum(dtype=np.float64)),
        })

    tolerances = {
        "gas_volume_m3": 1.0e-12,
        "analytic_geometry_m3": 1.0e-11,
        "liquid_formula_m3": 1.0e-14,
        "radius_inflation_m": 2.0e-9,
        "packing_displacement_m": maximum_displacement + 2.0e-7,
        "packing_displacement_record_m": 2.0e-7,
        "film_budget_fraction": 1.0 + 1.0e-9,
        "cell_height_to_radius": maximum_cell_aspect + 2.0e-5,
    }
    for name, value in counts.items():
        if value:
            failures.append(f"{name}={value}")
    checks = (
        ("parent gas", maxima["parent_gas_residual_m3"], tolerances["gas_volume_m3"]),
        ("frame gas", maxima["frame_gas_residual_m3"], tolerances["gas_volume_m3"]),
        ("analytic geometry", maxima["analytic_geometry_residual_m3"], tolerances["analytic_geometry_m3"]),
        ("radius inflation", maxima["radius_inflation_m"], tolerances["radius_inflation_m"]),
        ("packing displacement", maxima["packing_displacement_m"], tolerances["packing_displacement_m"]),
        ("packing displacement record", maxima["packing_displacement_record_residual_m"], tolerances["packing_displacement_record_m"]),
        ("film liquid formula", maxima["film_liquid_formula_residual_m3"], tolerances["liquid_formula_m3"]),
        ("border liquid formula", maxima["border_liquid_formula_residual_m3"], tolerances["liquid_formula_m3"]),
        ("node liquid formula", maxima["node_liquid_formula_residual_m3"], tolerances["liquid_formula_m3"]),
        ("film budget", maxima["film_budget_fraction"], tolerances["film_budget_fraction"]),
        ("cell height/radius", maxima["cell_height_to_radius"], tolerances["cell_height_to_radius"]),
    )
    for label, value, limit in checks:
        if value > limit:
            failures.append(f"{label}: {value} > {limit}")

    report = {
        "schema": 1,
        "product": "whitewater_v6_plateau_cells_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "valid": not failures,
        "failures": failures,
        "input": {
            "plateau_manifest": str(plateau_directory / "manifest.json"),
            "plateau_manifest_sha256": sha256_file(plateau_directory / "manifest.json"),
        },
        "criteria": {
            "input_manifests_are_hash_locked": not any("manifest hash" in item for item in failures),
            "parent_child_gas_volume_closes": maxima["parent_gas_residual_m3"] <= tolerances["gas_volume_m3"],
            "physical_radii_are_not_inflated": maxima["radius_inflation_m"] <= tolerances["radius_inflation_m"],
            "analytic_cell_geometry_closes": maxima["analytic_geometry_residual_m3"] <= tolerances["analytic_geometry_m3"],
            "shared_film_pairs_are_unique": counts["duplicate_film_pairs"] == 0,
            "film_area_is_budgeted_separately": maxima["film_budget_fraction"] <= tolerances["film_budget_fraction"],
            "film_border_node_liquid_formulas_close": max(maxima["film_liquid_formula_residual_m3"], maxima["border_liquid_formula_residual_m3"], maxima["node_liquid_formula_residual_m3"]) <= tolerances["liquid_formula_m3"],
            "nodes_have_at_least_three_incident_films": counts["invalid_node_incidence"] == 0,
            "topology_events_match_frame_differences": counts["topology_event_mismatches"] == 0,
            "render_mesh_is_finite_indexed_and_owned": not any(counts[name] for name in ("nonfinite_records", "invalid_mesh_indices", "invalid_triangle_contract", "invalid_triangle_owners")),
            "packing_stays_inside_tether": maxima["packing_displacement_m"] <= tolerances["packing_displacement_m"],
            "no_constraints_remain_unresolved": counts["unresolved_constraints"] == 0,
            "production_has_no_unsupported_rupture": counts["unsupported_ruptures"] == 0,
            "gas_film_area_and_liquid_are_separate_ledgers": True,
        },
        "counts": counts,
        "maxima": maxima,
        "tolerances": tolerances,
        "totals_over_samples_not_inventory": totals,
        "frame_ledgers": frame_ledgers,
    }
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / "audit_report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
