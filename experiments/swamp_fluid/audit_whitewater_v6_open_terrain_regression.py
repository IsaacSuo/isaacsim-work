"""Audit a schema-2 open terrain against the locked heightfield baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.liquid_fields import GridSpec
from whitewater.scene_contract import SceneContract
from whitewater.terrain_fields import open_terrain_collision_sdf


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_contract", type=Path)
    parser.add_argument("baseline_manifest", type=Path)
    parser.add_argument("seed_selection_record", type=Path)
    parser.add_argument("output_report", type=Path)
    parser.add_argument("--near-surface-band", type=float, default=0.05)
    parser.add_argument("--near-surface-tolerance", type=float, default=0.002)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main():
    args = parse_args()
    scene_contract_path = args.scene_contract.resolve()
    baseline_manifest_path = args.baseline_manifest.resolve()
    seed_selection_path = args.seed_selection_record.resolve()
    for path in (scene_contract_path, baseline_manifest_path, seed_selection_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    near_surface_band = float(args.near_surface_band)
    near_surface_tolerance = float(args.near_surface_tolerance)
    if near_surface_band <= 0.0 or near_surface_tolerance <= 0.0:
        raise ValueError("Audit distances must be positive")

    contract = SceneContract.load(scene_contract_path)
    if contract.schema != 2 or contract.terrain is None:
        raise ValueError("Audit requires a schema-2 scene with embedded terrain")
    baseline_manifest = load_json(baseline_manifest_path)
    grid = baseline_manifest["grid"]
    spec = GridSpec(tuple(grid["origin"]), float(grid["spacing"]), tuple(grid["shape"]))
    baseline_static_path = (
        baseline_manifest_path.parent / baseline_manifest["static_fields"]["file"]
    ).resolve()
    if sha256_file(baseline_static_path) != baseline_manifest["static_fields"]["sha256"]:
        raise ValueError("Locked baseline static-field hash mismatch")
    with np.load(baseline_static_path, allow_pickle=False) as cache:
        baseline_sdf = np.asarray(cache["terrain_collision_sdf"], dtype=np.float32)

    terrain_sdf, terrain_valid, terrain_metadata = open_terrain_collision_sdf(
        spec, contract.terrain, length_scale=contract.metres_per_unit
    )
    terrain_valid_bool = terrain_valid.astype(bool)
    comparable = terrain_valid_bool & np.isfinite(baseline_sdf)
    near_surface = comparable & (np.abs(baseline_sdf) <= near_surface_band)
    if not np.any(comparable) or not np.any(near_surface):
        raise ValueError("Terrain regression has no comparable nodes")

    expanded_selection = contract.terrain.parameters["selection"]
    seed_selection = load_json(seed_selection_path)
    expanded_seed_faces = expanded_selection["selection"]["support_seed_face_indices"]
    seed_faces = seed_selection["selection"]["source_face_indices"]
    difference = np.abs(terrain_sdf[comparable] - baseline_sdf[comparable])
    near_difference = np.abs(
        terrain_sdf[near_surface] - baseline_sdf[near_surface]
    )
    sign_mismatch = int(
        np.count_nonzero(
            (terrain_sdf[comparable] < 0.0) != (baseline_sdf[comparable] < 0.0)
        )
    )
    metrics = {
        "grid_nodes": int(spec.cell_count),
        "valid_nodes": int(np.count_nonzero(terrain_valid_bool)),
        "valid_fraction": float(np.mean(terrain_valid_bool)),
        "sign_mismatch_nodes": sign_mismatch,
        "near_surface_nodes": int(np.count_nonzero(near_surface)),
        "near_surface_absolute_error_maximum": float(np.max(near_difference)),
        "near_surface_absolute_error_mean": float(np.mean(near_difference)),
        "all_nodes_absolute_error_maximum": float(np.max(difference)),
        "all_nodes_absolute_error_mean": float(np.mean(difference)),
        "support_seed_triangle_count": len(expanded_seed_faces),
        "selected_triangle_count": int(terrain_metadata["triangle_count"]),
        "selected_vertex_count": int(terrain_metadata["vertex_count"]),
        "selected_component_count": int(terrain_metadata["connected_component_count"]),
        "boundary_edge_count": int(terrain_metadata["boundary_edge_count"]),
        "seed_face_indices_match_reference": expanded_seed_faces == seed_faces,
    }
    selection_guards = expanded_selection.get("selection_guards", {})
    criteria = {
        "scene_contract_owns_open_terrain": (
            contract.schema == 2
            and contract.terrain.representation == "open_triangle_mesh"
        ),
        "selection_does_not_use_global_first_hit": (
            expanded_selection.get("selector", {}).get("global_first_hit_ray_cast")
            is False
            and selection_guards.get("global_first_hit_ray_cast") is False
        ),
        "historical_62_face_support_is_exact": (
            metrics["support_seed_triangle_count"] == 62
            and metrics["seed_face_indices_match_reference"]
        ),
        "dry_bank_expansion_is_explicit": (
            metrics["selected_triangle_count"] > metrics["support_seed_triangle_count"]
            and expanded_selection.get("selector", {}).get("component_policy")
            == "expand_components_connected_to_level_seeds"
        ),
        "one_connected_top_shell_selected": metrics["selected_component_count"] == 1,
        "production_grid_is_fully_covered": metrics["valid_fraction"] == 1.0,
        "solid_side_classification_matches_heightfield": sign_mismatch == 0,
        "near_surface_distance_matches_within_tolerance": (
            metrics["near_surface_absolute_error_maximum"] <= near_surface_tolerance
        ),
        "euclidean_distance_deviation_is_subvoxel": (
            metrics["all_nodes_absolute_error_maximum"] <= spec.spacing
        ),
    }
    report = {
        "schema": 1,
        "product": "whitewater_v6_open_terrain_regression_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "valid": all(criteria.values()),
        "criteria": criteria,
        "metrics": metrics,
        "tolerances": {
            "near_surface_band": near_surface_band,
            "near_surface_absolute_error_maximum": near_surface_tolerance,
            "all_nodes_absolute_error_maximum": spec.spacing,
        },
        "provenance": {
            "scene_contract": str(scene_contract_path),
            "scene_contract_sha256": sha256_file(scene_contract_path),
            "terrain_mesh": contract.terrain.parameters["path"],
            "terrain_mesh_sha256": contract.terrain.parameters["sha256"],
            "terrain_selection": contract.terrain.parameters["selection_path"],
            "terrain_selection_sha256": contract.terrain.parameters[
                "selection_sha256"
            ],
            "seed_selection": str(seed_selection_path),
            "seed_selection_sha256": sha256_file(seed_selection_path),
            "baseline_manifest": str(baseline_manifest_path),
            "baseline_manifest_sha256": sha256_file(baseline_manifest_path),
            "baseline_static_fields": str(baseline_static_path),
            "baseline_static_fields_sha256": sha256_file(baseline_static_path),
        },
        "terrain_metadata": terrain_metadata,
        "interpretation": {
            "baseline": "vertical height difference",
            "candidate": "oriented Euclidean closest-surface distance",
            "expected_nonzero_difference": (
                "Sloped surfaces have a shorter Euclidean normal distance than "
                "the historical vertical height difference. Sign and zero surface "
                "must agree; bounded sub-voxel magnitude differences are expected."
            ),
        },
    }
    atomic_json(args.output_report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
