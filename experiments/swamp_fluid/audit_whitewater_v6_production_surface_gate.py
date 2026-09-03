"""Audit an artist-configured Splashsurf gate against the locked domain build."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import meshio
import numpy as np
from scipy.spatial import cKDTree

from whitewater.render_surface_support import load_render_surface_support_recipe


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_manifest", type=Path)
    parser.add_argument("locked_manifest", type=Path)
    parser.add_argument("legacy_surface", type=Path)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output_report", type=Path)
    parser.add_argument("--source-sample", type=int, required=True)
    parser.add_argument("--legacy-frame", type=int, required=True)
    parser.add_argument("--source-fps", type=float, default=120.0)
    parser.add_argument("--legacy-fps", type=float, default=30.0)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_diagnostics(value):
    return {
        name: value
        for name, value in re.findall(r"([a-z_]+)=([^ ]+)", str(value))
    }


def load_surface(path):
    mesh = meshio.read(path)
    points = np.asarray(mesh.points, dtype=np.float64)
    triangles = [cell.data for cell in mesh.cells if cell.type == "triangle"]
    if not triangles:
        raise ValueError(f"No triangle cells in {path}")
    faces = np.concatenate(triangles, axis=0).astype(np.int64, copy=False)
    coordinates = points[faces]
    cross = np.cross(coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0])
    face_area = 0.5 * np.linalg.norm(cross, axis=1)

    directed = np.concatenate(
        (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0
    )
    undirected = np.sort(directed, axis=1)
    unique_edges, inverse, counts = np.unique(
        undirected, axis=0, return_inverse=True, return_counts=True
    )
    orientation = np.where(directed[:, 0] == undirected[:, 0], 1, -1)
    orientation_sum = np.bincount(inverse, weights=orientation, minlength=len(unique_edges))
    boundary_edges = unique_edges[counts == 1]
    canonical_faces = np.sort(faces, axis=1)
    unique_faces = np.unique(canonical_faces, axis=0)
    return {
        "path": Path(path).resolve(),
        "points": points,
        "faces": faces,
        "area": float(face_area.sum()),
        "bounds_min": points.min(axis=0),
        "bounds_max": points.max(axis=0),
        "boundary_edges": boundary_edges,
        "boundary_edge_count": int(len(boundary_edges)),
        "nonmanifold_edge_count": int(np.count_nonzero(counts > 2)),
        "winding_conflict_count": int(
            np.count_nonzero((counts == 2) & (np.abs(orientation_sum) == 2))
        ),
        "degenerate_face_count": int(
            np.count_nonzero(face_area <= np.finfo(np.float32).eps**2)
        ),
        "duplicate_face_count": int(len(faces) - len(unique_faces)),
    }


def nearest_metrics(first, second):
    forward = cKDTree(second).query(first, workers=-1)[0]
    reverse = cKDTree(first).query(second, workers=-1)[0]
    combined = np.concatenate((forward, reverse))
    return {
        "maximum_m": float(combined.max(initial=0.0)),
        "p99_m": float(np.quantile(combined, 0.99)) if len(combined) else 0.0,
        "rms_m": float(np.sqrt(np.mean(np.square(combined)))) if len(combined) else 0.0,
    }


def directed_nearest_metrics(source, target):
    distances = cKDTree(target).query(source, workers=-1)[0]
    return {
        "maximum_m": float(distances.max(initial=0.0)),
        "p99_m": float(np.quantile(distances, 0.99)) if len(distances) else 0.0,
        "rms_m": float(np.sqrt(np.mean(np.square(distances)))) if len(distances) else 0.0,
    }


def raster_arrays(path):
    with np.load(path, allow_pickle=False) as cache:
        return {name: np.asarray(cache[name]) for name in cache.files}


def arrays_exact(first, second):
    if set(first) != set(second):
        return False
    for name in first:
        left = first[name]
        right = second[name]
        if np.issubdtype(left.dtype, np.inexact):
            matches = np.array_equal(left, right, equal_nan=True)
        else:
            matches = np.array_equal(left, right)
        if not matches:
            return False
    return True


def write_once(path, payload):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    args = parse_args()
    new_manifest_path = args.new_manifest.resolve()
    locked_manifest_path = args.locked_manifest.resolve()
    legacy_surface_path = args.legacy_surface.resolve()
    source_manifest_path = args.source_manifest.resolve()
    for path in (
        new_manifest_path, locked_manifest_path, legacy_surface_path, source_manifest_path
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    new_manifest = load_json(new_manifest_path)
    locked_manifest = load_json(locked_manifest_path)
    frame_id = f"{args.source_sample:06d}"
    new_frame = next(
        frame for frame in new_manifest["state"]["frames"]
        if str(frame["frame_id"]) == frame_id
    )
    locked_frame = next(
        frame for frame in locked_manifest["state"]["frames"]
        if str(frame["frame_id"]) == frame_id
    )
    new_surface_path = new_manifest_path.parent / "surface" / new_frame["surface_file"]
    locked_surface_path = locked_manifest_path.parent / "surface" / locked_frame["surface_file"]
    new_surface = load_surface(new_surface_path)
    locked_surface = load_surface(locked_surface_path)
    legacy_surface = load_surface(legacy_surface_path)

    new_configuration = new_manifest["configuration"]
    locked_configuration = locked_manifest["configuration"]
    new_terrain = new_configuration["terrain"]
    locked_terrain = locked_configuration["terrain"]
    new_raster_path = Path(new_terrain["raster"]["path"])
    locked_raster_path = Path(locked_terrain["raster"]["path"])
    raster_exact = arrays_exact(
        raster_arrays(new_raster_path), raster_arrays(locked_raster_path)
    )
    new_recipe = load_render_surface_support_recipe(new_manifest_path)
    locked_recipe = load_render_surface_support_recipe(locked_manifest_path)
    shoreline_different_cells = int(
        np.count_nonzero(new_recipe.static_wet_mask != locked_recipe.static_wet_mask)
    )

    locked_distance = nearest_metrics(new_surface["points"], locked_surface["points"])
    water_level = float(new_configuration["water_level"])
    spacing = float(new_configuration["spacing"])
    impact_x, impact_z, impact_radius = map(float, new_configuration["impact"])

    def quiet_top(surface):
        points = surface["points"]
        radial = np.hypot(points[:, 0] - impact_x, points[:, 2] - impact_z)
        return points[
            (radial >= impact_radius + 2.0 * spacing)
            & (points[:, 1] >= water_level - 0.04)
            & (points[:, 1] <= water_level + 0.04)
        ]

    new_quiet = quiet_top(new_surface)
    legacy_quiet = quiet_top(legacy_surface)
    legacy_to_new_quiet_distance = directed_nearest_metrics(legacy_quiet, new_quiet)
    new_to_legacy_quiet_distance = directed_nearest_metrics(new_quiet, legacy_quiet)

    source_manifest = load_json(source_manifest_path)
    source_row = next(
        row for row in source_manifest["samples"]
        if int(row["sample_index"]) == args.source_sample
    )
    source_time = float(source_row["simulation_time"])
    expected_source_time = args.source_sample / args.source_fps
    legacy_time = args.legacy_frame / args.legacy_fps
    new_diagnostics = parse_diagnostics(new_frame["diagnostics"])
    locked_diagnostics = parse_diagnostics(locked_frame["diagnostics"])

    terrain_hashes_match = all(
        new_terrain[name] == locked_terrain[name]
        for name in (
            "scene_contract_sha256", "terrain_mesh_sha256", "terrain_selection_sha256"
        )
    )
    terrain_files_authentic = all(
        sha256_file(Path(new_terrain[path_name])) == new_terrain[hash_name]
        for path_name, hash_name in (
            ("scene_contract", "scene_contract_sha256"),
            ("terrain_mesh", "terrain_mesh_sha256"),
            ("terrain_selection", "terrain_selection_sha256"),
        )
    )
    topology_clean = all(
        new_surface[name] == 0
        for name in (
            "nonmanifold_edge_count", "winding_conflict_count",
            "degenerate_face_count", "duplicate_face_count",
        )
    )
    manifest_topology_clean = all(
        new_diagnostics.get(name) == "0"
        for name in (
            "removed_degenerate_faces", "removed_duplicate_faces",
            "nonmanifold_edges", "winding_conflicts",
        )
    )
    area_relative_delta = abs(new_surface["area"] - locked_surface["area"]) / locked_surface["area"]
    boundary_edge_delta = abs(
        new_surface["boundary_edge_count"] - locked_surface["boundary_edge_count"]
    )
    criteria = {
        "source_to_legacy_time_mapping_is_exact": (
            abs(source_time - expected_source_time) <= 1.0e-12
            and abs(source_time - legacy_time) <= 1.0e-12
        ),
        "selected_terrain_provenance_matches_locked_build": (
            terrain_hashes_match and terrain_files_authentic
        ),
        "terrain_raster_is_exact": raster_exact,
        "fixed_shoreline_mask_is_exact": shoreline_different_cells == 0,
        "surface_recipe_is_exact": (
            all(
                abs(float(new_configuration[name]) - float(locked_configuration[name]))
                <= 1.0e-14
                for name in (
                    "water_level", "spacing", "particle_radius", "smoothing_length",
                    "cube_size", "surface_threshold",
                )
            )
            and all(
                new_configuration[name] == locked_configuration[name]
                for name in (
                    "mesh_smoothing_iters", "normal_smoothing_iters",
                    "shoreline_mode", "minimum_layers", "shoreline_erosion_cells",
                )
            )
            and all(
                np.allclose(
                    new_configuration[name], locked_configuration[name],
                    rtol=0.0, atol=1.0e-14,
                )
                for name in ("particle_aabb_min", "particle_aabb_max")
            )
        ),
        "surface_topology_is_clean": topology_clean and manifest_topology_clean,
        "surface_counts_match_locked_build": (
            len(new_surface["points"]) == len(locked_surface["points"])
            and len(new_surface["faces"]) == len(locked_surface["faces"])
        ),
        "surface_geometry_matches_locked_build": (
            locked_distance["maximum_m"] <= 1.0e-7
            and area_relative_delta <= 1.0e-10
            and boundary_edge_delta == 0
        ),
        "quiet_free_surface_remains_close_to_legacy_frame": (
            len(new_quiet) > 0
            and len(legacy_quiet) > 0
            and legacy_to_new_quiet_distance["p99_m"] <= 1.5 * spacing
        ),
        "no_regressed_rectangular_hole_signature": (
            shoreline_different_cells == 0
            and boundary_edge_delta == 0
            and new_diagnostics.get("top_shell_faces")
            == locked_diagnostics.get("top_shell_faces")
        ),
    }
    report = {
        "schema": 1,
        "product": "whitewater_v6_production_surface_gate_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "valid": all(criteria.values()),
        "criteria": criteria,
        "metrics": {
            "source_sample": args.source_sample,
            "source_time_s": source_time,
            "legacy_frame": args.legacy_frame,
            "legacy_time_s": legacy_time,
            "new_vertices": int(len(new_surface["points"])),
            "new_faces": int(len(new_surface["faces"])),
            "new_area_m2": new_surface["area"],
            "locked_area_m2": locked_surface["area"],
            "area_relative_delta": area_relative_delta,
            "new_boundary_edges": new_surface["boundary_edge_count"],
            "locked_boundary_edges": locked_surface["boundary_edge_count"],
            "boundary_edge_delta": boundary_edge_delta,
            "shoreline_different_cells": shoreline_different_cells,
            "locked_geometry_distance": locked_distance,
            "legacy_to_new_quiet_top_distance": legacy_to_new_quiet_distance,
            "new_to_legacy_quiet_top_distance": new_to_legacy_quiet_distance,
            "new_quiet_top_vertices": int(len(new_quiet)),
            "legacy_quiet_top_vertices": int(len(legacy_quiet)),
            "new_topology": {
                name: new_surface[name]
                for name in (
                    "boundary_edge_count", "nonmanifold_edge_count",
                    "winding_conflict_count", "degenerate_face_count",
                    "duplicate_face_count",
                )
            },
            "new_clip_diagnostics": new_diagnostics,
        },
        "provenance": {
            "new_manifest": str(new_manifest_path),
            "new_manifest_sha256": sha256_file(new_manifest_path),
            "locked_manifest": str(locked_manifest_path),
            "locked_manifest_sha256": sha256_file(locked_manifest_path),
            "new_surface": str(new_surface_path.resolve()),
            "new_surface_sha256": sha256_file(new_surface_path),
            "locked_surface": str(locked_surface_path.resolve()),
            "locked_surface_sha256": sha256_file(locked_surface_path),
            "legacy_surface": str(legacy_surface_path),
            "legacy_surface_sha256": sha256_file(legacy_surface_path),
        },
    }
    write_once(args.output_report, report)
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
