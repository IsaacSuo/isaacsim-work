"""Audit Splashsurf and foam-support migration to schema-2 open terrain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import meshio
import numpy as np

from whitewater.liquid_fields import GridSpec
from whitewater.render_surface_support import (
    build_render_surface_support,
    load_render_surface_support_recipe,
)
from whitewater.scene_contract import SceneContract


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_splashsurf_manifest", type=Path)
    parser.add_argument("open_terrain_splashsurf_manifest", type=Path)
    parser.add_argument("liquid_field_manifest", type=Path)
    parser.add_argument("liquid_field_file", type=Path)
    parser.add_argument("output_report", type=Path)
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
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def canonical_surface_faces(path):
    mesh = meshio.read(path)
    points = np.asarray(mesh.points, dtype=np.float32)
    cells = [cell.data for cell in mesh.cells if cell.type == "triangle"]
    if not cells:
        raise ValueError(f"Surface contains no triangles: {path}")
    faces = np.concatenate(cells, axis=0).astype(np.int64, copy=False)
    coordinates = points[faces]
    order = np.lexsort(
        (coordinates[:, :, 2], coordinates[:, :, 1], coordinates[:, :, 0]), axis=1
    )
    canonical = np.take_along_axis(
        coordinates, order[:, :, None], axis=1
    ).reshape((-1, 9))
    dtype = np.dtype([("vertices", np.float32, (9,))])
    records = np.ascontiguousarray(canonical).view(dtype).reshape(-1)
    return points, faces, np.unique(records)


def main():
    args = parse_args()
    old_manifest_path = args.legacy_splashsurf_manifest.resolve()
    new_manifest_path = args.open_terrain_splashsurf_manifest.resolve()
    liquid_manifest_path = args.liquid_field_manifest.resolve()
    liquid_field_path = args.liquid_field_file.resolve()
    for path in (
        old_manifest_path,
        new_manifest_path,
        liquid_manifest_path,
        liquid_field_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    old_manifest = load_json(old_manifest_path)
    new_manifest = load_json(new_manifest_path)
    old_configuration = old_manifest["configuration"]
    new_configuration = new_manifest["configuration"]
    terrain = new_configuration["terrain"]
    if terrain.get("representation") != "open_triangle_mesh_raster_adapter":
        raise ValueError("New Splashsurf manifest does not use open terrain")
    scene_contract_path = Path(terrain["scene_contract"]).resolve()
    contract = SceneContract.load(scene_contract_path)
    selection = contract.terrain.parameters["selection"]

    old_recipe = load_render_surface_support_recipe(old_manifest_path)
    new_recipe = load_render_surface_support_recipe(new_manifest_path)
    liquid_manifest = load_json(liquid_manifest_path)
    grid = liquid_manifest["grid"]
    spec = GridSpec(tuple(grid["origin"]), float(grid["spacing"]), tuple(grid["shape"]))
    with np.load(liquid_field_path, allow_pickle=False) as cache:
        fluid_mask = np.asarray(cache["fluid_mask"], dtype=np.uint8)
    old_support, old_metrics = build_render_surface_support(
        fluid_mask, spec, old_recipe
    )
    new_support, new_metrics = build_render_surface_support(
        fluid_mask, spec, new_recipe
    )

    selected_frames = new_configuration.get("selected_frames")
    if not isinstance(selected_frames, list) or len(selected_frames) != 1:
        raise ValueError("Consumer regression requires exactly one selected frame")
    frame = int(selected_frames[0])
    old_surface = (
        old_manifest_path.parent / "surface" / f"surface_{frame:04d}_clipped.obj"
    ).resolve()
    new_surface = (
        new_manifest_path.parent / "surface" / f"surface_{frame:04d}_clipped.obj"
    ).resolve()
    old_points, old_faces, old_records = canonical_surface_faces(old_surface)
    new_points, new_faces, new_records = canonical_surface_faces(new_surface)
    common_records = np.intersect1d(old_records, new_records, assume_unique=True)
    union_faces = len(old_records) + len(new_records) - len(common_records)

    exact_configuration_keys = (
        "mesh_smoothing_iters",
        "normal_smoothing_iters",
        "shoreline_mode",
        "minimum_layers",
        "shoreline_erosion_cells",
    )
    exact_configuration_mismatches = [
        key
        for key in exact_configuration_keys
        if old_configuration.get(key) != new_configuration.get(key)
    ]
    scalar_configuration_keys = (
        "water_level",
        "spacing",
        "particle_radius",
        "smoothing_length",
        "cube_size",
        "surface_threshold",
    )
    scalar_configuration_deltas = {
        key: abs(float(old_configuration[key]) - float(new_configuration[key]))
        for key in scalar_configuration_keys
    }
    vector_configuration_keys = ("impact", "particle_aabb_min", "particle_aabb_max")
    vector_configuration_deltas = {
        key: float(
            np.max(
                np.abs(
                    np.asarray(old_configuration[key], dtype=np.float64)
                    - np.asarray(new_configuration[key], dtype=np.float64)
                )
            )
        )
        for key in vector_configuration_keys
    }
    metrics = {
        "historical_support_seed_faces": len(
            selection["selection"]["support_seed_face_indices"]
        ),
        "selected_terrain_faces": int(selection["selected_mesh"]["triangle_count"]),
        "legacy_native_wet_cells": int(np.count_nonzero(old_recipe.static_wet_mask)),
        "open_terrain_native_wet_cells": int(
            np.count_nonzero(new_recipe.static_wet_mask)
        ),
        "native_wet_mask_different_cells": int(
            np.count_nonzero(old_recipe.static_wet_mask != new_recipe.static_wet_mask)
        ),
        "legacy_render_support_nodes": int(np.count_nonzero(old_support)),
        "open_terrain_render_support_nodes": int(np.count_nonzero(new_support)),
        "render_support_different_nodes": int(
            np.count_nonzero(old_support != new_support)
        ),
        "legacy_surface_vertices": int(len(old_points)),
        "open_terrain_surface_vertices": int(len(new_points)),
        "legacy_surface_faces": int(len(old_faces)),
        "open_terrain_surface_faces": int(len(new_faces)),
        "surface_common_faces": int(len(common_records)),
        "surface_old_only_faces": int(len(old_records) - len(common_records)),
        "surface_new_only_faces": int(len(new_records) - len(common_records)),
        "surface_face_jaccard": float(len(common_records) / union_faces),
        "exact_configuration_mismatches": exact_configuration_mismatches,
        "scalar_configuration_absolute_deltas": scalar_configuration_deltas,
        "vector_configuration_absolute_deltas_maximum": vector_configuration_deltas,
    }
    raster = terrain["raster"]
    criteria = {
        "open_terrain_provenance_is_complete": (
            contract.schema == 2
            and contract.terrain is not None
            and sha256_file(scene_contract_path) == terrain["scene_contract_sha256"]
            and sha256_file(Path(terrain["terrain_mesh"]))
            == terrain["terrain_mesh_sha256"]
            and sha256_file(Path(terrain["terrain_selection"]))
            == terrain["terrain_selection_sha256"]
            and sha256_file(Path(raster["path"])) == raster["sha256"]
            and sha256_file(Path(new_configuration["builder_script"]))
            == new_configuration["builder_script_sha256"]
            and sha256_file(Path(new_configuration["terrain_module"]))
            == new_configuration["terrain_module_sha256"]
        ),
        "global_first_hit_is_forbidden": (
            selection["selector"].get("global_first_hit_ray_cast") is False
            and raster["metadata"].get("global_first_hit_ray_cast") is False
        ),
        "support_seed_and_dry_bank_expansion_are_preserved": (
            metrics["historical_support_seed_faces"] == 62
            and metrics["selected_terrain_faces"] == 112
        ),
        "legacy_heightfield_is_not_a_new_dependency": (
            "heightfield" not in new_configuration
            and terrain["representation"] == "open_triangle_mesh_raster_adapter"
        ),
        "reconstruction_and_clip_configuration_is_numerically_exact": (
            not exact_configuration_mismatches
            and max(scalar_configuration_deltas.values()) <= 1.0e-12
            and max(vector_configuration_deltas.values()) <= 2.0e-7
        ),
        "native_shoreline_mask_is_exact": (
            metrics["native_wet_mask_different_cells"] == 0
        ),
        "render_surface_support_is_exact": (
            metrics["render_support_different_nodes"] == 0
            and old_metrics == new_metrics
        ),
        "splashsurf_free_surface_geometry_is_exact": (
            metrics["surface_old_only_faces"] == 0
            and metrics["surface_new_only_faces"] == 0
            and metrics["surface_face_jaccard"] == 1.0
        ),
    }
    report = {
        "schema": 1,
        "product": "whitewater_v6_open_terrain_consumer_regression_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "valid": all(criteria.values()),
        "criteria": criteria,
        "metrics": metrics,
        "old_render_support_metrics": old_metrics,
        "new_render_support_metrics": new_metrics,
        "provenance": {
            "legacy_manifest": str(old_manifest_path),
            "legacy_manifest_sha256": sha256_file(old_manifest_path),
            "open_terrain_manifest": str(new_manifest_path),
            "open_terrain_manifest_sha256": sha256_file(new_manifest_path),
            "legacy_surface": str(old_surface),
            "legacy_surface_sha256": sha256_file(old_surface),
            "open_terrain_surface": str(new_surface),
            "open_terrain_surface_sha256": sha256_file(new_surface),
            "liquid_manifest": str(liquid_manifest_path),
            "liquid_manifest_sha256": sha256_file(liquid_manifest_path),
            "liquid_field": str(liquid_field_path),
            "liquid_field_sha256": sha256_file(liquid_field_path),
        },
    }
    atomic_json(args.output_report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
