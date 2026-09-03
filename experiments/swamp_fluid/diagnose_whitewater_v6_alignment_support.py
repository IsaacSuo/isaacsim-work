"""Classify failed render anchors against terrain, liquid, and solid support."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from whitewater.liquid_fields import GridSpec
from whitewater.marker_birth import sample_scalar_trilinear
from whitewater.render_surface_support import (
    _map_native_wet_mask_to_grid,
    load_render_surface_support_recipe,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("alignment_audit", type=Path)
parser.add_argument("liquid_field", type=Path)
parser.add_argument("surface_recipe_manifest", type=Path)
args = parser.parse_args()

audit = json.loads(args.alignment_audit.read_text(encoding="utf-8"))
detail = next(
    item["detail"]
    for item in audit["checks"]
    if item["name"] == "surface_anchor_proximity"
)
recipe = load_render_surface_support_recipe(args.surface_recipe_manifest)
with np.load(args.liquid_field) as cache:
    origin = tuple(float(value) for value in cache["origin"]) if "origin" in cache else None
    fluid_mask = np.asarray(cache["fluid_mask"], dtype=bool)
    phi = np.asarray(cache["phi"])
    collision = np.asarray(cache["collision_sdf"])
    sphere = np.asarray(cache["sphere_collision_sdf"])
    sphere_center = np.asarray(cache["sphere_center"], dtype=float)

# The production v6 grid is fixed and recorded by the alignment audit's liquid
# manifest; this diagnostic is intentionally scoped to that audited grid.
spec = GridSpec((-2.256, -1.568, -0.848), 0.016, fluid_mask.shape)
static_wet = _map_native_wet_mask_to_grid(recipe, spec)
structure = np.zeros((3, 3, 3), dtype=np.uint8)
structure[1, 1, :] = 1
structure[1, :, 1] = 1
structure[:, 1, 1] = 1
components, _ = label(fluid_mask, structure=structure)
x_values, y_values, z_values = spec.axes()
bulk_seed = (
    fluid_mask
    & static_wet[:, None, :]
    & (y_values[None, :, None] <= recipe.water_level)
)
main_labels = set(np.unique(components[bulk_seed]).tolist()) - {0}
top_index = np.full((spec.shape[0], spec.shape[2]), -1, dtype=np.int32)
occupied = np.any(fluid_mask, axis=1)
top_index[occupied] = spec.shape[1] - 1 - np.argmax(
    fluid_mask[:, ::-1, :], axis=1
)[occupied]

records = []
for group_name, group in detail["groups"].items():
    for anchor in group["worst_anchors"]:
        position = np.asarray(anchor["position"], dtype=float)
        grid_index = np.rint(
            (position - np.asarray(spec.origin)) / spec.spacing
        ).astype(int)
        inside = np.all((grid_index >= 0) & (grid_index < np.asarray(spec.shape)))
        ix, iy, iz = np.clip(grid_index, 0, np.asarray(spec.shape) - 1)
        component = int(components[ix, iy, iz])
        records.append(
            {
                "group": group_name,
                "index": anchor["index"],
                "position": position.tolist(),
                "surface_distance_m": anchor["distance_m"],
                "grid_index": grid_index.tolist(),
                "inside_grid": bool(inside),
                "static_wet_column": bool(static_wet[ix, iz]),
                "impact_distance_m": float(
                    np.linalg.norm(
                        position[[0, 2]]
                        - np.asarray((recipe.impact_x, recipe.impact_z))
                    )
                ),
                "column_top_y_m": (
                    float(y_values[top_index[ix, iz]])
                    if top_index[ix, iz] >= 0
                    else None
                ),
                "vertical_offset_from_column_top_m": (
                    float(position[1] - y_values[top_index[ix, iz]])
                    if top_index[ix, iz] >= 0
                    else None
                ),
                "nearest_component": component,
                "nearest_component_is_main": component in main_labels,
                "phi_m": float(sample_scalar_trilinear(phi, position[None], spec)[0]),
                "collision_sdf_m": float(
                    sample_scalar_trilinear(collision, position[None], spec)[0]
                ),
                "sphere_collision_sdf_m": float(
                    sample_scalar_trilinear(sphere, position[None], spec)[0]
                ),
                "distance_to_sphere_center_m": float(
                    np.linalg.norm(position - sphere_center)
                ),
            }
        )

print(json.dumps(records, indent=2))
