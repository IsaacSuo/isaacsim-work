"""Build the marker-side analogue of the production free-surface support.

The production Splashsurf clipper keeps the terrain-derived main pool and,
inside the impact region, splash geometry that remains connected to that pool.
This module expresses the same contract on the coarser v6 liquid grid so an
entrained bubble cannot become renderable surface foam on a peripheral film or
an isolated droplet that the production surface deliberately removes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, label

from .liquid_fields import GridSpec
from .scene_contract import SceneContract


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _four_connected_component(mask, seed_x, seed_z, x_values, z_values):
    wet_indices = np.argwhere(mask)
    if not len(wet_indices):
        raise ValueError("The terrain shoreline recipe contains no wet cells")
    seed_ix = int(np.argmin(np.abs(x_values - seed_x)))
    seed_iz = int(np.argmin(np.abs(z_values - seed_z)))
    if not mask[seed_ix, seed_iz]:
        dx = x_values[wet_indices[:, 0]] - seed_x
        dz = z_values[wet_indices[:, 1]] - seed_z
        nearest = wet_indices[int(np.argmin(dx * dx + dz * dz))]
        seed_ix, seed_iz = map(int, nearest)
    selected = np.zeros_like(mask, dtype=bool)
    selected[seed_ix, seed_iz] = True
    queue = deque(((seed_ix, seed_iz),))
    nx, nz = mask.shape
    while queue:
        ix, iz = queue.popleft()
        for next_ix, next_iz in (
            (ix - 1, iz),
            (ix + 1, iz),
            (ix, iz - 1),
            (ix, iz + 1),
        ):
            if (
                0 <= next_ix < nx
                and 0 <= next_iz < nz
                and mask[next_ix, next_iz]
                and not selected[next_ix, next_iz]
            ):
                selected[next_ix, next_iz] = True
                queue.append((next_ix, next_iz))
    return selected


def _erode_four_connected(mask, iterations):
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(int(iterations)):
        eroded = np.zeros_like(result)
        eroded[1:-1, 1:-1] = (
            result[1:-1, 1:-1]
            & result[:-2, 1:-1]
            & result[2:, 1:-1]
            & result[1:-1, :-2]
            & result[1:-1, 2:]
        )
        result = eroded
    return result


@dataclass(frozen=True)
class RenderSurfaceSupportRecipe:
    manifest_path: Path
    manifest_sha256: str
    terrain_representation: str
    terrain_path: Path
    terrain_sha256: str
    terrain_selection_path: Path | None
    terrain_selection_sha256: str | None
    terrain_raster_path: Path
    terrain_raster_sha256: str
    water_level: float
    particle_spacing: float
    minimum_layers: int
    shoreline_erosion_cells: int
    impact_x: float
    impact_z: float
    impact_radius: float
    x_values: np.ndarray
    z_values: np.ndarray
    static_wet_mask: np.ndarray

    def metadata(self):
        return {
            "recipe_manifest": str(self.manifest_path),
            "recipe_manifest_sha256": self.manifest_sha256,
            "terrain": {
                "representation": self.terrain_representation,
                "path": str(self.terrain_path),
                "sha256": self.terrain_sha256,
                "selection_path": (
                    str(self.terrain_selection_path)
                    if self.terrain_selection_path is not None
                    else None
                ),
                "selection_sha256": self.terrain_selection_sha256,
                "raster_path": str(self.terrain_raster_path),
                "raster_sha256": self.terrain_raster_sha256,
            },
            "shoreline_mode": "terrain",
            "water_level": self.water_level,
            "particle_spacing": self.particle_spacing,
            "minimum_layers": self.minimum_layers,
            "shoreline_erosion_cells": self.shoreline_erosion_cells,
            "impact": [self.impact_x, self.impact_z, self.impact_radius],
            "static_wet_cells": int(np.count_nonzero(self.static_wet_mask)),
            "dynamic_contract": (
                "six-connected fluid component touching the main terrain pool, "
                "dilated by one liquid-field cell, inside the production impact region"
            ),
            "static_contract": (
                "highest locally reconstructed liquid top shell inside the eroded "
                "terrain main-pool columns"
            ),
        }


def load_render_surface_support_recipe(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    configuration = payload.get("configuration", payload)
    if configuration.get("shoreline_mode") != "terrain":
        raise ValueError("Marker support requires the production terrain shoreline mode")
    if configuration.get("shoreline_particles_ply") is not None:
        raise ValueError("Reference-Ply shorelines are forbidden for marker support")
    impact = configuration.get("impact")
    if not isinstance(impact, list) or len(impact) != 3:
        raise ValueError("Surface recipe is missing its three-value impact region")
    terrain_configuration = configuration.get("terrain")
    if terrain_configuration is None:
        terrain_representation = "regular_heightfield_legacy_adapter"
        terrain_path = Path(configuration["heightfield"]).resolve()
        terrain_sha256 = configuration.get("heightfield_sha256")
        terrain_selection_path = None
        terrain_selection_sha256 = None
        terrain_raster_path = terrain_path
        terrain_raster_sha256 = terrain_sha256
    else:
        terrain_representation = terrain_configuration.get("representation")
        raster = terrain_configuration.get("raster", {})
        terrain_raster_path = Path(raster["path"]).resolve()
        terrain_raster_sha256 = raster.get("sha256")
        if terrain_representation == "regular_heightfield_legacy_adapter":
            terrain_path = Path(terrain_configuration["heightfield"]).resolve()
            terrain_sha256 = terrain_configuration.get("heightfield_sha256")
            terrain_selection_path = None
            terrain_selection_sha256 = None
        elif terrain_representation == "open_triangle_mesh_raster_adapter":
            scene_contract_path = Path(terrain_configuration["scene_contract"]).resolve()
            if sha256_file(scene_contract_path) != terrain_configuration.get(
                "scene_contract_sha256"
            ):
                raise ValueError("Surface recipe scene-contract hash mismatch")
            contract = SceneContract.load(scene_contract_path)
            if contract.schema != 2 or contract.terrain is None:
                raise ValueError("Surface recipe does not resolve schema-2 terrain")
            terrain_path = Path(terrain_configuration["terrain_mesh"]).resolve()
            terrain_sha256 = terrain_configuration.get("terrain_mesh_sha256")
            terrain_selection_path = Path(
                terrain_configuration["terrain_selection"]
            ).resolve()
            terrain_selection_sha256 = terrain_configuration.get(
                "terrain_selection_sha256"
            )
            if (
                str(terrain_path) != contract.terrain.parameters["path"]
                or terrain_sha256 != contract.terrain.parameters["sha256"]
                or str(terrain_selection_path)
                != contract.terrain.parameters["selection_path"]
                or terrain_selection_sha256
                != contract.terrain.parameters["selection_sha256"]
            ):
                raise ValueError("Surface recipe terrain differs from its scene contract")
        else:
            raise ValueError(
                f"Unsupported surface recipe terrain {terrain_representation!r}"
            )
    for path, expected, label_name in (
        (terrain_path, terrain_sha256, "terrain"),
        (terrain_raster_path, terrain_raster_sha256, "terrain raster"),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Surface recipe {label_name} hash does not match the file")
    if terrain_selection_path is not None and (
        not terrain_selection_path.is_file()
        or sha256_file(terrain_selection_path) != terrain_selection_sha256
    ):
        raise ValueError("Surface recipe terrain selection hash mismatch")
    with np.load(terrain_raster_path, allow_pickle=False) as cache:
        x_values = np.asarray(cache["x_values"], dtype=np.float64)
        z_values = np.asarray(cache["z_values"], dtype=np.float64)
        terrain_y = np.asarray(cache["terrain_y"], dtype=np.float64)
        if terrain_representation == "open_triangle_mesh_raster_adapter":
            if str(np.asarray(cache["terrain_mesh_sha256"]).item()) != terrain_sha256:
                raise ValueError("Terrain raster does not reference the terrain mesh")
            if (
                str(np.asarray(cache["terrain_selection_sha256"]).item())
                != terrain_selection_sha256
            ):
                raise ValueError("Terrain raster does not reference the selection record")
    particle_spacing = float(configuration["spacing"])
    water_level = float(configuration["water_level"])
    minimum_layers = int(configuration["minimum_layers"])
    erosion = int(configuration["shoreline_erosion_cells"])
    fluid_rest_offset = 0.5 * particle_spacing
    particle_contact_offset = fluid_rest_offset / 0.6
    floor_clearance = particle_contact_offset + 0.001
    top = water_level - 0.45 * particle_spacing
    maximum_floor_y = top - floor_clearance - (minimum_layers - 1) * particle_spacing
    terrain_wet = np.isfinite(terrain_y) & (terrain_y <= maximum_floor_y)
    main_pool = _four_connected_component(
        terrain_wet, float(impact[0]), float(impact[1]), x_values, z_values
    )
    static_wet_mask = _erode_four_connected(main_pool, erosion)
    return RenderSurfaceSupportRecipe(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        terrain_representation=terrain_representation,
        terrain_path=terrain_path,
        terrain_sha256=terrain_sha256,
        terrain_selection_path=terrain_selection_path,
        terrain_selection_sha256=terrain_selection_sha256,
        terrain_raster_path=terrain_raster_path,
        terrain_raster_sha256=terrain_raster_sha256,
        water_level=water_level,
        particle_spacing=particle_spacing,
        minimum_layers=minimum_layers,
        shoreline_erosion_cells=erosion,
        impact_x=float(impact[0]),
        impact_z=float(impact[1]),
        impact_radius=float(impact[2]),
        x_values=x_values,
        z_values=z_values,
        static_wet_mask=static_wet_mask,
    )


def _map_native_wet_mask_to_grid(recipe, spec):
    grid_x, _, grid_z = spec.axes()
    dx = float(np.median(np.diff(recipe.x_values)))
    dz = float(np.median(np.diff(recipe.z_values)))
    ix = np.rint((grid_x - recipe.x_values[0]) / dx).astype(np.int64)
    iz = np.rint((grid_z - recipe.z_values[0]) / dz).astype(np.int64)
    valid_x = (ix >= 0) & (ix < len(recipe.x_values))
    valid_z = (iz >= 0) & (iz < len(recipe.z_values))
    mapped = np.zeros((spec.shape[0], spec.shape[2]), dtype=bool)
    mapped[np.ix_(valid_x, valid_z)] = recipe.static_wet_mask[
        np.ix_(ix[valid_x], iz[valid_z])
    ]
    return mapped


def build_render_surface_support(fluid_mask, spec: GridSpec, recipe):
    """Return a boolean 3-D field on which surface bubbles may exist."""
    fluid_mask = np.asarray(fluid_mask, dtype=bool)
    if fluid_mask.shape != spec.shape:
        raise ValueError("Fluid mask does not match the marker grid")
    static_wet = _map_native_wet_mask_to_grid(recipe, spec)
    structure = np.zeros((3, 3, 3), dtype=np.uint8)
    structure[1, 1, :] = 1
    structure[1, :, 1] = 1
    structure[:, 1, 1] = 1
    components, component_count = label(fluid_mask, structure=structure)
    _, y_values, _ = spec.axes()
    bulk_seed = (
        fluid_mask
        & static_wet[:, None, :]
        & (y_values[None, :, None] <= recipe.water_level)
    )
    seed_labels = np.unique(components[bulk_seed])
    seed_labels = seed_labels[seed_labels != 0]
    if not len(seed_labels):
        raise ValueError("No liquid component touches the production main-pool mask")
    main_connected = np.isin(components, seed_labels)
    # Projected surface markers sit just outside the binary carrier mask. The
    # dilation follows already-labelled main liquid and cannot bridge to a
    # separately labelled droplet before connectivity is decided.
    main_proximity = binary_dilation(main_connected, structure=structure, iterations=1)
    x_values, _, z_values = spec.axes()
    impact_xz = (
        (x_values[:, None] - recipe.impact_x) ** 2
        + (z_values[None, :] - recipe.impact_z) ** 2
        <= recipe.impact_radius**2
    )
    minimum_splash_y = recipe.water_level - max(0.04, 5.0 * recipe.particle_spacing)
    # The liquid SDF has zero crossings on both the liquid-air free surface and
    # liquid-solid contact boundaries. Mirror the clipper's per-column top-shell
    # selection so the pool bottom and wetted sphere wall cannot host foam.
    occupied_columns = np.any(fluid_mask, axis=1)
    top_index = np.full((spec.shape[0], spec.shape[2]), -1, dtype=np.int32)
    top_index[occupied_columns] = spec.shape[1] - 1 - np.argmax(
        fluid_mask[:, ::-1, :], axis=1
    )[occupied_columns]
    y_index = np.arange(spec.shape[1], dtype=np.int32)[None, :, None]
    # One coarse liquid-field cell below the highest occupied node covers the
    # 12 mm production top-shell thickness on a 16 mm marker grid. Nodes above
    # remain eligible because the projected zero crossing lies outside the
    # binary carrier mask.
    static_top_shell = (
        occupied_columns[:, None, :]
        & (y_index >= top_index[:, None, :] - 1)
    )
    static = static_wet[:, None, :] & static_top_shell
    dynamic = (
        main_proximity
        & impact_xz[:, None, :]
        & (y_values[None, :, None] >= minimum_splash_y)
    )
    support = static | dynamic
    metrics = {
        "fluid_component_count": int(component_count),
        "main_component_count": int(len(seed_labels)),
        "static_wet_grid_columns": int(np.count_nonzero(static_wet)),
        "static_top_support_nodes": int(np.count_nonzero(static)),
        "dynamic_support_nodes": int(np.count_nonzero(dynamic & ~static_wet[:, None, :])),
        "support_nodes": int(np.count_nonzero(support)),
    }
    return support.astype(np.uint8), metrics
