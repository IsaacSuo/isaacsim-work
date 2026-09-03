"""Synthetic conservation and topology gates for Plateau-cell foam."""

from __future__ import annotations

import json

import numpy as np

from whitewater.liquid_fields import GridSpec
from whitewater.plateau_cells import PlateauCellModel, build_plateau_cells
from whitewater.surface_raft import RAFT_DTYPE


spec = GridSpec((-0.05, -0.05, -0.05), 0.005, (21, 21, 21))
x, y, z = spec.axes()
phi = np.broadcast_to(y[None, :, None], spec.shape).astype(np.float32).copy()
normal = np.zeros(spec.shape + (3,), dtype=np.float32)
normal[..., 1] = 1.0
fields = {
    "phi": phi,
    "normal": normal,
    "collision_sdf": np.ones(spec.shape, dtype=np.float32),
}
raft = np.zeros(3, dtype=RAFT_DTYPE)
raft["marker_id"] = (101, 202, 303)
raft["physical_radius"] = 0.004
raft["representative_count"] = (2.5, 1.0, 1.0)
raft["phase_volume"] = (
    (4.0 / 3.0)
    * np.pi
    * raft["physical_radius"].astype(np.float64) ** 3
    * raft["representative_count"].astype(np.float64)
)
raft["raft_position"] = (
    (-0.003, 0.0, 0.0),
    (0.003, 0.0, 0.0),
    (0.0, 0.0, 0.0052),
)
raft["raw_anchor_position"] = raft["raft_position"]
raft["support_value"] = 1.0
raft["random_key"] = (1111, 2222, 3333)

model = PlateauCellModel(polygon_sides=24)
topology, metrics = build_plateau_cells(
    raft,
    fields,
    spec,
    active_surface_film_area_m2=1.0,
    event_time=0.1,
    dt=0.1,
    model=model,
)
source_volume = float(raft["phase_volume"].sum(dtype=np.float64))
cell_volume = float(topology["cells"]["gas_volume"].sum(dtype=np.float64))
analytic_volume = float(
    np.sum(
        topology["cells"]["footprint_area"]
        * (
            topology["cells"]["base_height"]
            + (
                topology["cells"]["peak_height"]
                - topology["cells"]["base_height"]
            )
            / 3.0
        ),
        dtype=np.float64,
    )
)

continued, continued_metrics = build_plateau_cells(
    raft,
    fields,
    spec,
    active_surface_film_area_m2=1.0,
    event_time=0.2,
    dt=0.1,
    previous_films=topology["films"],
    previous_nodes=topology["nodes"],
    previous_ruptured_pairs=topology["ruptured_pair_ids"],
    model=model,
)
shared_ids = np.intersect1d(topology["films"]["id"], continued["films"]["id"])
if len(shared_ids):
    first_index = np.searchsorted(topology["films"]["id"], shared_ids)
    second_index = np.searchsorted(continued["films"]["id"], shared_ids)
    ages_advance = np.all(
        continued["films"]["age"][second_index]
        > topology["films"]["age"][first_index]
    )
    films_drain = np.all(
        continued["films"]["thickness"][second_index]
        < topology["films"]["thickness"][first_index]
    )
else:
    ages_advance = films_drain = False

rupture_model = PlateauCellModel(
    polygon_sides=24,
    initial_film_thickness_m=1.0e-6,
    minimum_film_thickness_m=9.0e-7,
    film_drainage_seconds=0.01,
)
rupture_start, _ = build_plateau_cells(
    raft,
    fields,
    spec,
    active_surface_film_area_m2=1.0,
    event_time=0.1,
    dt=0.1,
    model=rupture_model,
)
ruptured, _ = build_plateau_cells(
    raft,
    fields,
    spec,
    active_surface_film_area_m2=1.0,
    event_time=0.2,
    dt=0.1,
    previous_films=rupture_start["films"],
    previous_nodes=rupture_start["nodes"],
    model=rupture_model,
)

criteria = {
    "representative_packets_expand_without_radius_inflation": bool(
        len(topology["cells"]) > len(raft)
        and np.max(topology["cells"]["physical_radius"])
        <= np.max(raft["physical_radius"]) + 1.0e-9
    ),
    "parent_gas_volume_is_exact": abs(source_volume - cell_volume) <= 1.0e-14,
    "analytic_cell_geometry_volume_is_exact": abs(cell_volume - analytic_volume)
    <= 1.0e-12,
    "shared_films_are_unique_unordered_pairs": bool(
        len(np.unique(topology["films"]["id"])) == len(topology["films"])
        and np.all(topology["films"]["cell_a"] < topology["films"]["cell_b"])
    ),
    "film_area_and_liquid_volume_are_physical": bool(
        np.all(topology["films"]["area"] > 0.0)
        and np.allclose(
            topology["films"]["liquid_volume"],
            topology["films"]["area"] * topology["films"]["thickness"],
            rtol=1.0e-12,
        )
    ),
    "film_age_and_drainage_are_persistent": bool(ages_advance and films_drain),
    "plateau_nodes_require_three_incident_films": bool(
        np.all(topology["nodes"]["incident_films"] >= 3)
    ),
    "rupture_creates_coalesced_groups_without_gas_loss": bool(
        np.any(ruptured["events"]["kind"] == 3)
        and len(np.unique(ruptured["cells"]["coalesced_group_id"]))
        < len(ruptured["cells"])
        and np.isclose(
            ruptured["cells"]["gas_volume"].sum(dtype=np.float64),
            source_volume,
            rtol=0.0,
            atol=1.0e-14,
        )
    ),
    "render_geometry_is_finite_and_indexed": bool(
        np.isfinite(topology["vertices"]).all()
        and len(topology["triangles"])
        and topology["triangles"].min() >= 0
        and topology["triangles"].max() < len(topology["vertices"])
    ),
}
report = {
    "schema": 1,
    "suite": "whitewater_v6_plateau_cells",
    "valid": all(criteria.values()),
    "criteria": criteria,
    "metrics": {
        "source_marker_packets": len(raft),
        "expanded_cells": len(topology["cells"]),
        "contact_pairs": metrics["contact_pairs"],
        "shared_films": len(topology["films"]),
        "plateau_nodes": len(topology["nodes"]),
        "plateau_borders": len(topology["borders"]),
        "gas_volume_residual_m3": source_volume - cell_volume,
        "analytic_volume_residual_m3": cell_volume - analytic_volume,
        "persistent_films": len(shared_ids),
        "rupture_events": int(np.count_nonzero(ruptured["events"]["kind"] == 3)),
        "geometry_vertices": len(topology["vertices"]),
        "geometry_triangles": len(topology["triangles"]),
        "continued_shared_films": continued_metrics["shared_films"],
    },
}
print(json.dumps(report, indent=2))
if not report["valid"]:
    raise SystemExit(1)
