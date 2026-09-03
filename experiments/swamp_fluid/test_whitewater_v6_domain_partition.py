"""Synthetic gates for sparse multi-water domains and escaped-particle rejection."""

from __future__ import annotations

import json

import numpy as np

from whitewater.domain_partition import (
    aligned_domain_bounds,
    body_frame_core,
    reference_water_bodies,
    sparse_particle_components,
)


axis = np.arange(5, dtype=np.float64) * 0.01
local = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape((-1, 3))
body_a = local + np.asarray((0.0, 0.0, 0.0))
body_b = local + np.asarray((1.0, 0.0, 0.0))
reference = np.concatenate((body_a, body_b), axis=0)

components = sparse_particle_components(reference, cell_size=0.02, connectivity=26)
bodies, reference_metadata = reference_water_bodies(
    reference,
    cell_size=0.02,
    minimum_body_particles=50,
    minimum_body_fraction=0.1,
)

moved = reference.copy()
moved[0] = (100.0, 100.0, 100.0)
frame_a = body_frame_core(
    moved,
    bodies[0]["particle_indices"],
    cell_size=0.02,
    minimum_fragment_particles=4,
    minimum_fragment_fraction=0.05,
)
domain = aligned_domain_bounds(
    frame_a["bounds_minimum"],
    frame_a["bounds_maximum"],
    spacing=0.01,
    padding=(0.02, 0.02, 0.02),
)

metrics = {
    "reference_component_count": int(len(components["component_particle_counts"])),
    "reference_body_count": len(bodies),
    "reference_assigned_particles": reference_metadata["assigned_particles"],
    "reference_unassigned_particles": reference_metadata["unassigned_particles"],
    "body_particle_counts": [body["particle_count"] for body in bodies],
    "detached_particles": frame_a["detached_particle_count"],
    "core_particles": frame_a["core_particle_count"],
    "core_maximum": frame_a["bounds_maximum"],
    "domain_maximum": domain["maximum"],
    "domain_cell_count": domain["cell_count"],
}
criteria = {
    "two_disconnected_water_bodies_are_identified": (
        metrics["reference_component_count"] == 2
        and metrics["reference_body_count"] == 2
        and metrics["body_particle_counts"] == [125, 125]
    ),
    "stable_membership_is_complete": (
        metrics["reference_assigned_particles"] == 250
        and metrics["reference_unassigned_particles"] == 0
    ),
    "single_escaped_particle_is_detached": (
        metrics["detached_particles"] == 1 and metrics["core_particles"] == 124
    ),
    "escaped_particle_cannot_expand_domain": (
        max(metrics["core_maximum"]) < 1.0
        and max(metrics["domain_maximum"]) < 1.0
    ),
    "domain_is_finite_and_small": metrics["domain_cell_count"] < 10_000,
}
report = {
    "schema": 1,
    "suite": "whitewater_v6_domain_partition",
    "valid": all(criteria.values()),
    "criteria": criteria,
    "metrics": metrics,
}
print(json.dumps(report, indent=2, sort_keys=True))
if not report["valid"]:
    raise SystemExit(1)
