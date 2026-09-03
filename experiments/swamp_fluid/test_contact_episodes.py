"""Regression tests for cumulative measured-surface contact episodes."""

from __future__ import annotations

import json

import numpy as np

from whitewater.contact_episodes import MeasuredSurfaceContactTracker


tracker = MeasuredSurfaceContactTracker(
    points=[[1.0, 0.0, 0.0]],
    normals=[[-1.0, 0.0, 0.0]],
    aabb_minimum=[0.8, -0.2, -0.2],
    aabb_maximum=[1.1, 0.2, 0.2],
    initial_particle_count=10,
    maximum_contact_distance=0.10,
    episode_entry_distance=0.12,
    minimum_incoming_normal_speed=0.2,
    minimum_cumulative_outward_change=0.5,
    maximum_episode_gap_steps=2,
)

# ID 10 slows against the wall by 0.2 m/s per sample.  No individual sample
# reaches the 0.5 m/s criterion, but the contact episode does cumulatively.
# ID 11 passes through the measured neighbourhood without any deflection.
states = [
    (0, [0.84, 0.84], [1.0, 1.0]),
    (1, [0.90, 0.90], [0.8, 1.0]),
    (2, [0.93, 0.96], [0.6, 1.0]),
    (3, [0.95, 1.02], [0.4, 1.0]),
]
for step, x_positions, x_velocities in states:
    particle_ids = np.arange(12, dtype=np.int64)
    positions = np.zeros((12, 3), dtype=np.float32)
    velocities = np.zeros((12, 3), dtype=np.float32)
    positions[:10, 0] = -1.0
    positions[10:, 0] = x_positions
    velocities[10:, 0] = x_velocities
    tracker.update(step, particle_ids, positions, velocities)

metrics = tracker.metrics()
criteria = {
    "append_only_emitted_particles_are_tracked": metrics["aabb_unique_particles"] == 2,
    "near_surface_denominator_excludes_broad_aabb_only_entries": (
        metrics["near_surface_unique_particles"] == 2
    ),
    "cumulative_contact_detects_gradual_physx_response": (
        metrics["contact_unique_particles"] == 1 and 10 in tracker.contact_particle_ids
    ),
    "unchanged_flythrough_is_not_contact": 11 not in tracker.contact_particle_ids,
    "contact_step_records_evidence": (
        len(metrics["contact_steps"]) == 1
        and metrics["contact_steps"][0]["maximum_cumulative_outward_change_m_s"]
        >= 0.5
    ),
}
criteria = {name: bool(value) for name, value in criteria.items()}
report = {
    "schema": 1,
    "suite": "measured_surface_contact_episodes",
    "valid": all(criteria.values()),
    "criteria": criteria,
    "metrics": metrics,
}
print(json.dumps(report, indent=2, sort_keys=True))
if not report["valid"]:
    raise SystemExit(1)
