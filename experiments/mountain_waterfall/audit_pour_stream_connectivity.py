"""Audit whether emitted PhysX particles form a connected source-to-wall jet."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source_sample", type=Path)
parser.add_argument("pour_source", type=Path)
parser.add_argument("wall_target", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--initial-particles", type=int, required=True)
parser.add_argument("--spacing", type=float, required=True)
parser.add_argument("--global-time", type=float, required=True)
parser.add_argument("--physics-fps", type=int, default=240)
args = parser.parse_args()

for path in (args.source_sample, args.pour_source, args.wall_target):
    if not path.is_file():
        raise FileNotFoundError(path)
if args.output.exists():
    raise FileExistsError(f"Refusing to overwrite {args.output}")
if args.initial_particles < 0 or args.spacing <= 0.0 or args.physics_fps <= 0:
    raise ValueError("Initial particle count and spacing are invalid")

shared_module_root = Path(__file__).resolve().parents[1] / "swamp_fluid"
sys.path.insert(0, str(shared_module_root))
from whitewater.pour_source import PourSource


def components_for_radius(points, radius):
    count = len(points)
    parent = np.arange(count, dtype=np.int64)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first, second):
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    radius_squared = radius * radius
    block = 256
    for first_start in range(0, count, block):
        first_stop = min(count, first_start + block)
        for second_start in range(first_start, count, block):
            second_stop = min(count, second_start + block)
            difference = (
                points[first_start:first_stop, None, :]
                - points[None, second_start:second_stop, :]
            )
            pairs = np.argwhere(np.sum(difference * difference, axis=2) <= radius_squared)
            for local_first, local_second in pairs:
                first = first_start + int(local_first)
                second = second_start + int(local_second)
                if first < second:
                    union(first, second)
    labels = np.asarray([find(index) for index in range(count)], dtype=np.int64)
    roots, inverse, sizes = np.unique(labels, return_inverse=True, return_counts=True)
    return inverse, sizes


def adjacency_for_radius(points, radius):
    count = len(points)
    adjacency = np.zeros((count, count), dtype=bool)
    radius_squared = radius * radius
    block = 256
    for first_start in range(0, count, block):
        first_stop = min(count, first_start + block)
        for second_start in range(first_start, count, block):
            second_stop = min(count, second_start + block)
            difference = (
                points[first_start:first_stop, None, :]
                - points[None, second_start:second_stop, :]
            )
            within = np.sum(difference * difference, axis=2) <= radius_squared
            adjacency[first_start:first_stop, second_start:second_stop] = within
            adjacency[second_start:second_stop, first_start:first_stop] = within.T
    np.fill_diagonal(adjacency, False)
    return adjacency


def density_core_report(adjacency, minimum_neighbours, source_mask, wall_mask):
    active = np.ones(len(adjacency), dtype=bool)
    degrees = np.sum(adjacency, axis=1, dtype=np.int64)
    queue = list(np.flatnonzero(degrees < minimum_neighbours))
    queued = degrees < minimum_neighbours
    while queue:
        index = int(queue.pop())
        if not active[index]:
            continue
        active[index] = False
        neighbours = np.flatnonzero(adjacency[index] & active)
        degrees[neighbours] -= 1
        newly_weak = neighbours[(degrees[neighbours] < minimum_neighbours) & ~queued[neighbours]]
        queued[newly_weak] = True
        queue.extend(map(int, newly_weak))

    labels = np.full(len(adjacency), -1, dtype=np.int64)
    component_sizes = []
    for seed in np.flatnonzero(active):
        if labels[seed] >= 0:
            continue
        label = len(component_sizes)
        stack = [int(seed)]
        labels[seed] = label
        size = 0
        while stack:
            index = stack.pop()
            size += 1
            neighbours = np.flatnonzero(adjacency[index] & active & (labels < 0))
            labels[neighbours] = label
            stack.extend(map(int, neighbours))
        component_sizes.append(size)
    source_labels = set(map(int, labels[source_mask & active])) - {-1}
    wall_labels = set(map(int, labels[wall_mask & active])) - {-1}
    spanning_labels = source_labels & wall_labels
    return {
        "minimum_neighbours": minimum_neighbours,
        "core_particles": int(np.count_nonzero(active)),
        "core_fraction": float(np.count_nonzero(active) / len(active)),
        "core_components": len(component_sizes),
        "source_zone_core_particles": int(np.count_nonzero(source_mask & active)),
        "source_zone_core_fraction": float(
            np.count_nonzero(source_mask & active) / np.count_nonzero(source_mask)
        ),
        "wall_zone_core_particles": int(np.count_nonzero(wall_mask & active)),
        "source_to_wall_core_spanning_components": len(spanning_labels),
        "source_to_wall_core_spanning_particles": int(
            sum(component_sizes[label] for label in spanning_labels)
        ),
    }


with np.load(args.source_sample, allow_pickle=False) as cache:
    positions = np.asarray(cache["positions"], dtype=np.float64)
    velocities = np.asarray(cache["velocities"], dtype=np.float64)
    particle_ids = np.asarray(cache["particle_ids"], dtype=np.int64)
emitted_mask = particle_ids >= args.initial_particles
emitted_positions = positions[emitted_mask]
emitted_velocities = velocities[emitted_mask]
emitted_ids = particle_ids[emitted_mask]
if not len(emitted_positions):
    raise RuntimeError("Source sample contains no emitted particles")

source = PourSource.load(args.pour_source)
pose = source.pose_at(args.global_time)
relative = emitted_positions - pose.centre
axial = relative @ pose.normal
radial_u = relative @ pose.tangent_u
radial_v = relative @ pose.tangent_v
source_zone = (
    (axial >= -0.05)
    & (axial <= 0.15)
    & (np.abs(radial_u) <= source.outlet.radii_m[0] + args.spacing)
    & (np.abs(radial_v) <= source.outlet.radii_m[1] + args.spacing)
)
wall_payload = json.loads(args.wall_target.read_text(encoding="utf-8"))
wall_rows = [row for row in wall_payload["rays"] if row.get("accepted") is True]
wall_points = np.asarray(
    [row["location_isaac"] for row in wall_rows], dtype=np.float64
)
wall_offsets = emitted_positions[:, None, :] - wall_points[None, :, :]
nearest_wall_distance = np.sqrt(np.min(np.sum(wall_offsets * wall_offsets, axis=2), axis=1))
# Fluid is kept outside the collider by the PhysX particle contact offset.  The
# jet therefore reaches the wall when it approaches the independently measured
# surface, not when it penetrates the broad target AABB.
wall_zone = nearest_wall_distance <= 0.11

# Preserve the birth-layer identity in the audit.  A connected-component
# failure alone says where the stream is broken, but not whether the break was
# introduced by source cadence or later by a collision.  PourSource allocates
# stable IDs batch-by-batch, so the mapping is exact and does not rely on
# nearest-neighbour trajectory matching.
emitted_id_offsets = emitted_ids - args.initial_particles
birth_steps = source.schedule_for_spacing(
    args.physics_fps, args.spacing, args.global_time
)
scheduled_batch_counts = np.asarray(
    [
        source.batch_particle_count(args.spacing, batch_index)
        for batch_index in range(len(birth_steps))
    ],
    dtype=np.int64,
)
scheduled_batch_ends = np.cumsum(scheduled_batch_counts)
scheduled_batch_starts = np.concatenate(
    (np.asarray([0], dtype=np.int64), scheduled_batch_ends[:-1])
)
batch_indices = np.searchsorted(
    scheduled_batch_ends, emitted_id_offsets, side="right"
)
if len(batch_indices) and int(np.max(batch_indices)) >= len(birth_steps):
    raise RuntimeError("Emitted IDs exceed the PourSource schedule at this sample time")
batch_local_indices = emitted_id_offsets - scheduled_batch_starts[batch_indices]
batch_identity_valid = bool(
    np.all(emitted_id_offsets >= 0)
    and np.array_equal(emitted_ids, np.arange(emitted_ids[0], emitted_ids[-1] + 1))
    and int(np.max(batch_indices)) < len(birth_steps)
    and all(
        np.array_equal(
            np.sort(batch_local_indices[batch_indices == batch_index]),
            np.arange(scheduled_batch_counts[batch_index]),
        )
        for batch_index in np.unique(batch_indices)
    )
)
batch_reports = []
previous_positions = None
for batch_index in np.unique(batch_indices):
    mask = batch_indices == batch_index
    batch_positions = emitted_positions[mask]
    batch_axial = axial[mask]
    distance_to_previous = None
    if previous_positions is not None:
        offsets = batch_positions[:, None, :] - previous_positions[None, :, :]
        distance_to_previous = float(
            np.sqrt(np.min(np.sum(offsets * offsets, axis=2)))
        )
    batch_reports.append(
        {
            "batch_index": int(batch_index),
            "birth_step": int(birth_steps[int(batch_index)]),
            "birth_time_seconds": float(
                birth_steps[int(batch_index)] / args.physics_fps
            ),
            "age_seconds": float(
                args.global_time - birth_steps[int(batch_index)] / args.physics_fps
            ),
            "particles": int(np.count_nonzero(mask)),
            "centroid": np.mean(batch_positions, axis=0).tolist(),
            "axial_minimum_m": float(np.min(batch_axial)),
            "axial_maximum_m": float(np.max(batch_axial)),
            "minimum_wall_distance_m": float(np.min(nearest_wall_distance[mask])),
            "minimum_distance_to_previous_batch_m": distance_to_previous,
        }
    )
    previous_positions = batch_positions

radius_reports = []
for multiplier in (1.25, 1.5, 1.75, 2.0):
    radius = multiplier * args.spacing
    labels, sizes = components_for_radius(emitted_positions, radius)
    source_labels = set(map(int, labels[source_zone]))
    wall_labels = set(map(int, labels[wall_zone]))
    spanning_labels = source_labels & wall_labels
    source_counts = np.bincount(labels[source_zone], minlength=len(sizes))
    largest_source_connected_count = int(np.max(source_counts, initial=0))
    main_source_label = int(np.argmax(source_counts))
    main_source_mask = labels == main_source_label
    main_source_batch_counts = [
        {
            "batch_index": int(batch_index),
            "particles": int(
                np.count_nonzero(main_source_mask & (batch_indices == batch_index))
            ),
        }
        for batch_index in np.unique(batch_indices)
        if np.count_nonzero(main_source_mask & (batch_indices == batch_index))
    ]
    radius_reports.append(
        {
            "radius_multiplier": multiplier,
            "radius_m": radius,
            "component_count": int(len(sizes)),
            "largest_component_particles": int(np.max(sizes)),
            "largest_component_fraction": float(np.max(sizes) / len(emitted_positions)),
            "source_zone_particles": int(np.count_nonzero(source_zone)),
            "near_wall_surface_particles": int(np.count_nonzero(wall_zone)),
            "largest_source_component_source_particles": largest_source_connected_count,
            "largest_source_component_source_fraction": (
                largest_source_connected_count / int(np.count_nonzero(source_zone))
                if np.count_nonzero(source_zone)
                else 0.0
            ),
            "main_source_component_particles": int(np.count_nonzero(main_source_mask)),
            "main_source_component_batch_counts": main_source_batch_counts,
            "main_source_component_minimum_wall_distance_m": float(
                np.min(nearest_wall_distance[main_source_mask])
            ),
            "main_source_component_bounds_minimum": emitted_positions[
                main_source_mask
            ].min(axis=0).tolist(),
            "main_source_component_bounds_maximum": emitted_positions[
                main_source_mask
            ].max(axis=0).tolist(),
            "source_to_wall_spanning_components": len(spanning_labels),
            "source_to_wall_spanning_particles": int(
                sum(int(sizes[label]) for label in spanning_labels)
            ),
        }
    )

density_core_radius_multiplier = 1.5
density_core_adjacency = adjacency_for_radius(
    emitted_positions, density_core_radius_multiplier * args.spacing
)
density_core_reports = [
    density_core_report(
        density_core_adjacency, minimum_neighbours, source_zone, wall_zone
    )
    for minimum_neighbours in (3, 5, 7)
]

criteria = {
    "emitted_ids_are_unique_and_sorted": (
        len(np.unique(emitted_ids)) == len(emitted_ids)
        and bool(np.all(np.diff(emitted_ids) > 0))
    ),
    "emitted_ids_resolve_to_complete_birth_batches": batch_identity_valid,
    "source_zone_is_populated": bool(np.count_nonzero(source_zone)),
    "wall_zone_is_populated": bool(np.count_nonzero(wall_zone)),
    "source_to_wall_particle_path_exists_at_1_5_spacing": (
        radius_reports[1]["source_to_wall_spanning_components"] > 0
    ),
    "at_least_75_percent_of_outlet_particles_share_the_main_1_5_spacing_component": (
        radius_reports[1]["largest_source_component_source_fraction"] >= 0.75
    ),
    "spanning_component_contains_the_resolved_outlet": (
        radius_reports[1]["source_to_wall_spanning_particles"]
        >= radius_reports[1]["source_zone_particles"]
    ),
    "source_to_wall_seven_neighbour_density_core_exists": (
        density_core_reports[2]["source_to_wall_core_spanning_components"] > 0
    ),
    "at_least_75_percent_of_outlet_particles_survive_the_seven_neighbour_core": (
        density_core_reports[2]["source_zone_core_fraction"] >= 0.75
    ),
}
criteria = {name: bool(value) for name, value in criteria.items()}
report = {
    "schema": 1,
    "product": "pour_stream_particle_connectivity_audit",
    "valid": all(criteria.values()),
    "criteria": criteria,
    "source_sample": str(args.source_sample.resolve()),
    "pour_source": str(args.pour_source.resolve()),
    "wall_target": str(args.wall_target.resolve()),
    "global_time_seconds": args.global_time,
    "particle_spacing_m": args.spacing,
    "physics_fps": args.physics_fps,
    "emitted_particles": len(emitted_positions),
    "emitted_bounds_minimum": emitted_positions.min(axis=0).tolist(),
    "emitted_bounds_maximum": emitted_positions.max(axis=0).tolist(),
    "emitted_speed_minimum_m_s": float(np.min(np.linalg.norm(emitted_velocities, axis=1))),
    "emitted_speed_maximum_m_s": float(np.max(np.linalg.norm(emitted_velocities, axis=1))),
    "birth_batch_diagnostics": {
        "minimum_particles_per_batch": int(np.min(scheduled_batch_counts)),
        "maximum_particles_per_batch": int(np.max(scheduled_batch_counts)),
        "resolved_batches": len(batch_reports),
        "maximum_adjacent_batch_distance_m": float(
            max(
                row["minimum_distance_to_previous_batch_m"] or 0.0
                for row in batch_reports
            )
        ),
        "batches": batch_reports,
    },
    "density_core_radius_multiplier": density_core_radius_multiplier,
    "density_core_reports": density_core_reports,
    "radius_reports": radius_reports,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
temporary = args.output.with_name(args.output.name + ".tmp")
with temporary.open("w", encoding="utf-8", newline="\n") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, args.output)
print(json.dumps(report, indent=2))
if not report["valid"]:
    raise SystemExit(1)
