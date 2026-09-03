"""Build deterministic spray, foam, and bubble caches from stable PhysX particles."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCHEMA = 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("particle_directory", type=Path)
    parser.add_argument("heightfield_npz", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--physics-report", type=Path)
    parser.add_argument("--water-level", type=float, required=True)
    parser.add_argument("--spacing", type=float, default=0.008)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--impact-x", type=float, default=-0.73)
    parser.add_argument("--impact-z", type=float, default=0.78)
    parser.add_argument("--sphere-radius", type=float, default=0.08)
    parser.add_argument("--spray-speed", type=float, default=0.25)
    parser.add_argument("--spray-neighbors", type=int, default=6)
    parser.add_argument("--foam-speed", type=float, default=0.08)
    parser.add_argument("--foam-acceleration", type=float, default=2.0)
    parser.add_argument("--foam-radius", type=float, default=0.55)
    parser.add_argument("--foam-emission-fraction", type=float, default=0.02)
    parser.add_argument("--foam-ring-speed", type=float, default=0.32)
    parser.add_argument("--foam-ring-width", type=float, default=0.10)
    parser.add_argument("--foam-cluster-min", type=int, default=2)
    parser.add_argument("--foam-cluster-max", type=int, default=5)
    parser.add_argument("--foam-cluster-radius", type=float, default=0.006)
    parser.add_argument("--bubble-radius", type=float, default=0.42)
    parser.add_argument("--bubble-emission-fraction", type=float, default=0.012)
    parser.add_argument("--bubble-min-size", type=float, default=0.0003)
    parser.add_argument("--bubble-regular-max-size", type=float, default=0.0035)
    parser.add_argument("--bubble-large-max-size", type=float, default=0.006)
    parser.add_argument("--bubble-large-fraction", type=float, default=0.025)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_binary_ply(path):
    with path.open("rb") as stream:
        count = None
        properties = []
        while True:
            line = stream.readline()
            if not line:
                raise RuntimeError(f"Invalid PLY header: {path}")
            words = line.strip().split()
            if words[:2] == [b"element", b"vertex"]:
                count = int(words[-1])
            elif words[:1] == [b"property"]:
                properties.append(words[-1].decode("ascii"))
            elif words == [b"end_header"]:
                break
        points = np.fromfile(stream, dtype="<f4")
    if count is None or properties[-3:] != ["x", "y", "z"] or points.size != count * 3:
        raise RuntimeError(f"Expected packed float32 XYZ PLY: {path}")
    return points.reshape(count, 3)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_digest(paths):
    digest = hashlib.sha256()
    total_bytes = 0
    for path in paths:
        stat = path.stat()
        total_bytes += stat.st_size
        digest.update(
            f"{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8")
        )
    return digest.hexdigest(), total_bytes


def hash01(ids, salt):
    values = ids.astype(np.uint64) + np.uint64(salt)
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    return ((values >> np.uint64(11)).astype(np.float64) / float(1 << 53)).astype(
        np.float32
    )


def neighbor_counts(points, bounds_minimum, bounds_maximum, cell_size):
    grid_shape = (
        np.floor((bounds_maximum - bounds_minimum) / cell_size).astype(np.int64) + 1
    )
    cells = np.floor((points - bounds_minimum) / cell_size).astype(np.int64)
    valid = np.all((cells >= 0) & (cells < grid_shape), axis=1)
    valid_indices = np.flatnonzero(valid)
    valid_cells = cells[valid]
    codes = (
        (valid_cells[:, 0] * grid_shape[1] + valid_cells[:, 1]) * grid_shape[2]
        + valid_cells[:, 2]
    )
    unique_codes, occupancy = np.unique(codes, return_counts=True)
    counts = np.zeros(len(points), dtype=np.int16)
    local_counts = np.zeros(len(valid_indices), dtype=np.int16)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                shifted = valid_cells + np.asarray((dx, dy, dz), dtype=np.int64)
                shifted_valid = np.all(
                    (shifted >= 0) & (shifted < grid_shape), axis=1
                )
                shifted_codes = (
                    (shifted[:, 0] * grid_shape[1] + shifted[:, 1])
                    * grid_shape[2]
                    + shifted[:, 2]
                )
                locations = np.searchsorted(unique_codes, shifted_codes)
                found = shifted_valid & (locations < len(unique_codes))
                found_indices = np.flatnonzero(found)
                found[found_indices] &= (
                    unique_codes[locations[found_indices]]
                    == shifted_codes[found_indices]
                )
                local_counts[found] += occupancy[locations[found]].astype(np.int16)
    counts[valid_indices] = local_counts
    return counts, valid


def local_bulk_top(points, neighbors, x_values, z_values, water_level, spacing):
    dx = float(np.median(np.diff(x_values)))
    dz = float(np.median(np.diff(z_values)))
    ix = np.rint((points[:, 0] - x_values[0]) / dx).astype(np.int64)
    iz = np.rint((points[:, 2] - z_values[0]) / dz).astype(np.int64)
    inside = (
        (ix >= 0)
        & (ix < len(x_values))
        & (iz >= 0)
        & (iz < len(z_values))
    )
    # Exclude isolated droplets before constructing the carrier free-surface
    # envelope, otherwise every spray particle defines its own local top.
    bulk = inside & (neighbors >= 8) & (points[:, 1] <= water_level + 0.12)
    top = np.full((len(x_values), len(z_values)), -np.inf, dtype=np.float32)
    np.maximum.at(top, (ix[bulk], iz[bulk]), points[bulk, 1])
    expanded = top.copy()
    for ox in (-1, 0, 1):
        for oz in (-1, 0, 1):
            source_x = slice(max(0, -ox), min(len(x_values), len(x_values) - ox))
            source_z = slice(max(0, -oz), min(len(z_values), len(z_values) - oz))
            target_x = slice(max(0, ox), min(len(x_values), len(x_values) + ox))
            target_z = slice(max(0, oz), min(len(z_values), len(z_values) + oz))
            expanded[target_x, target_z] = np.maximum(
                expanded[target_x, target_z], top[source_x, source_z]
            )
    fallback = water_level - 0.45 * spacing
    expanded[~np.isfinite(expanded)] = fallback
    particle_top = np.full(len(points), fallback, dtype=np.float32)
    particle_top[inside] = expanded[ix[inside], iz[inside]]
    return particle_top, expanded, ix, iz, inside


def write_npz(path, **arrays):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def write_manifest(path, configuration, state):
    payload = {
        "schema": SCHEMA,
        "configuration": configuration,
        "state": state,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def selected_payload(
    ids,
    positions,
    velocities,
    radii,
    birth,
    death,
    frame,
    opacity,
    shape=None,
):
    age = (frame - birth[ids]).astype(np.float32)
    lifetime = np.maximum(1, death[ids] - birth[ids]).astype(np.float32)
    payload = {
        "id": ids.astype(np.int32),
        "position": np.asarray(positions, dtype=np.float32),
        "velocity": np.asarray(velocities, dtype=np.float32),
        "radius": np.asarray(radii, dtype=np.float32),
        "age": age,
        "lifetime": lifetime,
        "opacity": np.asarray(opacity, dtype=np.float32),
    }
    if shape is not None:
        payload["shape"] = np.asarray(shape, dtype=np.float32)
    return payload


def expanded_payload(
    ids, positions, velocities, radii, age, lifetime, opacity, shape=None
):
    payload = {
        "id": np.asarray(ids, dtype=np.int32),
        "position": np.asarray(positions, dtype=np.float32),
        "velocity": np.asarray(velocities, dtype=np.float32),
        "radius": np.asarray(radii, dtype=np.float32),
        "age": np.asarray(age, dtype=np.float32),
        "lifetime": np.asarray(lifetime, dtype=np.float32),
        "opacity": np.asarray(opacity, dtype=np.float32),
    }
    if shape is not None:
        payload["shape"] = np.asarray(shape, dtype=np.float32)
    return payload


def main():
    args = parse_args()
    if args.spacing <= 0 or args.fps <= 0:
        raise ValueError("Spacing and FPS must be positive")
    if not 0 <= args.foam_emission_fraction <= 1:
        raise ValueError("--foam-emission-fraction must be within 0..1")
    if not 0 <= args.bubble_emission_fraction <= 1:
        raise ValueError("--bubble-emission-fraction must be within 0..1")
    if not 0 <= args.bubble_large_fraction <= 1:
        raise ValueError("--bubble-large-fraction must be within 0..1")
    if not 1 <= args.foam_cluster_min <= args.foam_cluster_max:
        raise ValueError("Foam cluster sizes must satisfy 1 <= minimum <= maximum")
    if args.foam_cluster_max > 32:
        raise ValueError("Foam cluster maximum is unreasonably large")
    if not (
        0 < args.bubble_min_size
        <= args.bubble_regular_max_size
        <= args.bubble_large_max_size
    ):
        raise ValueError("Bubble sizes must be positive and monotonically increasing")

    paths = sorted(args.particle_directory.glob("particles_*.ply"))
    if args.frames is not None:
        paths = paths[: args.frames]
    if len(paths) < 3:
        raise RuntimeError("At least three consecutive particle frames are required")

    with np.load(args.heightfield_npz) as heightfield:
        x_values = np.asarray(heightfield["x_values"], dtype=np.float32)
        z_values = np.asarray(heightfield["z_values"], dtype=np.float32)
        terrain_y = np.asarray(heightfield["terrain_y"], dtype=np.float32)
    finite_terrain = terrain_y[np.isfinite(terrain_y)]
    bounds_minimum = np.asarray(
        (x_values[0], float(finite_terrain.min() - 0.05), z_values[0]),
        dtype=np.float32,
    )
    bounds_maximum = np.asarray(
        (x_values[-1], args.water_level + 0.8, z_values[-1]),
        dtype=np.float32,
    )

    report_metrics = {}
    sphere_radius = args.sphere_radius
    if args.physics_report is not None:
        report = json.loads(args.physics_report.read_text(encoding="utf-8"))
        report_metrics = {
            int(row["output_frame"]): row for row in report.get("metrics", [])
        }
        sphere_radius = float(report.get("impactor", {}).get("radius", sphere_radius))

    input_digest, total_bytes = metadata_digest(paths)
    configuration = {
        "schema": SCHEMA,
        "particle_directory": str(args.particle_directory.resolve()),
        "particle_count": len(paths),
        "particle_metadata_sha256": input_digest,
        "particle_total_bytes": total_bytes,
        "heightfield": str(args.heightfield_npz.resolve()),
        "heightfield_sha256": sha256_file(args.heightfield_npz),
        "physics_report": (
            str(args.physics_report.resolve()) if args.physics_report else None
        ),
        "physics_report_sha256": (
            sha256_file(args.physics_report) if args.physics_report else None
        ),
        "script_sha256": sha256_file(Path(__file__)),
        "water_level": args.water_level,
        "spacing": args.spacing,
        "fps": args.fps,
        "impact": [args.impact_x, args.impact_z],
        "sphere_radius": sphere_radius,
        "spray": {
            "speed": args.spray_speed,
            "maximum_neighbors": args.spray_neighbors,
        },
        "foam": {
            "speed": args.foam_speed,
            "acceleration": args.foam_acceleration,
            "impact_radius": args.foam_radius,
            "emission_fraction": args.foam_emission_fraction,
            "ring_speed": args.foam_ring_speed,
            "ring_width": args.foam_ring_width,
            "cluster_size": [args.foam_cluster_min, args.foam_cluster_max],
            "cluster_radius": args.foam_cluster_radius,
        },
        "bubbles": {
            "impact_radius": args.bubble_radius,
            "emission_fraction": args.bubble_emission_fraction,
            "size_range": [args.bubble_min_size, args.bubble_large_max_size],
            "regular_max_size": args.bubble_regular_max_size,
            "large_fraction": args.bubble_large_fraction,
            "shape_model": "volume-preserving-oblate-oscillation",
        },
    }

    args.output_directory.mkdir(parents=True, exist_ok=True)
    directories = {
        name: args.output_directory / name for name in ("spray", "foam", "bubbles")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_directory / "secondary_manifest.json"
    existing = list(args.output_directory.glob("*/*.npz"))
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("configuration") != configuration and not args.force:
            raise RuntimeError(
                f"Secondary cache configuration changed: {manifest_path}. "
                "Use a new output directory or pass --force."
            )
    elif existing and not args.force:
        raise RuntimeError(
            f"{args.output_directory} contains unmanifested secondary caches"
        )
    write_manifest(
        manifest_path,
        configuration,
        {"complete": False, "completed_frames": 0, "total_frames": len(paths)},
    )

    current = read_binary_ply(paths[0])
    particle_count = len(current)
    ids = np.arange(particle_count, dtype=np.int64)
    foam_hash = hash01(ids, 0xF0A4)
    bubble_hash = hash01(ids, 0xBABB1E)
    spray_radius_all = 0.001 + 0.0025 * hash01(ids, 0x5A9A)
    foam_radius_all = 0.0007 + 0.0016 * hash01(ids, 0xF04D)
    bubble_size_hash = hash01(ids, 0xB0BB1E)
    bubble_radius_all = args.bubble_min_size + (
        args.bubble_regular_max_size - args.bubble_min_size
    ) * bubble_size_hash**2
    large_bubble_hash = hash01(ids, 0x1A49E)
    large_bubble_mask = large_bubble_hash <= args.bubble_large_fraction
    bubble_radius_all[large_bubble_mask] = args.bubble_regular_max_size + (
        args.bubble_large_max_size - args.bubble_regular_max_size
    ) * hash01(ids[large_bubble_mask], 0x1A29E)
    normalized_bubble_radius = np.clip(
        (bubble_radius_all - args.bubble_min_size)
        / max(1e-9, args.bubble_large_max_size - args.bubble_min_size),
        0.0,
        1.0,
    )
    bubble_rise_all = 0.03 + 0.19 * np.sqrt(normalized_bubble_radius)
    foam_lifetime_all = 18 + (58 * hash01(ids, 0x1AFE)).astype(np.int32)
    bubble_lifetime_all = 15 + (55 * hash01(ids, 0xB17E)).astype(np.int32)

    spray_active = np.zeros(particle_count, dtype=bool)
    spray_hits = np.zeros(particle_count, dtype=np.uint8)
    spray_misses = np.zeros(particle_count, dtype=np.uint8)
    spray_birth = np.full(particle_count, -1, dtype=np.int32)
    spray_death = np.full(particle_count, len(paths) + 1, dtype=np.int32)
    foam_birth = np.full(particle_count, -1, dtype=np.int32)
    foam_death = np.full(particle_count, -1, dtype=np.int32)
    bubble_birth = np.full(particle_count, -1, dtype=np.int32)
    bubble_death = np.full(particle_count, -1, dtype=np.int32)
    impact_started = False
    impact_frame = None

    previous = current
    for frame, path in enumerate(paths):
        following = (
            read_binary_ply(paths[frame + 1]) if frame + 1 < len(paths) else current
        )
        if len(following) != particle_count:
            raise RuntimeError("Particle identity/count changed across the sequence")
        velocity = (following - previous) * (0.5 * args.fps)
        acceleration = (following - 2.0 * current + previous) * (args.fps**2)
        speed = np.linalg.norm(velocity, axis=1)
        acceleration_magnitude = np.linalg.norm(acceleration, axis=1)
        neighbors, domain_valid = neighbor_counts(
            current, bounds_minimum, bounds_maximum, args.spacing
        )
        particle_top, _top_grid, _ix, _iz, terrain_inside = local_bulk_top(
            current,
            neighbors,
            x_values,
            z_values,
            args.water_level,
            args.spacing,
        )

        metric = report_metrics.get(frame)
        if metric is not None:
            sphere_center = np.asarray(metric["sphere_center"], dtype=np.float32)
            impact_xz = sphere_center[[0, 2]]
            contact_now = (
                float(sphere_center[1]) - sphere_radius
                <= args.water_level + args.spacing
            )
            impact_started |= contact_now
        else:
            sphere_center = None
            impact_xz = np.asarray((args.impact_x, args.impact_z), dtype=np.float32)
            impact_started |= frame >= 1
        if impact_started and impact_frame is None:
            impact_frame = frame
        impact_distance = np.linalg.norm(
            current[:, (0, 2)] - impact_xz[None, :], axis=1
        )

        spray_candidate = (
            impact_started
            & domain_valid
            & terrain_inside
            & (current[:, 1] > particle_top + 0.75 * args.spacing)
            & (neighbors <= args.spray_neighbors)
            & (speed >= args.spray_speed)
        )
        spray_hits[spray_candidate] = np.minimum(
            3, spray_hits[spray_candidate] + 1
        )
        spray_hits[~spray_candidate] = 0
        spray_misses[~spray_candidate] = np.minimum(
            3, spray_misses[~spray_candidate] + 1
        )
        spray_misses[spray_candidate] = 0
        spray_birth_mask = ~spray_active & (spray_hits >= 2)
        spray_active[spray_birth_mask] = True
        spray_birth[spray_birth_mask] = frame
        spray_death[spray_birth_mask] = len(paths) + 1
        spray_die = spray_active & (
            (spray_misses >= 2)
            | (current[:, 1] <= particle_top + 0.25 * args.spacing)
            | ~domain_valid
        )
        spray_active[spray_die] = False
        spray_death[spray_die] = frame

        surface_band = (
            domain_valid
            & terrain_inside
            & (neighbors >= 8)
            & (current[:, 1] >= particle_top - 1.5 * args.spacing)
            & (current[:, 1] <= particle_top + 0.75 * args.spacing)
        )
        energetic_surface = (
            (speed >= args.foam_speed)
            | (acceleration_magnitude >= args.foam_acceleration)
        )
        impact_age_seconds = (
            max(0, frame - impact_frame) / args.fps if impact_frame is not None else 0.0
        )
        ring_center = sphere_radius + args.foam_ring_speed * impact_age_seconds
        ring_weight = np.exp(
            -0.5
            * ((impact_distance - ring_center) / max(1e-6, args.foam_ring_width))
            ** 2
        )
        foam_probability = np.clip(
            args.foam_emission_fraction * (0.12 + 2.8 * ring_weight), 0.0, 1.0
        )
        foam_birth_mask = (
            impact_started
            & surface_band
            & energetic_surface
            & (impact_distance <= args.foam_radius)
            & (foam_hash <= foam_probability)
            & (foam_death <= frame)
        )
        foam_birth[foam_birth_mask] = frame
        foam_death[foam_birth_mask] = frame + foam_lifetime_all[foam_birth_mask]
        foam_active = (foam_birth >= 0) & (foam_death > frame) & terrain_inside

        depth = particle_top - current[:, 1]
        entraining = (velocity[:, 1] < -0.03) | (
            acceleration_magnitude >= 1.5 * args.foam_acceleration
        )
        bubble_birth_mask = (
            impact_started
            & domain_valid
            & terrain_inside
            & (neighbors >= 8)
            & (depth >= 0.010)
            & (depth <= 0.080)
            & entraining
            & (impact_distance <= args.bubble_radius)
            & (bubble_hash <= args.bubble_emission_fraction)
            & (bubble_death <= frame)
        )
        bubble_birth[bubble_birth_mask] = frame
        bubble_death[bubble_birth_mask] = (
            frame + bubble_lifetime_all[bubble_birth_mask]
        )
        bubble_active = (
            (bubble_birth >= 0) & (bubble_death > frame) & domain_valid & terrain_inside
        )
        bubble_age_seconds = np.maximum(0, frame - bubble_birth) / args.fps
        bubble_positions = current.copy()
        bubble_positions[:, 1] += bubble_rise_all * bubble_age_seconds
        if sphere_center is not None:
            bubble_indices = np.flatnonzero(bubble_active)
            sphere_delta = bubble_positions[bubble_indices] - sphere_center[None, :]
            sphere_distance = np.linalg.norm(sphere_delta, axis=1)
            bubble_shape_margin = 1.0 + np.clip(
                (bubble_radius_all[bubble_indices] - 0.001) / 0.005,
                0.0,
                0.38,
            )
            minimum_distance = sphere_radius + (
                bubble_radius_all[bubble_indices] * bubble_shape_margin
            ) + 0.00075
            overlapping = sphere_distance < minimum_distance
            overlapping_indices = np.flatnonzero(overlapping)
            if len(overlapping_indices):
                safe_direction = sphere_delta[overlapping_indices] / np.maximum(
                    sphere_distance[overlapping_indices, None], 1e-8
                )
                bubble_positions[bubble_indices[overlapping_indices]] = (
                    sphere_center[None, :]
                    + safe_direction * minimum_distance[overlapping_indices, None]
                )
        reached_surface = bubble_active & (
            bubble_positions[:, 1] >= particle_top - bubble_radius_all
        )
        bubble_death[reached_surface] = frame
        bubble_active[reached_surface] = False

        spray_ids = np.flatnonzero(spray_active)
        foam_ids = np.flatnonzero(foam_active)
        bubble_ids = np.flatnonzero(bubble_active)
        foam_positions = current[foam_ids].copy()
        foam_positions[:, 1] = particle_top[foam_ids] + 0.0008
        foam_progress = np.clip(
            (frame - foam_birth[foam_ids])
            / np.maximum(1, foam_death[foam_ids] - foam_birth[foam_ids]),
            0.0,
            1.0,
        )
        foam_opacity = np.sin(np.pi * foam_progress).astype(np.float32)
        bubble_progress = np.clip(
            (frame - bubble_birth[bubble_ids])
            / np.maximum(1, bubble_death[bubble_ids] - bubble_birth[bubble_ids]),
            0.0,
            1.0,
        )
        bubble_opacity = np.minimum(1.0, 4.0 * (1.0 - bubble_progress)).astype(
            np.float32
        )
        spray_opacity = np.ones(len(spray_ids), dtype=np.float32)

        # Expand each stable carrier into a small, stable surface cluster.  The
        # offsets remain tied to the carrier ID, while the expanding impact-ring
        # score controls only parent birth and avoids frame-to-frame resampling.
        foam_cluster_count_all = args.foam_cluster_min + np.floor(
            (args.foam_cluster_max - args.foam_cluster_min + 1)
            * hash01(ids, 0xC1057E)
        ).astype(np.int32)
        cluster_parent_indices = []
        cluster_child_indices = []
        for child in range(args.foam_cluster_max):
            enabled = child < foam_cluster_count_all[foam_ids]
            cluster_parent_indices.append(np.flatnonzero(enabled))
            cluster_child_indices.append(
                np.full(np.count_nonzero(enabled), child, dtype=np.int32)
            )
        cluster_parent_indices = np.concatenate(cluster_parent_indices)
        cluster_child_indices = np.concatenate(cluster_child_indices)
        foam_parent_ids = foam_ids[cluster_parent_indices]
        foam_cluster_positions = foam_positions[cluster_parent_indices].copy()
        noncentral = cluster_child_indices > 0
        cluster_key = (
            foam_parent_ids.astype(np.int64) * args.foam_cluster_max
            + cluster_child_indices.astype(np.int64)
        )
        cluster_angle = 2.0 * np.pi * hash01(cluster_key, 0xA091E)
        cluster_distance = args.foam_cluster_radius * np.sqrt(
            hash01(cluster_key, 0xD157)
        )
        foam_cluster_positions[noncentral, 0] += (
            cluster_distance[noncentral] * np.cos(cluster_angle[noncentral])
        )
        foam_cluster_positions[noncentral, 2] += (
            cluster_distance[noncentral] * np.sin(cluster_angle[noncentral])
        )
        grid_dx = float(np.median(np.diff(x_values)))
        grid_dz = float(np.median(np.diff(z_values)))
        cluster_ix = np.rint(
            (foam_cluster_positions[:, 0] - x_values[0]) / grid_dx
        ).astype(np.int64)
        cluster_iz = np.rint(
            (foam_cluster_positions[:, 2] - z_values[0]) / grid_dz
        ).astype(np.int64)
        cluster_inside = (
            (cluster_ix >= 0)
            & (cluster_ix < len(x_values))
            & (cluster_iz >= 0)
            & (cluster_iz < len(z_values))
        )
        valid_cluster_indices = np.flatnonzero(cluster_inside)
        foam_cluster_positions[valid_cluster_indices, 1] = (
            _top_grid[
                cluster_ix[valid_cluster_indices], cluster_iz[valid_cluster_indices]
            ]
            + 0.0008
        )
        foam_cluster_ids = cluster_key[cluster_inside]
        foam_parent_ids = foam_parent_ids[cluster_inside]
        cluster_parent_indices = cluster_parent_indices[cluster_inside]
        foam_cluster_positions = foam_cluster_positions[cluster_inside]
        foam_cluster_velocity = velocity[foam_parent_ids]
        foam_cluster_radius = foam_radius_all[foam_parent_ids] * (
            0.58 + 0.52 * hash01(foam_cluster_ids, 0x51CE)
        )
        foam_cluster_age = (frame - foam_birth[foam_parent_ids]).astype(np.float32)
        foam_cluster_lifetime = np.maximum(
            1, foam_death[foam_parent_ids] - foam_birth[foam_parent_ids]
        ).astype(np.float32)
        foam_cluster_opacity = foam_opacity[cluster_parent_indices]
        foam_cluster_order = np.argsort(foam_cluster_ids)
        foam_cluster_ids = foam_cluster_ids[foam_cluster_order]
        foam_cluster_positions = foam_cluster_positions[foam_cluster_order]
        foam_cluster_velocity = foam_cluster_velocity[foam_cluster_order]
        foam_cluster_radius = foam_cluster_radius[foam_cluster_order]
        foam_cluster_age = foam_cluster_age[foam_cluster_order]
        foam_cluster_lifetime = foam_cluster_lifetime[foam_cluster_order]
        foam_cluster_opacity = foam_cluster_opacity[foam_cluster_order]

        bubble_deformation = np.clip(
            (bubble_radius_all[bubble_ids] - 0.001) / 0.005, 0.0, 0.38
        )
        bubble_phase = 2.0 * np.pi * (
            hash01(bubble_ids, 0x0B1A7E)
            + bubble_age_seconds[bubble_ids]
            * (1.6 + 1.8 * hash01(bubble_ids, 0xF2E9))
        )
        vertical_scale = 1.0 - bubble_deformation * (
            0.72 + 0.28 * np.sin(bubble_phase)
        )
        horizontal_scale = 1.0 / np.sqrt(vertical_scale)
        bubble_shape = np.column_stack(
            (horizontal_scale, vertical_scale, horizontal_scale)
        ).astype(np.float32)

        frame_name = f"frame_{frame:04d}.npz"
        write_npz(
            directories["spray"] / frame_name,
            **selected_payload(
                spray_ids,
                current[spray_ids],
                velocity[spray_ids],
                spray_radius_all[spray_ids],
                spray_birth,
                spray_death,
                frame,
                spray_opacity,
            ),
        )
        write_npz(
            directories["foam"] / frame_name,
            **expanded_payload(
                foam_cluster_ids,
                foam_cluster_positions,
                foam_cluster_velocity,
                foam_cluster_radius,
                foam_cluster_age,
                foam_cluster_lifetime,
                foam_cluster_opacity,
            ),
        )
        write_npz(
            directories["bubbles"] / frame_name,
            **selected_payload(
                bubble_ids,
                bubble_positions[bubble_ids],
                velocity[bubble_ids]
                + np.column_stack(
                    (
                        np.zeros(len(bubble_ids), dtype=np.float32),
                        bubble_rise_all[bubble_ids],
                        np.zeros(len(bubble_ids), dtype=np.float32),
                    )
                ),
                bubble_radius_all[bubble_ids],
                bubble_birth,
                bubble_death,
                frame,
                bubble_opacity,
                shape=bubble_shape,
            ),
        )
        print(
            f"[secondary] frame={frame:04d} spray={len(spray_ids)} "
            f"foam_seeds={len(foam_ids)} foam={len(foam_cluster_ids)} "
            f"bubbles={len(bubble_ids)} "
            f"max_speed={float(speed.max()):.3f}",
            flush=True,
        )
        previous, current = current, following

    write_manifest(
        manifest_path,
        configuration,
        {
            "complete": True,
            "completed_frames": len(paths),
            "total_frames": len(paths),
        },
    )
    print(f"SECONDARY_PARTICLES={args.output_directory} frames={len(paths)}")


if __name__ == "__main__":
    main()
