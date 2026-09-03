"""Extract a stable terrain-aware free surface from a closed Splashsurf OBJ."""

import argparse
from collections import deque
from pathlib import Path

import meshio
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surface_obj", type=Path)
    parser.add_argument("heightfield_npz", type=Path)
    parser.add_argument("output_obj", type=Path)
    parser.add_argument("--particles-ply", type=Path, required=True)
    parser.add_argument(
        "--shoreline-particles-ply",
        type=Path,
        help="Fixed reference particles used by reference-ply shoreline mode.",
    )
    parser.add_argument(
        "--shoreline-mode",
        choices=("auto", "terrain", "reference-ply", "dynamic"),
        default="auto",
        help=(
            "Shoreline source. auto uses reference-ply when supplied and otherwise "
            "uses the deterministic terrain/initial-layer mask."
        ),
    )
    parser.add_argument("--water-level", type=float, required=True)
    parser.add_argument("--spacing", type=float, default=0.008)
    parser.add_argument("--top-shell-layers", type=float, default=1.5)
    parser.add_argument("--minimum-layers", type=int, default=8)
    parser.add_argument("--shoreline-erosion-cells", type=int, default=4)
    parser.add_argument("--impact-x", type=float, default=-0.73)
    parser.add_argument("--impact-z", type=float, default=0.78)
    parser.add_argument("--impact-radius", type=float, default=0.40)
    parser.add_argument(
        "--voxel-size",
        type=float,
        help="Marching-cubes cell size in metres (default: 0.375 * spacing).",
    )
    return parser.parse_args()


def four_connected_component(mask, seed_x, seed_z, x_values, z_values):
    """Keep the wet component containing, or nearest to, the pool seed."""
    wet_indices = np.argwhere(mask)
    if len(wet_indices) == 0:
        raise RuntimeError("The shoreline mask contains no wet cells")

    seed_ix = int(np.argmin(np.abs(x_values - seed_x)))
    seed_iz = int(np.argmin(np.abs(z_values - seed_z)))
    if not mask[seed_ix, seed_iz]:
        dx = x_values[wet_indices[:, 0]] - seed_x
        dz = z_values[wet_indices[:, 1]] - seed_z
        nearest = wet_indices[int(np.argmin(dx * dx + dz * dz))]
        seed_ix, seed_iz = map(int, nearest)

    selected = np.zeros_like(mask, dtype=bool)
    selected[seed_ix, seed_iz] = True
    queue = deque([(seed_ix, seed_iz)])
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


def erode_four_connected(mask, iterations):
    result = mask.copy()
    for _ in range(iterations):
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


def particle_column_mask(path, x_values, z_values, minimum_layers):
    particle_mesh = meshio.read(path)
    particle_points = np.asarray(particle_mesh.points, dtype=np.float32)
    dx = float(np.median(np.diff(x_values)))
    dz = float(np.median(np.diff(z_values)))
    particle_ix = np.rint((particle_points[:, 0] - x_values[0]) / dx).astype(
        np.int64
    )
    particle_iz = np.rint((particle_points[:, 2] - z_values[0]) / dz).astype(
        np.int64
    )
    inside = (
        (particle_ix >= 0)
        & (particle_ix < len(x_values))
        & (particle_iz >= 0)
        & (particle_iz < len(z_values))
    )
    counts = np.zeros((len(x_values), len(z_values)), dtype=np.int32)
    np.add.at(counts, (particle_ix[inside], particle_iz[inside]), 1)
    return counts >= minimum_layers


def build_shoreline_mask(args, terrain_y, x_values, z_values):
    mode = args.shoreline_mode
    if mode == "auto":
        mode = "reference-ply" if args.shoreline_particles_ply else "terrain"

    terrain_valid = np.isfinite(terrain_y)
    if mode == "terrain":
        # Mirror the PhysX vertical initialization and the requirement that a
        # rendered column contain minimum_layers particles.
        fluid_rest_offset = 0.5 * args.spacing
        particle_contact_offset = fluid_rest_offset / 0.6
        floor_clearance = particle_contact_offset + 0.001
        top = args.water_level - 0.45 * args.spacing
        maximum_floor_y = (
            top - floor_clearance - (args.minimum_layers - 1) * args.spacing
        )
        mask = terrain_valid & (terrain_y <= maximum_floor_y)
    else:
        if mode == "reference-ply":
            if args.shoreline_particles_ply is None:
                raise ValueError(
                    "--shoreline-mode=reference-ply requires "
                    "--shoreline-particles-ply"
                )
            particle_path = args.shoreline_particles_ply
        elif mode == "dynamic":
            particle_path = args.particles_ply
        else:  # pragma: no cover - argparse enforces the choices
            raise ValueError(f"Unsupported shoreline mode: {mode}")
        mask = particle_column_mask(
            particle_path, x_values, z_values, args.minimum_layers
        ) & terrain_valid
        maximum_floor_y = float("nan")

    mask = four_connected_component(
        mask, args.impact_x, args.impact_z, x_values, z_values
    )
    return erode_four_connected(mask, args.shoreline_erosion_cells), mode, maximum_floor_y


def bilinear_sample(x, z, x_values, z_values, values):
    """Sample a possibly sparse regular heightfield without nearest-cell jumps."""
    x = np.asarray(x)
    z = np.asarray(z)
    inside = (
        (x >= x_values[0])
        & (x <= x_values[-1])
        & (z >= z_values[0])
        & (z <= z_values[-1])
    )
    ix0 = np.clip(np.searchsorted(x_values, x, side="right") - 1, 0, len(x_values) - 2)
    iz0 = np.clip(np.searchsorted(z_values, z, side="right") - 1, 0, len(z_values) - 2)
    ix1 = ix0 + 1
    iz1 = iz0 + 1
    tx = np.clip((x - x_values[ix0]) / (x_values[ix1] - x_values[ix0]), 0.0, 1.0)
    tz = np.clip((z - z_values[iz0]) / (z_values[iz1] - z_values[iz0]), 0.0, 1.0)

    corners = np.stack(
        (
            values[ix0, iz0],
            values[ix1, iz0],
            values[ix0, iz1],
            values[ix1, iz1],
        ),
        axis=1,
    )
    weights = np.stack(
        (
            (1.0 - tx) * (1.0 - tz),
            tx * (1.0 - tz),
            (1.0 - tx) * tz,
            tx * tz,
        ),
        axis=1,
    )
    finite = np.isfinite(corners)
    effective_weights = np.where(finite, weights, 0.0)
    weight_sum = effective_weights.sum(axis=1)
    samples = np.full(len(x), np.inf, dtype=np.float32)
    valid = inside & (weight_sum >= 0.999)
    samples[valid] = (
        np.where(finite, corners, 0.0)[valid] * effective_weights[valid]
    ).sum(axis=1) / weight_sum[valid]
    return samples, valid


def keep_candidates_connected_to_surface(triangles, candidate, surface_seed):
    """Keep candidate splash components that share edges with the free surface."""
    candidate_faces = np.flatnonzero(candidate)
    if len(candidate_faces) == 0:
        return np.zeros(len(triangles), dtype=bool)

    local_triangles = triangles[candidate_faces]
    edges = np.concatenate(
        (
            local_triangles[:, (0, 1)],
            local_triangles[:, (1, 2)],
            local_triangles[:, (2, 0)],
        ),
        axis=0,
    )
    face_ids = np.tile(np.arange(len(candidate_faces), dtype=np.int64), 3)
    edges.sort(axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[order]
    edge_faces = face_ids[order]

    parent = np.arange(len(candidate_faces), dtype=np.int64)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        root_first = find(first)
        root_second = find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    repeated = np.all(edges[1:] == edges[:-1], axis=1)
    for location in np.flatnonzero(repeated):
        union(int(edge_faces[location]), int(edge_faces[location + 1]))

    roots = np.fromiter((find(i) for i in range(len(parent))), dtype=np.int64)
    local_seeds = surface_seed[candidate_faces]
    seeded_roots = np.unique(roots[local_seeds])
    connected = np.zeros(len(triangles), dtype=bool)
    connected[candidate_faces] = np.isin(roots, seeded_roots)
    return connected


def remove_bad_faces(points, triangles, voxel_size):
    unique_indices = (
        (triangles[:, 0] != triangles[:, 1])
        & (triangles[:, 1] != triangles[:, 2])
        & (triangles[:, 2] != triangles[:, 0])
    )
    removed_repeated_vertices = int(np.count_nonzero(~unique_indices))
    triangles = triangles[unique_indices]

    canonical = np.sort(triangles, axis=1)
    _, first_occurrence = np.unique(canonical, axis=0, return_index=True)
    first_occurrence.sort()
    removed_duplicate_faces = len(triangles) - len(first_occurrence)
    triangles = triangles[first_occurrence]

    cross = np.cross(
        points[triangles[:, 1]] - points[triangles[:, 0]],
        points[triangles[:, 2]] - points[triangles[:, 0]],
    )
    twice_area_squared = np.einsum("ij,ij->i", cross, cross)
    twice_area_epsilon = max(1.0e-12, voxel_size * voxel_size * 1.0e-6)
    nondegenerate = twice_area_squared > twice_area_epsilon * twice_area_epsilon
    removed_small_area = int(np.count_nonzero(~nondegenerate))
    return (
        triangles[nondegenerate],
        removed_repeated_vertices + removed_small_area,
        removed_duplicate_faces,
    )


def edge_diagnostics(triangles):
    directed_edges = np.concatenate(
        (
            triangles[:, (0, 1)],
            triangles[:, (1, 2)],
            triangles[:, (2, 0)],
        ),
        axis=0,
    )
    edge_direction = directed_edges[:, 0] < directed_edges[:, 1]
    undirected_edges = np.sort(directed_edges, axis=1)
    order = np.lexsort((undirected_edges[:, 1], undirected_edges[:, 0]))
    undirected_edges = undirected_edges[order]
    edge_direction = edge_direction[order]
    _, starts, counts = np.unique(
        undirected_edges, axis=0, return_index=True, return_counts=True
    )
    paired_starts = starts[counts == 2]
    winding_conflicts = np.count_nonzero(
        edge_direction[paired_starts] == edge_direction[paired_starts + 1]
    )
    return (
        int(np.count_nonzero(counts == 1)),
        int(np.count_nonzero(counts > 2)),
        int(winding_conflicts),
    )


def write_obj(path, points, triangles, normals=None):
    """Write vertex normals with explicit OBJ face references.

    meshio writes ``vn`` records for point data but does not put the matching
    ``v//vn`` indices on faces. Blender therefore ignores those normals. The
    clipped mesh has one normal per vertex, so the OBJ indices can be shared.
    """
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# Splashsurf terrain-clipped free surface\n")
        stream.writelines(
            f"v {point[0]:.9g} {point[1]:.9g} {point[2]:.9g}\n"
            for point in points
        )
        if normals is not None:
            stream.writelines(
                f"vn {normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g}\n"
                for normal in normals
            )
            stream.writelines(
                f"f {face[0] + 1}//{face[0] + 1} "
                f"{face[1] + 1}//{face[1] + 1} "
                f"{face[2] + 1}//{face[2] + 1}\n"
                for face in triangles
            )
        else:
            stream.writelines(
                f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n"
                for face in triangles
            )


def main():
    args = parse_args()
    voxel_size = args.voxel_size or 0.375 * args.spacing
    mesh = meshio.read(args.surface_obj)
    points = np.asarray(mesh.points, dtype=np.float32)
    triangle_cells = [cell.data for cell in mesh.cells if cell.type == "triangle"]
    if not triangle_cells:
        raise RuntimeError(f"No triangles found in {args.surface_obj}")
    triangles = np.concatenate(triangle_cells, axis=0).astype(np.int64, copy=False)
    centroids = points[triangles].mean(axis=1)

    with np.load(args.heightfield_npz) as heightfield:
        x_values = np.asarray(heightfield["x_values"], dtype=np.float32)
        z_values = np.asarray(heightfield["z_values"], dtype=np.float32)
        terrain_y = np.asarray(heightfield["terrain_y"], dtype=np.float32)

    render_wet_mask, shoreline_mode, maximum_floor_y = build_shoreline_mask(
        args, terrain_y, x_values, z_values
    )
    sampled_terrain, terrain_valid = bilinear_sample(
        centroids[:, 0], centroids[:, 2], x_values, z_values, terrain_y
    )

    dx = float(np.median(np.diff(x_values)))
    dz = float(np.median(np.diff(z_values)))
    ix = np.rint((centroids[:, 0] - x_values[0]) / dx).astype(np.int64)
    iz = np.rint((centroids[:, 2] - z_values[0]) / dz).astype(np.int64)
    inside_columns = (
        (ix >= 0)
        & (ix < len(x_values))
        & (iz >= 0)
        & (iz < len(z_values))
    )
    wet_column = np.zeros(len(triangles), dtype=bool)
    wet_column[inside_columns] = render_wet_mask[ix[inside_columns], iz[inside_columns]]
    wet_column &= terrain_valid

    flat_cell = ix * len(z_values) + iz
    top_y = np.full(len(x_values) * len(z_values), -np.inf, dtype=np.float32)
    np.maximum.at(top_y, flat_cell[wet_column], centroids[wet_column, 1])
    local_top = np.full(len(triangles), -np.inf, dtype=np.float32)
    local_top[wet_column] = top_y[flat_cell[wet_column]]
    top_shell = wet_column & (
        centroids[:, 1] >= local_top - args.top_shell_layers * args.spacing
    )

    distance_to_impact = np.linalg.norm(
        centroids[:, (0, 2)] - np.asarray((args.impact_x, args.impact_z)), axis=1
    )
    impact_region = distance_to_impact <= args.impact_radius
    splash_geometry = (
        impact_region
        & terrain_valid
        & (centroids[:, 1] >= sampled_terrain - 0.5 * args.spacing)
        & (centroids[:, 1] >= args.water_level - max(0.04, 5.0 * args.spacing))
    )
    splash_candidates = splash_geometry | (top_shell & impact_region)
    connected_splash = keep_candidates_connected_to_surface(
        triangles, splash_candidates, top_shell
    )
    dynamic_splash = connected_splash & splash_geometry & ~top_shell
    kept = triangles[top_shell | dynamic_splash]
    kept, removed_degenerate, removed_duplicates = remove_bad_faces(
        points, kept, voxel_size
    )
    if len(kept) == 0:
        raise RuntimeError("Clipping removed every triangle")

    used = np.unique(kept.reshape(-1))
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    compact_points = points[used]
    compact_triangles = remap[kept]

    compact_normals = None
    input_normals = mesh.point_data.get("obj:vn")
    if input_normals is not None and len(input_normals) == len(points):
        compact_normals = np.asarray(input_normals)[used]

    args.output_obj.parent.mkdir(parents=True, exist_ok=True)
    write_obj(
        args.output_obj, compact_points, compact_triangles, normals=compact_normals
    )
    boundary_edges, nonmanifold_edges, winding_conflicts = edge_diagnostics(
        compact_triangles
    )
    maximum_floor_text = (
        f"{maximum_floor_y:.9f}" if np.isfinite(maximum_floor_y) else "n/a"
    )
    print(
        f"input_vertices={len(points)} input_faces={len(triangles)} "
        f"output_vertices={len(compact_points)} output_faces={len(compact_triangles)} "
        f"shoreline_mode={shoreline_mode} "
        f"render_wet_cells={int(np.count_nonzero(render_wet_mask))} "
        f"maximum_floor_y={maximum_floor_text} "
        f"top_shell_faces={int(np.count_nonzero(top_shell))} "
        f"dynamic_faces={int(np.count_nonzero(dynamic_splash))} "
        f"removed_degenerate_faces={removed_degenerate} "
        f"removed_duplicate_faces={removed_duplicates} "
        f"boundary_edges={boundary_edges} nonmanifold_edges={nonmanifold_edges} "
        f"winding_conflicts={winding_conflicts} "
        f"normals_preserved={compact_normals is not None}"
    )


if __name__ == "__main__":
    main()
