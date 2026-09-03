"""Build a watertight dynamic water-medium proxy below Splashsurf free surfaces."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from clip_splashsurf_free_surface import (
    bilinear_sample,
    erode_four_connected,
    four_connected_component,
)


SCHEMA = 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surface_directory", type=Path)
    parser.add_argument("heightfield_npz", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--water-level", type=float, required=True)
    parser.add_argument("--spacing", type=float, default=0.008)
    parser.add_argument("--minimum-layers", type=int, default=8)
    parser.add_argument("--shoreline-erosion-cells", type=int, default=4)
    parser.add_argument("--impact-x", type=float, default=-0.73)
    parser.add_argument("--impact-z", type=float, default=0.78)
    parser.add_argument(
        "--surface-gap",
        type=float,
        default=0.0015,
        help="Place the transparent volume boundary below the rendered surface.",
    )
    parser.add_argument(
        "--bottom-overlap",
        type=float,
        default=0.010,
        help="Extend the medium below terrain to prevent optical cracks.",
    )
    parser.add_argument(
        "--maximum-top-offset",
        type=float,
        default=0.040,
        help="Exclude overturned splash/spray from the heightfield medium proxy.",
    )
    parser.add_argument("--frame-list", nargs="+", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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


def static_shoreline_mask(args, terrain_y, x_values, z_values):
    fluid_rest_offset = 0.5 * args.spacing
    particle_contact_offset = fluid_rest_offset / 0.6
    floor_clearance = particle_contact_offset + 0.001
    top = args.water_level - 0.45 * args.spacing
    maximum_floor_y = (
        top - floor_clearance - (args.minimum_layers - 1) * args.spacing
    )
    mask = np.isfinite(terrain_y) & (terrain_y <= maximum_floor_y)
    mask = four_connected_component(
        mask, args.impact_x, args.impact_z, x_values, z_values
    )
    return erode_four_connected(mask, args.shoreline_erosion_cells)


def read_points(path):
    # Clipped Splashsurf OBJ files contain all `v` records first, followed by
    # normals and faces.  The volume heightfield only needs positions; parsing
    # 600k+ faces with a general OBJ reader added several seconds per frame.
    vertex_rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("v "):
                vertex_rows.append(line[2:])
            elif vertex_rows:
                break
    values = np.fromstring("".join(vertex_rows), sep=" ", dtype=np.float32)
    if values.size == 0 or values.size % 3:
        raise RuntimeError(f"Invalid or empty OBJ vertex block: {path}")
    return values.reshape(-1, 3)


def build_top_field(
    points, wet_mask, x_values, z_values, water_level, spacing, maximum_top_offset
):
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
    selected = inside & (points[:, 1] <= water_level + maximum_top_offset)
    selected_indices = np.flatnonzero(selected)
    selected[selected_indices] &= wet_mask[ix[selected_indices], iz[selected_indices]]
    top = np.full(wet_mask.shape, -np.inf, dtype=np.float32)
    np.maximum.at(top, (ix[selected], iz[selected]), points[selected, 1])
    fallback = water_level - 0.45 * spacing
    missing = wet_mask & ~np.isfinite(top)
    top[missing] = fallback
    return top


def build_static_topology(wet_mask):
    wet_cells = np.argwhere(wet_mask)
    corner_keys = set()
    for ix, iz in wet_cells:
        corner_keys.update(
            (
                (int(ix), int(iz)),
                (int(ix + 1), int(iz)),
                (int(ix + 1), int(iz + 1)),
                (int(ix), int(iz + 1)),
            )
        )
    corners = sorted(corner_keys)
    corner_map = {key: index for index, key in enumerate(corners)}
    top_triangles = []
    for ix, iz in wet_cells:
        c00 = corner_map[(int(ix), int(iz))]
        c10 = corner_map[(int(ix + 1), int(iz))]
        c11 = corner_map[(int(ix + 1), int(iz + 1))]
        c01 = corner_map[(int(ix), int(iz + 1))]
        # Upward winding in Isaac's Y-up coordinate system.
        top_triangles.extend(((c00, c01, c11), (c00, c11, c10)))
    top_triangles = np.asarray(top_triangles, dtype=np.int64)

    directed_edges = np.concatenate(
        (
            top_triangles[:, (0, 1)],
            top_triangles[:, (1, 2)],
            top_triangles[:, (2, 0)],
        ),
        axis=0,
    )
    undirected = np.sort(directed_edges, axis=1)
    order = np.lexsort((undirected[:, 1], undirected[:, 0]))
    sorted_edges = undirected[order]
    sorted_directed = directed_edges[order]
    _, starts, counts = np.unique(
        sorted_edges, axis=0, return_index=True, return_counts=True
    )
    if np.any(counts > 2):
        raise RuntimeError("Static shoreline creates non-manifold top edges")
    boundary_edges = sorted_directed[starts[counts == 1]]

    vertex_count = len(corners)
    bottom_triangles = top_triangles[:, ::-1] + vertex_count
    side_triangles = []
    for first, second in boundary_edges:
        bottom_first = int(first + vertex_count)
        bottom_second = int(second + vertex_count)
        side_triangles.extend(
            (
                (int(first), int(second), bottom_second),
                (int(first), bottom_second, bottom_first),
            )
        )
    triangles = np.concatenate(
        (
            top_triangles,
            bottom_triangles,
            np.asarray(side_triangles, dtype=np.int64),
        ),
        axis=0,
    )
    return corners, triangles, top_triangles, boundary_edges


def corner_values(corners, cell_values, wet_mask, fallback):
    values = np.empty(len(corners), dtype=np.float32)
    for index, (corner_x, corner_z) in enumerate(corners):
        adjacent = []
        for ix in (corner_x - 1, corner_x):
            for iz in (corner_z - 1, corner_z):
                if (
                    0 <= ix < wet_mask.shape[0]
                    and 0 <= iz < wet_mask.shape[1]
                    and wet_mask[ix, iz]
                    and np.isfinite(cell_values[ix, iz])
                ):
                    adjacent.append(float(cell_values[ix, iz]))
        values[index] = min(adjacent) if adjacent else fallback
    return values


def edge_diagnostics(triangles):
    edges = np.concatenate(
        (
            triangles[:, (0, 1)],
            triangles[:, (1, 2)],
            triangles[:, (2, 0)],
        ),
        axis=0,
    )
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.count_nonzero(counts == 1)), int(np.count_nonzero(counts > 2))


def write_obj(path, points, triangles):
    temporary = path.with_suffix(".tmp.obj")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# Watertight water medium proxy\n")
        stream.writelines(
            f"v {point[0]:.9g} {point[1]:.9g} {point[2]:.9g}\n"
            for point in points
        )
        stream.writelines(
            f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n"
            for face in triangles
        )
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


def main():
    args = parse_args()
    if min(args.spacing, args.surface_gap, args.bottom_overlap) <= 0:
        raise ValueError("Spacing, surface gap, and bottom overlap must be positive")

    paths = sorted(args.surface_directory.glob("surface_*_clipped.obj"))
    if args.frame_list is not None:
        requested = set(args.frame_list)
        paths = [
            path
            for path in paths
            if int(path.stem.split("_")[1]) in requested
        ]
        found = {int(path.stem.split("_")[1]) for path in paths}
        if missing := sorted(requested - found):
            raise RuntimeError(f"Requested surface frames are missing: {missing}")
    if not paths:
        raise RuntimeError(f"No clipped OBJ sequence in {args.surface_directory}")

    with np.load(args.heightfield_npz) as heightfield:
        x_values = np.asarray(heightfield["x_values"], dtype=np.float32)
        z_values = np.asarray(heightfield["z_values"], dtype=np.float32)
        terrain_y = np.asarray(heightfield["terrain_y"], dtype=np.float32)
    wet_mask = static_shoreline_mask(args, terrain_y, x_values, z_values)
    corners, triangles, top_triangles, boundary_edges = build_static_topology(wet_mask)
    boundary_count, nonmanifold_count = edge_diagnostics(triangles.copy())
    if boundary_count or nonmanifold_count:
        raise RuntimeError(
            "Static volume topology failed: "
            f"boundary={boundary_count} nonmanifold={nonmanifold_count}"
        )
    corner_grid = np.asarray(corners, dtype=np.int64)
    corner_x = x_values[0] + (corner_grid[:, 0] - 0.5) * args.spacing
    corner_z = z_values[0] + (corner_grid[:, 1] - 0.5) * args.spacing
    sampled_bottom, bottom_valid = bilinear_sample(
        corner_x, corner_z, x_values, z_values, terrain_y
    )
    if not np.all(bottom_valid):
        raise RuntimeError(
            f"Terrain is missing below {np.count_nonzero(~bottom_valid)} volume corners"
        )
    sampled_bottom -= args.bottom_overlap

    input_digest, total_bytes = metadata_digest(paths)
    configuration = {
        "schema": SCHEMA,
        "surface_directory": str(args.surface_directory.resolve()),
        "surface_count": len(paths),
        "surface_metadata_sha256": input_digest,
        "surface_total_bytes": total_bytes,
        "heightfield": str(args.heightfield_npz.resolve()),
        "heightfield_sha256": sha256_file(args.heightfield_npz),
        "script_sha256": sha256_file(Path(__file__)),
        "water_level": args.water_level,
        "spacing": args.spacing,
        "minimum_layers": args.minimum_layers,
        "shoreline_erosion_cells": args.shoreline_erosion_cells,
        "impact": [args.impact_x, args.impact_z],
        "surface_gap": args.surface_gap,
        "bottom_overlap": args.bottom_overlap,
        "maximum_top_offset": args.maximum_top_offset,
        "wet_cells": int(np.count_nonzero(wet_mask)),
        "topology": {
            "corners": len(corners),
            "top_triangles": len(top_triangles),
            "boundary_edges": len(boundary_edges),
            "closed_triangles": len(triangles),
            "boundary_edges_after_closure": boundary_count,
            "nonmanifold_edges": nonmanifold_count,
        },
    }

    args.output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_directory / "water_volume_manifest.json"
    existing = list(args.output_directory.glob("volume_*.obj"))
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("configuration") != configuration and not args.force:
            raise RuntimeError(
                f"Water-volume configuration changed: {manifest_path}. "
                "Use a new directory or pass --force."
            )
    elif existing and not args.force:
        raise RuntimeError(
            f"{args.output_directory} contains unmanifested water-volume meshes"
        )
    write_manifest(
        manifest_path,
        configuration,
        {"complete": False, "completed_frames": 0, "total_frames": len(paths)},
    )

    for completed, path in enumerate(paths, start=1):
        frame_id = path.stem.split("_")[1]
        output = args.output_directory / f"volume_{frame_id}.obj"
        if output.is_file() and not args.force:
            print(f"[water-volume] frame={frame_id} cached", flush=True)
            continue
        points = read_points(path)
        top_field = build_top_field(
            points,
            wet_mask,
            x_values,
            z_values,
            args.water_level,
            args.spacing,
            args.maximum_top_offset,
        )
        top_y = corner_values(
            corners,
            top_field,
            wet_mask,
            args.water_level - 0.45 * args.spacing,
        )
        top_y -= args.surface_gap
        if np.any(top_y <= sampled_bottom + 0.5 * args.spacing):
            raise RuntimeError(f"Invalid water thickness in frame {frame_id}")
        top_points = np.column_stack((corner_x, top_y, corner_z))
        bottom_points = np.column_stack((corner_x, sampled_bottom, corner_z))
        volume_points = np.asarray(
            np.concatenate((top_points, bottom_points), axis=0), dtype=np.float32
        )
        write_obj(output, volume_points, triangles)
        print(
            f"[water-volume] frame={frame_id} vertices={len(volume_points)} "
            f"faces={len(triangles)} wet_cells={int(np.count_nonzero(wet_mask))} "
            f"top=({float(top_y.min()):.4f},{float(top_y.max()):.4f}) "
            f"closed=true",
            flush=True,
        )

    write_manifest(
        manifest_path,
        configuration,
        {
            "complete": True,
            "completed_frames": len(paths),
            "total_frames": len(paths),
        },
    )
    print(f"WATER_VOLUME_SEQUENCE={args.output_directory} frames={len(paths)}")


if __name__ == "__main__":
    main()
