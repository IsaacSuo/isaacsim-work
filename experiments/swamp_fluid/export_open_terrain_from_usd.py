"""Export a reproducible fluid-facing open terrain adapter from a USD mesh.

Run this script with a Python environment that provides Pixar USD (``pxr``).
It never chooses a surface by casting a first-hit ray over the full asset.
Selection is an explicit ROI + maximum surface level, and every selected source
face index is written to the companion audit record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_simulation_app = None
try:
    from pxr import Usd, UsdGeom
except ModuleNotFoundError:
    from isaacsim import SimulationApp

    _simulation_app = SimulationApp({"headless": True})
    from pxr import Usd, UsdGeom

from whitewater.terrain_fields import select_open_terrain_faces


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_usd", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--prim-path",
        action="append",
        required=True,
        help="USD mesh prim to include; repeat for a multi-mesh terrain adapter.",
    )
    parser.add_argument("--roi-minimum", type=float, nargs=3, required=True)
    parser.add_argument("--roi-maximum", type=float, nargs=3, required=True)
    parser.add_argument("--maximum-surface-y", type=float, required=True)
    parser.add_argument(
        "--expand-supported-components",
        action="store_true",
        help=(
            "Use below-level faces as liquid-support seeds, then retain their "
            "entire connected ROI components so dry banks share the same asset."
        ),
    )
    parser.add_argument(
        "--coordinate-conversion",
        choices=("identity", "blender_z_up_to_isaac_y_up"),
        default="identity",
    )
    parser.add_argument(
        "--orientation-reference",
        type=float,
        nargs=3,
        default=(0.0, 1.0, 0.0),
        help="Fluid-facing reference used to orient each selected component.",
    )
    parser.add_argument("--mesh-name", default="selected_open_terrain.obj")
    parser.add_argument("--selection-name", default="selection_record.json")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def triangulated_faces(counts, indices):
    triangles = []
    source_polygon = []
    cursor = 0
    for polygon_index, count in enumerate(counts):
        count = int(count)
        face = indices[cursor : cursor + count]
        cursor += count
        if count < 3:
            raise ValueError(f"USD polygon {polygon_index} has fewer than three vertices")
        for offset in range(1, count - 1):
            triangles.append((int(face[0]), int(face[offset]), int(face[offset + 1])))
            source_polygon.append(int(polygon_index))
    if cursor != len(indices):
        raise ValueError("USD face counts do not consume all face indices")
    return np.asarray(triangles, dtype=np.int64), source_polygon


def write_obj(path, vertices, faces):
    lines = ["# Audited fluid-facing open terrain; units follow the scene contract."]
    for vertex in np.asarray(vertices, dtype=np.float64):
        lines.append("v " + " ".join(format(float(value), ".17g") for value in vertex))
    for face in np.asarray(faces, dtype=np.int64):
        lines.append("f " + " ".join(str(int(value) + 1) for value in face))
    atomic_text(path, "\n".join(lines) + "\n")


def main():
    args = parse_args()
    source_usd = args.source_usd.resolve()
    output_directory = args.output_directory.resolve()
    if not source_usd.is_file():
        raise FileNotFoundError(source_usd)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.Open(str(source_usd))
    if stage is None:
        raise ValueError(f"Unable to open USD stage: {source_usd}")
    coordinate_matrix = np.eye(3, dtype=np.float64)
    if args.coordinate_conversion == "blender_z_up_to_isaac_y_up":
        # Blender (x, y, z) -> Isaac (x, z, -y), a right-handed rotation.
        coordinate_matrix = np.asarray(
            ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
            dtype=np.float64,
        )
    vertex_blocks = []
    face_blocks = []
    face_sources = []
    source_meshes = []
    vertex_offset = 0
    face_offset = 0
    xform_cache = UsdGeom.XformCache()
    for prim_path in args.prim_path:
        mesh = UsdGeom.Mesh.Get(stage, prim_path)
        if not mesh:
            raise ValueError(f"USD prim is not a mesh: {prim_path}")
        points = mesh.GetPointsAttr().Get() or []
        counts = mesh.GetFaceVertexCountsAttr().Get() or []
        indices = mesh.GetFaceVertexIndicesAttr().Get() or []
        if not points or not counts:
            raise ValueError(f"USD terrain mesh has no geometry: {prim_path}")
        transform = xform_cache.GetLocalToWorldTransform(mesh.GetPrim())
        mesh_vertices = np.asarray(
            [transform.Transform(point) for point in points], dtype=np.float64
        )
        mesh_vertices = mesh_vertices @ coordinate_matrix.T
        mesh_faces, source_polygon_indices = triangulated_faces(counts, indices)
        vertex_blocks.append(mesh_vertices)
        face_blocks.append(mesh_faces + vertex_offset)
        face_sources.extend(
            (prim_path, int(polygon_index))
            for polygon_index in source_polygon_indices
        )
        source_meshes.append(
            {
                "prim_path": prim_path,
                "authored_vertex_count": int(len(mesh_vertices)),
                "authored_polygon_count": int(len(counts)),
                "triangulated_face_count": int(len(mesh_faces)),
                "combined_face_range": [face_offset, face_offset + len(mesh_faces)],
            }
        )
        vertex_offset += len(mesh_vertices)
        face_offset += len(mesh_faces)
    vertices = np.concatenate(vertex_blocks, axis=0)
    faces = np.concatenate(face_blocks, axis=0)
    selected_vertices, selected_faces, selection = select_open_terrain_faces(
        vertices,
        faces,
        args.roi_minimum,
        args.roi_maximum,
        args.maximum_surface_y,
        orientation_reference=args.orientation_reference,
        expand_supported_components=args.expand_supported_components,
    )

    mesh_path = output_directory / args.mesh_name
    selection_path = output_directory / args.selection_name
    write_obj(mesh_path, selected_vertices, selected_faces)
    mesh_sha256 = sha256_file(mesh_path)
    selected_source_faces = selection["source_face_indices"]
    support_seed_faces = selection["support_seed_face_indices"]
    selected_source_polygons = [face_sources[index] for index in selected_source_faces]
    source_record = {
        "path": str(source_usd),
        "sha256": sha256_file(source_usd),
        "usd_up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "coordinate_conversion": args.coordinate_conversion,
        "coordinate_matrix": coordinate_matrix.astype(float).tolist(),
        "authored_vertex_count": int(len(vertices)),
        "authored_polygon_count": int(
            sum(row["authored_polygon_count"] for row in source_meshes)
        ),
        "triangulated_face_count": int(len(faces)),
    }
    if len(source_meshes) == 1:
        source_record["prim_path"] = source_meshes[0]["prim_path"]
    else:
        source_record["prim_paths"] = list(args.prim_path)
        source_record["meshes"] = source_meshes
    record = {
        "schema": 1,
        "product": "whitewater_open_terrain_selection",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_record,
        "selector": selection["selector"],
        "selection": {
            "source_face_indices": selected_source_faces,
            "support_seed_face_indices": support_seed_faces,
            "support_seed_face_count": len(support_seed_faces),
            "source_polygon_indices": (
                [polygon for _, polygon in selected_source_polygons]
                if len(source_meshes) == 1
                else [
                    {"prim_path": prim_path, "polygon_index": polygon}
                    for prim_path, polygon in selected_source_polygons
                ]
            ),
            "connected_component_count": selection["connected_component_count"],
            "flipped_component_indices": selection["flipped_component_indices"],
        },
        "normal_convention": "toward_fluid",
        "selected_mesh": {
            "path": str(mesh_path),
            "sha256": mesh_sha256,
            "vertex_count": selection["selected_vertex_count"],
            "triangle_count": selection["selected_face_count"],
            "connected_component_count": selection["connected_component_count"],
            "boundary_edge_count": selection["boundary_edge_count"],
            "bounds_minimum": selection["bounds_minimum"],
            "bounds_maximum": selection["bounds_maximum"],
            "watertight_expected": False,
        },
        "selection_guards": {
            "global_first_hit_ray_cast": False,
            "source_faces_are_explicit": True,
            "fluid_facing_orientation_is_explicit": True,
        },
    }
    atomic_text(selection_path, json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "mesh": str(mesh_path),
                "mesh_sha256": mesh_sha256,
                "selection": str(selection_path),
                "selection_sha256": sha256_file(selection_path),
                "triangles": selection["selected_face_count"],
                "components": selection["connected_component_count"],
                "bounds_minimum": selection["bounds_minimum"],
                "bounds_maximum": selection["bounds_maximum"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        if _simulation_app is not None:
            _simulation_app.close()
