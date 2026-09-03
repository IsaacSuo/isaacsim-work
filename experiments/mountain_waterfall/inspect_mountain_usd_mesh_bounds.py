"""Measure every Mountain USD mesh in Isaac Y-up world coordinates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from isaacsim import SimulationApp


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source_usd", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--wall-target", type=Path)
parser.add_argument("--physx-reference-layout", action="store_true")
args = parser.parse_args()
if not args.source_usd.is_file():
    raise FileNotFoundError(args.source_usd)
if args.output.exists():
    raise FileExistsError(f"Refusing to overwrite {args.output}")
wall_minimum = None
wall_maximum = None
wall_median = None
if args.wall_target is not None:
    wall_payload = json.loads(args.wall_target.read_text(encoding="utf-8"))
    measured = wall_payload["measured_wall"]
    wall_minimum = np.asarray(
        measured["contact_gate_aabb_minimum_isaac"], dtype=np.float64
    )
    wall_maximum = np.asarray(
        measured["contact_gate_aabb_maximum_isaac"], dtype=np.float64
    )
    wall_median = np.asarray(
        measured["location_median_isaac"], dtype=np.float64
    )

app = SimulationApp({"headless": True})
try:
    from pxr import Usd, UsdGeom

    if args.physx_reference_layout:
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        environment = UsdGeom.Xform.Define(stage, "/World/Environment")
        environment.GetPrim().GetReferences().AddReference(str(args.source_usd.resolve()))
        UsdGeom.Xformable(environment.GetPrim()).AddRotateXOp().Set(-90.0)
        traversal_root = environment.GetPrim()
        conversion = np.eye(3, dtype=np.float64)
        conversion_label = "physx_reference_under_rotate_x_minus_90"
    else:
        stage = Usd.Stage.Open(str(args.source_usd.resolve()))
        if stage is None:
            raise RuntimeError("Unable to open Mountain USD")
        traversal_root = stage.GetPseudoRoot()
        conversion = np.asarray(
            ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
            dtype=np.float64,
        )
        conversion_label = "blender_z_up_to_isaac_y_up"
    cache = UsdGeom.XformCache()
    rows = []
    for prim in Usd.PrimRange(traversal_root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get() or []
        counts = mesh.GetFaceVertexCountsAttr().Get() or []
        indices = mesh.GetFaceVertexIndicesAttr().Get() or []
        transform = cache.GetLocalToWorldTransform(prim)
        vertices = np.asarray(
            [transform.Transform(point) for point in points], dtype=np.float64
        )
        vertices = vertices @ conversion.T
        row = {
                "prim_path": str(prim.GetPath()),
                "vertices": int(len(vertices)),
                "polygons": int(len(counts)),
                "bounds_minimum_isaac": vertices.min(axis=0).tolist(),
                "bounds_maximum_isaac": vertices.max(axis=0).tolist(),
            }
        if wall_minimum is not None:
            vertex_distances = np.linalg.norm(vertices - wall_median, axis=1)
            closest_vertex = int(np.argmin(vertex_distances))
            row["minimum_vertex_distance_to_wall_median_m"] = float(
                vertex_distances[closest_vertex]
            )
            row["closest_vertex_to_wall_median_isaac"] = vertices[
                closest_vertex
            ].tolist()
            triangles = []
            cursor = 0
            for count in counts:
                face = indices[cursor : cursor + int(count)]
                cursor += int(count)
                for offset in range(1, int(count) - 1):
                    triangles.append((int(face[0]), int(face[offset]), int(face[offset + 1])))
            triangle_points = vertices[np.asarray(triangles, dtype=np.int64)]
            overlaps = np.all(
                triangle_points.max(axis=1) >= wall_minimum, axis=1
            ) & np.all(triangle_points.min(axis=1) <= wall_maximum, axis=1)
            row["wall_aabb_overlapping_triangles"] = int(np.count_nonzero(overlaps))
        rows.append(row)
    payload = {
        "schema": 1,
        "product": "mountain_usd_mesh_bounds_inspection",
        "source_usd": str(args.source_usd.resolve()),
        "coordinate_conversion": conversion_label,
        "meshes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))
finally:
    app.close()
