"""List mesh triangles intersected by a vertical ray at an X/Z scene location."""

import argparse

from isaacsim import SimulationApp


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("usd")
parser.add_argument("x", type=float)
parser.add_argument("z", type=float)
args = parser.parse_args()

app = SimulationApp({"headless": True, "renderer": "MinimalRendering"})

try:
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(args.usd)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    hits = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get() or []
        counts = mesh.GetFaceVertexCountsAttr().Get() or []
        indices = mesh.GetFaceVertexIndicesAttr().Get() or []
        matrix = cache.GetLocalToWorldTransform(prim)
        cursor = 0
        for face_index, count in enumerate(counts):
            face = indices[cursor : cursor + count]
            cursor += count
            if count < 3:
                continue
            p0 = matrix.Transform(Gf.Vec3d(points[face[0]]))
            for offset in range(1, count - 1):
                p1 = matrix.Transform(Gf.Vec3d(points[face[offset]]))
                p2 = matrix.Transform(Gf.Vec3d(points[face[offset + 1]]))
                x0, z0 = float(p0[0]), float(p0[2])
                x1, z1 = float(p1[0]), float(p1[2])
                x2, z2 = float(p2[0]), float(p2[2])
                denominator = (z1 - z2) * (x0 - x2) + (x2 - x1) * (z0 - z2)
                if abs(denominator) < 1e-10:
                    continue
                a = ((z1 - z2) * (args.x - x2) + (x2 - x1) * (args.z - z2)) / denominator
                b = ((z2 - z0) * (args.x - x2) + (x0 - x2) * (args.z - z2)) / denominator
                c = 1.0 - a - b
                if min(a, b, c) < -1e-6:
                    continue
                cross = Gf.Cross(p1 - p0, p2 - p0)
                length = cross.GetLength()
                normal_y = abs(float(cross[1])) / length if length > 1e-12 else 0.0
                y = a * float(p0[1]) + b * float(p1[1]) + c * float(p2[1])
                hits.append((y, normal_y, str(prim.GetPath()), face_index))
    for y, normal_y, path, face_index in sorted(hits, reverse=True):
        print(f"y={y:.6f} normal_y={normal_y:.4f} face={face_index} prim={path}")
finally:
    app.close()
