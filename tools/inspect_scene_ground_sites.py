"""Find the real rendered mesh under each configured experiment site."""

import argparse
import json
from pathlib import Path

from isaacsim import SimulationApp


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--manifest", default=r"Y:\scenes\static_scene_manifest.json")
parser.add_argument("--config", default=r"Y:\isaacsim_work\scene_experiments.json")
parser.add_argument("--only", nargs="*", default=None)
parser.add_argument("--output", default=r"Y:\isaacsim_work\output\scene_ground_sites.json")
args = parser.parse_args()

app = SimulationApp({"headless": True, "renderer": "MinimalRendering"})

try:
    from pxr import Gf, Usd, UsdGeom

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    config_path = Path(args.config)
    overrides = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    selected = set(args.only or [])
    results = []
    for row in manifest["results"]:
        name = row["scene"]
        if selected and name not in selected:
            continue
        experiment = dict(row["experiment"])
        experiment.update(overrides.get(name, {}))
        x = float(experiment["spawn_x"])
        z = float(experiment["spawn_z"])
        expected_y = float(experiment["support_top_y"])
        stage = Usd.Stage.Open(row["source_usd"])
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
                    a = ((z1 - z2) * (x - x2) + (x2 - x1) * (z - z2)) / denominator
                    b = ((z2 - z0) * (x - x2) + (x0 - x2) * (z - z2)) / denominator
                    c = 1.0 - a - b
                    if min(a, b, c) < -1e-6:
                        continue
                    cross = Gf.Cross(p1 - p0, p2 - p0)
                    length = cross.GetLength()
                    normal_y = abs(float(cross[1])) / length if length > 1e-12 else 0.0
                    if normal_y < 0.5:
                        continue
                    y = a * float(p0[1]) + b * float(p1[1]) + c * float(p2[1])
                    hits.append(
                        {
                            "y": y,
                            "normal_y": normal_y,
                            "source_prim": str(prim.GetPath()),
                            "face": face_index,
                        }
                    )
        hits.sort(key=lambda hit: (abs(hit["y"] - expected_y), -hit["normal_y"]))
        candidate = hits[0] if hits else None
        if candidate:
            default_path = str(stage.GetDefaultPrim().GetPath())
            relative_path = candidate["source_prim"][len(default_path) :]
            candidate["composed_prim"] = "/World/Environment" + relative_path
        result = {
            "scene": name,
            "x": x,
            "z": z,
            "expected_y": expected_y,
            "candidate": candidate,
            "nearest_hits": hits[:5],
        }
        results.append(result)
        if candidate:
            print(
                f"[ground] {name} expected={expected_y:.4f} hit={candidate['y']:.4f} "
                f"normal_y={candidate['normal_y']:.3f} prim={candidate['composed_prim']}"
            )
        else:
            print(f"[ground-missing] {name} x={x:.4f} z={z:.4f}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if selected and output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        updated_by_name = {row["scene"]: row for row in results}
        merged_results = [
            updated_by_name.pop(row["scene"], row)
            for row in previous.get("results", [])
        ]
        merged_results.extend(updated_by_name.values())
        results = sorted(merged_results, key=lambda row: row["scene"])
    output_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print(f"[complete] output={output_path}")
finally:
    app.close()
