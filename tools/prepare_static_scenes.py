"""Audit exported scene USDs and build static-collision wrappers in one Isaac run."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from isaacsim import SimulationApp


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes-root", default=r"Y:\scenes")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", nargs="*", default=None)
    return parser.parse_args()


ARGS = parse_args()
SCENES_ROOT = Path(ARGS.scenes_root).resolve()
MANIFEST_PATH = SCENES_ROOT / "static_scene_manifest.json"

app = SimulationApp({"headless": True, "renderer": "MinimalRendering"})

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    def horizontal_surface_levels(stage):
        levels = defaultdict(lambda: {"area": 0.0, "x": 0.0, "z": 0.0, "faces": 0})
        xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get() or []
            counts = mesh.GetFaceVertexCountsAttr().Get() or []
            indices = mesh.GetFaceVertexIndicesAttr().Get() or []
            matrix = xforms.GetLocalToWorldTransform(prim)
            cursor = 0
            for count in counts:
                face = indices[cursor : cursor + count]
                cursor += count
                if count < 3:
                    continue
                anchor = matrix.Transform(Gf.Vec3d(points[face[0]]))
                for offset in range(1, count - 1):
                    p1 = matrix.Transform(Gf.Vec3d(points[face[offset]]))
                    p2 = matrix.Transform(Gf.Vec3d(points[face[offset + 1]]))
                    cross = Gf.Cross(p1 - anchor, p2 - anchor)
                    twice_area = cross.GetLength()
                    if twice_area <= 1e-9:
                        continue
                    if abs(float(cross[1])) / twice_area < 0.72:
                        continue
                    area = 0.5 * twice_area
                    center = (anchor + p1 + p2) / 3.0
                    level = round(float(center[1]) * 10.0) / 10.0
                    entry = levels[level]
                    entry["area"] += area
                    entry["x"] += area * float(center[0])
                    entry["z"] += area * float(center[2])
                    entry["faces"] += 1
        rows = []
        for y, entry in levels.items():
            if entry["area"] <= 0.0:
                continue
            rows.append(
                {
                    "y": y,
                    "area": entry["area"],
                    "x": entry["x"] / entry["area"],
                    "z": entry["z"] / entry["area"],
                    "faces": entry["faces"],
                }
            )
        return rows

    def choose_experiment_site(levels, bounds):
        if not levels:
            return {
                "support_top_y": float(bounds[0][1]),
                "spawn_x": float((bounds[0][0] + bounds[1][0]) * 0.5),
                "spawn_z": float((bounds[0][2] + bounds[1][2]) * 0.5),
                "basis": "bounds_fallback",
            }
        maximum_area = max(row["area"] for row in levels)
        candidates = [
            row for row in levels if row["area"] >= max(1.0, maximum_area * 0.05)
        ]
        chosen = min(candidates, key=lambda row: (row["y"], -row["area"]))
        return {
            "support_top_y": chosen["y"],
            "spawn_x": chosen["x"],
            "spawn_z": chosen["z"],
            "basis": "large_horizontal_surface",
            "surface_area": chosen["area"],
            "surface_faces": chosen["faces"],
        }

    def inspect_assets(stage, source_path):
        assets = set()
        for prim in stage.Traverse():
            for attr in prim.GetAttributes():
                value = attr.Get()
                if isinstance(value, Sdf.AssetPath) and value.path:
                    assets.add(value.path)
        missing = []
        for raw in sorted(assets):
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = source_path.parent / raw
            if not candidate.exists():
                missing.append(raw)
        return sorted(assets), missing

    def build_wrapper(source_path, output_path):
        wrapper = Usd.Stage.CreateNew(str(output_path))
        UsdGeom.SetStageMetersPerUnit(wrapper, 1.0)
        UsdGeom.SetStageUpAxis(wrapper, UsdGeom.Tokens.y)
        UsdPhysics.SetStageKilogramsPerUnit(wrapper, 1.0)
        world = UsdGeom.Xform.Define(wrapper, "/World")
        wrapper.SetDefaultPrim(world.GetPrim())
        environment = UsdGeom.Xform.Define(wrapper, "/World/Environment")
        environment.GetPrim().GetReferences().AddReference(f"./{source_path.name}")
        physics_scene = UsdPhysics.Scene.Define(wrapper, "/World/PhysicsScene")
        physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, -1.0, 0.0))
        physics_scene.CreateGravityMagnitudeAttr().Set(9.81)
        colliders = 0
        for prim in wrapper.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(True)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("none")
                colliders += 1
        if colliders == 0:
            raise RuntimeError("No referenced meshes were found")
        wrapper.GetRootLayer().Save()
        return colliders

    selected = set(ARGS.only or [])
    scene_dirs = sorted(path for path in SCENES_ROOT.iterdir() if path.is_dir())
    results = []
    for scene_dir in scene_dirs:
        name = scene_dir.name
        if selected and name not in selected:
            continue
        source_path = scene_dir / f"{name}.usdc"
        output_path = scene_dir / f"{name}_sim.usda"
        row = {"scene": name, "source_usd": str(source_path), "sim_usd": str(output_path)}
        try:
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            stage = Usd.Stage.Open(str(source_path))
            if not stage:
                raise RuntimeError("Usd.Stage.Open returned None")
            cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            )
            aligned = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedBox()
            lo, hi = aligned.GetMin(), aligned.GetMax()
            bounds = ([float(lo[i]) for i in range(3)], [float(hi[i]) for i in range(3)])
            meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
            points = sum(len(UsdGeom.Mesh(prim).GetPointsAttr().Get() or []) for prim in meshes)
            faces = sum(
                len(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get() or [])
                for prim in meshes
            )
            nontri = sum(
                sum(1 for count in (UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get() or []) if count != 3)
                for prim in meshes
            )
            assets, missing = inspect_assets(stage, source_path)
            levels = horizontal_surface_levels(stage)
            site = choose_experiment_site(levels, bounds)
            support = site["support_top_y"]
            x, z = site["spawn_x"], site["spawn_z"]
            site.update(
                {
                    "drop_height": support + 3.49,
                    "camera_eye": [x + 5.5, support + 4.49, z + 8.1],
                    "camera_target": [x, support + 2.24, z],
                }
            )
            should_build = ARGS.force or not output_path.is_file()
            collider_count = build_wrapper(source_path, output_path) if should_build else len(meshes)
            row.update(
                {
                    "valid": True,
                    "built": should_build,
                    "default_prim": str(stage.GetDefaultPrim().GetPath()),
                    "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
                    "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
                    "bounds": bounds,
                    "mesh_count": len(meshes),
                    "point_count": points,
                    "face_count": faces,
                    "nontri_face_count": nontri,
                    "asset_count": len(assets),
                    "missing_assets": missing,
                    "collider_count": collider_count,
                    "experiment": site,
                    "horizontal_levels": sorted(levels, key=lambda item: item["area"], reverse=True)[:12],
                }
            )
            print(
                f"[scene] {name} meshes={len(meshes)} faces={faces} "
                f"assets={len(assets)} missing={len(missing)} support={support:.3f} "
                f"built={should_build}"
            )
        except Exception as exc:
            row.update({"valid": False, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[scene-failed] {name} {row['error']}")
        results.append(row)

    if selected and MANIFEST_PATH.is_file():
        previous = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        updated_by_name = {row["scene"]: row for row in results}
        merged_results = [
            updated_by_name.pop(row["scene"], row)
            for row in previous.get("results", [])
        ]
        merged_results.extend(updated_by_name.values())
        results = sorted(merged_results, key=lambda row: row["scene"])
    manifest = {
        "valid": all(row.get("valid", False) for row in results),
        "scenes_root": str(SCENES_ROOT),
        "scene_count": len(results),
        "results": results,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[complete] valid={manifest['valid']} scenes={len(results)} manifest={MANIFEST_PATH}")
    if not manifest["valid"]:
        raise SystemExit(1)
finally:
    app.close()
