"""Verify local scene collision assets at each configured experiment landing area."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "scene_experiments.json"
CAMERAS = ROOT / "configs" / "blender_camera_selections.json"
ASSETS = ROOT / "output" / "scene_collision_assets"
BASELINES = ROOT / "output" / "blender_scene_videos" / "isaac"


def baseline_values(scene_name):
    candidates = [
        BASELINES / scene_name / "run_complete.json",
        ROOT / "output" / "blender_silicone_exact_collision" / "isaac" / scene_name / "run_complete.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        patch = report.get("ground_patch") or {}
        initial = (report.get("keyframes") or {}).get("initial") or {}
        center = initial.get("center") or [None, None, None]
        return {
            "support_y": report.get("support_top_y"),
            "spawn_x": patch.get("center_x", center[0]),
            "spawn_z": patch.get("center_z", center[2]),
        }
    return {}


def intersections_at(points, triangles, x, z, np):
    tri = points[triangles]
    x0, z0 = tri[:, 0, 0], tri[:, 0, 2]
    x1, z1 = tri[:, 1, 0], tri[:, 1, 2]
    x2, z2 = tri[:, 2, 0], tri[:, 2, 2]
    denominator = (z1 - z2) * (x0 - x2) + (x2 - x1) * (z0 - z2)
    usable = np.abs(denominator) > 1.0e-10
    a = np.zeros_like(denominator)
    b = np.zeros_like(denominator)
    a[usable] = ((z1[usable] - z2[usable]) * (x - x2[usable]) + (x2[usable] - x1[usable]) * (z - z2[usable])) / denominator[usable]
    b[usable] = ((z2[usable] - z0[usable]) * (x - x2[usable]) + (x0[usable] - x2[usable]) * (z - z2[usable])) / denominator[usable]
    c = 1.0 - a - b
    inside = usable & (a >= -1.0e-7) & (b >= -1.0e-7) & (c >= -1.0e-7)
    return a[inside] * tri[inside, 0, 1] + b[inside] * tri[inside, 1, 1] + c[inside] * tri[inside, 2, 1]


def main():
    app = SimulationApp({"headless": True, "renderer": "MinimalRendering"})
    try:
        import numpy as np
        from pxr import Usd, UsdGeom

        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        scenes = list(json.loads(CAMERAS.read_text(encoding="utf-8")))
        results = []
        for scene_name in scenes:
            scene = config[scene_name]
            fallback = baseline_values(scene_name)
            support_y = float(scene.get("support_top_y", fallback.get("support_y")))
            spawn_x = float(scene.get("spawn_x", fallback.get("spawn_x")))
            spawn_z = float(scene.get("spawn_z", fallback.get("spawn_z")))
            asset = ASSETS / f"{scene_name}_exact_collision.usdc"
            stage = Usd.Stage.Open(str(asset))
            mesh = UsdGeom.Mesh.Get(stage, "/World/ExperimentGroundMeshes/ExactLocalEnvironment")
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
            triangle_points = points[indices]
            triangle_normals = np.cross(
                triangle_points[:, 1] - triangle_points[:, 0],
                triangle_points[:, 2] - triangle_points[:, 0],
            )
            upward_triangles = int(np.count_nonzero(triangle_normals[:, 1] > 1.0e-10))
            downward_triangles = int(np.count_nonzero(triangle_normals[:, 1] < -1.0e-10))
            vertical_triangles = int(len(indices) - upward_triangles - downward_triangles)
            center_hits = intersections_at(points, indices, spawn_x, spawn_z, np)
            center_y = None
            if len(center_hits):
                center_y = float(center_hits[np.argmin(np.abs(center_hits - support_y))])
            sampled_heights = []
            for dx in np.linspace(-0.35, 0.35, 5):
                for dz in np.linspace(-0.35, 0.35, 5):
                    hits = intersections_at(points, indices, spawn_x + dx, spawn_z + dz, np)
                    if len(hits):
                        sampled_heights.append(float(hits[np.argmin(np.abs(hits - support_y))]))
                    else:
                        sampled_heights.append(None)
            near_support = [
                value for value in sampled_heights
                if value is not None and abs(value - support_y) <= (1.0 if scene_name == "mountain" else 0.25)
            ]
            center_delta = None if center_y is None else center_y - support_y
            valid = bool(
                len(points) > 0
                and len(indices) > 0
                and center_y is not None
                and abs(center_delta) <= (1.0 if scene_name == "mountain" else 0.25)
                and len(near_support) >= 20
            )
            row = {
                "scene": scene_name,
                "asset": str(asset),
                "vertices": int(len(points)),
                "triangles": int(len(indices)),
                "upward_triangles": upward_triangles,
                "downward_triangles": downward_triangles,
                "vertical_triangles": vertical_triangles,
                "spawn": [spawn_x, spawn_z],
                "configured_support_y": support_y,
                "mesh_surface_y_at_spawn": center_y,
                "surface_delta": center_delta,
                "footprint_samples": len(sampled_heights),
                "near_support_samples": len(near_support),
                "footprint_coverage": len(near_support) / len(sampled_heights),
                "sample_height_min": min(near_support) if near_support else None,
                "sample_height_max": max(near_support) if near_support else None,
                "valid": valid,
            }
            results.append(row)
            print(
                f"[collision-audit] {scene_name} valid={valid} triangles={len(indices)} "
                f"normals=+{upward_triangles}/-{downward_triangles}/|{vertical_triangles} "
                f"surface={center_y} configured={support_y} coverage={len(near_support)}/25"
            )
        payload = {"valid": all(row["valid"] for row in results), "results": results}
        output = ASSETS / "audit.json"
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[collision-audit] complete valid={payload['valid']} report={output}")
        return 0 if payload["valid"] else 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
