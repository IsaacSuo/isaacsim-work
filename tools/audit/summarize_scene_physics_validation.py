"""Combine per-scene PhysX reports with the exact collision geometry audit."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT / "output" / "scene_collision_assets"
GEOMETRY_AUDIT = ASSET_DIR / "audit.json"
FINAL_RUNS = ROOT / "output" / "local_collision_final_validation" / "runs"
BEDROOM_OVERRIDE = (
    ROOT
    / "output"
    / "local_collision_bedroom_refine"
    / "runs"
    / "bedroom_cat_silicone_firm"
    / "isaac"
    / "bedroom"
    / "run_complete.json"
)
OUTPUT = ASSET_DIR / "physics_audit.json"
IMPACT_TOLERANCE_METERS = 0.04


def find_reports():
    reports = {}
    for path in FINAL_RUNS.glob("*/isaac/*/run_complete.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        scene = Path(report["environment_usd"]).parent.name
        reports[scene] = (path, report)
    bedroom = json.loads(BEDROOM_OVERRIDE.read_text(encoding="utf-8"))
    reports["bedroom"] = (BEDROOM_OVERRIDE, bedroom)
    return reports


def main():
    geometry = json.loads(GEOMETRY_AUDIT.read_text(encoding="utf-8"))
    surfaces = {
        row["scene"]: row["mesh_surface_y_at_spawn"] for row in geometry["results"]
    }
    reports = find_reports()
    rows = []
    for scene in sorted(surfaces):
        path, report = reports[scene]
        impact_y = float(report["keyframes"]["impact"]["minimum_y"])
        surface_y = float(surfaces[scene])
        impact_delta = impact_y - surface_y
        collision_asset = Path(report["prebuilt_collision_usd"])
        expected_asset = ASSET_DIR / f"{scene}_exact_collision.usdc"
        metadata = json.loads(expected_asset.with_suffix(".json").read_text(encoding="utf-8"))
        allow_roll_off = scene == "mountain"
        minimum_delta = None
        if not allow_roll_off:
            minimum_delta = float(report["minimum_surface_y"]) - surface_y
        valid = bool(
            report.get("valid")
            and collision_asset.resolve() == expected_asset.resolve()
            and metadata.get("winding_policy")
            == "non_vertical_triangles_face_positive_y_v1"
            and abs(impact_delta) <= IMPACT_TOLERANCE_METERS
            and (allow_roll_off or minimum_delta >= -IMPACT_TOLERANCE_METERS)
        )
        material = report.get("physics_material") or {}
        rows.append(
            {
                "scene": scene,
                "valid": valid,
                "report": str(path),
                "collision_asset": str(collision_asset),
                "mesh_surface_y_at_spawn": surface_y,
                "impact_minimum_y": impact_y,
                "impact_surface_delta": impact_delta,
                "minimum_surface_delta": minimum_delta,
                "allow_roll_off": allow_roll_off,
                "collision_rest_offset": material.get("collision_rest_offset"),
                "collision_contact_offset": material.get("collision_contact_offset"),
                "deformable_resolution": material.get("deformable_resolution"),
            }
        )
    payload = {
        "valid": len(rows) == 14 and all(row["valid"] for row in rows),
        "impact_tolerance_meters": IMPACT_TOLERANCE_METERS,
        "results": rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for row in rows:
        print(
            f"[physics-audit] {row['scene']} valid={row['valid']} "
            f"impact_delta={row['impact_surface_delta']:+.4f} "
            f"minimum_delta={row['minimum_surface_delta']}"
        )
    print(f"[physics-audit] complete valid={payload['valid']} report={OUTPUT}")
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
