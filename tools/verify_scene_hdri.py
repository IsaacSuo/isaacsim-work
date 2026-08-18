"""Verify that every prepared scene composes with the workspace HDRI."""

import argparse
import json
import os
import traceback
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes-root", default=r"Y:\scenes")
    parser.add_argument("--hdri", default=r"Y:\scenes\HDRI\bambanani_sunset_8k.exr")
    parser.add_argument("--intensity", type=float, default=1000.0)
    parser.add_argument("--updates", type=int, default=12)
    parser.add_argument("--output", default=r"Y:\isaacsim_work\output\scene_hdri_validation.json")
    return parser.parse_args()


ARGS = parse_args()
APP = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": 320, "height": 320})

import omni.usd
from pxr import Sdf, UsdLux


def validate_scene(context, scene_path, hdri_path):
    context.new_stage()
    APP.update()
    stage = context.get_stage()
    stage.GetRootLayer().subLayerPaths.append(scene_path.as_posix())

    imported_paths = []
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdLux.DomeLight):
            imported_paths.append(str(prim.GetPath()))
            UsdLux.DomeLight(prim).GetPrim().SetActive(False)

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/UnifiedHDRI")
    dome.CreateTextureFileAttr(Sdf.AssetPath(hdri_path.as_posix()))
    dome.CreateTextureFormatAttr("latlong")
    dome.CreateIntensityAttr(float(ARGS.intensity))
    UsdGeom.Xformable(dome.GetPrim()).AddRotateXOp().Set(-90.0)
    for _ in range(max(1, ARGS.updates)):
        APP.update()

    texture = dome.GetTextureFileAttr().Get()
    active_domes = [str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdLux.DomeLight)]
    imported_inactive = all(not stage.GetPrimAtPath(path).IsActive() for path in imported_paths)
    expected = str(hdri_path.resolve())
    resolved = str(texture.resolvedPath) if isinstance(texture, Sdf.AssetPath) else ""
    valid = (
        imported_inactive
        and active_domes == ["/World/Lights/UnifiedHDRI"]
        and Path(resolved).resolve() == hdri_path.resolve()
        and dome.GetTextureFormatAttr().Get() == "latlong"
        and float(dome.GetIntensityAttr().Get()) == float(ARGS.intensity)
    )
    return {
        "scene": scene_path.parent.name,
        "usd": str(scene_path),
        "valid": valid,
        "imported_domes_disabled": imported_paths,
        "active_domes": active_domes,
        "texture": texture.path if isinstance(texture, Sdf.AssetPath) else str(texture),
        "resolved_texture": resolved,
        "expected_texture": expected,
        "texture_format": dome.GetTextureFormatAttr().Get(),
        "intensity": dome.GetIntensityAttr().Get(),
    }


def main():
    scenes_root = Path(ARGS.scenes_root).resolve()
    hdri_path = Path(ARGS.hdri).resolve()
    output_path = Path(ARGS.output).resolve()
    if not hdri_path.is_file():
        raise FileNotFoundError(f"HDRI does not exist: {hdri_path}")
    scene_paths = sorted(scenes_root.glob("*/*_sim.usda"), key=lambda path: path.parent.name)
    if not scene_paths:
        raise RuntimeError(f"No prepared scene USD files found under {scenes_root}")

    context = omni.usd.get_context()
    results = []
    for scene_path in scene_paths:
        try:
            row = validate_scene(context, scene_path.resolve(), hdri_path)
        except BaseException as exc:
            traceback.print_exc()
            row = {"scene": scene_path.parent.name, "usd": str(scene_path), "valid": False, "error": repr(exc)}
        results.append(row)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"valid": all(item["valid"] for item in results), "results": results}, indent=2), encoding="utf-8")
        print(f"[hdri-verify] scene={row['scene']} valid={row['valid']}", flush=True)

    payload = {"valid": all(row["valid"] for row in results), "count": len(results), "hdri": str(hdri_path), "results": results}
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[complete] valid={payload['valid']} count={payload['count']} report={output_path}", flush=True)
    return 0 if payload["valid"] else 1


try:
    raise SystemExit(main())
finally:
    APP.close()
