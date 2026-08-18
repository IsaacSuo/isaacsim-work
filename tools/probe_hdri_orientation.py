"""Render a bare DomeLight toward horizon, zenith, and nadir to diagnose HDRI orientation."""

import argparse
import asyncio
import json
import os
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--hdri", default=r"Y:\scenes\HDRI\bambanani_sunset_8k.exr")
parser.add_argument("--intensity", type=float, default=1000.0)
parser.add_argument("--exposure", type=float, default=-1.0)
parser.add_argument("--size", type=int, default=512)
parser.add_argument("--output", default=r"Y:\isaacsim_work\output\hdri_orientation_probe")
args = parser.parse_args()

output = Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp

app = SimulationApp(
    {
        "headless": True,
        "renderer": "RaytracedLighting",
        "width": args.size,
        "height": args.size,
    }
)

import carb
import omni.usd
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from pxr import Gf, Sdf, UsdGeom, UsdLux


def set_look_at(prim, eye, target, up):
    matrix = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up)
    ).GetInverse()
    xformable = UsdGeom.Xformable(prim)
    ops = [
        op
        for op in xformable.GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeTransform
    ]
    (ops[0] if ops else xformable.AddTransformOp()).Set(matrix)


def capture(viewport, path):
    if path.exists():
        path.unlink()
    request = capture_viewport_to_file(viewport, file_path=str(path))
    task = asyncio.ensure_future(request.wait_for_result(completion_frames=2))
    for _ in range(600):
        app.update()
        if task.done():
            break
    if not task.done() or not task.result():
        raise RuntimeError(f"Capture failed: {path}")


try:
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
    dome.CreateTextureFileAttr(Sdf.AssetPath(Path(args.hdri).resolve().as_posix()))
    dome.CreateTextureFormatAttr("latlong")
    dome.CreateIntensityAttr(args.intensity)
    rotate = UsdGeom.Xformable(dome.GetPrim()).AddRotateXOp()

    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateFocalLengthAttr(18.0)
    camera.CreateHorizontalApertureAttr(20.955)

    viewport = get_active_viewport()
    viewport.camera_path = camera.GetPath()
    viewport.set_texture_resolution((args.size, args.size))
    carb.settings.get_settings().set("/rtx/post/tonemap/exposure", args.exposure)
    carb.settings.get_settings().set("/persistent/app/viewport/displayOptions", 0)

    directions = {
        "horizon_pos_z": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        "zenith_pos_y": ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
        "nadir_neg_y": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    }
    files = []
    for x_degrees in (0.0, -90.0, 180.0):
        rotate.Set(x_degrees)
        for _ in range(8):
            app.update()
        for name, (target, up) in directions.items():
            set_look_at(camera.GetPrim(), (0.0, 0.0, 0.0), target, up)
            for _ in range(4):
                app.update()
            path = output / f"x{x_degrees:03.0f}_{name}.png"
            capture(viewport, path)
            files.append(path.name)
            print(f"[capture] {path}", flush=True)

    report = {
        "hdri": str(Path(args.hdri).resolve()),
        "intensity": args.intensity,
        "exposure": args.exposure,
        "stage_up_axis": "Y",
        "files": files,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
finally:
    app.close()
