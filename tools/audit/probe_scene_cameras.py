"""Render an azimuth ring of camera probes from an existing hero USD stage."""

import argparse
import asyncio
import math
import os
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--spawn-x", required=True, type=float)
    parser.add_argument("--spawn-z", required=True, type=float)
    parser.add_argument("--support-y", required=True, type=float)
    parser.add_argument("--radius", type=float, default=5.5)
    parser.add_argument("--angles", type=int, default=8)
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=400)
    return parser.parse_args()


ARGS = parse_args()
OUTPUT = Path(ARGS.output)
OUTPUT.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {"headless": True, "renderer": "RaytracedLighting", "width": ARGS.width, "height": ARGS.height}
)

import carb
import omni.usd
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from pxr import Gf, UsdGeom


def set_look_at(prim, eye, target):
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 1, 0)).GetInverse()
    xformable.AddTransformOp().Set(matrix)


def capture(viewport, path):
    if path.exists():
        path.unlink()
    request = capture_viewport_to_file(viewport, file_path=str(path))
    task = asyncio.ensure_future(request.wait_for_result(completion_frames=2))
    for _ in range(600):
        simulation_app.update()
        if task.done():
            break
    if not task.done() or not task.result():
        raise RuntimeError(f"Capture failed: {path}")
    for _ in range(600):
        if path.is_file() and path.stat().st_size:
            return
        simulation_app.update()
    raise RuntimeError(f"Capture file missing: {path}")


try:
    context = omni.usd.get_context()
    if not context.open_stage(ARGS.usd):
        raise RuntimeError(f"Could not open {ARGS.usd}")
    for _ in range(24):
        simulation_app.update()
    stage = context.get_stage()
    camera = stage.GetPrimAtPath("/World/HeroCamera")
    if not camera or not camera.IsValid():
        raise RuntimeError("/World/HeroCamera not found")
    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "RaytracedLighting")
    settings.set("/persistent/app/viewport/displayOptions", 0)
    settings.set("/rtx/post/tonemap/exposure", 1.0)
    viewport = get_active_viewport()
    viewport.camera_path = camera.GetPath()
    viewport.set_texture_resolution((ARGS.width, ARGS.height))
    target = (ARGS.spawn_x, ARGS.support_y + 0.9, ARGS.spawn_z)
    for ring, eye_y in (("low", ARGS.support_y + 1.9), ("high", ARGS.support_y + 3.0)):
        for index in range(ARGS.angles):
            angle = 2.0 * math.pi * index / ARGS.angles
            eye = (
                ARGS.spawn_x + ARGS.radius * math.cos(angle),
                eye_y,
                ARGS.spawn_z + ARGS.radius * math.sin(angle),
            )
            set_look_at(camera, eye, target)
            for _ in range(8):
                simulation_app.update()
            path = OUTPUT / f"{ring}_{index:02d}_{math.degrees(angle):03.0f}.png"
            capture(viewport, path)
            print(f"[probe] {path} eye={eye} target={target}", flush=True)
finally:
    simulation_app.close()
