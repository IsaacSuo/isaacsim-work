"""Render one cached soft-body frame with rebuilt normals and an OmniGlass control sphere."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from fixed_topology_video import job_path, load_frame, load_job, load_topology
from liquid_video_cache import read_jsonl, validate_png

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--job", required=True)
parser.add_argument("--frame", type=int, default=60)
parser.add_argument("--spp", type=int, default=64)
args = parser.parse_args()

job_dir = Path(args.job).resolve()
job = load_job(job_dir)
video, simulation = job["video"], job["simulation"]
width, height = int(video["width"]), int(video["height"])
topology = load_topology(job_path(job_dir, job, "topology"))
triangles = np.asarray(topology["face_vertex_indices"], dtype=np.int64).reshape(-1, 3)
rows = {int(row["output_index"]): row for row in read_jsonl(job_path(job_dir, job, "simulation_manifest"))}
row = rows[args.frame]
cached = load_frame(
    job_path(job_dir, job, "cache_dir") / row["cache_file"],
    expected_sha256=row["cache_sha256"],
    expected_vertex_count=int(topology["metadata"]["vertex_count"]),
)
points = np.asarray(cached["points"], dtype=np.float64)
face_normals = np.cross(
    points[triangles[:, 1]] - points[triangles[:, 0]],
    points[triangles[:, 2]] - points[triangles[:, 0]],
)
normals = np.zeros_like(points)
for corner in range(3):
    np.add.at(normals, triangles[:, corner], face_normals)
lengths = np.linalg.norm(normals, axis=1)
if np.any(lengths <= 1e-12):
    raise RuntimeError("Degenerate vertex normal")
normals = (normals / lengths[:, None]).astype(np.float32)

from isaacsim import SimulationApp

app = SimulationApp({
    "headless": True,
    "renderer": "PathTracing",
    "width": width,
    "height": height,
    "samples_per_pixel_per_frame": args.spp,
    "max_bounces": 16,
    "max_specular_transmission_bounces": 16,
    "max_volume_bounces": 16,
    "denoiser": True,
})

import carb
import omni.usd
from omni.kit.material.library import CreateAndBindMdlMaterialFromLibrary
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt


def update(count):
    for _ in range(count):
        app.update()


try:
    context = omni.usd.get_context()
    if not context.open_stage(str(job_path(job_dir, job, "render_template"))):
        raise RuntimeError("Cannot open render template")
    update(6)
    stage = context.get_stage()
    visual = UsdGeom.Mesh.Get(stage, Sdf.Path(simulation["visual_mesh_prim"]))
    root = stage.GetPrimAtPath(Sdf.Path(simulation["dynamic_root_prim"]))
    camera = UsdGeom.Camera.Get(stage, Sdf.Path(simulation["camera_prim"]))
    visual.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(topology["face_vertex_counts"]))
    visual.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(topology["face_vertex_indices"]))
    visual.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    visual.CreateNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals))
    visual.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    visual.CreateDoubleSidedAttr().Set(False)
    visual.GetExtentAttr().Set(Vt.Vec3fArray([
        Gf.Vec3f(*(float(v) for v in cached["bounds_min"])),
        Gf.Vec3f(*(float(v) for v in cached["bounds_max"])),
    ]))
    root.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*(float(v) for v in cached["translate"])))
    for prim_path in simulation.get("hidden_prims", []):
        prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
        if prim:
            UsdGeom.Imageable(prim).MakeInvisible()

    CreateAndBindMdlMaterialFromLibrary(
        mdl_name="OmniGlass.mdl", mtl_name="OmniGlass",
        bind_selected_prims=False, prim_name="OmniGlassSiliconeCheck",
    ).do()
    material_path = Sdf.Path("/World/Looks/OmniGlassSiliconeCheck")
    material = UsdShade.Material.Get(stage, material_path)
    shader = UsdShade.Shader.Get(stage, material_path.AppendChild("Shader"))
    shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.97, 0.995, 1.0))
    shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.41)
    shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(0.035)
    shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(False)
    shader.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(2.0)
    UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(material)

    sphere = UsdGeom.Sphere.Define(stage, "/World/OmniGlassReferenceSphere")
    sphere.CreateRadiusAttr().Set(0.42)
    UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(-2.25, 1.1, 0.25))
    UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)

    camera.GetPrim().CreateAttribute("omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool).Set(False)
    viewport = get_active_viewport()
    viewport.camera_path = camera.GetPath()
    viewport.set_texture_resolution((width, height))
    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "PathTracing")
    settings.set("/rtx/pathtracing/spp", args.spp)
    settings.set("/rtx/pathtracing/totalSpp", args.spp)
    settings.set("/rtx/pathtracing/clampSpp", args.spp)
    settings.set("/rtx/pathtracing/maxBounces", 16)
    settings.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 16)
    settings.set("/rtx/pathtracing/maxVolumeBounces", 16)
    settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
    settings.set("/rtx/post/tonemap/exposure", 0.7)
    update(64)

    output = job_dir / "previews" / "omniglass_normals_check_f000060.png"
    output.unlink(missing_ok=True)
    request = capture_viewport_to_file(viewport, file_path=str(output))
    task = asyncio.ensure_future(request.wait_for_result(completion_frames=2))
    for _ in range(1200):
        app.update()
        if task.done():
            break
    if not task.done() or not task.result():
        raise RuntimeError("Capture failed")
    for _ in range(1200):
        if output.is_file() and output.stat().st_size:
            break
        app.update()
    validate_png(output, width=width, height=height)
    print(output, flush=True)
finally:
    app.close()
