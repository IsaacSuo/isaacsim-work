"""Render a cached deformable video as a closed, thick OmniGlass volume."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from fixed_topology_video import job_path, load_frame, load_job, load_topology
from liquid_video_cache import read_jsonl, validate_png

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--job", required=True)
parser.add_argument("--spp", type=int, default=64)
parser.add_argument("--max-bounces", type=int, default=16)
parser.add_argument("--ior", type=float, default=1.41)
parser.add_argument("--roughness", type=float, default=0.04)
parser.add_argument("--absorption", type=float, default=0.005)
parser.add_argument("--settle-updates", type=int, default=6)
args = parser.parse_args()

job_dir = Path(args.job).resolve()
job = load_job(job_dir)
video, simulation = job["video"], job["simulation"]
width, height = int(video["width"]), int(video["height"])
frame_count = int(video["output_frames"])
fps = int(video["output_fps"])
topology = load_topology(job_path(job_dir, job, "topology"))
triangles = np.asarray(topology["face_vertex_indices"], dtype=np.int64).reshape(-1, 3)
vertex_count = int(topology["metadata"]["vertex_count"])
rows = {
    int(row["output_index"]): row
    for row in read_jsonl(job_path(job_dir, job, "simulation_manifest"))
}
if sorted(rows) != list(range(frame_count)):
    raise RuntimeError("Simulation cache is incomplete or non-contiguous")

profile = (
    f"thick_omniglass_{width}x{height}_{fps}fps_{args.spp}spp_"
    f"ior{args.ior:.3f}_r{args.roughness:.3f}_a{args.absorption:.4f}"
)
output_dir = job_dir / "previews" / profile
frames_dir = output_dir / "frames"
frames_dir.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp

app = SimulationApp({
    "headless": True,
    "renderer": "PathTracing",
    "width": width,
    "height": height,
    "samples_per_pixel_per_frame": args.spp,
    "max_bounces": args.max_bounces,
    "max_specular_transmission_bounces": args.max_bounces,
    "max_volume_bounces": args.max_bounces,
    "denoiser": True,
})

import carb
import omni.usd
from omni.kit.material.library import CreateAndBindMdlMaterialFromLibrary
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt


def update(count: int) -> None:
    for _ in range(max(0, count)):
        app.update()


def rebuild_normals(mesh: UsdGeom.Mesh, points: np.ndarray) -> None:
    face_normals = np.cross(
        points[triangles[:, 1]] - points[triangles[:, 0]],
        points[triangles[:, 2]] - points[triangles[:, 0]],
    )
    normals = np.zeros_like(points, dtype=np.float64)
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1e-12):
        raise RuntimeError("Cannot rebuild normals for degenerate mesh vertices")
    normals = (normals / lengths[:, None]).astype(np.float32)
    mesh.CreateNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)


def capture(viewport, path: Path) -> None:
    temporary = path.with_name(f".{path.stem}.part.png")
    temporary.unlink(missing_ok=True)
    request = capture_viewport_to_file(viewport, file_path=str(temporary))
    task = asyncio.ensure_future(request.wait_for_result(completion_frames=2))
    for _ in range(1200):
        app.update()
        if task.done():
            break
    if not task.done() or not task.result():
        raise RuntimeError(f"Capture failed: {path}")
    for _ in range(1200):
        if temporary.is_file() and temporary.stat().st_size:
            break
        app.update()
    validate_png(temporary, width=width, height=height)
    os.replace(temporary, path)


try:
    context = omni.usd.get_context()
    if not context.open_stage(str(job_path(job_dir, job, "render_template"))):
        raise RuntimeError("Cannot open render template")
    update(6)
    stage = context.get_stage()
    visual = UsdGeom.Mesh.Get(stage, Sdf.Path(simulation["visual_mesh_prim"]))
    root = stage.GetPrimAtPath(Sdf.Path(simulation["dynamic_root_prim"]))
    camera = UsdGeom.Camera.Get(stage, Sdf.Path(simulation["camera_prim"]))
    if not visual or not root or not camera:
        raise RuntimeError("Render template lacks the visual mesh, root, or camera")
    visual.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(topology["face_vertex_counts"]))
    visual.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(topology["face_vertex_indices"]))
    visual.CreateDoubleSidedAttr().Set(False)
    for prim_path in simulation.get("hidden_prims", []):
        prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
        if prim:
            UsdGeom.Imageable(prim).MakeInvisible()

    CreateAndBindMdlMaterialFromLibrary(
        mdl_name="OmniGlass.mdl",
        mtl_name="OmniGlass",
        bind_selected_prims=False,
        prim_name="ThickSiliconeGlass",
    ).do()
    material_path = Sdf.Path("/World/Looks/ThickSiliconeGlass")
    material = UsdShade.Material.Get(stage, material_path)
    shader = UsdShade.Shader.Get(stage, material_path.AppendChild("Shader"))
    if not material or not shader:
        raise RuntimeError("OmniGlass material was not created")
    shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.97, 0.99, 1.0))
    shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(args.ior)
    shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(args.roughness)
    shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(False)
    shader.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(args.absorption)
    UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(material)

    stripe_materials = []
    for name, color in (("RefWhite", (0.92, 0.94, 0.96)), ("RefDark", (0.025, 0.035, 0.05))):
        stripe_material = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
        stripe_shader = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/PreviewSurface")
        stripe_shader.CreateIdAttr("UsdPreviewSurface")
        stripe_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        stripe_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        stripe_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        stripe_material.CreateSurfaceOutput().ConnectToSource(stripe_shader.ConnectableAPI(), "surface")
        stripe_materials.append(stripe_material)
    for index in range(13):
        stripe = UsdGeom.Cube.Define(stage, f"/World/RefractionStripes/S{index:02d}")
        stripe.CreateSizeAttr(1.0)
        stripe_xform = UsdGeom.Xformable(stripe)
        stripe_xform.AddTranslateOp().Set(Gf.Vec3d((index - 6) * 0.62, 2.7, -3.02))
        stripe_xform.AddScaleOp().Set(Gf.Vec3f(0.31, 3.8, 0.025))
        UsdShade.MaterialBindingAPI.Apply(stripe.GetPrim()).Bind(stripe_materials[index % 2])

    backdrop = UsdShade.Shader.Get(stage, Sdf.Path("/World/Looks/WarmBackdrop/PreviewSurface"))
    floor = UsdShade.Shader.Get(stage, Sdf.Path("/World/Looks/WarmStudio/PreviewSurface"))
    backdrop.GetInput("diffuseColor").Set(Gf.Vec3f(0.78, 0.81, 0.84))
    floor.GetInput("diffuseColor").Set(Gf.Vec3f(0.56, 0.59, 0.62))
    ambient = stage.GetPrimAtPath(Sdf.Path("/World/Lights/Ambient"))
    ambient.GetAttribute("inputs:intensity").Set(1700.0)
    ambient.GetAttribute("inputs:color").Set(Gf.Vec3f(0.9, 0.95, 1.0))

    camera.GetPrim().CreateAttribute("omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool).Set(False)
    viewport = get_active_viewport()
    viewport.camera_path = camera.GetPath()
    viewport.set_texture_resolution((width, height))
    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "PathTracing")
    settings.set("/rtx/pathtracing/spp", args.spp)
    settings.set("/rtx/pathtracing/totalSpp", args.spp)
    settings.set("/rtx/pathtracing/clampSpp", args.spp)
    settings.set("/rtx/pathtracing/maxBounces", args.max_bounces)
    settings.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", args.max_bounces)
    settings.set("/rtx/pathtracing/maxVolumeBounces", args.max_bounces)
    settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
    settings.set("/rtx/post/tonemap/exposure", 0.0)
    update(24)

    for index in range(frame_count):
        row = rows[index]
        cached = load_frame(
            job_path(job_dir, job, "cache_dir") / row["cache_file"],
            expected_sha256=row["cache_sha256"],
            expected_vertex_count=vertex_count,
        )
        points = np.asarray(cached["points"], dtype=np.float32)
        visual.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
        rebuild_normals(visual, points.astype(np.float64))
        visual.GetExtentAttr().Set(Vt.Vec3fArray([
            Gf.Vec3f(*(float(v) for v in cached["bounds_min"])),
            Gf.Vec3f(*(float(v) for v in cached["bounds_max"])),
        ]))
        root.GetAttribute("xformOp:translate").Set(
            Gf.Vec3d(*(float(v) for v in cached["translate"]))
        )
        update(48 if index == 0 else args.settle_updates)
        output = frames_dir / f"rgb_{index:06d}.png"
        capture(viewport, output)
        if index % 15 == 0 or index == frame_count - 1:
            print(f"[thick-glass-video] {index + 1}/{frame_count}", flush=True)

    metadata = {
        "job": str(job_dir),
        "profile": profile,
        "frame_count": frame_count,
        "fps": fps,
        "resolution": [width, height],
        "material": {
            "mdl": "OmniGlass.mdl::OmniGlass",
            "thin_walled": False,
            "ior": args.ior,
            "roughness": args.roughness,
            "volume_absorption": args.absorption,
        },
        "path_tracing": {
            "spp": args.spp,
            "max_bounces": args.max_bounces,
            "max_specular_transmission_bounces": args.max_bounces,
            "max_volume_bounces": args.max_bounces,
            "denoiser": True,
            "auto_exposure": False,
        },
        "surface": {
            "closed_volume": True,
            "normals": "area_weighted_vertex_normals_recomputed_per_frame",
        },
    }
    (output_dir / "render.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
finally:
    app.close()
