"""Render selected cached deformable frames with an OmniGlass silicone preset."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from fixed_topology_video import job_path, load_frame, load_job, load_topology
from liquid_video_cache import read_jsonl, validate_png


os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--job", required=True)
parser.add_argument("--frames", default="0,36,60,120")
parser.add_argument("--spp", type=int, default=64)
parser.add_argument("--max-bounces", type=int, default=16)
parser.add_argument("--roughness", type=float, default=0.06)
parser.add_argument("--ior", type=float, default=1.41)
parser.add_argument("--depth", type=float, default=0.8)
parser.add_argument("--settle-updates", type=int, default=48)
args = parser.parse_args()

job_dir = Path(args.job).resolve()
job = load_job(job_dir)
video, simulation = job["video"], job["simulation"]
width, height = int(video["width"]), int(video["height"])
indices = [int(value.strip()) for value in args.frames.split(",") if value.strip()]
if not indices or len(indices) != len(set(indices)):
    raise ValueError("--frames must contain unique comma-separated frame indices")
if min(indices) < 0 or max(indices) >= int(video["output_frames"]):
    raise ValueError("Preview frame index is outside the cached video")

cache_dir = job_path(job_dir, job, "cache_dir")
template_path = job_path(job_dir, job, "render_template")
topology = load_topology(job_path(job_dir, job, "topology"))
vertex_count = int(topology["metadata"]["vertex_count"])
cache_rows = {
    int(row["output_index"]): row
    for row in read_jsonl(job_path(job_dir, job, "simulation_manifest"))
}
output_dir = job_dir / "previews" / (
    f"omniglass_{args.spp}spp_{args.max_bounces}b_ior{args.ior:.3f}_r{args.roughness:.3f}"
)
output_dir.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": True,
        "renderer": "PathTracing",
        "width": width,
        "height": height,
        "samples_per_pixel_per_frame": args.spp,
        "max_bounces": args.max_bounces,
        "max_specular_transmission_bounces": args.max_bounces,
        "max_volume_bounces": args.max_bounces,
        "denoiser": True,
    }
)

import carb
import omni.usd
from omni.kit.material.library import CreateAndBindMdlMaterialFromLibrary
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt


def update(count: int) -> None:
    for _ in range(max(0, count)):
        simulation_app.update()


def capture(viewport, path: Path) -> None:
    path.unlink(missing_ok=True)
    request = capture_viewport_to_file(viewport, file_path=str(path))
    task = asyncio.ensure_future(request.wait_for_result(completion_frames=2))
    for _ in range(1200):
        simulation_app.update()
        if task.done():
            break
    if not task.done() or not task.result():
        raise RuntimeError(f"Capture failed: {path}")
    for _ in range(1200):
        if path.is_file() and path.stat().st_size:
            return
        simulation_app.update()
    raise RuntimeError(f"Capture did not write a file: {path}")


try:
    context = omni.usd.get_context()
    if not context.open_stage(str(template_path)):
        raise RuntimeError(f"Failed to open template: {template_path}")
    update(6)
    stage = context.get_stage()
    visual = UsdGeom.Mesh.Get(stage, Sdf.Path(simulation["visual_mesh_prim"]))
    root_prim = stage.GetPrimAtPath(Sdf.Path(simulation["dynamic_root_prim"]))
    camera = UsdGeom.Camera.Get(stage, Sdf.Path(simulation["camera_prim"]))
    if not visual or not root_prim or not camera:
        raise RuntimeError("Template lacks the cached visual mesh, root, or camera")
    visual.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(topology["face_vertex_counts"]))
    visual.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(topology["face_vertex_indices"]))
    for prim_path in simulation.get("hidden_prims", []):
        prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
        if prim:
            UsdGeom.Imageable(prim).MakeInvisible()

    CreateAndBindMdlMaterialFromLibrary(
        mdl_name="OmniGlass.mdl",
        mtl_name="OmniGlass",
        bind_selected_prims=False,
        prim_name="OmniGlassSilicone",
    ).do()
    material_path = Sdf.Path("/World/Looks/OmniGlassSilicone")
    material = UsdShade.Material.Get(stage, material_path)
    shader = UsdShade.Shader.Get(stage, material_path.AppendChild("Shader"))
    if not material or not shader:
        raise RuntimeError("OmniGlass material was not created")
    shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.93, 0.985, 1.0))
    shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(args.ior)
    shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(args.roughness)
    shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(False)
    shader.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(args.depth)
    UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(material)

    # Stripes behind the object make refraction and thickness immediately visible.
    stripe_scope = UsdGeom.Scope.Define(stage, "/World/OmniGlassPreviewStripes")
    del stripe_scope
    stripe_materials = []
    for name, color in (("StripeLight", (0.88, 0.91, 0.93)), ("StripeDark", (0.035, 0.055, 0.07))):
        mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
        surface = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/PreviewSurface")
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
        stripe_materials.append(mat)
    for stripe_index in range(13):
        stripe = UsdGeom.Cube.Define(stage, f"/World/OmniGlassPreviewStripes/S{stripe_index:02d}")
        stripe.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(stripe)
        xform.AddTranslateOp().Set(Gf.Vec3d((stripe_index - 6) * 0.62, 2.7, -3.02))
        xform.AddScaleOp().Set(Gf.Vec3f(0.31, 3.8, 0.025))
        UsdShade.MaterialBindingAPI.Apply(stripe.GetPrim()).Bind(stripe_materials[stripe_index % 2])

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
    settings.set("/rtx/post/tonemap/exposure", 0.7)
    update(12)

    rendered = []
    for index in indices:
        row = cache_rows[index]
        cached = load_frame(
            cache_dir / str(row["cache_file"]),
            expected_sha256=str(row["cache_sha256"]),
            expected_vertex_count=vertex_count,
        )
        visual.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(cached["points"]))
        visual.GetExtentAttr().Set(Vt.Vec3fArray([
            Gf.Vec3f(*(float(v) for v in cached["bounds_min"])),
            Gf.Vec3f(*(float(v) for v in cached["bounds_max"])),
        ]))
        translate = root_prim.GetAttribute("xformOp:translate")
        if not translate:
            translate = UsdGeom.Xformable(root_prim).AddTranslateOp().GetAttr()
        translate.Set(Gf.Vec3d(*(float(v) for v in cached["translate"])))
        update(args.settle_updates)
        output = output_dir / f"omniglass_{index:06d}.png"
        capture(viewport, output)
        validate_png(output, width=width, height=height)
        rendered.append(str(output))

    metadata = {
        "job": str(job_dir),
        "frames": indices,
        "outputs": rendered,
        "path_tracing": {
            "spp": args.spp,
            "max_bounces": args.max_bounces,
            "max_specular_transmission_bounces": args.max_bounces,
            "max_volume_bounces": args.max_bounces,
            "denoiser": True,
            "auto_exposure": False,
        },
        "material": {
            "mdl": "OmniGlass.mdl::OmniGlass",
            "ior": args.ior,
            "roughness": args.roughness,
            "depth": args.depth,
            "thin_walled": False,
            "glass_color": [0.93, 0.985, 1.0],
        },
    }
    (output_dir / "preview.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
finally:
    simulation_app.close()
