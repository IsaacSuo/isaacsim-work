"""Unmodified NVIDIA ParticlePostProcessingDemo, headless keyframe capture."""

import argparse
import asyncio
import os

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=121)
parser.add_argument("--capture-every", type=int, default=20)
parser.add_argument("--capture-last-only", action="store_true")
parser.add_argument("--capture-material-pair", action="store_true")
parser.add_argument("--export-particles", action="store_true")
parser.add_argument("--external-surface", default="")
parser.add_argument(
    "--isosurface-settle-frames",
    type=int,
    default=120,
    help="Render updates allowed for the asynchronous PhysX isosurface to settle.",
)
parser.add_argument(
    "--material-settle-frames",
    type=int,
    default=8,
    help="Render updates after changing a material, without advancing physics.",
)
parser.add_argument(
    "--capture-timeout-updates",
    type=int,
    default=600,
    help="Maximum app updates to wait for one viewport capture.",
)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=360)
parser.add_argument("--output", default=r"Y:\isaacsim_work\output\official_baseline")
parser.add_argument(
    "--renderer",
    choices=["RaytracedLighting", "PathTracing"],
    default="RaytracedLighting",
)
parser.add_argument("--path-spp", type=int, default=32)
parser.add_argument("--hq-surface", action="store_true")
parser.add_argument(
    "--hq-exclude",
    action="append",
    choices=[
        "smoothing",
        "anisotropy",
        "surface-distance",
        "grid-spacing",
        "mesh-smoothing",
        "mesh-budget",
    ],
    default=[],
)
parser.add_argument(
    "--water-material", choices=["glass", "demo"], default="glass"
)
parser.add_argument("--thin-walled-water", action="store_true")
parser.add_argument("--grid-filtering-passes", default="")
parser.add_argument(
    "--surface-test",
    choices=[
        "smoothing",
        "anisotropy",
        "surface-distance",
        "grid-spacing",
        "mesh-smoothing",
        "mesh-budget",
    ],
)
args = parser.parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": True,
        "renderer": args.renderer,
        "width": args.width,
        "height": args.height,
    }
)

import carb
import omni.kit.app
import omni.kit.commands
import omni.usd
from omni.kit.material.library import CreateAndBindMdlMaterialFromLibrary
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from omni.physx import get_physx_simulation_interface
from omni.physx.scripts import particleUtils
from pxr import Gf, Sdf, UsdGeom, UsdShade, UsdUtils
import numpy as np


def set_camera(camera, eye, target):
    matrix = Gf.Matrix4d().SetLookAt(
        eye, target, Gf.Vec3d(0, 1, 0)
    ).GetInverse()
    UsdGeom.Xformable(camera).AddTransformOp().Set(matrix)


extension_manager = omni.kit.app.get_app().get_extension_manager()
extension_manager.set_extension_enabled_immediate("omni.physx.demos", True)

from omni.physxdemos.scenes.ParticlePostProcessingDemo import ParticlePostProcessingDemo

os.makedirs(args.output, exist_ok=True)
context = omni.usd.get_context()
context.new_stage()
stage = context.get_stage()
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())

# This call is the complete, unmodified NVIDIA sample scene.
fluid_demo = ParticlePostProcessingDemo()
fluid_demo.create(stage, Anisotropy=True, Smoothing=True, Isosurface=True)

# The particle spheres are only a debug representation. Rendering them together
# with the generated isosurface causes the visible bead-like surface.
particle_prototype = UsdGeom.Imageable.Get(
    stage, "/World/particles/particlePrototype0"
)
if particle_prototype:
    particle_prototype.MakeInvisible()

# The four tank walls remain physical colliders but are never rendered.
for wall_path in (
    "/World/box/front",
    "/World/box/right",
    "/World/box/back",
    "/World/box/left",
):
    wall = UsdGeom.Imageable.Get(stage, wall_path)
    if wall:
        wall.MakeInvisible()

# Optional, independently testable render-only surface reconstruction.
particle_system_path = Sdf.Path("/World/particleSystem")
particle_system_prim = stage.GetPrimAtPath(particle_system_path)

external_surface_path = Sdf.Path("/World/SplashSurfSurface")
if args.external_surface:
    surface_data = np.load(args.external_surface)
    surface_vertices = np.asarray(surface_data["vertices"], dtype=np.float32)
    surface_triangles = np.asarray(surface_data["triangles"], dtype=np.int32)
    splashsurf_mesh = UsdGeom.Mesh.Define(stage, external_surface_path)
    splashsurf_mesh.CreatePointsAttr(surface_vertices.tolist())
    splashsurf_mesh.CreateFaceVertexCountsAttr(
        np.full(len(surface_triangles), 3, dtype=np.int32).tolist()
    )
    splashsurf_mesh.CreateFaceVertexIndicesAttr(
        surface_triangles.reshape(-1).tolist()
    )
    splashsurf_mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    if "normals" in surface_data:
        surface_normals = np.asarray(surface_data["normals"], dtype=np.float32)
        splashsurf_mesh.CreateNormalsAttr(surface_normals.tolist())
        splashsurf_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    UsdGeom.Imageable(particle_system_prim).MakeInvisible()
    print(
        f"[splashsurf] loaded vertices={len(surface_vertices)}, "
        f"triangles={len(surface_triangles)}"
    )

if args.hq_surface or args.surface_test == "smoothing":
    particleUtils.add_physx_particle_smoothing(
        stage, particle_system_path, enabled=True, strength=0.7
    )

if args.hq_surface or args.surface_test == "anisotropy":
    particleUtils.add_physx_particle_anisotropy(
        stage, particle_system_path, enabled=True, scale=5.0, min=1.0, max=2.0
    )

if args.hq_surface:
    particleUtils.add_physx_particle_isosurface(
        stage,
        particle_system_path,
        enabled=True,
        grid_spacing=0.25,
        surface_distance=0.9,
        num_mesh_smoothing_passes=3,
        num_mesh_normal_smoothing_passes=4,
        max_vertices=1_048_576,
        max_triangles=1_048_576,
        max_subgrids=4096,
    )
    hq_attribute_groups = {
        "smoothing": [
            "physxParticleSmoothing:particleSmoothingEnabled",
            "physxParticleSmoothing:strength",
        ],
        "anisotropy": [
            "physxParticleAnisotropy:particleAnisotropyEnabled",
            "physxParticleAnisotropy:scale",
            "physxParticleAnisotropy:min",
            "physxParticleAnisotropy:max",
        ],
        "surface-distance": [
            "physxParticleIsosurface:surfaceDistance",
        ],
        "grid-spacing": [
            "physxParticleIsosurface:gridSpacing",
        ],
        "mesh-smoothing": [
            "physxParticleIsosurface:numMeshSmoothingPasses",
            "physxParticleIsosurface:numMeshNormalSmoothingPasses",
        ],
        "mesh-budget": [
            "physxParticleIsosurface:maxVertices",
            "physxParticleIsosurface:maxTriangles",
            "physxParticleIsosurface:maxSubgrids",
        ],
    }
    for excluded_group in args.hq_exclude:
        for attribute_name in hq_attribute_groups[excluded_group]:
            particle_system_prim.RemoveProperty(attribute_name)
else:
    if args.surface_test == "grid-spacing":
        particle_system_prim.CreateAttribute(
            "physxParticleIsosurface:gridSpacing", Sdf.ValueTypeNames.Float
        ).Set(0.25)
    elif args.surface_test == "surface-distance":
        particle_system_prim.CreateAttribute(
            "physxParticleIsosurface:surfaceDistance",
            Sdf.ValueTypeNames.Float,
        ).Set(0.9)
    elif args.surface_test == "mesh-smoothing":
        particle_system_prim.CreateAttribute(
            "physxParticleIsosurface:numMeshSmoothingPasses",
            Sdf.ValueTypeNames.Int,
        ).Set(3)
        particle_system_prim.CreateAttribute(
            "physxParticleIsosurface:numMeshNormalSmoothingPasses",
            Sdf.ValueTypeNames.Int,
        ).Set(4)
    elif args.surface_test == "mesh-budget":
        particle_system_prim.CreateAttribute(
            "physxParticleIsosurface:maxVertices", Sdf.ValueTypeNames.Int
        ).Set(1_048_576)
        particle_system_prim.CreateAttribute(
            "physxParticleIsosurface:maxTriangles", Sdf.ValueTypeNames.Int
        ).Set(1_048_576)
        particle_system_prim.CreateAttribute(
            "physxParticleIsosurface:maxSubgrids", Sdf.ValueTypeNames.Int
        ).Set(4096)

# Match the known-good retry4 configuration exactly: neither filtering
# attribute is authored after the helper has finished.
particle_system_prim.RemoveProperty(
    "physxParticleIsosurface:gridFilteringPasses"
)
particle_system_prim.RemoveProperty(
    "physxParticleIsosurface:gridSmoothingRadius"
)
if args.grid_filtering_passes:
    particle_system_prim.CreateAttribute(
        "physxParticleIsosurface:gridFilteringPasses",
        Sdf.ValueTypeNames.String,
    ).Set(args.grid_filtering_passes)
    particle_system_prim.CreateAttribute(
        "physxParticleIsosurface:gridSmoothingRadius",
        Sdf.ValueTypeNames.Float,
    ).Set(1.0)

# Optional transmissive water; "demo" leaves NVIDIA's original material untouched.
if args.water_material == "glass":
    CreateAndBindMdlMaterialFromLibrary(
        mdl_name="OmniGlass.mdl",
        mtl_name="OmniGlass",
        bind_selected_prims=False,
        prim_name="WaterMaterial",
    ).do()
    water_material_path = Sdf.Path("/World/Looks/WaterMaterial")
    water_shader = UsdShade.Shader.Get(
        stage, water_material_path.AppendChild("Shader")
    )
    water_shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.93, 0.98, 1.0)
    )
    water_shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.333)
    water_shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(
        0.015
    )
    water_shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(
        args.thin_walled_water
    )
    water_shader.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(0.002)
    omni.kit.commands.execute(
        "BindMaterialCommand",
        prim_path=(
            external_surface_path if args.external_surface else particle_system_path
        ),
        material_path=water_material_path,
        strength=None,
    )

settings = carb.settings.get_settings()
settings.set("/rtx/translucency/maxRefractionBounces", 12)
settings.set("/rtx/reflections/enabled", True)
settings.set("/rtx/indirectDiffuse/enabled", True)
if args.renderer == "PathTracing":
    settings.set("/rtx/pathtracing/spp", args.path_spp)
    settings.set("/rtx/pathtracing/totalSpp", args.path_spp)
    settings.set("/rtx/pathtracing/maxBounces", 12)

camera = UsdGeom.Camera.Define(stage, "/World/RenderCamera")
camera.CreateFocalLengthAttr(45.0)
camera.CreateHorizontalApertureAttr(20.955)
camera.CreateClippingRangeAttr(Gf.Vec2f(1.0, 100000.0))
set_camera(
    camera.GetPrim(),
    Gf.Vec3d(190.0, 125.0, 220.0),
    Gf.Vec3d(0.0, 20.0, 0.0),
)
viewport = get_active_viewport()
viewport.camera_path = camera.GetPath()
viewport.set_texture_resolution((args.width, args.height))


def run_render_updates(count):
    """Advance Kit/rendering only; never advance the PhysX simulation."""
    for _ in range(max(0, count)):
        simulation_app.update()


def capture_current_viewport(file_path):
    """Capture a fresh frame and wait for the capture request itself to finish."""
    # A pre-existing image is not evidence that this capture completed. Removing
    # it also prevents a rerun in the same output directory from racing ahead.
    if os.path.exists(file_path):
        os.remove(file_path)

    capture = capture_viewport_to_file(viewport, file_path=file_path)
    capture_task = asyncio.ensure_future(
        capture.wait_for_result(completion_frames=2)
    )
    for _ in range(args.capture_timeout_updates):
        simulation_app.update()
        if capture_task.done():
            break
    if not capture_task.done():
        capture_task.cancel()
        raise RuntimeError(f"Timed out waiting for viewport capture: {file_path}")

    captured_aovs = capture_task.result()
    if not captured_aovs:
        raise RuntimeError(f"Viewport capture returned no AOVs: {file_path}")

    # The capture helper completes after submitting the renderer file write.
    # Wait for that new file to become visible and non-empty as a final guard.
    for _ in range(args.capture_timeout_updates):
        if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
            print(
                f"[capture] renderer_frame={capture.frame_number}, "
                f"path={file_path}"
            )
            return
        simulation_app.update()
    raise RuntimeError(f"Timed out waiting for captured file: {file_path}")


stage.GetRootLayer().Export(os.path.join(args.output, "official_baseline.usda"))
simulation_app.update()
simulation_app.update()

simulation = get_physx_simulation_interface()
stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
simulation.attach_stage(stage_id)

capture_index = 0
for frame in range(args.frames):
    simulation.simulate(1.0 / 60.0, frame / 60.0)
    simulation.fetch_results()
    simulation_app.update()
    should_capture = (
        frame == args.frames - 1
        if args.capture_last_only
        else frame % args.capture_every == 0 or frame == args.frames - 1
    )
    if should_capture:
        if args.export_particles:
            particle_instancer = UsdGeom.PointInstancer.Get(
                stage, "/World/particles"
            )
            particle_positions = np.asarray(
                particle_instancer.GetPositionsAttr().Get(), dtype=np.float32
            )
            particle_radius = float(
                UsdGeom.Sphere.Get(
                    stage, "/World/particles/particlePrototype0"
                ).GetRadiusAttr().Get()
            )
            particle_path = os.path.join(
                args.output, f"particles_{capture_index:04d}.npz"
            )
            np.savez_compressed(
                particle_path,
                positions=particle_positions,
                particle_radius=np.float32(particle_radius),
            )
            print(
                f"[particles] exported count={len(particle_positions)}, "
                f"radius={particle_radius}, path={particle_path}"
            )
        # PhysX updates the generated isosurface asynchronously. Let it finish
        # before the first material capture; these are render updates only.
        run_render_updates(args.isosurface_settle_frames)
        frame_path = os.path.join(args.output, f"rgb_{capture_index:04d}.png")
        capture_current_viewport(frame_path)
        if args.capture_material_pair and args.water_material == "glass":
            render_prim_path = (
                external_surface_path
                if args.external_surface
                else particle_system_path
            )
            try:
                omni.kit.commands.execute(
                    "BindMaterialCommand",
                    prim_path=render_prim_path,
                    material_path=Sdf.Path("/World/Looks/OmniPBR"),
                    strength=None,
                )
                run_render_updates(args.material_settle_frames)
                opaque_path = os.path.join(
                    args.output, f"rgb_{capture_index:04d}_opaque.png"
                )
                capture_current_viewport(opaque_path)
            finally:
                # Keep later keyframes transparent. Previously every capture
                # after the first material pair accidentally remained opaque.
                omni.kit.commands.execute(
                    "BindMaterialCommand",
                    prim_path=render_prim_path,
                    material_path=water_material_path,
                    strength=None,
                )
                run_render_updates(args.material_settle_frames)
        capture_index += 1
    if frame % 30 == 0:
        print(f"[official-baseline] frame {frame}/{args.frames}")

simulation.detach_stage()
fluid_demo.on_shutdown()
simulation_app.close()
